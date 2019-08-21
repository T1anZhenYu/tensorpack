#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: svhn-digit-dorefa.py
# Author: Yuxin Wu

import argparse
import os
import sys
import tensorflow as tf
from imagenet_utils import ImageNetModel, eval_classification, fbresnet_augmentor, get_imagenet_dataflow
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.varreplace import remap_variables

from dorefa_nb import get_dorefa

"""
这个代码是用对角采样计算均值和方差。模型的变动就是用fg代替BN和Activation
To Run:
    ./svhn-digit-dorefa.py --dorefa 1,2,4
"""

BITW = 1
BITA = 2
BITG = 4

class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, 40, 40, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]


    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        fw, fa, quan_bn = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fc' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        image = image / 256.0

        with remap_variables(binarize_weight), \
        argscope(BatchNorm, momentum=0.9, epsilon=1e-4),\
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 48, 5, padding='VALID', use_bias=True)
                      .MaxPooling('pool0', 2, padding='SAME')
                      .apply(activate)
                      # 18
                      .Conv2D('conv1', 64, 3, padding='SAME')
                      .apply(quan_bn,'quan_bn1',is_training)#模型的核心变动，用quan_bn代替bn和activate
                      #.BatchNorm('bn1')
                      #.apply(activate)

                      .Conv2D('conv2', 64, 3, padding='SAME')
                      .MaxPooling('pool1', 2, padding='SAME')
                      .apply(quan_bn,'quan_bn2',training=is_training)#注意，这里要先maxpooling再做量化。
                      #因为原来的模型maxpooling是在bn之后的
                      #.BatchNorm('bn2')
                      #.MaxPooling('pool1', 2, padding='SAME')
                      #.apply(activate)
                      # 9
                      .Conv2D('conv3', 128, 3, padding='VALID')
                      .apply(quan_bn,'quan_bn3',is_training)
                      #.BatchNorm('bn3')
                      #.apply(activate)
                      # 7

                      .Conv2D('conv4', 128, 3, padding='SAME')
                      .apply(quan_bn,'quan_bn4',is_training)
                      #.BatchNorm('bn4')
                      #.apply(activate)

                      .Conv2D('conv5', 128, 3, padding='VALID')
                      .apply(quan_bn,'quan_bn5',is_training)
                      #.BatchNorm('bn5').apply(activate)
                      # 5
                      .Dropout(rate=0.5 if is_training else 0.0)
                      .Conv2D('conv6', 512, 5, padding='VALID')
                      #最后一层不做量化
                      .BatchNorm('bn6')
                      .apply(nonlin)
                      .FullyConnected('fc1', 100)())

        tf.nn.softmax(logits, name='output')
    
        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong-top1')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))
        add_param_summary(('.*/my_bm', ['histogram', 'rms']))
        add_param_summary(('.*/my_bv', ['histogram', 'rms']))
        add_param_summary(('.*/real_bm', ['histogram', 'rms']))
        add_param_summary(('.*/real_bv', ['histogram', 'rms']))
        add_param_summary(('.*/diff_bm', ['histogram', 'rms']))
        add_param_summary(('.*/diff_bv', ['histogram', 'rms']))
        add_param_summary(('.*/ratio_bm', ['histogram', 'rms']))        
        add_param_summary(('.*/ratio_bv', ['histogram', 'rms']))
        add_param_summary(('.*/W', ['histogram', 'rms']))
        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_config():
    logger.set_logger_dir(os.path.join('train_log', 'svhn-dorefa-{}'.format(args.dorefa)))

    # prepare dataset
    d1 = dataset.CifarBase('train',cifar_classnum=100)
    #d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1])
    data_test = dataset.CifarBase('test',cifar_classnum=100)

    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = MultiProcessRunnerZMQ(data_train, 5)

    augmentors = [imgaug.Resize((40, 40))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)

    return TrainConfig(
        data=QueueInput(data_train),
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(10, 0.001), (60, 0.0001), (120, 0.000001)]),
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong-top1')]),
            #DumpTensors(['fg1/batch_mean:0','fg1/batch_var:0','fg1/realbatch_mean:0','fg1/realbatch_var:0'])
        ],
        model=Model(),
        max_epoch=200,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npz (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--dorefa', required=True,
                        help='number of bits for W,A,G, separated by comma. W="t" means TTQ')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if args.eval:
        BATCH_SIZE = 128
        data_test = dataset.SVHNDigit('test')
        augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
        ]
        data_test = AugmentImageComponent(data_test, augmentors)
        data_test = BatchData(data_test, 128, remainder=True)
        eval_classification(Model(), get_model_loader(args.load), data_test)
        sys.exit()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    launch_train_with_config(config, SimpleTrainer())


