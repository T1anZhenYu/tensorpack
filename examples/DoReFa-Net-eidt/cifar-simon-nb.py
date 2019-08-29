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

        fw, fa, fg,quan_bn = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv1' in name or 'fc2' in name:
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

        image = image / 4

        with remap_variables(binarize_weight), \
        argscope(BatchNorm, momentum=0.9, epsilon=1e-4),\
                argscope(Conv2D, use_bias=False,kernel_size=3):
            logits = (LinearWrap(image)
                      .Conv2D('conv1', filters=64)
                      #.BatchNorm('bn0')
                      #.apply(activate)
                      .apply(quan_bn,'quan_bn_1',is_training)
                      # 18
                      .Conv2D('conv2', filters=64)
                      .apply(quan_bn,'quan_bn_2',is_training)
                      #.apply(fg)
                      #.BatchNorm('bn1').apply(activate)
                      .MaxPooling('pool1', 3, stride=2, padding='SAME') 

                      .Conv2D('conv3', filters=128)
                      .apply(fg)
                      .apply(quan_bn,'quan_bn_3',is_training)
                      #.BatchNorm('bn2').apply(activate)
                      # 9
                      .Conv2D('conv4', filters=128)
                      .apply(fg)
                      .apply(quan_bn,'quan_bn_4',is_training)
                      #.BatchNorm('bn3').apply(activate)
                      .MaxPooling('pool2', 3, stride=2, padding='SAME') 
                      # 7

                      .Conv2D('conv5' , filters=128, padding='VALID')
                      .apply(fg)
                      .apply(quan_bn,'quan_bn_5',is_training)
                      #.BatchNorm('bn4').apply(activate)

                      .Conv2D('conv6' , filters=128, padding='VALID')
                      .apply(fg)
                      .apply(quan_bn,'quan_bn_6',is_training)
                      #.BatchNorm('bn5').apply(activate)
                      # 5
                      .FullyConnected('fc0', 1024 + 512)
                      .apply(fg)
                      .BatchNorm('bn6').apply(activate)
                      .Dropout(rate=0.5 if is_training else 0.0)
                      .FullyConnected('fc1', 512) 
                      .apply(fg)
                      .BatchNorm('bn7')
                      .tf.nn.relu()
                      .FullyConnected('fc2', 10)())

        tf.nn.softmax(logits, name='output')
    
        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong-top1')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))

        add_param_summary(('.*/W', ['histogram', 'rms']))
        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)
        tf.summary.scalar('lr', lr)
        return opt

def get_data(train_or_test, dir):
    BATCH_SIZE = 128
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test, dir=dir)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    logger.set_logger_dir(os.path.join('train_log', 'svhn-dorefa-{}'.format(args.dorefa)))

    # prepare dataset
    data_train = get_data('train', dir = './cifar10_data/')
    data_test = get_data('test', dir = './cifar10_data/')

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
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong-top1')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.01), (82, 0.001), (123, 0.0002), (200, 0.0001)]),
            #DumpTensors(['fg1/batch_mean:0','fg1/batch_var:0','fg1/realbatch_mean:0','fg1/realbatch_var:0'])
        ],
        model=Model(),
        max_epoch=250,
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


