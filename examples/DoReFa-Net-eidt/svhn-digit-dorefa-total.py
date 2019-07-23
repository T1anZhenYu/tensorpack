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

from dorefa_eidt import get_dorefa

"""
This is a tensorpack script for the SVHN results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160
The original experiements are performed on a proprietary framework.
This is our attempt to reproduce it on tensorpack.
Accuracy:
    With (W,A,G)=(1,1,4), can reach 3.1~3.2% error after 150 epochs.
    With (W,A,G)=(1,2,4), error is 3.0~3.1%.
    With (W,A,G)=(32,32,32), error is about 2.3%.
Speed:
    With quantization, 60 batch/s on 1 1080Ti. (4721 batch / epoch)
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

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

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
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 48, 5, padding='VALID', use_bias=True)
                      .MaxPooling('pool0', 2, padding='SAME')
                      #.apply(activate)
                      # 18
                      .Conv2D('conv1', 64, 3, padding='SAME')
                      .apply(fg,'fg1',is_training)
                      #.BatchNorm('bn1')
                      #.apply(activate)

                      .Conv2D('conv2', 64, 3, padding='SAME')
                      .apply(fg,'fg2',is_training)
                      #.BatchNorm('bn2')
                      .MaxPooling('pool1', 2, padding='SAME')
                      #.apply(activate)
                      # 9
                      .Conv2D('conv3', 128, 3, padding='VALID')
                      .apply(fg,'fg3',is_training)
                      #.BatchNorm('bn3')
                      #.apply(activate)
                      # 7

                      .Conv2D('conv4', 128, 3, padding='SAME')
                      .apply(fg,'fg4',is_training)
                      #.BatchNorm('bn4')
                      #.apply(activate)

                      .Conv2D('conv5', 128, 3, padding='VALID')
                      .apply(fg,'fg5',is_training)
                      #.BatchNorm('bn5').apply(activate)
                      # 5
                      .Dropout(rate=0.5 if is_training else 0.0)
                      .Conv2D('conv6', 512, 5, padding='VALID')
                      .apply(fg,'fg6',is_training)
                      #.BatchNorm('bn6')
                      .apply(nonlin)
                      .FullyConnected('fc1', 10)())

        tf.nn.softmax(logits, name='output')
        '''
        conv_out = tf.get_default_graph().get_tensor_by_name("conv1/output:0")[0]
        fg_out = tf.get_default_graph().get_tensor_by_name("fg1/output:0")[0]
        grad = tf.identity(tf.gradients(fg_out,conv_out),name = 'grad')
        '''
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
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=4721 * 100,
            decay_rate=0.5, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_config():
    logger.set_logger_dir(os.path.join('train_log', 'svhn-dorefa-{}'.format(args.dorefa)))

    # prepare dataset
    d1 = dataset.SVHNDigit('train')
    d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1, d2])
    data_test = dataset.SVHNDigit('test')

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
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong-top1')]),
            DumpTensors(['fg1/moving_mean:0','fg1/moving_var:0','fg1/batch_mean:0','fg1/batch_var:0','conv1/output:0','fg1/output:0','grad'])
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