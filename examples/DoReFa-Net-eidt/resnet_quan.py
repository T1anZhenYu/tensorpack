#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet-dorefa.py

import argparse
import numpy as np
import os
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from dorefa_nb import get_dorefa
from imagenet_utils import ImageNetModel, eval_classification, fbresnet_augmentor

"""
This script loads the pre-trained ResNet-18 model with (W,A,G) = (1,4,32)
It has 59.2% top-1 and 81.5% top-5 validation error on ILSVRC12 validation set.

To run on images:
    ./resnet-dorefa.py --load ResNet-18-14f.npz --run a.jpg b.jpg

To eval on ILSVRC validation set:
    ./resnet-dorefa.py --load ResNet-18-14f.npz --eval --data /path/to/ILSVRC
"""

BITW = 1
BITA = 4
BITG = 32


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, 32,32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 256.0
        is_training = get_current_tower_context().is_training
        fw, fa, fg, quan_bn = get_dorefa(BITW, BITA, BITG)

        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv1' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        def resblock(x, channel, stride):
            def get_stem_full(x):
                return (LinearWrap(x)
                        .Conv2D('c3x3a', channel, 3)
                        .BatchNorm('stembn')
                        .apply(activate)
                        #.apply(quan_bn,'stembn',is_training)
                        .Conv2D('c3x3b', channel, 3)())
            channel_mismatch = channel != x.get_shape().as_list()[3]
            if stride != 1 or channel_mismatch or 'pool1' in x.name:
                # handling pool1 is to work around an architecture bug in our model
                if stride != 1 or 'pool1' in x.name:
                    x = AvgPooling('pool', x, stride, stride)
                # x = BatchNorm('bn', x)
                # x = activate(x)
                x = quan_bn(x,'bn',is_training)
                shortcut = Conv2D('shortcut', x, channel, 1)
                stem = get_stem_full(x)
            else:
                shortcut = x
                # x = BatchNorm('bn', x)
                # x = activate(x)
                x = quan_bn(x,'bn',is_training)
                stem = get_stem_full(x)
            return shortcut + stem

        def group(x, name, channel, nr_block, stride):
            with tf.variable_scope(name + 'blk1'):
                x = resblock(x, channel, stride)
            for i in range(2, nr_block + 1):
                with tf.variable_scope(name + 'blk{}'.format(i)):
                    x = resblock(x, channel, 1)
            return x

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      # use explicit padding here, because our private training framework has
                      # different padding mechanisms from TensorFlow
                      .tf.pad([[0, 0], [3, 2], [3, 2], [0, 0]])
                      .Conv2D('conv1', 64, 7, stride=2, padding='VALID', use_bias=True)
                      .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
                      .MaxPooling('pool1', 3, 2, padding='VALID')
                      .apply(group, 'conv2', 64, 2, 1)
                      .apply(group, 'conv3', 128, 2, 2)
                      .apply(group, 'conv4', 256, 2, 2)
                      .apply(group, 'conv5', 512, 2, 2)
                      .BatchNorm('lastbn')
                      .apply(nonlin)
                      .GlobalAvgPooling('gap')
                      .tf.multiply(49)  # this is due to a bug in our model design
                      .FullyConnected('fct', 10)())
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
        #add_param_summary(relax, ['scalar'])
        #tf.summary.scalar('relax_para', relax)
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
    #logger.auto_set_dir()
    logger.set_logger_dir('./train_log')
    # prepare dataset
    dataset_train = get_data('train', dir = '.cifar10_data/')
    dataset_test = get_data('test', dir = '.cifar10_data/')


    '''
    class AddTBinTrain(TrainingMonitor):
        def __init__(name,value):
        def _trigger_step(self):
    '''
    return TrainConfig(
        starting_epoch = args.starting_epoch,
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong-top1')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.01), (82, 0.001), (123, 0.0002), (200, 0.0001)]),
            DumpTensors(['conv2blk1/bn/bnMyrangenorm/Myrangenorm/my_bm',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/my_bv',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/real_bv',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/real_bm',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/diff_bm',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/diff_bv',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/ratio_bm',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/ratio_bv',\
                            'conv2blk1/bn/bnMyrangenorm/Myrangenorm/ratio_bv2',\
                ]),
            MergeAllSummaries(),
            #MergeAllSummaries(period=1, key='relax')
        ],
        #monitors=DEFAULT_MONITORS() + [ScalarPrinter(enable_step=True)],
        model=Model(),
        max_epoch=args.epoches,
    )





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='2,2,32')
    parser.add_argument('--root_dir', action='store', default='trash/', help='root dir for different experiments',
              type=str)
    parser.add_argument('--starting_epoch',default='1',type=int)
    parser.add_argument('--load', help='load a checkpoint, or a npz (given as the pretrained model)')
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--epoches', default='300', type=int)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    if args.eval:
        BATCH_SIZE = 128
        data_test = dataset.Cifar10('test')
        pp_mean = data_test.get_per_pixel_mean()
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
        data_test = AugmentImageComponent(data_test, augmentors)
        data_test = BatchData(data_test, 128, remainder=True)
        eval_classification(Model(), get_model_loader(args.load), data_test)
        sys.exit()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    print('check...................')
    launch_train_with_config(config, SimpleTrainer())