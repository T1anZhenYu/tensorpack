#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: svhn-digit-dorefa.py
# Author: Yuxin Wu

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.varreplace import remap_variables

from dorefa import get_dorefa#这里实现了对W，A，G的量化，原版。

"""
这个代码是用原版的测试svhn数据的模型来测试cifar。
"""

BITW = 1#对W的量化bit
BITA = 2#对A的量化bit
BITG = 4#对G的量化bit
#我在测试的时候没有考虑量化导数的问题，所以三个参量设置成1，2，32.

class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, 40, 40, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)#获取对三个参量量化的函数变量

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):#注意，对模型的第一层和最后一层，一般是不做任何量化的。
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fc' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):#这里是clip_Relu
            if BITA == 32:
                return tf.nn.relu(x)
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):#这里是对A先做clip_Relu，再做量化
            return fa(nonlin(x))

        image = image / 256.0

        with remap_variables(binarize_weight), \
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4,center=True, scale=True,), \
                argscope(Conv2D, use_bias=False):#这行代码是对所有的variables对过binarize_weight函数、设置BN和Conv的参数

            logits = (LinearWrap(image)#LinearWrap用来搭建线性模型，其中apply的是函数句柄，可以向其中传递参数；
                      .Conv2D('conv0', 48, 5, padding='VALID', use_bias=True)#conv0 input:[none,40,40,3] output:[none,36,36,48]
                      .MaxPooling('pool0', 2, padding='SAME')#pooling input[none:36,36,48] output:[18,18,48]
                      .apply(activate)#对Activation进行量化。
                      # 18
                      .Conv2D('conv1', 64, 3, padding='SAME')#input[none,18,18,48] output[none,18,18,64]
                      .apply(fg)#对导数进行量化
                      .BatchNorm('bn1').apply(activate)

                      .Conv2D('conv2', 64, 3, padding='SAME')#input[none 18,18,64] output[none,18,18,64]
                      .apply(fg)
                      .BatchNorm('bn2')
                      .MaxPooling('pool1', 2, padding='SAME')#input[none,18,18,64] output[none,9,9,64]
                      .apply(activate)
                      # 9
                      .Conv2D('conv3', 128, 3, padding='VALID')#input[none,9,9,64] output[none,7,7,128]
                      .apply(fg)
                      .BatchNorm('bn3').apply(activate)
                      # 7

                      .Conv2D('conv4', 128, 3, padding='SAME')#input[none,7,7,128] output[none,7,7,128]
                      .apply(fg)
                      .BatchNorm('bn4').apply(activate)

                      .Conv2D('conv5', 128, 3, padding='VALID')#input[none,7,7,128] output[none,5,5,128]
                      .apply(fg)
                      .BatchNorm('bn5').apply(activate)
                      # 5
                      .Dropout(rate=0.5 if is_training else 0.0)
                      .Conv2D('conv6', 512, 5, padding='VALID')#input[none,5,5,128] output[none,1,1,512]
                      .apply(fg).BatchNorm('bn6')
                      .apply(nonlin)#这里只做了clip_relu.并没有过量化。
                      .FullyConnected('fc1', 10)())#fc1 output[none,10]
        tf.nn.softmax(logits, name='output')

        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_tensor')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))

        add_param_summary(('.*/W', ['histogram', 'rms']))
        add_param_summary(('.*/beta', ['histogram', 'rms']))
        add_param_summary(('.*/mean/EMA', ['histogram']))   
        add_param_summary(('.*/gamma', ['histogram', 'rms']))   
        add_param_summary(('.*/variance/EMA', ['histogram']))           
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


def get_config():#这里是用来声明train的参数
    logger.set_logger_dir(os.path.join('train_log', 'svhn-dorefa-{}'.format(args.dorefa)))#设置log地址

    # prepare dataset
    d1 = dataset.CifarBase('train')#设置trian数据集
    #d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1])#这里是将两个以上的数据集mix，对单独的数据集没效果
    data_test = dataset.CifarBase('test')#设置test数据集
    #设置train的时候的augmentor的参数。
    augmentors = [
        imgaug.Resize((40, 40)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = MultiProcessRunnerZMQ(data_train, 5)#这个是用来处理多线程的。

    augmentors = [imgaug.Resize((40, 40))]#这里是设置test的时候的augmentor
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)#remainder的含义：当batchdata不足一个batch时，
    #是否构建小的batch。True构建，默认False不构建

    return TrainConfig(
        data=QueueInput(data_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong_tensor')]),
            #如果想要在train的时候获取中间变量，可以用下面的指令
            #DumpTensors(['bn1/mean/EMA:0','conv1/output:0','bn6/mean/EMA:0','conv6/output:0'])

        ],
        model=Model(),
        max_epoch=200,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='1,2,4')
    args = parser.parse_args()
    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    launch_train_with_config(config, SimpleTrainer())