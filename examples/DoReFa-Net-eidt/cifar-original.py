#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: alexnet-dorefa.py
# Author: Yuxin Wu, Yuheng Zou ({wyx,zyh}@megvii.com)

import argparse
import numpy as np
import os
import sys
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.sessinit import get_model_loader
from tensorpack.tfutils.summary import add_param_summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from dorefa import get_dorefa, ternarize
from imagenet_utils import ImageNetModel, eval_classification, fbresnet_augmentor, get_imagenet_dataflow

"""
This is a tensorpack script for the ImageNet results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

The original experiements are performed on a proprietary framework.
This is our attempt to reproduce it on tensorpack & TensorFlow.

To Train:
    ./alexnet-dorefa.py --dorefa 1,2,6 --data PATH --gpu 0,1

    PATH should look like:
    PATH/
      train/
        n02134418/
          n02134418_198.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...

    And you'll need the following to be able to fetch data efficiently
        Fast disk random access (Not necessarily SSD. I used a RAID of HDD, but not sure if plain HDD is enough)
        More than 20 CPU cores (for data processing)
        More than 10G of free memory
    On 8 P100s and dorefa==1,2,6, the training should take about 30 minutes per epoch.

To run pretrained model:
    ./alexnet-dorefa.py --load alexnet-126.npz --run a.jpg --dorefa 1,2,6
"""

BITW = 1
BITA = 2
BITG = 6
TOTAL_BATCH_SIZE = 256
BATCH_SIZE = None


class Model(ModelDesc):
    weight_decay = 5e-6
    weight_decay_pattern = 'fc.*/W'
    def inputs(self):
        return [tf.TensorSpec([None, 40, 40, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fct' in name:
                return v
            else:
                logger.info("Quantizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        image = image / 256.0

        with remap_variables(new_get_variable), \
                argscope([Conv2D, BatchNorm, MaxPooling]), \
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 48, 12, strides=4, padding='VALID', use_bias=True)
                      .apply(activate)
                      .Conv2D('conv1', 128, 5, padding='SAME', split=2)
                      .apply(fg)
                      .BatchNorm('bn1')
                      .MaxPooling('pool1', 3, 2, padding='SAME')
                      .apply(activate)

                      .Conv2D('conv2', 256, 3)
                      .apply(fg)
                      .BatchNorm('bn2')
                      .MaxPooling('pool2', 3, 1, padding='SAME')
                      .apply(activate)

                      .Conv2D('conv3', 256, 3, split=2)
                      .apply(fg)
                      .BatchNorm('bn3')
                      .apply(activate)

                      .Conv2D('conv4', 128, 3, split=2)
                      .apply(fg)
                      .BatchNorm('bn4')
                      .MaxPooling('pool4', 3, 1, padding='VALID')
                      .apply(activate)

                      .FullyConnected('fc0', 512)
                      .apply(fg)
                      .BatchNorm('bnfc0')
                      .apply(activate)

                      .FullyConnected('fc1', 256, use_bias=False)
                      .apply(fg)
                      .BatchNorm('bnfc1')
                      .apply(nonlin)
                      .FullyConnected('fct', 10, use_bias=True)())

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
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)



def get_config():
    logger.set_logger_dir(os.path.join('train_log', 'svhn-dorefa-{}'.format(args.dorefa)))

    # prepare dataset
    d1 = dataset.CifarBase('train',cifar_classnum=10)
    #d2 = dataset.SVHNDigit('extra')
    data_train = RandomMixData([d1])
    data_test = dataset.CifarBase('test',cifar_classnum=10)

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
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter(
                'learning_rate', [(1, 0.1), (82, 0.01), (123, 0.001), (200, 0.0001)]),
            InferenceRunner(data_test,
                            [ClassificationError('wrong-top1', 'val-error-top1')])
        ],
        model=Model(),
        max_epoch=300,
    )


def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output']
    )
    predictor = OfflinePredictor(pred_config)
    meta = dataset.ILSVRCMeta()
    words = meta.get_synset_words_1000()

    transformers = imgaug.AugmentorList(fbresnet_augmentor(isTrain=False))
    for f in inputs:
        assert os.path.isfile(f), f
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        outputs = predictor(img)[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


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

    dorefa = args.dorefa.split(',')
    if dorefa[0] == 't':
        assert dorefa[1] == '32' and dorefa[2] == '32'
        BITW, BITA, BITG = 't', 32, 32
    else:
        BITW, BITA, BITG = map(int, dorefa)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npz')
        run_image(Model(), DictRestore(dict(np.load(args.load))), args.run)
        sys.exit()
    if args.eval:
        BATCH_SIZE = 128
        ds = get_data('val')
        eval_classification(Model(), get_model_loader(args.load), ds)
        sys.exit()

    nr_tower = max(get_num_gpu(), 1)
    BATCH_SIZE = TOTAL_BATCH_SIZE // nr_tower
    logger.set_logger_dir(os.path.join(
        'train_log', 'alexnet-dorefa-{}'.format(args.dorefa)))
    logger.info("Batch per tower: {}".format(BATCH_SIZE))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerReplicated(nr_tower))
