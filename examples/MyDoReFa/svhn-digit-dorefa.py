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
from tensorpack.callbacks import DumpTensors
from imagenet_utils import ImageNetModel, eval_classification, fbresnet_augmentor, get_imagenet_dataflow

from dorefa import get_dorefa
import inspect 

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

def get_mean(x):
    #[batch,height,width,channels]
    return tf.reduce_mean(tf.reduce_mean(x,1),1)


def naive_bn(x):
    m , var = tf.nn.moments(x,-1,keep_dims = True)  
    return (x-m)/(var+0.00001)

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

        def activate(x,name = 'otherQa'):
            
            y = tf.identity(fa(nonlin(x)),name=name)

            return y

        def bn_loss(x,name='loss'):
            label = naive_bn(x)
            loss = tf.identity(tf.losses.mean_squared_error(x,label),name=name)
            return x,loss

        image = image / 256.0

        with remap_variables(binarize_weight): 
            x = Conv2D('conv0',image,48,5,padding='VALID', use_bias=True)
            x = MaxPooling('pool0', x,2, padding='SAME')
            x = activate(x)

            x = Conv2D('conv1', x,64, 3, padding='SAME',use_bias=False)
            x = fg(x)
            x = BatchNorm('bn1',x)
            x = activate(x)

            x = Conv2D('conv2', x,64, 3, padding='SAME',use_bias=False)
            x = fg(x)
            x = BatchNorm('bn2',x)
            x = activate(x)

            x = Conv2D('conv3',x,128, 3, padding='VALID',use_bias=False)
            x = fg(x)
            x = BatchNorm('bn3',x)
            x = activate(x)

            x = Conv2D('conv4',x,128, 3, padding='SAME',use_bias = False)
            x = fg(x)
            x = BatchNorm('bn4',x)
            x = activate(x)

            x = Conv2D('conv5',x, 128, 3, padding='VALID',use_bias = False)
            x = fg(x)
            x = BatchNorm('bn5',x)
            x = activate(x)

            x = Dropout(x,rate=0.5 if is_training else 0.0)
            x = Conv2D('conv6',x, 512, 5, padding='VALID',use_bias = False)
            x = fg(x)
            x = BatchNorm('bn6',x)
            x = nonlin(x)
            logits = FullyConnected('fc1',x,10)

        tf.nn.softmax(logits, name='output')

        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_tensor_top1')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

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

def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, dataset_name, BATCH_SIZE, augmentors)
def get_config():
    logger.set_logger_dir(os.path.join('dorefa_log', 'svhn-dorefa-{}'.format(args.dorefa)))

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
            #DumpTensors(['conv5/output:0','bn5/output:0','bn5Qa:0']),
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong_tensor_top1'),
                        ])
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

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npz')
        run_image(Model(), DictRestore(dict(np.load(args.load))), args.run)
        sys.exit()

    config = get_config()
    if args.eval:
        print('####################################################in eval')
        BATCH_SIZE = 128
        #ds = get_data('test')
        data_test = dataset.SVHNDigit('test')
        augmentors = [imgaug.Resize((40, 40))]
        data_test = AugmentImageComponent(data_test, augmentors)
        data_test = BatchData(data_test, 128, remainder=True)
        eval_classification(Model(), get_model_loader(args.load), data_test)
        sys.exit()
    launch_train_with_config(config, SimpleTrainer())

