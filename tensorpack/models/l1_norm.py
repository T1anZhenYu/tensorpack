# -*- coding: utf-8 -*-
# File: batch_norm.py


import re
import six
import os
from ..compat import tfv1 as tf  # this should be avoided first in model code
from tensorflow.python.training import moving_averages
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from ..tfutils.collection import backup_collection, restore_collection
from ..tfutils.common import get_tf_version_tuple
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.argtools import get_data_format
from ..utils.develop import log_deprecated
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args, rename_get_variable
import numpy as np 

__all__ = ['L2norm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum',
        'use_local_stat': 'training'
    })
def L2norm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              bit_activation=2):
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training


    training = bool(training)

    shape = inputs.get_shape().as_list()
    with tf.variable_scope('L2norm',reuse=tf.AUTO_REUSE):
        gamma =  tf.get_variable('gamma',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.ones_initializer())

        beta = tf.get_variable('beta',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.zeros_initializer())

        batch_mean = tf.get_variable('batch_mean',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)

        batch_std = tf.get_variable('batch_std',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.ones_initializer(),trainable = False)

        moving_mean = tf.get_variable('moving_mean',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)

        moving_std = tf.get_variable('moving_std',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.ones_initializer(),trainable = False)
        before_mean = tf.get_variable('before_mean',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)
        before_std = tf.get_variable('before_std',shape=shape[-1],\
            dtype = tf.float32,initializer=tf.ones_initializer(),trainable = False)
        if training:
            bm, bv = tf.nn.moments(inputs, axes=[0,1,2])

            temp_mean = moving_mean+bm

            batch_mean = tf.assign(batch_mean,bm)
            batch_std = tf.assign(batch_std,tf.sqrt(bv)+0.000001)

            before_mean_op = tf.assign(before_mean,moving_mean)
            moving_mean_op = tf.assign(moving_mean,momentum*before_mean_op + (1-momentum)*bm ) 

            before_std_op= tf.assign(before_std,moving_std)
            moving_std_op = tf.assign(moving_std, momentum*before_std_op + (1-momentum)*tf.sqrt(bm))


            x_ = (inputs-batch_mean)/batch_std +\
                    tf.stop_gradient(moving_mean_op -moving_mean_op +moving_std_op-moving_std_op)

            output = gamma * x_ + beta 
            

        else:
            x_ = (inputs-moving_mean)/moving_std
            output = gamma * x_ + beta
        return output,gamma,beta,moving_mean,moving_std
