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
    '''
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    '''
    training = bool(training)

    shape = inputs.get_shape().as_list()
    with tf.variable_scope('L2norm',reuse=tf.AUTO_REUSE):
        gamma =  tf.get_variable('gamma',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.ones_initializer())

        beta = tf.get_variable('beta',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer())

        batch_mean = tf.get_variable('batch_mean',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)

        batch_std = tf.get_variable('batch_std',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)

        m_mean = tf.get_variable('mm',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer())

        m_std = tf.get_variable('ms',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer())
        if training:
            bm, bv = tf.nn.moments(x, axes=[0,1,2])
            batch_mean = tf.assign(batch_mean,tf.expand_dims(bm,axis=-1))
            batch_std = tf.assign(batch_std,tf.expand_dims(tf.sqrt(bv),axis=-1))
            m_mean = tf.assign(m_mean,momentum*m_mean+(1-momentum)*tf.expand_dims(bm,axis=-1))
            m_std = tf.assign(m_std,momentum*m_std+(1-momentum)*tf.expand_dims(tf.sqrt(bv),axis=-1))  

            x_ = (inputs-batch_mean)/batch_std +m_mean -m_mean +m_std-m_std

            output = gamma * x_ + beta 
            

        else:
            x_ = (inputs-m_mean)/m_std
            output = gamma * x_ + beta
        return output 


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
def L1norm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,

              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              bit_activation=2):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope('L1norm',reuse=tf.AUTO_REUSE,use_resource=True):
        gamma =  tf.get_variable('gamma',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.ones_initializer())

        beta = tf.get_variable('beta',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer())

        moving_mean = tf.get_variable('moving_mean',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)

        moving_std = tf.get_variable('moving_std',shape=[shape[-1],1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable = False)

        bm, bv = tf.nn.moments(x, axes=[0,1,2])

        x_ = (inputs-tf.expand_dims(bm,axis=-1))/(tf.expand_dims(tf.sqrt(bv)),axis=-1)

        output = gamma * x_ + beta
        return output 