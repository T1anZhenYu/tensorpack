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
from tensorflow.python.training.moving_averages import assign_moving_average
__all__ = ['L2norm','L1norm','L2norm_quan_train']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4
def nonlin(x):
    return tf.clip_by_value(x, 0.0, 4.0)
def quantize(x):
    k = 2
    n = float(2 ** k - 1)

    @tf.custom_gradient
    def _quantize(x):
        return tf.round(x * n) / n, lambda dy: dy

    return _quantize(x)
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
def L2norm_quan_train(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()
        params_shape = params_shape[-1:]
        moving_mean = tf.get_variable('mean', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        
        def mean_var_with_update():

            mean, variance = tf.nn.moments(quantize(nonlin(x)), [0,1,2], name='moments')
            mean = quantize(nonlin(mean))
            variance = quantize(nonlin(variance))
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),#计算滑动平均值
                                         assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        if train:#亲测tf.cond的第一个函数不能直接写成ture or false，所以只好用一个很蠢的方法。
            xx = tf.constant(3)
            yy = tf.constant(4)
        else:
            xx = tf.constant(4)
            yy = tf.constant(3)
        mean, variance = tf.cond(xx<yy, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            '''
            x = tf.nn.batch_normalization(tf.to_bfloat16(x),tf.to_bfloat16(mean),\
             tf.to_bfloat16(variance), tf.to_bfloat16(beta), tf.to_bfloat16(gamma), tf.to_bfloat16(eps))
            x = tf.cast(x,dtype=tf.float32)
            '''

            x = tf.nn.batch_normalization(quantize(nonlin(x)), quantize(nonlin(mean)), \
                quantize(nonlin(variance)), quantize(nonlin(beta)), quantize(nonlin(gamma)), eps)

        else:
            x = tf.nn.batch_normalization(x, quantize(nonlin(mean)), quantize(nonlin(variance)),\
             None, None, eps)
        return x


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
def L2norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()
        params_shape = params_shape[-1:]
        moving_mean = tf.get_variable('mean', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        
        def mean_var_with_update():

            mean, variance = tf.nn.moments(x, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),#计算滑动平均值
                                         assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        if train:#亲测tf.cond的第一个函数不能直接写成ture or false，所以只好用一个很蠢的方法。
            xx = tf.constant(3)
            yy = tf.constant(4)
        else:
            xx = tf.constant(4)
            yy = tf.constant(3)
        mean, variance = tf.cond(xx<yy, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x,gamma,beta,moving_mean,moving_variance



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
def L1norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    def get_l1norm(x,ave):
        return 4/5*tf.reduce_mean(tf.abs(x-ave),axis=[0,1,2])

    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()
        params_shape = params_shape[-1:]
        moving_mean = tf.get_variable('mean', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        
        def mean_var_with_update():

            mean, v_ = tf.nn.moments(x, [0,1,2], name='moments')
            variance = get_l1norm(x,mean)
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),#计算滑动平均值
                                         assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        if train:#亲测tf.cond的第一个函数不能直接写成ture or false，所以只好用一个很蠢的方法。
            xx = tf.constant(3)
            yy = tf.constant(4)
        else:
            xx = tf.constant(4)
            yy = tf.constant(3)
        mean, variance = tf.cond(xx<yy, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x,gamma,beta,mean, variance