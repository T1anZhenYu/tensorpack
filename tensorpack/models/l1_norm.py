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
__all__ = ['L2norm','L1norm','L2norm_quan_train','Lmaxnorm1','Lmaxnorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4
bitG = 8

def nonlin(x):
    return tf.clip_by_value(x, -2, 2)
def quantize(x,k):
    n = float(2 ** k - 1)

    @tf.custom_gradient
    def _quantize(x):
        return tf.round(x * n) / n, lambda dy: dy

    return _quantize(x)
def quan_(x,max_value):

    rank = x.get_shape().ndims
    assert rank is not None
    maxx = max_value
    x = x / maxx
    n = float(2**bitG - 1)
    x = x * 0.5 + 0.5 + tf.random_uniform(
        tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = quantize(x, bitG) - 0.5
    return x * maxx * 2
def quan(x):

    rank = x.get_shape().ndims
    assert rank is not None
    maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
    x = x / maxx
    n = float(2**bitG - 1)
    x = x * 0.5 + 0.5 + tf.random_uniform(
        tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = quantize(x, bitG) - 0.5
    return x * maxx * 2


def near_2(x):
    def log2(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator
    return tf.sign(x)*tf.pow(2,tf.round(log2(tf.abs(x))))

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
def BNN(x, train, eps=1e-05, decay=0.9, affine=True, name=None):

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

            mean, variance_ = tf.nn.moments(x, [0,1,2], name='moments')
            variance = tf.reduce_sum((x - mean)*near_2((x-mean)),[0,1,2])

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
            x_ = (x-mean)*(1/(tf.sqrt(variance)+eps))
            x = x_ * near_2(gamma)
            #x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x,gamma,beta,moving_mean,moving_variance,mean,variance
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
def Lmaxnorm1(x, train, eps=1e-05, decay=0.9, affine=True, name=None):

    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()
        params_shape = params_shape[-1:]
        moving_mean = tf.get_variable('mean', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        my_bm = tf.get_variable('my_bm', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        my_bv = tf.get_variable('my_bv', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)
        real_bm = tf.get_variable('real_bm', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        real_bv = tf.get_variable('real_bv', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)
        diff_bm = tf.get_variable('diff_bm', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        diff_bv = tf.get_variable('diff_bv', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)
        ratio_bm = tf.get_variable('ratio_bm', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        ratio_bv = tf.get_variable('ratio_bv', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)


        c_max = tf.reduce_max(x,[0,1,2])
        c_min = tf.reduce_min(x,[0,1,2])

        mean_, variance_ = tf.nn.moments(x, [0,1,2], name='moments')

        my_bm = tf.assign(my_bm,(c_max+c_min)/2)
        my_bv = tf.assign(my_bv,(c_max-c_min))            
        real_bm = tf.assign(real_bm,mean_)
        real_bv = tf.assign(real_bv,variance_)
        diff_bm = tf.assign(diff_bm,((c_max+c_min)/2)-mean_)
        diff_bv = tf.assign(diff_bv,(c_max-c_min)-variance_)
        ratio_bm = tf.assign(ratio_bm,(((c_max+c_min)/2))/mean_)
        ratio_bv = tf.assign(ratio_bv,(c_max-c_min)/variance_)




        def mean_var_with_update():

            mean = tf.stop_gradient((c_max+c_min)/2 - mean_)+mean_
            variance = tf.stop_gradient(c_max - c_min- variance_)+variance_
            with tf.control_dependencies([assign_moving_average(moving_mean, mean_, decay),#计算滑动平均值
                                         assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean_), tf.identity(variance)
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
        return x,gamma,beta,moving_mean,moving_variance,mean,variance
        #return x
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
def Lmaxnorm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):

    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.get_shape().as_list()
        params_shape = params_shape[-1:]
        moving_mean = tf.get_variable('mean', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        c_max = tf.reduce_max(x,[0,1,2])
        c_min = tf.reduce_min(x,[0,1,2])

        

        def mean_var_with_update():

            mean_, variance_ = tf.nn.moments(x, [0,1,2], name='moments')
            mean = tf.stop_gradient((c_max+c_min)/2 - mean_)+mean_
            variance = tf.stop_gradient(c_max - c_min- variance_)+variance_
            with tf.control_dependencies([assign_moving_average(moving_mean, mean_, decay),#计算滑动平均值
                                         assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean_), tf.identity(variance)
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
def L2norm_quan_train(x, train, layer_num,eps=1e-05, decay=0.9, affine=True, name=None):
    if layer_num == 1:
        beta_max,gamma_max,variance_max,mean_max = 0.4,1.3,12,8
    elif layer_num == 2:
        beta_max,gamma_max,variance_max,mean_max = 0.6,1.2,18,6
    elif layer_num == 3:
        beta_max,gamma_max,variance_max,mean_max = 0.5,1.1,15,12
    elif layer_num == 4:
        beta_max,gamma_max,variance_max,mean_max = 0.6,1.2,13,6
    elif layer_num == 5:
        beta_max,gamma_max,variance_max,mean_max = 0.5,1.1,12,3
    elif layer_num == 6:
        beta_max,gamma_max,variance_max,mean_max = 0.2,1.5,55,8   

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

            mean = quan_(mean,mean_max)
            variance = quan_(variance,tf.reduce_max(variance)-tf.reduce_min(variance))+eps
            
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

            x = tf.nn.batch_normalization(x, mean,variance, beta,gamma, eps)

        else:
            x = tf.nn.batch_normalization(x, mean, variance,None, None, eps)

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