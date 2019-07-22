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
logger.set_logger_dir(os.path.join('dorefa_log', 'bn'))
__all__ = ['QuanBatchNorm', 'BatchRenormEidt2']

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
def QuanBatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              ema_update='default',
              sync_statistics=None,
              internal_update=None,
              bit_activation=2):

    def get_quan_point():
        return np.array([(2**bit_activation-i+0.5)/(2**bit_activation-1) \
            for i in range(2**bit_activation,1,-1)])

    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)

    # parse shapes
    shape = inputs.get_shape().as_list()

    num_chan = shape[axis]

    moving_mean = tf.get_variable('moving_mean',shape=[num_chan],dtype=tf.float32, initializer=tf.zeros_initializer())
    moving_var = tf.get_variable('moving_var',shape=[num_chan],dtype=tf.float32, initializer=tf.ones_initializer())

    if training:
        batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0,1,2])
        moving_mean.assign(momentum*moving_mean+batch_mean)
        moving_var.assign(momentum*moving_var+batch_variance)

        output = (inputs-batch_mean)/(tf.math.sqrt(batch_variance))

    else:

        #quantize BN during inference
        print('in quantize BN')
        quan_points = get_quan_point()

        #add_moving_summary(tf.identity(quan_points[3],name='origin_quan_points_3')) 
        quan_values = np.array([round((quan_points[i]-0.005)*(2**bit_activation-1))\
        /(float(2**bit_activation-1)) for i in range(len(quan_points))])
        quan_values = np.append(quan_values,np.array([1.]),axis=-1)

        moving_mean_ = tf.identity(moving_mean,name='moving_mean_')
        moving_mean_ = tf.expand_dims(moving_mean_,axis=-1)
        moving_var_ = tf.identity(moving_var,name='moving_var')
        moving_var_ = tf.expand_dims(moving_var_,axis = -1)

        quan_points = moving_var_*quan_points + moving_mean_
     
        b,w,h,c = inputs.shape

        inputs = tf.transpose(tf.reshape(inputs,[-1,c]))

        label1 = tf.cast(tf.less_equal(inputs,tf.expand_dims(quan_points[:,0],axis=-1)),dtype=tf.float32)
        label2 = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,1],axis=-1)),\
            tf.math.greater(inputs,tf.expand_dims(quan_points[:,0],axis=-1))),dtype=tf.float32)
        label3 = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,2],axis=-1)),\
            tf.math.greater(inputs,tf.expand_dims(quan_points[:,1],axis=-1))),dtype=tf.float32)
        label4 = tf.cast(tf.math.greater(inputs,tf.expand_dims(quan_points[:,2],axis=-1)),dtype=tf.float32)
        xn = label1*quan_values[0]+label2*quan_values[1]+label3*quan_values[2]+\
        label4*quan_values[3]
        output = tf.reshape(tf.transpose(xn),[-1,w,h,c])
    return tf.identity(output,name='output')
  
       
