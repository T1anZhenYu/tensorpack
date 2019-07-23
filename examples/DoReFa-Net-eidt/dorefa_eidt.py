# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu

import tensorflow as tf
import numpy as np 

def get_dorefa(bitW, bitA, bitG):
    """
    Return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    """
    def get_quan_point():
        return np.array([(2**bitA-i+0.5)/(2**bitA-1) \
            for i in range(2**bitA,1,-1)])
    def quantize(x, k):
        n = float(2 ** k - 1)

        @tf.custom_gradient
        def _quantize(x):
            return tf.round(x * n) / n, lambda dy: dy

        return _quantize(x)

    def fw(x):
        if bitW == 32:
            return x

        if bitW == 1:   # BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy

            return _sign(x)

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    def fg(x,name,training,momentum = 0.9):#bitG == 32
                    #quantize BN during inference

        quan_points = get_quan_point().astype(np.float32)

        #add_moving_summary(tf.identity(quan_points[3],name='origin_quan_points_3')) 
        quan_values = np.array([round((quan_points[i]-0.005)*(2**bitA-1))\
        /(float(2**bitA-1)) for i in range(len(quan_points))])
        quan_values = np.append(quan_values,np.array([1.]),axis=-1).astype(np.float32)

        @tf.custom_gradient
        def my_grad(x,quan_points0,quan_values):
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE,use_resource=True):
                shape = x.get_shape().as_list()
                num_chan = shape[-1]
                batch_size = shape[0]
                moving_mean = tf.get_variable('moving_mean',shape=[num_chan,1],\
                    dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
                moving_var = tf.get_variable('moving_var',shape=[num_chan,1],\
                    dtype=tf.float32, initializer=tf.ones_initializer(),trainable=False)
                batch_mean = tf.get_variable('batch_mean',shape=[num_chan,1],\
                dtype = tf.float32,initializer=tf.zeros_initializer(),trainable=False\
                                     ,collections=[tf.GraphKeys.LOCAL_VARIABLES])

                batch_var = tf.get_variable('batch_var',shape=[num_chan,1],\
                dtype = tf.float32,initializer=tf.zeros_initializer(),trainable=False\
                                     ,collections=[tf.GraphKeys.LOCAL_VARIABLES])
                if training:
                    print('in training')
                    bm, bv = tf.nn.moments(x, axes=[0,1,2])

                    batch_mean = batch_mean.assign(tf.expand_dims(bm,axis=-1))
                    batch_var = batch_var.assign(tf.expand_dims(tf.math.sqrt(bv),axis=-1))

                    moving_mean = moving_mean.assign(momentum*moving_mean+batch_mean)
                    moving_var = moving_var.assign(momentum*moving_var+batch_var)

                    quan_points = batch_var*quan_points0 + batch_mean
                    #output = (x-batch_mean)/(tf.math.sqrt(batch_var))
                else:

                    print('in inference')

                    quan_points = moving_var *quan_points0 + moving_mean


                b,w,h,c = x.shape

                inputs = tf.transpose(tf.reshape(x,[-1,c]))

                label1 = tf.cast(tf.less_equal(inputs,tf.expand_dims(quan_points[:,0],axis=-1)),dtype=tf.float32)
                label2 = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,1],axis=-1)),\
                    tf.math.greater(inputs,tf.expand_dims(quan_points[:,0],axis=-1))),dtype=tf.float32)
                label3 = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,2],axis=-1)),\
                    tf.math.greater(inputs,tf.expand_dims(quan_points[:,1],axis=-1))),dtype=tf.float32)
                label4 = tf.cast(tf.math.greater(inputs,tf.expand_dims(quan_points[:,2],axis=-1)),dtype=tf.float32)
                xn = label1*quan_values[0]+label2*quan_values[1]+label3*quan_values[2]+\
                label4*quan_values[3]
                output = tf.reshape(tf.transpose(xn),[-1,w,h,c])
            def grad_fg(d):
                rank = d.get_shape().ndims
                assert rank is not None
                bn_z = 1/(batch_var)*(batch_size-1)/batch_size  \
                -tf.math.square((inputs-batch_mean)/(batch_var))*2/batch_size
                
                
                bn_z = tf.identity(tf.expand_dims(tf.expand_dims(tf.transpose(bn_z),\
                                                       axis=0),axis = 0),name='bnz')

                return d * bn_z,tf.ones(quan_points0.shape,name='fake0'),tf.ones(quan_values.shape,name='fake1')

            return output,grad_fg 

        return my_grad(x,quan_points,quan_values)

    return fw, fa, fg


def ternarize(x, thresh=0.05):
    """
    Implemented Trained Ternary Quantization:
    https://arxiv.org/abs/1612.01064

    Code modified from the authors' at:
    https://github.com/czhu95/ternarynet/blob/master/examples/Ternary-Net/ternary.py
    """
    shape = x.get_shape()

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thresh)

    w_p = tf.get_variable('Wp', initializer=1.0, dtype=tf.float32)
    w_n = tf.get_variable('Wn', initializer=1.0, dtype=tf.float32)

    tf.summary.scalar(w_p.op.name + '-summary', w_p)
    tf.summary.scalar(w_n.op.name + '-summary', w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    @tf.custom_gradient
    def _sign_mask(x):
        return tf.sign(x) * mask_z, lambda dy: dy

    w = _sign_mask(x)

    w = w * mask_np

    tf.summary.histogram(w.name, w)
    return w
