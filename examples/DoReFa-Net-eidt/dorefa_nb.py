# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu
#这个代码实现了量化bn的核心功能，主要是fg函数。
import tensorflow as tf
import numpy as np 
from tensorpack import *


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

    quan_points0 = get_quan_point().astype(np.float32)#standard quan_points

    #add_moving_summary(tf.identity(quan_points[3],name='origin_quan_points_3')) 
    quan_values = np.array([round((quan_points0[i]-0.005)*(2**bitA-1))\
    /(float(2**bitA-1)) for i in range(len(quan_points0))])#values after quantization 
    
    quan_values = np.append(quan_values,np.array([1.]),axis=-1).astype(np.float32)#append 1 to quan_values
    quan_points0 = np.append(np.insert(quan_points0,0,-1000.),np.array([1000.]))
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

    def my_sigmoid(x):
        s_ = tf.get_variable('sigmoid_', params_shape,
                       initializer=tf.zeros_initializer)
        return s_*(tf.stop_gradient(tf.math.sigmoid(x)-tf.nn.relu(x))+tf.nn.relu(x))
    def nonlin(x):
        if bitA == 32:
            return tf.nn.relu(x)
        #return tf.clip_by_value(x, 0.0, 1.0)
        return my_sigmoid(x)
    def fg(x):
        if bitG == 32:
            return x

        @tf.custom_gradient
        def _identity(input):
            def grad_fg(x):

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

            return input, grad_fg

        return _identity(x)
    def activate(x):
        return fa(nonlin(x))
    def quan_bn(x,name,training,momentum = 0.9):#bitG == 32

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE,use_resource=True):

            shape = x.get_shape().as_list()#x is input, get input shape[batchsize,width,height,channel]

            num_chan = shape[-1]#channel number
            batch_size0 = tf.shape(x)[0]#because placehoder,batch size is always None,needs this operation to use it as a real number
            w = shape[1]
            h = shape[2]
            #batch_mean 用来存储当前batch的均值
            batch_mean = tf.get_variable('batch_mean',shape=[num_chan,1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable=False)
            #batch_var 用来存储当前batch的标准差
            batch_var = tf.get_variable('batch_var',shape=[num_chan,1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable=False)



            #为了方便计算导数，这里引入了bn。momentum是用来计算movingmean和movingvar的。center表示
            #是否使用beta，scale表示是否使用gamma
  

            #fake_output代表这不是真实的输出
            #当使用L2norm时，是用的L2norm做方差；用L1norm时，是用的L1norm做方差
            #fake_output,layer_gamma,layer_beta,layer_mm,layer_ms =  L2norm(name+'L2norm',x, train=training)
            fake_output,layer_gamma,layer_beta,layer_mm,layer_mv,bm,bv = 
             Myrangenorm(name+'Myrangenorm',x, train=training)
            # if training:#在train的时候
            #     print('in training')

            #     batch_mean = batch_mean.assign(tf.expand_dims(bm,axis=-1))
            #     batch_var = batch_var.assign(tf.expand_dims(tf.sqrt(bv),axis=-1))
            #     #计算量化区间的起止点。
            #     quan_points = batch_var*quan_points0/tf.expand_dims(layer_gamma,axis=-1)+\
            #     batch_mean - batch_var*tf.expand_dims(layer_beta/layer_gamma,axis=-1)
            #     # adjust quan_points
            # else:
            #     print('in inference')

            #     #xnn,layer_gamma,layer_beta,layer_mm,layer_ms = L2norm(x, training=training)

            #     batch_mean = batch_mean.assign(tf.expand_dims(layer_mm,axis=-1))
            #     batch_var = batch_var.assign(tf.expand_dims(tf.sqrt(layer_mv),axis=-1))

            #     quan_points = batch_var*quan_points0/tf.expand_dims(layer_gamma,axis=-1)+\
            #     batch_mean - batch_var*tf.expand_dims(layer_beta/layer_gamma,axis=-1)

            # '''
            # the following part is to use quan_points to quantizate inputs.
            # '''
            # inputs = tf.transpose(tf.reshape(x,[-1,num_chan]))

            # label = []

            # for i in range(1,len(quan_points0)):
            #     label.append(tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,i],axis=-1)),\
            #                 tf.math.greater(inputs,tf.expand_dims(quan_points[:,i-1],axis=-1))),dtype=tf.float32))
            # xn = label[0]*quan_values[0]
            # for i in range(1,len(label)):
            #     xn += label[i]*quan_values[i]

            # quan_output = tf.reshape(tf.transpose(xn),[-1,w,h,num_chan])
            
            if training:
                return activate(fake_output)

            else:
                return activate(fake_output)

    return fw, fa, fg,quan_bn


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
