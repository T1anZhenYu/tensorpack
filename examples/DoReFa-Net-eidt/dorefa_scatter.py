# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu
#这个代码是利用对角采样计算均值，是基于dorefa_total的
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

    quan_points0 = get_quan_point().astype(np.float32)#standard quan_points

    #add_moving_summary(tf.identity(quan_points[3],name='origin_quan_points_3')) 
    quan_values = np.array([round((quan_points0[i]-0.005)*(2**bitA-1))\
    /(float(2**bitA-1)) for i in range(len(quan_points0))])#values after quantization 
    
    quan_values = np.append(quan_values,np.array([1.]),axis=-1).astype(np.float32)#append 1 to quan_values

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
    def nonlin(x):
        if bitA == 32:
            return tf.nn.relu(x)
        return tf.clip_by_value(x, 0.0, 1.0)

    def activate(x):
        return fa(nonlin(x))
    def fg(x,name,training,momentum = 0.9,kernel_size=4,scatter_rate=0):#bitG == 32
        def get_std(x,ave):#根据采样后的均值计算方差
            inputs = tf.reshape(x,[-1,tf.shape(x)[-1]])
            return tf.sqrt(tf.reduce_mean(tf.square(inputs-tf.expand_dims(ave,axis=0)),axis=0))
        def dignoal(x,kernel_size,scatter_rate):#利用对角采样计算均值，
            b = tf.shape(x)[0]
            w = tf.shape(x)[1]
            h = tf.shape(x)[2]

            dig = tf.matrix_diag((([0]*scatter_rate+[1])*kernel_size)[:kernel_size])

            dig = tf.tile(dig,[tf.cast(tf.math.ceil(w/kernel_size),dtype=tf.int32),tf.cast(tf.math.ceil(w/kernel_size),dtype=tf.int32)])[:w,:h]
            dig = tf.tile(tf.expand_dims(tf.expand_dims(dig,axis=0),axis=-1),[b,1,1,tf.shape(x)[-1]])

            x_ = x*tf.cast(dig,dtype=tf.float32)
            num =tf.cast(b,dtype=tf.float64)*tf.math.floor(w/kernel_size)*tf.math.floor(h/kernel_size)*tf.cast(tf.math.floor(kernel_size/(scatter_rate+1)),dtype=tf.float64)+\
            tf.cast(b,dtype=tf.float64)*tf.math.floor(tf.floormod(w,kernel_size)/(scatter_rate+1))*tf.math.floor(h/kernel_size)+\
            tf.cast(b,dtype=tf.float64)*tf.math.floor(tf.floormod(h,kernel_size)/(scatter_rate+1))*tf.math.floor(w/kernel_size)+\
            tf.cast(b,dtype=tf.float64)*tf.math.floor(tf.reduce_min([tf.floormod(w,kernel_size),tf.floormod(h,kernel_size)])/(scatter_rate+1))
            ave = tf.reduce_sum(x_,axis=[0,1,2])/tf.expand_dims(tf.cast(num,dtype=tf.float32),axis=-1)
            return ave
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE,use_resource=True):

            shape = x.get_shape().as_list()#x is input, get input shape[batchsize,width,height,channel]

            num_chan = shape[-1]#channel number
            batch_size0 = tf.shape(x)[0]#because placehoder,batch size is always None,needs this operation to use it as a real number
            w = shape[1]
            h = shape[2]

            batch_mean = tf.get_variable('batch_mean',shape=[num_chan,1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable=False)

            batch_var = tf.get_variable('batch_var',shape=[num_chan,1],\
            dtype = tf.float32,initializer=tf.zeros_initializer(),trainable=False)
            tf_args = dict(
                momentum=momentum,center=True, scale=True,name = name+'/bn')   
            layer = tf.layers.BatchNormalization(**tf_args)  

            fake_output =  layer.apply(x, training=training, scope=tf.get_variable_scope())
            print([n.name for n in tf.trainable_variables()])
            if training:
                print('in training')
                #bm, bv = tf.nn.moments(x, axes=[0,1,2])
                bm = dignoal(x,kernel_size,scatter_rate)
                bv = get_std(x,bm)
                batch_mean = batch_mean.assign(tf.expand_dims(bm,axis=-1))
                batch_var = batch_var.assign(tf.expand_dims(bv,axis=-1))

                quan_points = batch_var*quan_points0/tf.expand_dims(layer.gamma,axis=-1)+\
                batch_mean - batch_var*tf.expand_dims(layer.beta/layer.gamma,axis=-1)

                layer.moving_mean = layer.moving_mean.assign(layer.moving_mean*momentum+(1-momentum)*bm)
                layer.moving_variance = layer.moving_variance.assign(layer.moving_variance*momentum+(1-momentum)*tf.square(bv))

                # adjust quan_points
            else:
                print('in inference')
                xnn = layer.apply(x, training=training, scope=tf.get_variable_scope())

                i1 = x[0,0,0,:]
                i2 = x[1,1,1,:]
                x1 = xnn[0,0,0,:]
                x2 = xnn[1,1,1,:]

                mean0 = i1-x1*(i1-i2)/(x1-x2)
                var0 = (i1-i2)/(x1-x2)
                #quantize BN during inference

                #add_moving_summary(tf.identity(quan_points[3],name='origin_quan_points_3')) 

                moving_mean_ = tf.identity(mean0,name='moving_mean_')
                moving_mean_ = tf.expand_dims(moving_mean_,axis=-1)
                moving_var_ = tf.identity(var0,name='moving_var')
                moving_var_ = tf.expand_dims(moving_var_,axis = -1)

                quan_points = moving_var_ *quan_points0 + moving_mean_

            '''
            the following part is to use quan_points to quantizate inputs.
            '''
            inputs = tf.transpose(tf.reshape(x,[-1,num_chan]))

            label1 = tf.cast(tf.less_equal(inputs,tf.expand_dims(quan_points[:,0],axis=-1)),dtype=tf.float32)

            label2 = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,1],axis=-1)),\
                tf.math.greater(inputs,tf.expand_dims(quan_points[:,0],axis=-1))),dtype=tf.float32)

            label3 = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs,tf.expand_dims(quan_points[:,2],axis=-1)),\
                tf.math.greater(inputs,tf.expand_dims(quan_points[:,1],axis=-1))),dtype=tf.float32)

            label4 = tf.cast(tf.math.greater(inputs,tf.expand_dims(quan_points[:,2],axis=-1)),dtype=tf.float32)

            xn = label1*quan_values[0]+label2*quan_values[1]+label3*quan_values[2]+\
            label4*quan_values[3]

            quan_output = tf.reshape(tf.transpose(xn),[-1,w,h,num_chan])
            if training:
                return tf.stop_gradient(quan_output - activate(fake_output)) + activate(fake_output)

            else:
                return quan_output

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
