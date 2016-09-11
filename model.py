#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

def _var(name, shape, wd=0.001,initializer=None):
    #sqrt(3. / (in + out))
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
        
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

class Layer(object):
    """
    Layer with convolution and pooling
    """
    def __init__(self, name, output_ch, num_conv=1, retain_ratio=0.5, is_train=False):
        self.name = name
        self.output_ch = output_ch
        self.num_conv = num_conv
        self.retain_ratio = retain_ratio
        self.is_train = is_train

    def inference(self, in_feat):
        self.input_shape = in_feat.get_shape()
        N, H, W, C = self.input_shape
        feat = in_feat

        with tf.variable_scope(self.name):
            with tf.variable_scope("conv0"):
                self.w = _var("W", [3,3,C,self.output_ch])
                #self.b = _var("b", [self.output_ch],initializer=tf.constant_initializer())
                    
                feat = tf.nn.conv2d(feat, self.w, strides=[1,1,1,1],padding="SAME")
                #feat = feat + self.b
                feat = tf.nn.relu(feat)
                    

            self.pre_pool = feat
            feat = tf.nn.max_pool(feat, [1,2,2,1], strides=[1,2,2,1],padding="SAME")
            self.aft_pool = feat

            if self.is_train:
                feat = tf.nn.dropout(feat, keep_prob=self.retain_ratio)
                    
        return feat

    def deconv(self, top):
        """
_max_pool_grad(orig_input, orig_output, grad, ksize, strides, padding, data_format=None, name=None)
    Computes gradients of the maxpooling function.
    
    Args:
      orig_input: A `Tensor`. Must be one of the following types: `float32`, `half`.
        The original input tensor.
      orig_output: A `Tensor`. Must have the same type as `orig_input`.
        The original output tensor.
      grad: A `Tensor`. Must have the same type as `orig_input`.
        4-D.  Gradients w.r.t. the output of `max_pool`.
      ksize: A list of `ints` that has length `>= 4`.
        The size of the window for each dimension of the input tensor.
      strides: A list of `ints` that has length `>= 4`.
        The stride of the sliding window for each dimension of the
        input tensor.
      padding: A `string` from: `"SAME", "VALID"`.
        The type of padding algorithm to use.
      data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Specify the data format of the input and output data. With the
        default format "NHWC", the data is stored in the order of:
            [batch, in_height, in_width, in_channels].
        Alternatively, the format could be "NCHW", the data storage order of:
            [batch, in_channels, in_height, in_width].
      name: A name for the operation (optional).
    
    Returns:
      A `Tensor`. Has the same type as `orig_input`.
      Gradients w.r.t. the input to `max_pool`.
        """
        self.pre_unpool = top
        unpool = gen_nn_ops._max_pool_grad(
            orig_input=self.pre_pool, orig_output=self.aft_pool, grad=top,
            ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"
        )

        self.unpool = unpool
        
        # not reverse operation        
        feat = tf.nn.relu(unpool)
        
        #feat = feat - self.b
        
        feat = tf.nn.conv2d_transpose(feat, self.w, self.input_shape, [1,1,1,1], padding='SAME')
        
        return feat

class Network(object):
    def __init__(self):
        pass
    
    def inference(self, x, is_train=False):
        """
        recover network
        """
        layers = []
        feats = []
        
        D = x.get_shape()[3]
        output_ch = 32
        num_classes = 10
        feat = x

        # 28x28
        for idx_layer in range(4):
            name = "layer{}".format(idx_layer)

            if idx_layer == 0:
                layer = Layer(name, output_ch, retain_ratio=0.8, is_train=is_train)
            else:
                layer = Layer(name, output_ch, is_train=is_train)

            feat = layer.inference(feat)            

            feats.append(feat)
            layers.append(layer)


        self.conv_outputs = feats
        self.conv_layers = layers
        output_ch *= 2

        # FC layer
        with tf.variable_scope("FC"):
            flat = tf.contrib.layers.flatten(feat)
            N, D = flat.get_shape()
            output_ch = 512
        
            W = _var("W", [D, output_ch])
            b = _var("b", [output_ch], initializer=tf.constant_initializer())

            fc1 = tf.nn.relu(tf.matmul(flat, W) + b)

            W2 = _var("W2", [output_ch, num_classes])
            b2 = _var("b2", [num_classes], initializer=tf.constant_initializer())

            logits = tf.matmul(fc1, W2) + b2
    
        return logits

    def deconv(self):
        feat = self.conv_output
        
        for layer in reversed(self.conv_layers):
            feat = layer.deconv(feat)

        tf.image_summary("deconv", feat, max_images=5)

        return feat

    def deconv_from_layer(self, idx_layer, layer_value):
        """
        start deconvolution from layer at idx_layer.
        the value of the layer is layer_value.
        """
        for layer in reversed(self.conv_layers[:idx_layer+1]):
            layer_value = layer.deconv(layer_value)
            
        tf.image_summary("layer_deconv", layer_value, max_images=5)

        return layer_value

def get_loss(label, logits):
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, label), name="entropy")
    decays = tf.add_n(tf.get_collection('losses'), name="weight_loss")
    total_loss = tf.add(entropy, decays, "total_loss")

    tf.scalar_summary("entropy", entropy)
    tf.scalar_summary("total_loss",total_loss)
    
    return entropy, total_loss
