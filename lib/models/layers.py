#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

@add_arg_scope
def conv(filter_size,
         stride,
         out_dim,
         layer_dict,
         bn=False,
         nl=tf.identity,
         init_w=None,
         init_b=tf.zeros_initializer(),
         padding='SAME',
         pad_type='ZERO',
         trainable=True,
         is_training=None,
         name='conv'):
    inputs = layer_dict['cur_input']
    stride = get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]
    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]

    if padding == 'SAME' and pad_type == 'REFLECT':
        pad_size_1 = int((filter_shape[0] - 1) / 2)
        pad_size_2 = int((filter_shape[1] - 1) / 2)
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
            "REFLECT")
        padding = 'VALID'

    with tf.variable_scope(name):
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)

        outputs = tf.nn.conv2d(inputs,
                               filter=weights,
                               strides=stride,
                               padding=padding,
                               use_cudnn_on_gpu=True,
                               data_format='NHWC',
                               dilations=[1, 1, 1, 1],
                               name='conv2d')
        outputs += biases

        if bn is True:
            # outputs = layers.batch_norm(outputs, train=is_training, name='bn')
            outputs = tf.contrib.layers.instance_norm(
                outputs,
                center=True,
                scale=True,
                epsilon=1e-06,
                activation_fn=None,
                param_initializers=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                # data_format=DATA_FORMAT_NHWC,
                scope='bn'
            )

        layer_dict['cur_input'] = nl(outputs)
        return layer_dict['cur_input']

@add_arg_scope
def transpose_conv(
                   filter_size,
                   out_dim,
                   layer_dict,
                   stride=2,
                   out_shape=None,
                   init_w=None,
                   init_b=tf.zeros_initializer(),
                   padding='SAME',
                   trainable=True,
                   nl=tf.identity,
                   bn=False,
                   is_training=None,
                   name='dconv'):
    inputs = layer_dict['cur_input']
    stride = get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(inputs)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]), 
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])        

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
        
        outputs = tf.nn.conv2d_transpose(inputs,
                                        weights, 
                                        output_shape=out_shape, 
                                        strides=stride, 
                                        padding=padding, 
                                        name=scope.name)

        outputs = tf.nn.bias_add(outputs, biases)
        if bn == True:
            # outputs = layers.batch_norm(outputs, train=is_training, name='bn')
            outputs = tf.contrib.layers.instance_norm(
                outputs,
                center=True,
                scale=True,
                epsilon=1e-06,
                activation_fn=None,
                param_initializers=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope='bn'
            )

        outputs.set_shape([None, None, None, out_dim])
        layer_dict['cur_input'] = nl(outputs)
        return layer_dict['cur_input']

@add_arg_scope
def residual_block(out_dim, layer_dict, name, is_training, init_w=None):
    def conv_layer(name, relu=False):
        if relu == True:
            nl = tf.nn.relu
        else:
            nl = tf.identity
        
        return conv(
            filter_size=3,
            stride=1,
            out_dim=out_dim,
            layer_dict=layer_dict,
            bn=True,
            nl=nl,
            init_w=init_w,
            init_b=tf.zeros_initializer(),
            padding='SAME',
            pad_type='REFLECT',
            trainable=True,
            is_training=is_training,
            name=name)

    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']
        conv_layer('conv1', relu=True)
        conv_layer('conv2', relu=False)

        layer_dict['cur_input'] += inputs
        return layer_dict['cur_input']


# @add_arg_scope
# def deconv_bn_drop_relu(filter_size,
#                         out_dim,
#                         stride,
#                         keep_prob,
#                         layer_dict,
#                         name,
#                         is_training,
#                         fusion_id=None,
#                         bn=True):
#     with tf.variable_scope(name):
#         inputs = layer_dict['cur_input']
#         deconv_out = transpose_conv(
#             inputs,
#             filter_size=filter_size,
#             out_dim=out_dim,
#             stride=stride,
#             padding='SAME',
#             name='dconv')
#         if bn == True:
#             # bn_deconv_out = _instance_norm(deconv_out, train=is_training, name='bn')
#             bn_deconv_out = layers.batch_norm(deconv_out, train=is_training, name='bn')
#         else:
#             bn_deconv_out = deconv_out
#         drop_out_bn = dropout(
#             bn_deconv_out, keep_prob, is_training=is_training, name='dropout')

#         if fusion_id is not None:
#             layer_dict['cur_input'] = tf.concat(
#               (drop_out_bn, layer_dict['encoder_{}'.format(fusion_id)]),
#               axis=-1)
#         else:
#             layer_dict['cur_input'] = drop_out_bn

#         layer_dict['cur_input'] = tf.nn.relu(layer_dict['cur_input'])

#         return layer_dict['cur_input']


def max_pool(x,
             name='max_pool',
             filter_size=2,
             stride=None,
             padding='VALID',
             switch=False):
    """ 
    Max pooling layer 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.

    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    if switch == True:
        return tf.nn.max_pool_with_argmax(
            x,
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            Targmax=tf.int64,
            name=name)
    else:
        return tf.nn.max_pool(
            x,
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            name=name), None

def get_shape2D(in_val):
    """
    Return a 2D shape 

    Args:
        in_val (int or list with length 2)

    Returns:
        list with length 2
    """
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

def get_shape4D(in_val):
    """
    Return a 4D shape

    Args:
        in_val (int or list with length 2)

    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

# def dropout(x, keep_prob, is_training, name='dropout'):
#     """ 
#     Dropout 

#     Args:
#         x (tf.tensor): a tensor 
#         keep_prob (float): keep prbability of dropout
#         is_training (bool): whether training or not
#         name (str): name scope

#     Returns:
#         tf.tensor with name 'name'
#     """

#     return tf.layers.dropout(x, rate=1 - keep_prob, 
#                             training=is_training, name=name)

# def batch_norm(x, train=True, name='bn'):
#     """ 
#     batch normal 

#     Args:
#         x (tf.tensor): a tensor 
#         name (str): name scope
#         train (bool): whether training or not

#     Returns:
#         tf.tensor with name 'name'
#     """
#     return tf.contrib.layers.batch_norm(x, decay=0.9, 
#                           updates_collections=None,
#                           epsilon=1e-5, scale=False,
#                           is_training=train, scope=name)

