#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel

import lib.models.layers as L


VGG_MEAN = [103.939, 116.779, 123.68]


def resize_tensor_image_with_smallest_side(image, small_size):
    """
    Resize image tensor with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (tf.tensor): 4-D Tensor of shape
            [batch, height, width, channels]
            or 3-D Tensor of shape [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        Image tensor with the original aspect ratio and
        smallest side = small_size.
        If images was 4-D, a 4-D float Tensor of shape
        [batch, new_height, new_width, channels].
        If images was 3-D, a 3-D float Tensor of shape
        [new_height, new_width, channels].
    """
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(small_size, tf.float32)
    new_height, new_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
        lambda: ((height / width) * new_shorter_edge, new_shorter_edge))

    return tf.image.resize_images(
        tf.cast(image, tf.float32),
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])


class BaseVGG16(BaseModel):
    def __init__(self):

        self._trainable = False
        self._switch = False

    def _sub_mean(self, inputs):
        VGG_MEAN = [103.939, 116.779, 123.68]
        red, green, blue = tf.split(axis=3,
                                    num_or_size_splits=3,
                                    value=inputs)
        input_bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        return input_bgr

    def _create_vgg_conv(self, inputs, layer_dict, data_dict={}):

        self.receptive_s = 1
        self.stride_t = 1
        self.receptive_size = {}
        self.stride = {}
        self.cur_input = self._sub_mean(inputs)

        def conv_layer(filter_size, out_dim, name):
            init_w = tf.keras.initializers.he_normal()
            # init_w = None
            layer_dict[name] = conv(self.cur_input, filter_size, out_dim, name, init_w=init_w)
            self.receptive_s = self.receptive_s + (filter_size - 1) * self.stride_t
            self.receptive_size[name] = self.receptive_s
            self.stride[name] = self.stride_t
            self.cur_input = layer_dict[name]

        def pool_layer(name, switch=False, padding='SAME'):
            layer_dict[name], _ =\
                L.max_pool(self.cur_input, name, padding=padding, switch=switch)
            self.receptive_s = self.receptive_s + self.stride_t
            self.receptive_size[name] = self.receptive_s
            self.stride_t = self.stride_t * 2
            self.stride[name] = self.stride_t
            self.cur_input = layer_dict[name]

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu,
                       trainable=False, data_dict=data_dict):

            conv_layer(3, 64, 'conv1_1')
            conv_layer(3, 64, 'conv1_2')
            pool_layer('pool1', switch=self._switch)

            conv_layer(3, 128, 'conv2_1')
            conv_layer(3, 128, 'conv2_2')
            pool_layer('pool2', switch=self._switch)

            conv_layer(3, 256, 'conv3_1')
            conv_layer(3, 256, 'conv3_2')
            conv_layer(3, 256, 'conv3_3')
            pool_layer('pool3', switch=self._switch)

            conv_layer(3, 512, 'conv4_1')
            conv_layer(3, 512, 'conv4_2')
            conv_layer(3, 512, 'conv4_3')
            pool_layer('pool4', switch=self._switch)

            conv_layer(3, 512, 'conv5_1')
            conv_layer(3, 512, 'conv5_2')
            conv_layer(3, 512, 'conv5_3')
            pool_layer('pool5', switch=self._switch)

        return self.cur_input

class BaseVGG19(BaseVGG16):
    def _create_vgg_conv(self, inputs, layer_dict, data_dict={}):

        self.receptive_s = 1
        self.stride_t = 1
        self.receptive_size = {}
        self.stride = {}
        self.cur_input = self._sub_mean(inputs)

        def conv_layer(filter_size, out_dim, name):
            init_w = tf.keras.initializers.he_normal()
            # init_w = None
            layer_dict[name] = conv(self.cur_input, filter_size, out_dim, name, init_w=init_w)
            self.receptive_s = self.receptive_s + (filter_size - 1) * self.stride_t
            self.receptive_size[name] = self.receptive_s
            self.stride[name] = self.stride_t
            self.cur_input = layer_dict[name]

        def pool_layer(name, switch=False, padding='SAME'):
            layer_dict[name], _ =\
                L.max_pool(self.cur_input, name, padding=padding, switch=switch)
            self.receptive_s = self.receptive_s + self.stride_t
            self.receptive_size[name] = self.receptive_s
            self.stride_t = self.stride_t * 2
            self.stride[name] = self.stride_t
            self.cur_input = layer_dict[name]

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu,
                       trainable=False, data_dict=data_dict):

            conv_layer(3, 64, 'conv1_1')
            conv_layer(3, 64, 'conv1_2')
            pool_layer('pool1', switch=self._switch)

            conv_layer(3, 128, 'conv2_1')
            conv_layer(3, 128, 'conv2_2')
            pool_layer('pool2', switch=self._switch)

            conv_layer(3, 256, 'conv3_1')
            conv_layer(3, 256, 'conv3_2')
            conv_layer(3, 256, 'conv3_3')
            conv_layer(3, 256, 'conv3_4')
            pool_layer('pool3', switch=self._switch)

            conv_layer(3, 512, 'conv4_1')
            conv_layer(3, 512, 'conv4_2')
            conv_layer(3, 512, 'conv4_3')
            conv_layer(3, 512, 'conv4_4')
            pool_layer('pool4', switch=self._switch)

            conv_layer(3, 512, 'conv5_1')
            conv_layer(3, 512, 'conv5_2')
            conv_layer(3, 512, 'conv5_3')
            conv_layer(3, 512, 'conv5_4')
            pool_layer('pool5', switch=self._switch)

        return self.cur_input
 