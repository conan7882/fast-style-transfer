#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: faststyle.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import numpy as np
import tensorflow as tf
from tensorcv.models.base import BaseModel

import lib.models.layers as L
from lib.models.vgg import BaseVGG19
from lib.models.vgg import BaseVGG16
import lib.utils.viz as viz

INIT_W_STD = 0.1
INIT_W = tf.random_normal_initializer(stddev=INIT_W_STD)

class FastStyle(BaseVGG19):
    def __init__(self,
                 content_size=None,
                 style_size=None,
                 c_channel=None,
                 s_channel=None,
                 vgg_path=None,
                 s_weight=1e-3,
                 c_weight=100,
                 tv_weight=1e-6):

        self._switch = False

        if content_size == None:
            self._c_size = [None, None]
        else:
            self._c_size = L.get_shape2D(content_size)
        if style_size == None:
            self._s_size = [None, None]
        else:
            self._s_size = L.get_shape2D(style_size)
        self._n_c_c = c_channel
        self._n_s_c = s_channel
        self._vgg_path = vgg_path

        self._s_w = s_weight
        self._c_w = c_weight
        self._tv_w = tv_weight

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        self.layers = {}

    def _create_train_input(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.content_image = tf.placeholder(
            tf.float32,
            name='content',
            shape=[None, self._c_size[0], self._c_size[1], self._n_c_c])
        self.style_image = tf.placeholder(
            tf.float32,
            name='style',
            shape=[None, self._s_size[0], self._s_size[1], self._n_s_c])

    def create_train_model(self):
        self.set_is_training(True)
        self._create_train_input()

        self.vgg_layer = {}
        vgg_param = np.load(self._vgg_path, encoding='latin1').item()

        # Labels
        # style
        self._create_vgg_conv(
            self.style_image, self.vgg_layer, data_dict=vgg_param)
        self.s_feats_label = self._comp_style_feat('label', is_label=True)

        # content
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            self._create_vgg_conv(
                self.content_image, self.vgg_layer, data_dict=vgg_param)
        self.c_feats_label = self._comp_content_feat('label')

        # Generate image
        self.layers['gen_im'] = self._create_style_transfer_net(self.content_image)
        self.layers['gen_im'] = tf.reshape(
                self.layers['gen_im'],
                [-1, self._c_size[0], self._c_size[1], self._n_c_c])
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            scope.reuse_variables()
            self._create_vgg_conv(
                self.layers['gen_im'], self.vgg_layer, data_dict=vgg_param)

        self.c_feats_gen = self._comp_content_feat('generate')
        self.s_feats_gen = self._comp_style_feat('generate')

        self.train_op = self.train_op()
        self.train_summary_op = self.get_train_summary()
        self.loss_op = self.get_loss()

        self.global_step = 0

    def create_generate_model(self):
        self.set_is_training(False)
        self._create_generate_input()

        self.layers['gen_im'] = self._create_style_transfer_net(self.test_image)
        self.valid_summary_op = self.get_valid_summary()
        self.global_step = 0

    def _create_generate_input(self):
        self.test_image = tf.placeholder(
            tf.float32,
            name='test_content',
            shape=[None, None, None, self._n_c_c])

    def _create_style_transfer_net(self, inputs):
        with tf.variable_scope('style_net', reuse=tf.AUTO_REUSE) as scope:
            self.layers['cur_input'] = inputs
            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.conv],
                           layer_dict=self.layers,
                           is_training=self.is_training,
                           bn=True,
                           nl=tf.nn.relu,
                           init_w=INIT_W,
                           pad_type='REFLECT',
                           ):
                L.conv(filter_size=9, stride=1, out_dim=32, name='conv1')
                L.conv(filter_size=3, stride=2, out_dim=64, name='conv2')
                L.conv(filter_size=3, stride=2, out_dim=128, name='conv3')

            with arg_scope([L.residual_block],
                           layer_dict=self.layers,
                           is_training=self.is_training,
                           init_w=INIT_W):
                L.residual_block(128, name='res1')
                L.residual_block(128, name='res2')
                L.residual_block(128, name='res3')
                L.residual_block(128, name='res4')
                L.residual_block(128, name='res5')

            with arg_scope([L.transpose_conv],
                           stride=2,
                           layer_dict=self.layers,
                           bn=True,
                           init_w=INIT_W):
                L.transpose_conv(filter_size=3, out_dim=64, name='deconv1')
                L.transpose_conv(filter_size=3, out_dim=32, name='deconv2')

            L.conv(
                filter_size=9,
                stride=1,
                out_dim=self._n_c_c,
                layer_dict=self.layers,
                bn=True,
                init_w=INIT_W,
                padding='SAME',
                pad_type='REFLECT',
                trainable=True,
                is_training=self.is_training,
                name='convout')

            self.layers['cur_input'] = tf.tanh(self.layers['cur_input'])
            self.layers['cur_input'] = (self.layers['cur_input'] + 1.0) / 2.0 * 255.
            
            return self.layers['cur_input']

    def _comp_content_feat(self, name):
        with tf.name_scope('content_feat_{}'.format(name)):
            c_feats = {}
            for key in self.content_layers:
                shape = self.vgg_layer[key].shape.as_list()
                CHW = float(np.prod(shape[1:]))
                c_feats[key] = self.vgg_layer[key] * 1.0
                c_feats['{}_CHW'.format(key)] = CHW
            return c_feats

    def _comp_style_feat(self, name, is_label=False):
        with tf.name_scope('style_feat_{}'.format(name)):
            s_feats = {}
            if is_label == True:
                with tf.variable_scope('style_feat_label'):
                    for key in self.style_layers:
                        cur_gram_mat, _ = self._gram_matrix(key, 'G_{}'.format(key))
                        cur_gram_mat = tf.squeeze(cur_gram_mat, axis=0)
                        cur_gram_mat = tf.expand_dims(cur_gram_mat, axis=0)
                        s_feats[key] = tf.get_variable(
                            'style_feats_{}'.format(key),
                            initializer=cur_gram_mat,
                            trainable=False)
            else:
                for key in self.style_layers:
                    cur_gram_mat, N = self._gram_matrix(key, 'G_{}'.format(key))
                    s_feats[key] = cur_gram_mat
                    s_feats['{}_N'.format(key)] = N
            return s_feats

    def _gram_matrix(self, layer_name, name):
        with tf.name_scope(name):
            feat_map = self.vgg_layer[layer_name]
            shape = feat_map.shape.as_list()
            CHW = float(np.prod(shape[1:]))
            flatten_feat = tf.reshape(feat_map, (-1, shape[1] * shape[2], shape[-1]))
            g_mat = tf.matmul(tf.transpose(flatten_feat, perm=[0, 2, 1]), flatten_feat)
            return g_mat * 1.0 / CHW, shape[-1]

    def train_op(self):
        with tf.name_scope('train'):
            opt = tf.train.AdamOptimizer(beta1=0.5,
                                         learning_rate=self.lr)
            loss = self.get_loss()
            grads = opt.compute_gradients(loss)
            return opt.apply_gradients(grads)
            
    def get_loss(self):
        try:
            return self.loss
        except AttributeError:
            self.loss = self._get_loss()
            return self.loss

    def _get_loss(self):
        with tf.name_scope('loss'):
            shape = self.content_image.get_shape()
            b_size = shape[-1].value
            with tf.name_scope('style'):
                style_loss = 0
                dict_1 = self.s_feats_label
                dict_2 = self.s_feats_gen
                for key in self.style_layers:
                    N = dict_2['{}_N'.format(key)]
                    style_loss += 2. * tf.nn.l2_loss(dict_1[key] - dict_2[key]) / (b_size * N ** 2)

            with tf.name_scope('content'):
                content_loss = 0
                dict_1 = self.c_feats_label
                dict_2 = self.c_feats_gen
                for key in self.content_layers:
                    CHW = dict_2['{}_CHW'.format(key)]
                    content_loss += 2. * tf.nn.l2_loss(dict_1[key] - dict_2[key]) / (b_size * CHW)
            with tf.name_scope('total_variation'):
                tv_loss = 2. * self._total_variation(self.layers['gen_im']) / b_size

            self.style_loss = style_loss
            self.content_loss = content_loss
            return self._s_w * style_loss + self._c_w * content_loss + self._tv_w * tv_loss

    def _total_variation(self, image):
        var_x = tf.pow(image[:, 1:, :-1, :] - image[:, :-1, :-1, :], 2)
        var_y = tf.pow(image[:, :-1, 1:, :] - image[:, :-1, :-1, :], 2)
        return tf.reduce_sum(var_x + var_y)

    def get_train_summary(self):
        with tf.name_scope('train'):
            train_g = tf.clip_by_value(self.layers['gen_im'], 0, 255)
            tf.summary.image(
                'generate', tf.cast(train_g, tf.uint8),
                collections=['train'])
            tf.summary.image(
                'input', tf.cast(self.content_image, tf.uint8),
                collections=['train'])
            return tf.summary.merge_all(key='train')

    def get_valid_summary(self):
        with tf.name_scope('valid'):
            test_g = tf.clip_by_value(self.layers['gen_im'], 0, 255)
            tf.summary.image(
                'generate', tf.cast(test_g, tf.uint8),
                collections=['valid'])
            tf.summary.image(
                'input', tf.cast(self.test_image, tf.uint8),
                collections=['valid'])
            return tf.summary.merge_all(key='valid')

    def train(self, sess, train_data, num_iteration=100, summary_writer=None):
        display_name = ['loss', 'style loss', 'content loss']
        batch_data = train_data.next_batch_dict()

        loss_sum = 0
        style_loss_sum = 0
        content_loss_sum = 0
        step = 0
        current_summary = None
        for i in range(0, num_iteration):
            self.global_step += 1
            step += 1

            batch_data = train_data.next_batch_dict()

            _, loss, style_loss, content_loss = sess.run(
                [self.train_op, self.loss_op, self.style_loss, self.content_loss],
                feed_dict={self.lr: 1e-3,
                           self.content_image: batch_data['im']})

            loss_sum += loss
            style_loss_sum += style_loss
            content_loss_sum += content_loss

        current_summary= sess.run(
            self.train_summary_op,
            feed_dict={self.content_image: batch_data['im']})

        viz.display(
            global_step=self.global_step,
            step=step,
            scaler_sum_list=[loss_sum, style_loss_sum, content_loss_sum],
            name_list=display_name,
            collection='train',
            summary_val=current_summary,
            summary_writer=summary_writer)

    def generate(self, sess, test_im, summary_writer=None, save_name=None):
        self.global_step += 1
        if summary_writer is not None:
            current_summary = sess.run(self.valid_summary_op, feed_dict={self.test_image: test_im})
            summary_writer.add_summary(current_summary, self.global_step)

        transfered = sess.run(self.layers['gen_im'], feed_dict={self.test_image: test_im})
        if save_name is not None:
            scipy.misc.imsave(save_name, transfered[0])
            print('Result is saved in {}'.format(save_name))
        return transfered


        #         writer.add_summary(summary, i)
        #         model.set_is_training(True)
        #         summary = sess.run(
        #                 test_summary_op,
        #                 feed_dict={model.test_image: test_im})
        #             writer.add_summary(summary, i)
        #         print('step: {}, loss: {:0.2f}'.format(i, loss))
        # _, loss, gen_im = sess.run(
        #             [train_op, loss_op, train_summary_op, gen_op],
        #             feed_dict={model.lr: 1e-3,
        #                        model.content_image: batch_data['im']})

        #     display(global_step,
        #     step,
        #     scaler_sum_list,
        #     name_list,
        #     collection,
        #     summary_val=None,
        #     summary_writer=None,
        #     )

