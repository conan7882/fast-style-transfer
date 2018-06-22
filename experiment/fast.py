#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fast.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import platform
import scipy.misc
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorcv.dataflow.image import ImageFromFile

import sys
sys.path.append('../')

from lib.models.faststyle import FastStyle
# from lib.dataflow.facades import ImagePair


if platform.node() == 'Qians-MacBook-Pro.local':
    DATA_PATH = '/Users/gq/workspace/Dataset/CMP_facade_DB_base/'
    SAVE_PATH = '/Users/gq/tmp/dlout/pix2pix/'
    # RESULT_PATH = '/Users/gq/tmp/ram/center/result/'
elif platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/dataset/COCO/train2014/'
    SAVE_PATH = '/home/qge2/workspace/data/out/fast/'
    VGG_PATH = '/home/qge2/workspace/data/pretrain/vgg/vgg16.npy'
else:
    VGG_PATH = 'E:/GITHUB/workspace/CNN/pretrained/vgg16.npy'
    DATA_PATH = 'E:/Dataset/COCO/val2017/val2017/'
    SAVE_PATH = 'E:/GITHUB/workspace/CNN/fast/'

def resize_image_with_smallest_side(image, small_size):
    """
    Resize single image array with smallest side = small_size and
    keep the original aspect ratio.
    Args:
        image (np.array): 2-D image of shape
            [height, width] or 3-D image of shape
            [height, width, channels] or 4-D of shape
            [1, height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.
    """
    im_shape = image.shape
    shape_dim = len(im_shape)
    assert shape_dim <= 4 and shape_dim >= 2,\
        'Wrong format of image!Shape is {}'.format(im_shape)

    if shape_dim == 4:
        image = np.squeeze(image, axis=0)
        height = float(im_shape[1])
        width = float(im_shape[2])
    else:
        height = float(im_shape[0])
        width = float(im_shape[1])

    if height <= width:
        new_height = int(small_size)
        new_width = int(new_height/height * width)
    else:
        new_width = int(small_size)
        new_height = int(new_width/width * height)

    if shape_dim == 2:
        im = scipy.misc.imresize(image, (new_height, new_width))
    elif shape_dim == 3:
        im = scipy.misc.imresize(image, (new_height, new_width, image.shape[2]))
    else:
        im = scipy.misc.imresize(image, (new_height, new_width, im_shape[3]))
        im = np.expand_dims(im, axis=0)

    return im

def im_normalize(im):
    return scipy.misc.imresize(im, [256, 256])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test')

    parser.add_argument('--batch', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')

    parser.add_argument('--style', default=1e-3, type=float,
                        help='Style weight')
    parser.add_argument('--content', default=100, type=float,
                        help='Content weight')
    parser.add_argument('--tv', default=1e-6, type=float,
                        help='TV weight')

    parser.add_argument('--sim', default='la_muse.jpg', type=str,
                        help='Style image name')
    
    return parser.parse_args()

if __name__ == '__main__':
    FLAGS = get_args()

    style_im = scipy.misc.imread('../data/{}'.format(FLAGS.sim))
    style_im = [resize_image_with_smallest_side(style_im, 512)]
    style_shape = [style_im[0].shape[0], style_im[0].shape[1]]

    model = FastStyle(content_size=256,
                      style_size=style_shape,
                      c_channel=3,
                      s_channel=3,
                      vgg_path=VGG_PATH,
                      s_weight=FLAGS.style,
                      c_weight=FLAGS.content,
                      tv_weight=FLAGS.tv)

    if FLAGS.train:
        # style_im = [scipy.misc.imresize(scipy.misc.imread('../data/{}'.format(FLAGS.sim)),
        #             [256, 256])]

        train_data = ImageFromFile(
            ext_name='.jpg',
            data_dir=DATA_PATH, 
            num_channel=3,
            shuffle=True,
            batch_dict_name=['im'],
            pf=im_normalize)
        train_data.setup(epoch_val=0, batch_size=FLAGS.batch)

        test_im = scipy.misc.imread('../data/cat.png')
        test_im = [resize_image_with_smallest_side(test_im, 256)]

        model.create_model()
        model.create_generate_model()

        train_op = model.train_op()
        train_summary_op = model.get_train_summary()
        test_summary_op = model.get_test_summary()

        loss_op = model.get_loss()
        gen_op = model.layers['gen_im']

        # with tf.variable_scope(tf.get_variable_scope()) as scope:
        #     scope.reuse_variables()
        #     test_model.create_generate_model()
        #     test_summary_op = test_model.get_summary()

        writer = tf.summary.FileWriter(SAVE_PATH)
        # saver = tf.train.Saver()
        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        with tf.Session(config=sessconfig) as sess:
            sess.run(tf.global_variables_initializer(),
                     feed_dict={model.style_image: style_im})
            writer.add_graph(sess.graph)

            for i in range(0, 40000):
                model.set_is_training(True)
                batch_data = train_data.next_batch_dict()

                _, loss, summary, gen_im = sess.run(
                    [train_op, loss_op, train_summary_op, gen_op],
                    feed_dict={model.lr: 1e-3,
                               model.content_image: batch_data['im']})

                if i % 100 == 0:
                    writer.add_summary(summary, i)
                    model.set_is_training(True)
                    summary = sess.run(
                    test_summary_op,
                    feed_dict={
                               model.test_image: test_im})
                    writer.add_summary(summary, i)
                print('step: {}, loss: {}'.format(i, loss))

        writer.close()

    if FLAGS.test:
        model.create_generate_model()
        gen_op = model.layers['gen_im']
        summary_op = model.get_summary()

        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        with tf.Session(config=sessconfig) as sess:
            sess.run(tf.global_variables_initializer())
            
