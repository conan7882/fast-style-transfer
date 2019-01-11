#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf


def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    """ Display averaged intermediate results for a period during training.
    The intermediate result will be displayed as:
    [step: global_step] name_list[0]: scaler_sum_list[0]/step ...
    Those result will be saved as summary as well.
    Args:
        global_step (int): index of current iteration
        step (int): number of steps for this period
        scaler_sum_list (float): list of summation of the intermediate
            results for this period
        name_list (str): list of display name for each intermediate result
        collection (str): list of graph collections keys for summary
        summary_val : additional summary to be saved
        summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
    """
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)