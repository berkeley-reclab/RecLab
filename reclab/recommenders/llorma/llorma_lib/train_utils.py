from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import time


def init_session():
    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=GPU_MEMORY_FRAC)
    # gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    # session = tf.Session(config=gpu_config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    return session


def get_train_op(optimizer, loss, var_list):
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    # capped_gvs = [(tf.clip_by_value(grad, -100.0, 100.0), var)
    #               for grad, var in gvs]
    capped_gvs = gvs
    train_op = optimizer.apply_gradients(capped_gvs)
    return train_op


def init_latent_mat(n, rank, mu, std):
    _mu = math.sqrt(mu / rank)
    _std = math.sqrt((math.sqrt(mu * mu + std * std) - mu) / rank)
    return tf.Variable(
        tf.truncated_normal([n, rank], _mu, _std, dtype=tf.float64))
