""" LLORMA training utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


def init_session():
    """Initializes TF session

    Returns
    -------
    obj: tf.Session
        Returns TF Session
    """
    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=GPU_MEMORY_FRAC)
    # gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    # session = tf.Session(config=gpu_config)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)
    session.run(tf.compat.v1.global_variables_initializer())
    return session


def get_train_op(optimizer, loss, var_list):
    """ Get a train operation

    Parameters
    ----------
    optimizer : obj
        Valid TensorFlow optimizer,
        e.g. tf.train.GradientDescentOptimizer
    loss : obj
        TF variable
    var_list : obj
        List of TF variables
    """
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    # capped_gvs = [(tf.clip_by_value(grad, -100.0, 100.0), var)
    #               for grad, var in gvs]
    capped_gvs = gvs
    train_op = optimizer.apply_gradients(capped_gvs)
    return train_op


def init_latent_mat(n, rank, mu_val, std_val):
    """Initialize a matrix for the latent factors

    Parameters
    ----------
    n : int
        Number of user/items
    rank : int
        Size of the latent dimension
    mu_val : float
        Unscaled mean value
    std_val : float
        Unscaled standard deviation value
    """
    _mu = math.sqrt(mu_val / rank)
    _std = math.sqrt((math.sqrt(mu_val * mu_val + std_val * std_val) - mu_val) / rank)
    return tf.Variable(
        tf.compat.v1.truncated_normal([n, rank], _mu, _std, dtype=tf.float64))
