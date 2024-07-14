"""
MIT license notice
© 2024 Saurabh Pathak. All rights reserved
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Purpose: utility functions and classes
"""
import os
import shutil
import typing

import tensorflow as tf


def get_bw_fn(bw_init):
    """
    gets initializer for auxiliary layer fixed weights
    :param bw_init: string or callable (callable is returned as is)
    :return: weight initializer callable
    """
    if bw_init == 'ortho':
        bw_fn = tf.keras.initializers.Orthogonal
    elif bw_init == 'lecun':
        bw_fn = tf.keras.initializers.LecunNormal
    elif isinstance(bw_init, typing.Callable):
        return bw_init
    else:
        raise ValueError(f'Unknown initialization: {bw_init}')
    return bw_fn


def get_activation(act_str):
    """
    gets activation function from string
    :param act_str: string containing the name of activation function or a callable (callable is returned as is)
    :return: callable representing the activation function
    """
    if act_str == 'tanh':
        return tf.tanh
    elif act_str == 'relu':
        # adopt the following definition of relu to make it work with zero initialization
        return lambda x: tf.where(tf.greater_equal(x, 0.), x, 0.)
    elif act_str == 'lrelu':
        return tf.keras.layers.Activation('leaky_relu')
    elif act_str == 'sigmoid':
        return tf.sigmoid
    elif act_str == 'swish':
        return lambda x: x * tf.sigmoid(x)
    elif isinstance(act_str, typing.Callable):
        return act_str
    raise ValueError(f'Unrecognized activation: {act_str}')


def weight_alignment(w, f, f_cur=None):
    """
    measures layer weight alignment as defined in ref: https://arxiv.org/pdf/2011.12428
    :param w: forward weight matrix of the layer
    :param f: fixed projection matrix of the auxiliary layer corresponding to this layer
    :param f_cur: fixed projection matrix of the auxiliary layer corresponding to the previous layer
    :return: cosine of the weight alignment
    """
    if f_cur is not None:
        f = tf.matmul(f, f_cur, transpose_b=True)
    f = tf.keras.backend.flatten(f)
    w = tf.keras.backend.flatten(w)
    return cosines(w, f)


def cosines(v1, v2, axis=None, normalize=True, keepdims=False):
    """
    computes inner product or cosine of the angle between two vectors
    :param v1: vector or tensor
    :param v2: vector or tensor
    :param axis: which axis to consider
    :param normalize: whether to normalize the vectors in question along the axis given
    :param keepdims: keep the dimensionality of the result same as the original tensors
    :return: inner product or cosine angle
    """
    if normalize:
        v1 = tf.math.l2_normalize(v1, axis=axis)
        v2 = tf.math.l2_normalize(v2, axis=axis)
    cos = tf.reduce_sum(v1 * v2, axis=axis, keepdims=keepdims)
    return cos


def error_rate(yt, yp):
    """
    computes error rate as 100 - accuracy
    :param yt: true labels
    :param yp: predicted labels
    :return: error rate
    """
    return (1. - tf.keras.metrics.categorical_accuracy(yt, tf.math.softmax(yp))) * 1e2


def ensure_empty_dir(dirname):
    """
    creates a directory with a given name if it does not exist. clears its contents otherwise
    :param dirname: name of the directory to be created or cleared
    :return: the directory name argument
    """
    try:
        os.makedirs(dirname)
    except FileExistsError:
        shutil.rmtree(dirname, ignore_errors=True)
        os.makedirs(dirname)
    return dirname


def add_directory(newdir, basedir=None, clear=False):
    """
    adds a directory or subdirectory under a base directory if basedir is given, additionally clears it if clear is set
    and directory/subdirectory in question exists
    :param newdir: directory name
    :param basedir: base directory name or None
    :param clear: when set ensures that the directory is empty if it already exists
    :return: newdir or basedir/newdir pathname
    """
    directory = newdir if basedir is None else os.path.join(basedir, newdir)
    if clear or not os.path.exists(directory):
        ensure_empty_dir(directory)
    return directory
