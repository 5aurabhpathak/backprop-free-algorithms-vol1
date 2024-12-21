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

Purpose: WTF projection layer to output space using random projection matrices (aka feedback matrices from PoV of FA)
"""
import tensorflow as tf
from absl import logging

import hparams
import util


class DenseProject(tf.keras.layers.Layer):
    """linearly projects a hidden layer output to network output space"""

    def __init__(self, units, config, **kwargs):
        """
        init
        :param units: number of hidden units
        :param config: configuration dictionary
        :param kwargs: additional arguments for superclass
        """
        self.config = config
        self.units = units
        self.kernel = None
        super().__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        """
        build the projection layer
        :param input_shape: shape of input tensor
        """
        super().build(input_shape)
        bw_init = util.get_bw_fn(self.config.bw_init)(seed=hparams.RANDOM_SEED, gain=1.)

        if self.config.problem_type == "classification":
            shape = (self.units, self.config.output_shape[1])
        elif self.config.problem_type == "reconstruction":
            shape = (self.units, input_shape[1])
        else:
            raise ValueError(f'Unknown problem type: {self.config.problem_type}')

        self.kernel = bw_init(shape=shape, dtype=tf.float32)
        logging.info(f'Projection kernel shape: {self.kernel.shape}')

    def call(self, x):
        """
        project the hidden layer output to network output space
        :param x: input tensor
        :return: projected tensor in the network output space
        """
        return tf.matmul(x, self.kernel)
