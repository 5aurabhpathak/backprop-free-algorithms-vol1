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

Purpose: Sequential Model
"""
import tensorflow as tf

import hparams
import util


def get_model(config):
    """
    define dense sequential model based on architecture specified in config in the following string
    format: units-units-...-units
    :param config: config dict containing model specifications
    :return: keras model
    """
    model = tf.keras.Sequential(name='model')
    for i, item in enumerate(config.arch.strip().lower().split('-')):
        units = int(item)
        lyr = tf.keras.layers.Dense(units, use_bias=config.use_bias,
                                    kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.,
                                                                                        seed=hparams.RANDOM_SEED
                                                                                        )
                                    )

        model.add(lyr)
        if config.get('batchnorm'):
            model.add(tf.keras.layers.BatchNormalization())

        act = util.get_activation(config.activation) if isinstance(config.activation, str) else config.activation
        model.add(tf.keras.layers.Activation(act))
        if config.reg_rate:
            model.add(tf.keras.layers.Dropout(rate=config.reg_rate, seed=hparams.RANDOM_SEED))

    if config.problem_type == 'classification':
        model.add(tf.keras.layers.Dense(config.output_shape[1], name='DenseOut',
                                        kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.,
                                                                                            seed=hparams.RANDOM_SEED)))
        model.add(tf.keras.layers.Activation('softmax'))
    elif config.problem_type == 'reconstruction':
        model.add(tf.keras.layers.Dense(config.output_shape[1], name='DenseOut',
                                        kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.,
                                                                                            seed=hparams.RANDOM_SEED)))
        model.add(tf.keras.layers.Activation('tanh' if config.scale_data == 'standard' else 'sigmoid'))
    return model
