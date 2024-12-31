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

Purpose: Hypermodels for keras tuner hyperparameter tuning
"""
import keras_tuner
import tensorflow as tf

import dataloader
import hparams
import train
import util


class HyperModel(keras_tuner.HyperModel):
    """
    hypermodel class. This class can build models by varying following hyperparameters:
    architecture: from hparams.ARCHS
    algorithms: from hparams.ALGORITHM
    activations: from hparams.ACTIVATION
    batchnorm: [True, False]
    feedback_matrix init scheme: hparams.BW_INIT
    learning rate: min_value=1e-4, max_value=.1, step=10, sampling='log'
    dropout regularization rate: from hparams.DROPOUT_RATE
    Any of these hparams can be prefixed as well. In that case, they are not varied.
    """

    def __init__(self, config, *args, **kwargs):
        """
        init
        :param config: config dict
        :param args: superclass args
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, **kwargs)
        self.config = config

    def build(self, hp):
        """
        build the model using hp
        :param hp: keras tuner hyperparameters instance
        :return: keras model
        """
        self.config.algorithm = hp.Choice('algo', hparams.ALGORITHM)
        self.config.activation = hp.Choice('act', hparams.ACTIVATION)
        self.config.arch = hp.Choice('arch', hparams.ARCHS)
        self.config.batchnorm = hp.Boolean('bn')
        self.config.bw_init = hp.Choice('F', hparams.BW_INIT)

        if 'val' in self.config.objective:
            self.config.reg_rate = hp.Choice('reg_rate', hparams.DROPOUT_RATE)
        else:
            # self.config.use_bias = hp.Boolean('use_bias')
            self.config.reg_rate = False
            lr = hp.Float('lr', min_value=1e-4, max_value=.1, step=10, sampling='log')

        model = train.Model(self.config, name='model')
        if self.config.problem_type == 'classification':
            loss = 'categorical_crossentropy'
            metrics = util.error_rate
        elif self.config.problem_type == 'reconstruction':
            loss = 'mean_squared_error'
            metrics = None
        else:
            raise ValueError(f'Unknown problem type: {self.config.problem_type}')

        model.compile(run_eagerly=False,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=loss,
                      metrics=metrics)
        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        fit the model with keras tuner
        :param hp: keras tuner hyperparameters instance
        :param model: keras model
        :param args: passed on to model.fit() args
        :param kwargs: passed on to model.fit() kwargs
        :return: keras model.fit() result
        """
        # set tensorboard attribute in model instance. needed when visualizing histograms
        if self.config.histogram_freq:
            tboard_handle = [x for x in kwargs['callbacks'] if isinstance(x, tf.keras.callbacks.TensorBoard)]
            if len(tboard_handle):
                model.tboard = tboard_handle[0]

        dataset = dataloader.get_dataset(self.config, training=True)
        validation_data = dataloader.get_dataset(self.config, training=False)

        return super().fit(hp, model, dataset, *args, validation_data=validation_data, **kwargs)
