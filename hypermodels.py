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
import pandas as pd
import tensorflow as tf

import dataloader
import hparams
import util
import train


class HyperModel(keras_tuner.HyperModel):
    """
    hypermodel class. This class can build models by varying following hyperparameters:
    architecture: from hparams.ARCHS
    algorithms: from hparams.ALGORITHM
    activations: from hparams.ACTIVATION
    batchnorm: [True, False]
    feedback_matrix init scheme: hparams.BW_INIT
    learning rate: min_value=1e-4, max_value=.1, step=10, sampling='log'
    skew type: only when hp exists. from hparams.SKEW_TYPES
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
        self.architectures = hparams.ARCHS

        # if validation error is the tuning objective, then best learning rate should be known. If not, it must be set
        # Assumption: the best learning rate on training loss is known before tuning on the val loss or error
        # the above assumption can be violated by explicitly providing a fixed learning rate when tuning with val metric
        if ('val' in config.objective or 'error' in config.objective) and self.config.dataset in hparams.DATASETS:
            try:
                self.lr_df = pd.read_json(f'best_lr.json')
            except (ValueError, FileNotFoundError):
                print('could not read learning rate file. If it is not set otherwise, this script will crash')

        # similarly for dropout rate, the best regularization rate must be known with respect to a smaller dataset
        # before running more than 10 trials for the whole dataset
        if (self.config.executions_per_trial >= 10 and 'val' in self.config.objective and
                self.config.get('reg_rate') is None):
            self.reg_df = pd.read_json('best_reg_rate.json')

    def get_lr(self, hp):
        """
        get the learning rate
        :param hp: keras tuner hyperparameters instance
        :return: learning rate
        """
        if 'lr' not in hp:
            try:
                lr_df = self.lr_df[(self.lr_df.arch == self.config.arch) &
                                      (self.lr_df.dataset == self.config.dataset) &
                                      (self.lr_df.algo == self.config.algorithm) &
                                      (self.lr_df.F == self.config.bw_init) &
                                      (self.lr_df.act == self.config.activation)]
                if self.config.get('skew_type'):
                    lr_df = lr_df[lr_df.skew_type == self.config.skew_type]
                lr = float(lr_df.lr)
            except AttributeError:
                raise ValueError('either lr has to be set or lr_df must be present in the working directory')
        else:
            lr = hp.get('lr')
        return lr

    def build(self, hp):
        """
        build the model using hp
        :param hp: keras tuner hyperparameters instance
        :return: keras model
        """
        self.config.algorithm = hp.Choice('algo', hparams.ALGORITHM)
        self.config.activation = hp.Choice('act', hparams.ACTIVATION)
        self.config.arch = hp.Choice('arch', self.architectures)
        self.config.batchnorm = hp.Boolean('bn')
        self.config.bw_init = hp.Choice('F', hparams.BW_INIT)

        if 'val' in self.config.objective:
            if 'reg_rate' in hp:
                self.config.reg_rate = hp.get('reg_rate')
            else:
                if self.config.executions_per_trial < 10:
                    self.config.reg_rate = hp.Choice('reg_rate', hparams.DROPOUT_RATE)
                else:
                    reg_df = self.reg_df[(self.reg_df.arch == self.config.arch) &
                                           (self.reg_df.dataset == self.config.dataset) &
                                           (self.reg_df.algo == self.config.algorithm) &
                                           (self.reg_df.F == self.config.bw_init) &
                                           (self.reg_df.act == self.config.activation)]
                    if self.config.get('skew_type'):
                        reg_df = reg_df[reg_df.skew_type == self.config.skew_type]
                    reg_rate = float(reg_df.reg_rate)
                    self.config.reg_rate = reg_rate

            lr = self.get_lr(hp)
        else:
            # self.config.use_bias = hp.Boolean('use_bias')
            self.config.reg_rate = False

            if 'error' in self.config.objective:
                lr = self.get_lr(hp)
            else:
                lr = hp.Float('lr', min_value=1e-4, max_value=.1, step=10, sampling='log')

        model = train.Model(self.config, name='model')
        if self.config.problem_type == 'classification':
            loss = 'categorical_crossentropy'

            # if we are dealing with imbalanced classification
            if self.config.get('skew_type') or self.config.dataset in {'caltech256', 'eurosat'}:
                metrics = [tf.keras.metrics.F1Score(self.config.n_classes, average='micro'),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           util.error_rate]
            else:
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
