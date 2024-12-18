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

Purpose: Training algorithm implementations: BP, DFA, DRTP, LL
References:
    DFA: https://arxiv.org/pdf/1609.01596
    DRTP: https://www.zora.uzh.ch/id/eprint/217500/1/fnins-15-629892.pdf
    LL: https://arxiv.org/pdf/1711.06756
"""
import re
from queue import Queue

import tensorflow as tf
from absl import logging

import projection
import sequential
import util


class Layer(tf.keras.layers.Layer):
    """
    Layer class that encapsulates feedforward and projection layers
    """

    def __init__(self, layer, config, **kwargs):
        """
        init
        :param layer: instance of a feedforward dense layer
        :param config: configuration dictionary
        :param kwargs: additional keyword arguments for superclass
        """
        super().__init__(**kwargs)
        self.backend = layer
        self.config = config
        self.project = projection.DenseProject(layer.units, config)

        # trackers
        if config.trackers:
            # cosine of the angle between BP and local error
            if 'bp_cosine' in config.trackers:
                self.bp_cos_angle_tracker = tf.keras.metrics.Mean(name=f'{self.name}/bp_cosine')

            if 'gradients' in config.trackers:
                self.kernel_gradients_tracker = tf.keras.metrics.Mean(name=f'{self.name}/kernel_gradients')
                if config.use_bias:
                    self.bias_gradients_tracker = tf.keras.metrics.Mean(name=f'{self.name}/bias_gradients')

            if 'weights' in config.trackers:
                self.weight_norm_tracker = tf.keras.metrics.Mean(name=f'{self.name}/weight_norm')
                if config.use_bias:
                    self.bias_norm_tracker = tf.keras.metrics.Mean(name=f'{self.name}/bias_norm')

            if config.algorithm != 'bp':
                if 'local_loss' in config.trackers:
                    self.local_loss_tracker = tf.keras.metrics.Mean(name=f'{self.name}/local_loss')

                # tracks weight alignment. see ref. https://arxiv.org/pdf/2011.12428
                if 'weight_alignment' in config.trackers:
                    self.weight_alignment_tracker = tf.keras.metrics.Mean(name=f'{self.name}/weight_alignment')


    def build(self, input_shape):
        """
        build component layers
        :param input_shape: shape of input tensor
        """
        super().build(input_shape)
        self.backend.build(input_shape)
        self.project.build(input_shape=(None, self.config.n_classes))

    def call(self, x, training=False):
        """
        call feedforward layer only
        :param x: input tensor
        :param training: training flag
        :return: output tensor
        """
        return self.backend(x, training=training)


class Model(tf.keras.Model):
    """Model that can train a given FC network using specified training schemes"""

    def __init__(self, config, *args, **kwargs):
        """
        init
        :param config: configuration dictionary
        :param args: additional arguments for superclass
        :param kwargs: additional keyword arguments for superclass
        """
        super().__init__(*args, **kwargs)
        self.config = config

        # get the FC keras model
        self.model = sequential.get_model(config)

        # if specified algorithm is not BP, then create a copy of model with projection layers
        if config.algorithm != 'bp':
            self.lyrs = []
            self.weighted_layers = []
            for lyr in self.model.layers[:-2]:
                if isinstance(lyr, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                    lyr = Layer(lyr, config)
                    self.weighted_layers.append(lyr)
                self.lyrs.append(lyr)

            self.linear = self.model.layers[-2]
            self.out = self.model.layers[-1]

            # tracks weight alignment. see ref. https://arxiv.org/pdf/2011.12428
            if 'weight_alignment' in config.trackers:
                self.weight_alignment_tracker = tf.keras.metrics.Mean(name=f'{self.linear.name}/weight_alignment')

        if config.histogram_freq:
            self.tboard = None

        if config.weight_decay:
            self.l2_loss_tracker = tf.keras.metrics.Mean(name='reg')

    def build(self, input_shape):
        """
        builds the model. uses aligned init scheme if needed.
        :param input_shape: shape of input tensor
        """
        self.model.build(input_shape)
        self.model.summary()
        super().build(input_shape)

        # if init_fw_weight_aligned == True, initializes forward weights for i-th layer as W_i = F_{i-1}.T * F_i.
        # This promotes weight alignment at initialization.
        # Initialization motivated from observations in ref. https://arxiv.org/pdf/2011.12428
        if self.config.algorithm != 'bp' and self.config.init_fw_weight_aligned:
            for i, lyr in enumerate(self.weighted_layers):
                if not i:
                    continue
                prev_lyr = self.weighted_layers[i-1]
                weight = tf.matmul(prev_lyr.project.kernel, lyr.project.kernel, transpose_b=True)
                lyr.backend.kernel.assign(weight)

            self.linear.kernel.assign(self.weighted_layers[-1].project.kernel)

    def call(self, inputs, training=False, post_activation_outs:Queue = None, bn_outs:Queue = None):
        """
        calls the model
        :param inputs: input tensor
        :param training: training flag
        :param post_activation_outs: if not None, append post-activation outputs for each layer to the queue
        :param bn_outs: if not None, append batch normalization outputs for each layer to the queue
        :return: output tensor if algorithm != DFA else also output the final linear layer response
        """
        if self.config.algorithm == 'bp':
            return self.model(inputs, training=training)

        x = inputs
        for lyr in self.lyrs:
            x = lyr(x, training=training)
            if hasattr(lyr, 'backend') and isinstance(lyr.backend, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                logging.info(f'Featuremap: {x.shape}')
            elif isinstance(lyr, tf.keras.layers.Activation) and post_activation_outs is not None:
                post_activation_outs.put(x)
            elif isinstance(lyr, tf.keras.layers.BatchNormalization) and bn_outs is not None:
                bn_outs.put(x)
        x = self.linear(x, training=training)

        # if training with dfa algorithm, we need to return the linear layer output as well to compute the global error
        if self.config.algorithm == 'dfa' and training:
            return x, self.out(x, training=training)
        return self.out(x, training=training)

    def _reg_l2_loss(self, regex=r'.*(kernel|weight):0$'):
        """
        regularization l2 loss
        :param regex: regularization pattern to search for in variable names. only those matching will be affected by
        regularization l2 loss
        :return: regularization l2 loss multiplied by loss weight
        """
        var_match = re.compile(regex)
        return self.config.weight_decay * tf.add_n([
            tf.nn.l2_loss(v) for v in self.trainable_variables if var_match.match(v.name)
        ])

    def error_signal(self, lyr, y, y_local, e=None):
        """
        creates an error signal at layer level for DRTP, DFA and LL algorithms
        :param lyr: lyr in question
        :param y: global output target
        :param y_local: local layer output
        :param e:global error signal to be used only in case of DFA. Must not be None when DFA is used
        :return: error signal at this layer
        """
        with tf.GradientTape(watch_accessed_variables=False) as local_tape:
            local_tape.watch(y_local)
            dots = lyr.project(y_local)
            if self.config.algorithm == 'dfa':
                if e is None:
                    raise ValueError('Received None global error with DFA algorithm')
                local_loss = util.cosines(e, dots, axis=-1, normalize=False)
            elif 'drtp' == self.config.algorithm:
                local_loss = util.cosines(y, dots, axis=-1, normalize=False)
            elif self.config.algorithm == 'cce':
                local_loss = tf.losses.categorical_crossentropy(y, dots, from_logits=True)
            elif self.config.algorithm == 'mse':
                dots = tf.tanh(dots) if self.config.scale_data == 'standard' else tf.sigmoid(dots)
                local_loss = tf.losses.mse(y, dots)
            else:
                raise ValueError(f'Unknown algorithm: {self.config.algorithm}')

            local_loss = tf.reduce_sum(local_loss)

        dy = local_tape.gradient(local_loss, y_local)

        # track local loss for this layer
        # Note: while DRTP and DFA papers do not define the local loss, they can still be seen as local losses
        # See respective definitions of local losses above
        if hasattr(lyr, 'local_loss_tracker'):
            lyr.local_loss_tracker.update_state(local_loss / tf.cast(y.shape[0], local_loss.dtype))

        return dy

    @tf.function
    def train_step(self, data):
        """
        trains the model for one batch
        :param data: input tensor
        :return: training metrics
        """
        x, y = data
        grads = []
        post_activation_outs = Queue()
        bn_outs = None
        if self.config.trackers and self.config.batchnorm:
            bn_outs = Queue()

        # we need a persistent tape to address algorithms other than BP
        with tf.GradientTape(persistent=self.config.algorithm != 'bp') as tape:
            output = self(x, training=True, post_activation_outs=post_activation_outs, bn_outs=bn_outs)
            if self.config.algorithm == 'dfa':
                x, yp = output
            else:
                yp = output
            loss = self.compiled_loss(y, yp)

            if self.config.weight_decay:
                l2_loss = self._reg_l2_loss()
                loss += l2_loss

        # logic for DFA, DRTP and LL
        if self.config.algorithm != 'bp':
            if self.config.algorithm == 'dfa':
                e = tape.gradient(loss, x)
            else:
                e = None

            i = 0
            for lyr in self.lyrs:
                if isinstance(lyr, Layer):
                    lyr_output = post_activation_outs.get()
                    dy = self.error_signal(lyr, y, lyr_output, e=e)
                    g = tape.gradient(lyr_output, lyr.trainable_variables, output_gradients=dy)

                    # add weight decay gradient if needed
                    if self.config.weight_decay:
                        g_reg = tape.gradient(l2_loss, lyr.backend.kernel)
                        g[0] += g_reg

                    grads.extend(g)

                    # update trackers
                    if self.config.trackers:
                        kernel_grad = g[0]

                        if hasattr(lyr, 'bp_cos_angle_tracker'):
                            bp_grad = tape.gradient(loss, lyr.backend.kernel)
                            lyr.bp_cos_angle_tracker.update_state(util.cosines(kernel_grad, bp_grad))

                        if hasattr(lyr, 'bias_gradients_tracker'):
                            lyr.bias_gradients_tracker.update_state(tf.norm(g[1]))

                        if hasattr(lyr, 'kernel_gradients_tracker'):
                            lyr.kernel_gradients_tracker.update_state(tf.norm(kernel_grad))

                        if hasattr(lyr, 'weight_norm_tracker'):
                            lyr.weight_norm_tracker.update_state(tf.norm(lyr.backend.kernel))

                        if hasattr(lyr, 'bias_norm_tracker'):
                            lyr.bias_norm_tracker.update_state(tf.norm(lyr.backend.bias))

                        if hasattr(lyr, 'weight_alignment_tracker') and i:
                            lyr.weight_alignment_tracker.update_state(util.weight_alignment(
                                lyr.backend.kernel, self.weighted_layers[i-1].project.kernel,
                                lyr.project.kernel))

                        if hasattr(self, 'tboard') and not self.config.batchnorm:
                            with self.tboard._train_writer.as_default():
                                tf.summary.histogram(f'{lyr.name}/activation', lyr_output,
                                                     step=self._train_counter)
                        i += 1

                elif isinstance(lyr, tf.keras.layers.BatchNormalization):
                    # update trackers
                    if self.config.trackers and hasattr(self, 'tboard'):
                        bn_lyr_output = bn_outs.get()
                        with self.tboard._train_writer.as_default():
                            tf.summary.histogram(f'{lyr.name}/activation', bn_lyr_output,
                                                 step=self._train_counter)

                    # batchnorm grads
                    if lyr.center or lyr.scale:
                        g = tape.gradient(lyr_output, lyr.trainable_variables, output_gradients=dy)
                        grads.extend(g)

            # output layer grads
            g = tape.gradient(loss, self.linear.trainable_variables)
            grads.extend(g)
        else:
            grads = tape.gradient(loss, self.trainable_variables)

        # apply grads and update compiled metrics
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, yp)

        # update output layer tracker
        if hasattr(self, 'weight_alignment_tracker') and self.config.algorithm != 'bp':
            lyr = self.weighted_layers[-1]
            self.weight_alignment_tracker.update_state(util.weight_alignment(self.linear.kernel,
                                                                             tf.transpose(lyr.project.kernel)))
        # update l2_loss tracker
        if self.config.weight_decay:
            self.l2_loss_tracker.update_state(l2_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        tests the model on one batch
        :param data: input tensor
        :return: test metrics
        """
        x, y = data
        post_activation_outs, bn_outs = None, None
        if self.config.algorithm != 'bp' and self.config.trackers:
            post_activation_outs = Queue()
            if self.config.batchnorm:
                bn_outs = Queue()

        # call the model
        yp = self(x, training=False, post_activation_outs=post_activation_outs, bn_outs=bn_outs)

        # update trackers
        if self.config.algorithm != 'bp' and self.config.trackers:
            for lyr in self.lyrs:
                if isinstance(lyr, Layer):
                    lyr_output = post_activation_outs.get()

                    if not self.config.algorithm == 'dfa' and hasattr(lyr, 'local_loss_tracker'):
                        self.error_signal(lyr, y, lyr_output)

                    if hasattr(self, 'tboard'):
                        with self.tboard._val_writer.as_default():
                            tf.summary.histogram(f'{lyr.name}/activation', lyr_output, step=self._test_counter)
                elif isinstance(lyr, tf.keras.layers.BatchNormalization) and hasattr(self, 'tboard'):
                    lyr_output = bn_outs.get()
                    with self.tboard._val_writer.as_default():
                        tf.summary.histogram(f'{lyr.name}/activation', lyr_output, step=self._train_counter)

        # update loss and compiled metrics
        self.compiled_loss(y, yp)
        self.compiled_metrics.update_state(y, yp)

        # remove metrics that are not expected to change during test phase, e.g., weights
        mets = self.metrics if not self.config.trackers else [m for m in self.metrics if 'loss' in m.name or
                                                              'error' in m.name]
        return {m.name: m.result() for m in mets if 'reg' not in m.name and
                'weight' not in m.name}
