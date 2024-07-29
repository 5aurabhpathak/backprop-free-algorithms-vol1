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

Purpose: training with hyperparams tuning using keras tuner
"""
import json
import os

import keras_tuner
import tensorflow as tf
from absl import app, flags, logging
from easydict import EasyDict as Config

import dataloader
import hparams
import hypermodels
import util
import gridsearch

FLAGS = flags.FLAGS


def define_flags():
    """
    Define the flags. Note that setting these flags on command line overrides the current setting in hparams module.
    These flags can be used to freeze certain hyperparameters during the tuning process. The following are tuned by
    default:
    architecture: from hparams.ARCHS
    algorithms: from hparams.ALGORITHM
    activations: from hparams.ACTIVATION
    batchnorm: [True, False]
    feedback_matrix init scheme: hparams.BW_INIT
    learning rate: min_value=1e-4, max_value=.1, step=10, sampling='log'
    dropout regularization rate: from hparams.DROPOUT_RATE
    ALL of the above hparams can be fixed using flags defined in this function.
    Caution: The flags provide a way to override most but NOT all the other hyperparams than those described above in
    hparams.py. Double check hparams.py settings before training
    """
    flags.DEFINE_string('problem_type', 'classification', 'classification or reconstruction')
    flags.DEFINE_string('arch', None, 'dash-separated fully connected architecture: units-units-units....-units')
    flags.DEFINE_string('objective', None, 'objective to optimize', required=True)
    flags.DEFINE_enum('dataset', None, hparams.DATASETS, 'dataset to use', required=True)
    flags.DEFINE_enum('scaler', 'standard', hparams.DATA_SCALER + ['identity'], 'data scaler to use')
    flags.DEFINE_enum('algorithm', None, hparams.ALGORITHM, 'training algorithm to use')
    flags.DEFINE_enum('bw_init', None, hparams.BW_INIT, 'feedback matrix init type')
    flags.DEFINE_enum('activation', None, hparams.ACTIVATION, 'activation function to use')
    flags.DEFINE_multi_enum('tracker', [], hparams.TRACKERS, 'trackers to use. may make training slow')
    flags.DEFINE_integer('batch_size', hparams.BATCHSIZE, 'training batch size')
    flags.DEFINE_integer('num_epochs', hparams.NUM_EPOCHS, 'Number of epochs for training')
    flags.DEFINE_integer('executions_per_trial', hparams.EXECUTIONS_PER_TRIAL, 'Number of executions per trial')
    flags.DEFINE_integer('histogram_freq', hparams.HISTOGRAM_FREQ, 'Histogram frequency in tensorboard')
    flags.DEFINE_float('examples_frac', hparams.EXAMPLES_FRAC,
                       'Fraction of dataset examples to use for training', lower_bound=0., upper_bound=1.)
    flags.DEFINE_float('reg_rate', None, 'regularization rate', lower_bound=0., upper_bound=1.)
    flags.DEFINE_float('weight_decay', hparams.WEIGHT_DECAY, 'weight decay')
    flags.DEFINE_float('lr', None, 'learning rate')
    flags.DEFINE_bool('batch_norm', None, 'Use batch normalization ')
    flags.DEFINE_bool('use_early_stopping', hparams.USE_EARLY_STOPPING, 'Use early stopping')
    flags.DEFINE_bool('use_bias', hparams.USE_BIAS, 'Use bias in forward layers')
    flags.DEFINE_bool('use_xla', True, 'Use XLA')
    flags.DEFINE_bool('reset_training', False, 'start training from scratch')


def run_tuner(config, model_class, hp=None):
    """
    create a tuner using config and start the tuning process
    :param config: configuration dictionary
    :param model_class: hypermodel instance to create and tune. see hypermodels.py
    :param hp: keras tuner hyperparameters instance containing (possibly frozen) hparams to be passed to tuner
    :return: tuner instance
    """
    with open(os.path.join(config.project_name, 'config.json'), 'w') as f:
        json.dump(config, f)

    # we use gridsearch
    tuner = gridsearch.GridSearch(
        hypermodel=model_class(config),
        hyperparameters=hp,
        objective=keras_tuner.Objective(config.objective, direction='min'),
        executions_per_trial=config.executions_per_trial,
        overwrite=False,
        tune_new_entries=True,
        max_retries_per_trial=0,
        project_name=config.project_name,
        max_consecutive_failed_trials=10)

    tuner.search_space_summary(extended=True)

    log_dir = util.add_directory('log_dir', config.project_name)
    cb = [tf.keras.callbacks.TensorBoard(log_dir,
                                         histogram_freq=config.histogram_freq,
                                         write_graph=False,
                                         write_images=False,
                                         write_steps_per_second=False,
                                         update_freq='epoch',
                                         profile_batch=0,
                                         embeddings_freq=0,
                                         embeddings_metadata=None),
          tf.keras.callbacks.TerminateOnNaN()]
    if config.use_early_stopping:
        cb.append(tf.keras.callbacks.EarlyStopping(monitor=config.objective,
                                                   min_delta=1e-4,
                                                   patience=int((.05 if 'val' not in config.objective else .1)
                                                                * config.epochs),
                                                   start_from_epoch=0 if 'val' not in config.objective else 100,
                                                   verbose=1))

    # begin search
    tuner.search(epochs=config.epochs, callbacks=cb, batch_size=config.batchsize)
    return tuner

def create_project_name(config, hp):
    """
    creates a project directory name under which to store all the relevant files for execution
    :param config: configuration dictionary
    :param hp: keras tuner hyperparameters instance, used for creating directory names
    :return: string containing directory name to be created
    """
    def is_fixed(name):
        return isinstance(hp._hps[name][0], keras_tuner.engine.hyperparameters.Fixed)

    project_dir = '_'.join([config.dataset, config.objective])
    if 'algo' in hp and is_fixed('algo'):
        project_dir = '_'.join([project_dir, config.algorithm])
    if 'F' in hp and is_fixed('F'):
        project_dir = '_'.join([project_dir, config.bw_init])
    if 'skew_type' in hp and is_fixed('skew_type'):
        project_dir = '_'.join([project_dir, config.skew_type])
    if 'lr' in hp and is_fixed('lr'):
        project_dir = '_'.join([project_dir, str(hp.get("lr")).replace('.','')])
    if 'arch' in hp and is_fixed('arch'):
        project_dir = '_'.join([project_dir, config.arch])
    return project_dir


def main(_):
    """main function. reads in flags to config dict, and calls subroutines"""
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    if FLAGS.use_xla:
        tf.config.optimizer.set_jit(True)

    config = Config()
    config.problem_type = FLAGS.problem_type
    config.dataset = FLAGS.dataset
    config.epochs = FLAGS.num_epochs
    config.batchsize = FLAGS.batch_size
    config.executions_per_trial = hparams.EXECUTIONS_PER_TRIAL = FLAGS.executions_per_trial
    config.examples_frac = hparams.EXAMPLES_FRAC = FLAGS.examples_frac
    config.objective = FLAGS.objective
    config.histogram_freq = FLAGS.histogram_freq
    config.trackers = FLAGS.tracker
    config.use_bias = hparams.USE_BIAS = FLAGS.use_bias
    config.scale_data = FLAGS.scaler
    config.weight_decay = FLAGS.weight_decay
    config.use_early_stopping = hparams.USE_EARLY_STOPPING = FLAGS.use_early_stopping
    config.init_fw_weight_aligned = hparams.FW_INIT_WEIGHT_ALIGNED

    # fix hparams if specified
    hp = keras_tuner.HyperParameters()

    if FLAGS.algorithm:
        if config.problem_type != 'classification' and FLAGS.algorithm == 'drtp':
            raise ValueError(f'Invalid problem type {config.problem_type} for algorithm {FLAGS.algorithm}')

        config.algorithm = hp.Fixed('algo', FLAGS.algorithm)
    else:
        config.algorithm = None

    if FLAGS.bw_init:
        config.bw_init = hp.Fixed('F', FLAGS.bw_init)

    if FLAGS.arch:
        config.arch = hp.Fixed('arch', FLAGS.arch)
    if FLAGS.lr:
        hp.Fixed('lr', FLAGS.lr)

    config.project_name = create_project_name(config, hp)
    if not os.path.exists(config.project_name) or FLAGS.reset_training:
        util.ensure_empty_dir(config.project_name)

    if FLAGS.activation:
        hp.Fixed('act', FLAGS.activation)
    if FLAGS.batch_norm is not None:
        config.batchnorm = hp.Fixed('bn', FLAGS.batch_norm)
    if 'val' in FLAGS.objective:
        if FLAGS.reg_rate is not None:
            config.reg_rate = hp.Fixed('reg_rate', FLAGS.reg_rate)

    hc = hypermodels.HyperModel

    config.n_classes = dataloader.get_n_classes(config)
    tuner = run_tuner(config, hc, hp=hp)
    tuner.results_summary()
    tuner.save()


if __name__ == '__main__':
    define_flags()
    logging.set_verbosity(logging.INFO)
    app.run(main)
