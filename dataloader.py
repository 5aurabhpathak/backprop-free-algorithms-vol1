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

Purpose: Dataloader classes and utilities for loading them
"""
import abc
import functools
import os
import pickle
import typing

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import hparams


class Dataset(abc.ABC):
    """abstract class representing a dataset"""

    def __init__(self):
        """init.loads data from pickle. If not present, creates a pickle file from dataset in /tmp directory"""
        try:
            with open(f'/tmp/{self.__class__.__name__.lower()}.bin', 'rb') as f:
                self.x, self.y = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.x, self.y = self._preload_data()
            with open(f'/tmp/{self.__class__.__name__.lower()}.bin', 'wb') as f:
                pickle.dump((self.x, self.y), f)

    @abc.abstractmethod
    def _preload_data(self):
        """creates a pickle file from dataset in /tmp directory. must be implemented by subclasses"""
        raise NotImplementedError()

    def load_data(self):
        """
        create train and test sets
        :return: train and test sets
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=.1, shuffle=True,
                                                            stratify=self.y)
        return (x_train, y_train), (x_test, y_test)


class Caltech256(Dataset):
    """Represents Caltech256 dataset"""

    def __init__(self, scale_type='standard', image_size=64):
        """
        init
        :param scale_type: scaling scheme, see scaler() function in this module
        :param image_size: image size to resize the dataset to
        """
        self.scale_type = scale_type
        self.image_size = image_size, image_size
        super().__init__()

    def _preload_data(self):
        """
        creates a pickle file from dataset in /tmp directory and returns the entire dataset as x, y pair
        :return: dataset as x, y pair
        """
        root_dir = '/blue/data/datasets/caltech256'
        class_names = os.listdir(root_dir)

        images = []
        labels = []

        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith('.jpg'):
                    image = tf.keras.utils.load_img(os.path.join(class_path, filename))
                    image = tf.keras.utils.img_to_array(image)
                    image, _ = scaler(image, None, scale=self.scale_type)
                    image = tf.image.resize_with_pad(image, *self.image_size, antialias=True)
                    images.append(image)
                    labels.append(label)

        return np.array(images), np.array(labels)


class Eurosat(Dataset):
    """Represents Eurosat dataset"""

    def _preload_data(self):
        """
        creates a pickle file from dataset and returns the entire dataset as x, y pair
        :return: dataset as x, y pair
        """
        x, y = [], []
        for xx, yy in tfds.load('eurosat', as_supervised=True, split='all').as_numpy_iterator():
            x.append(xx), y.append(yy)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y


def get_data(config, training=True):
    """
    reads in dataset defined in config and returns training set or test set conditioned on whether training==True
    :param config: config dict
    :param training: True for training set, False for testing set
    :return: training set or test set conditioned on whether training==True
    """
    dataset_name = config.dataset
    if dataset_name == 'cifar100':
        dataset = tf.keras.datasets.cifar100
    elif dataset_name == 'cifar10':
        dataset = tf.keras.datasets.cifar10
    elif dataset_name == 'mnist':
        dataset = tf.keras.datasets.mnist
    elif dataset_name == 'fmnist':
        dataset = tf.keras.datasets.fashion_mnist
    elif dataset_name == 'caltech256':
        dataset = Caltech256(scale_type=config.scale_data, image_size=64)
    elif dataset_name == 'eurosat':
        dataset = Eurosat()
    elif isinstance(type(dataset_name), typing.Type):
        dataset = dataset_name
    else:
        raise ValueError(f'unknown dataset: {dataset_name}')

    try:
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
    except AttributeError:
        raise ValueError(f'{dataset.__name__} must have load_data() defined')

    x, y = (x_train, y_train) if training else (x_test, y_test)

    # select a fraction of data if 0 < config.examples_face < 1
    if config.examples_frac:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=config.examples_frac, random_state=hparams.RANDOM_SEED)
        _, inds = next(sss.split(x, y))
        x, y = x[inds], y[inds]

    x = x.astype(np.float32)
    y = tf.keras.utils.to_categorical(y).astype(np.float32)
    return x, y


def scaler(x, y, config):
    """
    scales data to specified range in config
    :param x: data x
    :param y: data y, passes through unchanged
    :param config: config dict
    :return: scaled data x and y
    """
    scale = config.scale_data

    if scale == 'standard':
        x = (x - 127.5) / 127.5
    elif scale == 'zero_one':
        x = x / 255.
    elif isinstance(scale, typing.Callable):
        x = scale(x)
    elif scale != 'identity':
        raise ValueError(f'Unknown scaler: {str(scale)}')

    return x, y


def get_shapes(config):
    """
    gets the shapes of input and outputs
    :param config: config dict
    :return: shapes as a two-tuple
    """
    #TODO: need to define a simpler logic. Currently this requires reading the whole dataset
    x, y = get_data(config)
    x = x[0].ravel()
    if config.problem_type == 'classification':
        return (config.batchsize, x.shape[0]), (config.batchsize, y.shape[1])
    elif config.problem_type == 'reconstruction':
        return (config.batchsize, x.shape[0]), (config.batchsize, x.shape[0])


def augmentations(config):
    """
    creates model that applies data augmentation
    :param config: config dict
    :return: keras model that performs data augmentation
    """
    data_augmentation = dict(brightness=tf.keras.layers.RandomBrightness(hparams.augmentations.brightness,
                                                                         value_range=(-1, 1.)),
                             contrast=tf.keras.layers.RandomRotation(hparams.augmentations.contrast,
                                                                     fill_mode='constant'),
                             zoom=tf.keras.layers.RandomZoom(hparams.augmentations.height_factor,
                                                             hparams.augmentations.width_factor, fill_mode='constant'),
                             translate=tf.keras.layers.RandomTranslation(hparams.augmentations.height_factor,
                                                                         hparams.augmentations.width_factor,
                                                                         fill_mode='constant'))

    data_augmentation = tf.keras.Sequential([v for k, v in data_augmentation.items()
                                             if hparams.augmentations.enabled[k] and
                                             not hparams.augmentations.disable_all])
    data_augmentation.add(tf.keras.layers.Lambda(lambda x:
                                                 tf.clip_by_value(x, -1. if config.scale_data == 'standard' else 0.,
                                                                  1.)))
    data_augmentation.add(tf.keras.layers.Flatten())
    data_augmentation.compile(run_eagerly=False)
    return data_augmentation


def get_dataset(config, training=False):
    """
    creates tf.data pipeline
    :param config: config dict
    :param training: True for training set, False for testing set
    :return: tf.data pipeline
    """
    x, y = get_data(config, training=training)
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if config.dataset != 'caltech256':
        scaler_fn = functools.partial(scaler, config=config)
        ds = ds.map(scaler_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()

    if training:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)

    ds = ds.batch(config.batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    data_augmentation = augmentations(config)

    def augment_batch(x, y):
        """
        adds data augmentations
        :param x: data x
        :param y: data y
        :return: augmented data x, where y is passed-through
        """
        x = data_augmentation(x, training=training)
        if config.problem_type == 'classification':
            return x, y
        elif config.problem_type == 'reconstruction':
            return x, x
        else:
            raise ValueError(f'unknown problem type: {config.problem_type}')

    ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)
