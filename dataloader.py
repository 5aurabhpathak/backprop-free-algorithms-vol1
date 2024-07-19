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


class DatasetLoader:
    """Dataloader class for loading dataset. Applies imbalanced sampling on train and test sets if requested"""

    def __init__(self, dataset, skew_type=None, imbalance_factor=1.):
        """
        init
        :param dataset: dataset class
        :param skew_type: 'long_tailed' induces exponentially decreasing samples across classes, 'step' induces stepwise
        decrement in number of samples across different classes
        :param imbalance_factor: desired ratio between the number of samples of the most frequent class and number of
        samples of the least frequent class. see ref 2
        ref 1: https://arxiv.org/pdf/1710.05381
        ref 2: https://arxiv.org/pdf/2006.07529
        """
        if skew_type not in {None, 'long_tailed', 'step'}:
            raise ValueError(f'Unknown skew type {skew_type}')
        assert (imbalance_factor > 1. and skew_type is not None) or (imbalance_factor == 1 and skew_type is None)
        self.dataset = dataset
        self.skew_type = skew_type
        self.imbalance_factor = imbalance_factor

    def load_data(self):
        """
        creates imbalanced dataset according to the set scheme
        :return: imbalanced dataset as train and test sets
        """
        (x_train, y_train), (x_test, y_test) = self.dataset.load_data()

        if self.skew_type == 'long_tailed':
            mu = 1./ self.imbalance_factor ** (1./ y_train.max())
            first = np.argwhere(y_train == 0)[:, 0]
            new_xtrain = [x_train[first]]
            new_ytrain = [y_train[first]]
            uniques = np.unique(y_train)
            for yi in uniques[1:]:
                x_train_yi = np.argwhere(y_train == yi)[:, 0]
                size = int(x_train_yi.shape[0] * mu ** yi)
                choice = np.random.choice(x_train_yi, size=size, replace=False)
                new_xtrain.append(x_train[choice])
                new_ytrain.append(y_train[choice])
            x_train = np.concatenate(new_xtrain, axis=0)
            y_train = np.concatenate(new_ytrain, axis=0)

        elif self.skew_type == 'step':
            median_cutoff = y_train <= np.median(np.unique(y_train))
            inds = np.argwhere(median_cutoff)[:, 0]
            new_xtrain = [x_train[inds]]
            new_ytrain = [y_train[inds]]

            inds = np.argwhere(~median_cutoff)[:, 0]
            size = int(inds.shape[0] / self.imbalance_factor)
            choice = np.random.choice(inds, size=size, replace=False)
            new_xtrain.append(x_train[choice])
            new_ytrain.append(y_train[choice])
            x_train = np.concatenate(new_xtrain, axis=0)
            y_train = np.concatenate(new_ytrain, axis=0)
        return (x_train, y_train), (x_test, y_test)


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
        loader = DatasetLoader(dataset, skew_type=config.get('skew_type'),
                               imbalance_factor=config.get('imbalance_factor', 1.))
        (x_train, y_train), (x_test, y_test) = loader.load_data()
    except AttributeError:
        raise ValueError(f'{loader.__name__} must have load_data() defined')

    x, y = (x_train, y_train) if training else (x_test, y_test)

    # select a fraction of data if 0 < config.examples_face < 1
    if config.examples_frac:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=config.examples_frac, random_state=hparams.RANDOM_SEED)
        _, inds = next(sss.split(x, y))
        x, y = x[inds], y[inds]

    x = x.astype(np.float32)

    if config.problem_type == 'reconstruction':
        return x, x

    y = tf.keras.utils.to_categorical(y).astype(np.float32)
    return x, y


def scaler(x, y, config):
    """
    scales data to specified range in config
    :param x: data x
    :param y: data y
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


def get_n_classes(config):
    """
    get the number of classes
    :param config: config dict
    :return: number of classes
    """
    #TODO: need to define a simpler logic. Currently this requires reading the whole dataset
    x, y = get_data(config)
    if config.problem_type == 'classification':
        return y.shape[1]
    elif config.problem_type == 'reconstruction':
        return x[0].ravel().shape[0]


def augmentations():
    """
    creates model that applies data augmentation
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
    data_augmentation.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1., 1.)))
    data_augmentation.add(tf.keras.layers.Flatten())
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
        ds = ds.shuffle(128, reshuffle_each_iteration=True)

    ds = ds.batch(config.batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    data_augmentation = augmentations()
    def augment_batch(x, y):
        x = data_augmentation(x, training=training)
        if config.problem_type == 'classification':
            return x, y
        elif config.problem_type == 'reconstruction':
            return x, x

    ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)
