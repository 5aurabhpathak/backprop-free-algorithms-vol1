# Backpropagation-Free Algorithms (Volume 1)
© 2024 Saurabh Pathak. All rights reserved

This repository contains implementations of backpropagation-
free algorithms for training neural networks. These
algorithms aim to provide alternative methods for updating
network weights without relying on traditional
backpropagation techniques.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Examples](#examples)
7. [Caveats](#caveats)
8. [Footnote](#footnote)

## Overview

This project implements and explores backpropagation-free
algorithms for training neural networks. The main focus in
this repository is on projection-based methods that update
network weights without explicitly computing gradients
through backpropagation and avoid the weight transport 
problem.

### Need for backpropagation free training

Backpropagation has been the cornerstone of neural network
training for decades. However, there are several reasons 
to explore backpropagation-free training methods:

1. **Biological Plausibility**: The brain doesn't perform
   backpropagation in the way artificial neural networks do.
   Backpropagation-free methods aim to create models that
   are more biologically plausible and may lead to insights
   about how the brain learns.
2. **Memory Efficiency**: Backpropagation requires storing
   intermediate activations during the forward pass, which
   can be memory-intensive for large models.
   Backpropagation-free methods can offer lower memory
   requirements.
3. **Real-Time Learning**: Some backpropagation-free methods
   allow for online or real-time learning, where the model
   can be updated immediately after each example, without
   waiting for a full backward pass.
4. **Overcoming Vanishing/Exploding Gradients**: Traditional
   backpropagation can suffer from vanishing or exploding
   gradients, especially in deep networks. Some
   backpropagation-free methods are less susceptible to
   these issues.
5. **Parallel Processing**: Certain backpropagation-free
   algorithms are more amenable to parallel processing,
   potentially leading to faster training on specialized
   hardware.
6. **Non-Differentiable Components**: Backpropagation
   requires all components of the network to be
   differentiable. Backpropagation-free methods can
   potentially work with non-differentiable activation
   functions or layers.
7. **Theoretical Insights**: Studying alternative training
   methods can provide new theoretical insights into
   learning dynamics and optimization in neural networks.

### Scope
This project explores three known methods as an
alternative to backpropagation: DFA(Direct Feedback 
Alignment)[1], DRTP(Direct Random Target Projection)[2] and 
Local Loss(LL)[3]. These methods update weights using 
random feedback matrices. In the case of DFA, DRTP and 
LL methods, these matrices are used to project global error,
output target, and local error respectively onto the
layer-local space. From another viewpoint, the random 
matrices can be seen projecting the layer output to the 
target space, on which a loss function is then applied. This
is obvious in the case of LL, but both DRTP and DFA can also
be seen as computing a loss function in the projected output
space.

In this implementation, we take the latter view and 
create a unified framework in TF2/Keras/Keras-tuner for all 
three 
algorithms in context of fully connected feedforward 
networks. While [1,2,3] consider the classification 
problem, however, in the case of LL, we additionally
consider the image reconstruction problem by using a 
local MSE loss in that case.
By implementing and studying these 
backpropagation-free algorithms, this project contributes to
the broader research effort in finding efficient,
biologically plausible, and potentially more powerful ways
to train neural networks.

### References
[1] [Nøkland, A., 2016. Direct feedback alignment 
provides learning in deep neural networks. Advances in 
neural information processing systems, 29.](https://proceedings.neurips.cc/paper/2016/file/d490d7b4576290fa60eb31b5fc917ad1-Paper.pdf)

[2] [Frenkel, C., Lefebvre, M. and Bol, D., 2021. Learning
without feedback: Fixed random learning signals allow for
feedforward training of deep neural networks. Frontiers in
neuroscience, 15, p.629892.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.629892/pdf)

[3] [Mostafa, H., Ramesh, V. and Cauwenberghs, G., 2018.
Deep supervised learning using local errors. Frontiers in
neuroscience, 12, p.608.](https://www.frontiersin.org/articles/10.3389/fnins.2018.00608/pdf)

## Installation
We use TF2.14 and Python 3.11 based setup. Using a docker 
environment is highly recommended. To that  end, a
`Dockerfile` and a pipenv `Pipfile` is provided. Please see
them for details. When built using the given `Dockerfile`,
it should automatically take care of all dependencies.
```bash
docker build -t <name>:<tag> .
```

Once the image is built, an interactive container should be 
created:
```bash
docker run -itv <mounts> --network host --gpus all 
--name <cname> <name>:<tag>
```

## Project Structure

- `projection.py`: Implements FC layer that 
  projects a local layer output into target space.
- `dataloader.py`: Handles data loading and 
  preprocessing `tf.data` pipeline. Currently provisioned to 
  work with 
  MNIST, CIFAR10/100, FMNIST, Caltech256, and Eurosat
- `hparams.py`: Defines hyperparameters for the models and
  training/tuning process.
- `hypermodels.py`: Implements hypermodels for tuning.
- `sequential.py`: Defines the sequential FC neural network
  model.
- `train.py`: Contains the unified training model and 
  logic for handling all three algorithms as well as 
  standard backpropagation for comparison.
- `tune.py`: Implements hyperparameter tuning 
  functionality using keras tuner's gridsearch.
- `train.bash`: Bash script for running training
  experiments. This script can be used to manage 
  tuning runs and can auto resume in case of crashes.
- `utils.py`: miscellaneous functions used throughout 
  the project.
- `gridsearch.py`: essentially a copy of keras_tuner's 
  gridsearch functionality except that we create trials 
  with random trial_id instead of sequential in the 
  keras_tuner's default implementation of gridsearch. 
  Random trial_ids help in visualizing multiple tuning runs 
  in tensorboard simultaneously without clash.

## Usage
The primary design objective is to be able to tune best 
possible values of hyperparameters through gridsearch. 
The following training command pattern is to be used:
```bash
./train.bash <experiment_data_dir> --objective 
<tuner_objective, eg., loss|val_loss> --dataset <dataset,
 e.g., mnist> [optional_args]
```
where `experiment_data_dir` will be created to store 
data for the experiment under execution, such as tuner runs
with different hparams settings under consideration. 
Optional arguments may include hparams that one wants to 
fix instead of tuning as well as other params that are
defined in `hparams.py`. Parameters defined in `hparams.py`
other than those that are tunable (see below) will 
take on their default value unless overridden through 
command line. See `hparams.py` for their default values. 

*Note: Not all parameters defined in `hparams.py` can be 
overridden through commandline, but one can always edit 
the file directly for such scenarios.*

## Hyperparameter Tuning
Specifically, we tune the following 
hyperparameters:
- architecture: chosen from `hparams.ARCHS`
- algorithms: chosen from `hparams.ALGORITHM`
- activations: from `hparams.ACTIVATION`
- batchnorm: `[True, False]`
- feedback_matrix init scheme: `hparams.BW_INIT`
- learning rate: `min_value=1e-4, max_value=.1, step=10, 
  sampling='log'`
- skew type: from `hparams.SKEW_TYPES`
- dropout regularization rate: from `hparams.DROPOUT_RATE`

We exclude those hparams from tuning that are set to a
fixed value by means of an optional command line flag.

## Examples
```bash
./train.bash data/mnist_test --dataset mnist --num_epochs 
100 \
 --lr 1e-3 --activation relu --bw_init ortho \
--algorithm dfa --problem_type reconstruction \
--examples_frac .5 --executions_per_trial 1 \
--batch_norm --objective val_loss --arch 500-500
```
```bash
./train.bash data/test --dataset cifar10 --num_epochs 50 \
 --lr 1e-3 --activation tanh --bw_init lecun \
--algorithm drtp --problem_type classification \
--examples_frac 0. --executions_per_trial 3 \
--nobatch_norm --objective loss --arch 500-500-1000-500 \
--tracker bp_cosine --tracker weight_alignment
```
```bash
./train.bash data/ll_test --dataset cifar10 --num_epochs 
150 \
 --lr 1e-4 --activation swish --bw_init lecun \
--algorithm cce --problem_type classification \
--examples_frac 0. --executions_per_trial 1 \
--nobatch_norm --objective loss --arch 500-500-1000-500 \
--tracker gradients --tracker weight_alignment
```
```bash
./train.bash data/ll_test_recons --dataset cifar10 
--num_epochs 
150 \
--activation swish --bw_init lecun --skew_type all\
--algorithm mse --problem_type reconstruction \
--examples_frac 0. --executions_per_trial 3 \
--batch_norm --objective val_loss --arch 500-500-1000
```

## Caveats
There are minimal checks in the code for parameter 
combinations. However, using them arbitrarily might lead to
failures and/or instability in training. It is advisable to 
read the reference papers and
use the knowledge of respective algorithms in experiment 
designs. Following are some scenarios off the top:

- DRTP does not work well for reconstruction problems by 
  design. Code throws an error on attempting to use it as so.
- DRTP fails with unbounded activation functions such as 
  ReLU. No checks in code. Training will likely explode to 
  NaN.
- Using MSE loss for classification might not work as 
  well as CCE.
- Batchnorm might hurt or boost performance depending on 
  other settings, check performance without it first 
  before using it.
- Dropout regularization hyperparameter is only 
  activated when `objective` is `val_loss` or 
  `val_error`. Providing a dropout rate hyperparameter 
  without a `val` substring in the `objective` will have 
  no effect.
- When `val` type `objective` is provided, either a 
  dataframe must already be present in the `data` folder 
  that lists the best learning rate for different 
  hparams from an earlier tuning run, or `lr` must be 
  fixed on the command line. See `hypermodels.py` for 
  details.

## Footnote
Thanks for reading the README. This software is "AS IS" 
and may contain issues. I will fix them as I come across 
or if you bring them to my notice. I am deeply 
interested in efficient training algorithms for deep 
learning that draw inspiration from but are not limited by 
natural counterparts of neural nets, as well as sustainable 
and decentralized AI in general. As such, I would greatly
welcome research and collaboration opportunities in this
domain. Please
[contact](mailto:saurabhpathak.personal@gmail.com) me if you
think I can be of value.

Cheers!