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

Purpose: Config file
"""
from easydict import EasyDict as Augmentations

ALGORITHM = ['bp', 'dfa', 'drtp', 'cce', 'mse']
DATASETS = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'caltech256', 'eurosat']
ARCHS = ['500', '500-500-500-500-500', '500-1000-500', '1000', '1000-1000', '1000-500-1000']
BATCHSIZE = 64
NUM_EPOCHS = 300
EXECUTIONS_PER_TRIAL = 3
EXAMPLES_FRAC = .5
BW_INIT = ['ortho', 'lecun']
DATA_SCALER = ['standard', 'zero_one']
DROPOUT_RATE = [.1, .2, .3, .4, .5]
WEIGHT_DECAY = 0.

# This promotes weight alignment at initialization.
# Initialization motivated from observations in ref. https://arxiv.org/pdf/2011.12428
FW_INIT_WEIGHT_ALIGNED = False

augmentations = Augmentations()
augmentations.disable_all = True
augmentations.enabled = dict(brightness=True,
                             contrast=False,
                             zoom=False,
                             translate=False)
augmentations.brightness = 0.2
augmentations.contrast = 0.1
augmentations.height_factor = 0.2
augmentations.width_factor = 0.2

ACTIVATION = ['tanh', 'relu', 'swish', 'lrelu']

HISTOGRAM_FREQ = 5
TRACKERS = [
    'bp_cosine',
    'local_loss',
    'weights',
    'gradients',
    'weight_alignment',
]
USE_BIAS = True
RANDOM_SEED = None
USE_EARLY_STOPPING = False
