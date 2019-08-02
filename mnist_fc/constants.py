# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for fully-connected MNIST experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from bedrock import paths
import tensorflow as tf

#import foundations as f9s

HYPERPARAMETERS = {'layers': [16, 32, 64, 128, 256, 128, 64, 32, 16, 1]}

# 16, tf.nn.relu), 32, tf.nn.relu), (64, tf.nn.relu), (128, tf.nn.relu), (256, tf.nn.relu),
#                           (128, tf.nn.relu), (64, tf.nn.relu), (32, tf.nn.relu), (16, tf.nn.relu), (1, None)]}

#FASHIONMNIST_LOCATION = locations.FASHIONMNIST_LOCATION
#OPTIMIZER_FN = functools.partial(tf.train.GradientDescentOptimizer, .1)

OPTIMIZER_FN = functools.partial(tf.train.AdamOptimizer, .001)
PRUNE_PERCENTS = {'encoder_conv_0_conv1': .1, 'encoder_conv_0_conv2': .1,
                  'encoder_conv_1_conv1': .1, 'encoder_conv_1_conv2': .1,
                  'encoder_conv_2_conv1': .2, 'encoder_conv_2_conv2': .2,
                  'encoder_conv_3_conv1': .2, 'encoder_conv_3_conv2': .2,
                  'bottom_layer_conv1': .2, 'bottom_layer_conv2': .2,
                  'transpose_conv2d_5': .2, 'decoder_conv_5_conv1': .2, 'decoder_conv_5_conv2': .2,
                  'transpose_conv2d_6': .2, 'decoder_conv_6_conv1': .2, 'decoder_conv_6_conv2': .2,
                  'transpose_conv2d_7': .1, 'decoder_conv_7_conv1': .1, 'decoder_conv_7_conv2': .1,
                  'transpose_conv2d_8': .1, 'decoder_conv_8_conv1': .1, 'decoder_conv_8_conv2': .1,
                  'output_layer': .1}

NUM_EPOCHS = 1
TRAINING_LEN = ('iterations', 50000)
NUM_ITERATIONS = 1  # Number of times to prune the network
EXPERIMENT_NAME = 'unet'
EXPERIMENT_PATH = 'unet'


# This logs the dictionary elements into the GUI under Parameters
params_dict = {**HYPERPARAMETERS, **PRUNE_PERCENTS,
               'epochs': NUM_EPOCHS,
               'iterations': TRAINING_LEN[1], 'times_pruned': NUM_ITERATIONS}

#f9s.log_params(params_dict)

def graph(category, filename):
  return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
  return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
  return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
  return paths.run(trial(trial_name), level, experiment_name, run_id)
