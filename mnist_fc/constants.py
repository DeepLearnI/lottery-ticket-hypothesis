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

import foundations as f9s

HYPERPARAMETERS = {'layers': [(256, tf.nn.relu), (128, tf.nn.relu), (64, tf.nn.relu), (32, tf.nn.relu), (10, None)]}
#FASHIONMNIST_LOCATION = locations.FASHIONMNIST_LOCATION
OPTIMIZER_FN = functools.partial(tf.train.GradientDescentOptimizer, .1)
PRUNE_PERCENTS = {'layer0': .3, 'layer1': .2, 'layer2': .1, 'layer3': .1, 'layer4': .1}
NUM_EPOCHS = 1
TRAINING_LEN = ('iterations', 50000)
NUM_ITERATIONS = 1  # Number of times to prune the network
EXPERIMENT_NAME = 'wheredoesstuffwrite'

EXPERIMENT_PATH = 'mnist_fc_data'
MNIST_LOCATION = '/home/rm/lottery_ticket/mnist_fc/datasets/mnist' #'datasets/mnist'  #'/home/rm/lottery_ticket/mnist_fc/datasets/mnist'

# This logs the dictionary elements into the GUI under Parameters
{f9s.log_param('layer_sizes', value) for key, value in HYPERPARAMETERS.items()}
{f9s.log_param('prune_pct_' + str(key), value) for key, value in PRUNE_PERCENTS.items()}
f9s.log_param('epochs', NUM_EPOCHS)
f9s.log_param('iterations', TRAINING_LEN[1])
f9s.log_param('times_pruned', NUM_ITERATIONS)
f9s.log_param('tag', EXPERIMENT_NAME)


def graph(category, filename):
  return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
  return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
  return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
  return paths.run(trial(trial_name), level, experiment_name, run_id)
