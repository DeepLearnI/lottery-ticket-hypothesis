'''
1. Randomly initialize a neural network f(x;θ0)(where θ0 ∼Dθ).
2. Train the network for j iterations, arriving at parameters θj .
3. Prune p% of the parameters in θj , creating a mask m.  Each unpruned connection’s value is then reset to its initialization from original network before it was trained.
4. Resettheremainingparameterstotheirvaluesinθ0,creatingthewinningticketf(x;m⊙θ0).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from datasets import dataset_mnist
from foundations import experiment
from foundations import model_fc
from foundations import paths
from foundations import pruning
from foundations import save_restore
from foundations import trainer
from mnist_fc import constants

NUM_ITERATIONS = 2  #TODO HN: what's an iteration? steps? an epoch? does it matter???

def train(output_dir,
          mnist_location=constants.MNIST_LOCATION,
          training_len=constants.TRAINING_LEN,
          iterations=30,
          experiment_name='same_init',
          presets=None,
          permute_labels=False,
          train_order_seed=None):
  """Perform the lottery ticket experiment.

  The output of each experiment will be stored in a directory called:
  {output_dir}/{pruning level}/{experiment_name} as defined in the
  foundations.paths module.

  Args:
    output_dir: Parent directory for all output files.
    mnist_location: The path to the NPZ file containing MNIST.
    training_len: How long to train on each iteration.
    iterations: How many iterative pruning steps to perform.
    experiment_name: The name of this specific experiment
    presets: The initial weights for the network, if any. Presets can come in
      one of three forms:
      * A dictionary of numpy arrays. Each dictionary key is the name of the
        corresponding tensor that is to be initialized. Each value is a numpy
        array containing the initializations.
      * The string name of a directory containing one file for each
        set of weights that is to be initialized (in the form of
        foundations.save_restore).
      * None, meaning the network should be randomly initialized.
    permute_labels: Whether to permute the labels on the dataset.
    train_order_seed: The random seed, if any, to be used to determine the
      order in which training examples are shuffled before being presented
      to the network.
  """
  # Define model and dataset functions.
  def make_dataset():
    return dataset_mnist.DatasetMnist(
        mnist_location,
        permute_labels=permute_labels,
        train_order_seed=train_order_seed)

  make_model = functools.partial(model_fc.ModelFc, constants.HYPERPARAMETERS)

  # Define a training function.
  def train_model(sess, level, dataset, model):
    params = {
        'test_interval': 100,
        'save_summaries': True,
        'save_network': True,
    }

    return trainer.train(
        sess,
        dataset,
        model,
        constants.OPTIMIZER_FN,
        training_len,
        output_dir=paths.run(output_dir, level, experiment_name),
        **params)

  print('Define a pruning function')
  prune_masks = functools.partial(pruning.prune_by_percent,
                                  constants.PRUNE_PERCENTS)

  print('Run the experiment.')
  experiment.experiment(
      make_dataset,
      make_model,
      train_model,
      prune_masks,
      iterations,
      presets=save_restore.standardize(presets))

print('Training')
train('/home/rm/lottery_ticket')

