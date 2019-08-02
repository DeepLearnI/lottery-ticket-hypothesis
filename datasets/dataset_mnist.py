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

"""The MNIST dataset."""

#import keras
import tensorflow as tf
from bedrock import dataset_base
from bedrock import save_restore
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import zipfile
from mnist_fc.constants import BATCH_SIZE

class DatasetSalt(dataset_base.DatasetBase):
  def __init__(self,
               train_order_seed=None):
    
    im_width = 128
    im_height = 128
    
    path, target_path = self.download_data()
    ids = os.listdir(path)
    print("No. of images = ", len(ids))

    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
      # Load images
      img = load_img(os.path.join(path, id_), grayscale=True)
      x_img = img_to_array(img)
      x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)
      # Load masks
      mask = img_to_array(load_img(os.path.join(target_path, id_), grayscale=True))
      mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
      # Save images
      X[n] = x_img / 255.0
      y[n] = mask / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=train_order_seed)
    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("y_test shape: {}".format(y_test.shape))
    
    # Prepare the dataset.
    super(DatasetSalt, self).__init__(
      (X_train, y_train),
      BATCH_SIZE, (X_test, y_test),
      train_order_seed=train_order_seed)
    
  def download_data(self):
    import urllib.request
  
    data_landing_path = "/tmp/salt.zip"
  
    urllib.request.urlretrieve("https://www.dropbox.com/s/5y7sv8yq8j4iam1/train.zip?dl=1", data_landing_path)

    zip_ref = zipfile.ZipFile(data_landing_path, 'r')
    zip_ref.extractall('/tmp/salt/')
    zip_ref.close()

    data_path = '/tmp/salt/images'
    target_path = '/tmp/salt/masks'
    return data_path, target_path


class DatasetMnist(dataset_base.DatasetBase):
  """The MNIST dataset."""

  def __init__(self,
               mnist_location,
               flatten=False,#True,
               permute_labels=False,
               train_order_seed=None):
    """Create an MNIST dataset object.

    Args:
      mnist_location: The directory that contains MNIST as four npy files.
      flatten: Whether to convert the 28x28 MNIST images into a 1-dimensional
        vector with 784 entries.
      permute_labels: Whether to randomly permute the labels.
      train_order_seed: (optional) The random seed for shuffling the training
        set.
    """
    #mnist = save_restore.restore_network(mnist_location)

    path = tf.keras.utils.get_file('mnist', 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
    with np.load(path) as f:
      x_train, y_train = f['x_train'], f['y_train']
      x_test, y_test = f['x_test'], f['y_test']

    #x_train = mnist['x_train']
    #x_test = mnist['x_test']
    #y_train = mnist['y_train']
    #y_test = mnist['y_test']

    if permute_labels:
      # Reassign labels according to a random permutation of the labels.
      permutation = np.random.permutation(10)
      y_train = permutation[y_train]
      y_test = permutation[y_test]

    # Flatten x_train and x_test.
    if flatten:
      x_train = x_train.reshape((x_train.shape[0], -1))
      x_test = x_test.reshape((x_test.shape[0], -1))
    else:
      x_train = np.expand_dims(x_train, axis=-1)
      x_test = np.expand_dims(x_test, axis=-1)

    # Normalize x_train and x_test.
    x_train = tf.keras.utils.normalize(x_train).astype(np.float32)
    x_test = tf.keras.utils.normalize(x_test).astype(np.float32)

    # Convert y_train and y_test to one-hot.
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Prepare the dataset.
    super(DatasetMnist, self).__init__(
        (x_train, y_train),
        64, (x_test, y_test),
        train_order_seed=train_order_seed)
