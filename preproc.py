'''
Bryan Medina
Convolutional Neural Network on cifar-100 dataset
'''

##################################
### Imports

from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import preproc
import random
import tensorflow as tf
import time

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 100

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 1

# Number of images for each batch-file in the training-set.
_images_per_file = 50000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """
    print("Loading data: " + filename)

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-100 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-100 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'fine_labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


def load_class_names():
    """
    Load the names for the classes in the CIFAR-100 data-set.
    Returns a list with the names.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="cifar-100-python/meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names

def load_training_data():
    """
    Load all the training-data for the CIFAR-100 data-set.
    Returns the images, class-numbers 
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index
    begin = 0
   
    # Load the images and class-numbers from the data-file.
    images_batch, cls_batch = _load_data(filename="cifar-100-python/train")

    # Number of images in this batch.
    num_images = len(images_batch)

    # End-index for the current batch.
    end = begin + num_images
	
    # Store the images into the array.
    images[begin:end, :] = images_batch

    # Store the class-numbers into the array.
    cls[begin:end] = cls_batch

    # The begin-index for the next batch is the current end-index.
    begin = end

    return images, cls


def load_test_data():
    """
    Load all the test-data for the CIFAR-100 data-set.
    Returns the images, class-numbers
    """

    images, cls = _load_data(filename="cifar-100-python/test")

    return images, cls

def create_model(layers, x):
    """
    Create a model with the 'layers' number of layers
    return given model
    """

    # we always need  input    and input and output
    #                 (to CNN)     (fully connected NN)
    layers = layers - (1       +   2)
    model = Sequential()

    # First layer always has to be the input to the CNN
    # it must include the input size into the layer
    
    # input is of dimensions [_, 32, 32, 3]
    # output is of dimensions [_, 32, 32, 32]
    # 5 X 5 filters, ReLU, 32 feature maps

    # 1
    model.add(Conv2D(32, (5,5), input_shape=x.shape[1:], padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (5,5), padding="same"))
    model.add(Activation("relu"))

    # 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3
    model.add(Dropout(0.25))

    # 4
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))

    # 5
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 6
    model.add(Dropout(0.25))

    # 7
    model.add(Flatten())
    model.add(Dense(1024))

    # 8
    model.add(Dropout(0.4))

    # 9
    model.add(Dense(512))

    # 10
    model.add(Dropout(0.4))

    # 11
    model.add(Dense(100))

    # 12
    model.add(Activation("softmax"))

    return model


def create_models(x, y):

    loss_func = ["categorical_crossentropy", "mean_squared_error"]
    models = []
    info   = []

    # Add all the models you want here...
    # We still need to account for 1) data set size and 2) model size (layers AND parameters)
    models.append(create_model(0, x))
    info.append( (loss_func[0], 0, 15) )

    models.append(create_model(0, x))
    info.append( (loss_func[1], 0, 200) )

    return models, info
