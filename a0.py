import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# one dataset (still need to get the other ones)
batch = unpickle("cifar-10-batches-py/data_batch_1")

# batch is a dictionary, w/ b'batch_label', 'labels', 'data', 'filenames'

