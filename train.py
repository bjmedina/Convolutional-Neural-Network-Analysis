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


# number of classes
classes = 100
layers = 3

##################################
### Loading Dataset

X_train, Y_train = preproc.load_training_data()
X_test,  Y_test  = preproc.load_test_data()

# one hot encoming...
Y_train = utils.to_categorical( Y_train, num_classes=classes )
Y_test  = utils.to_categorical( Y_test,  num_classes=classes )


##################################
### Building the Models

models, info = preproc.create_models(X_train, Y_train)

# models is a list of models
# info is a list of 3-tuples. Each 3-tuple is ( loss_function_name, num_layers, dataset_size )

##################################
### Compiling the Models

for i in range(0, len(models)):
    models[i].compile(loss=info[i][0],
                   optimizer="adam",
                   metrics=['accuracy', 'top_k_categorical_accuracy'])

    ##################################
    ### Training the Models

    with tf.device('/device:GPU:0'):
        
        # history object
        h = models[i].fit(X_train, Y_train,
                      batch_size=512,
                      epochs=50,
                      verbose=1,
                      validation_split=0.2)

        models[i].summary()

        ########################################
        ### Plotting and Saving

        Y = h.history['val_loss']
        Z = h.history['loss']
        X = [i for i in range(1, len(Y)+1)]

        plt.plot(X, Y, label="Validation Loss")
        plt.plot(X, Z, label="Training Loss")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")

        plt.legend()

        plt.title("Loss: " + info[i][0] +
                  ". Layers: " + str(info[i][1]) +
                  ". Dataset size: " + str(info[i][2]))

        model_name = info[i][0] + "_" + str(info[i][1]) + "l_" + str(info[i][2]) + "ds"

        plt.savefig("plots/" + model_name)
        plt.clf()
        plt.cla()

        models[i].save("models/" + model_name + ".h5")


        
        

