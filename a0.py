import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import preproc
#from keras import metrics
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint
import time # needed for calculating elapsed time for a given model


###############################################
# loading dataset

names = preproc.load_class_names()
X_train, Y_train = preproc.load_training_data()
X_test,  Y_test  = preproc.load_test_data()

# need the labels to be one hot...
Y_train = utils.to_categorical(Y_train)
Y_test  = utils.to_categorical(Y_test)

###############################################
# building the model

num_layers = range(3, 12, 2)

for layers in num_layers:

    ############################################################
    # Building the model
    
    model = Sequential()
    
    # 1st layer (needs input sizes)
    #                units  window_size  input_shape
    model.add(Conv2D(128,    (3,3),       input_shape = X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    for i in range(0, layers-1):
        model.add(Conv2D(128,    (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
    
    # last layer (fully connected nn)
    model.add(Flatten()) #3d feature maps to 1d feature vectors
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(20))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    
    # output layer
    model.add(Dense(10))
    model.add(Activation("softmax"))

    # compare between cross entropy and mse
    loss_functions = ["categorical_crossentropy", "mean_squared_error"]
    
    for loss_function in loss_functions:
        
        model.compile(loss=loss_function,
                      optimizer="adam",
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        
        
        #############################################
        # training the model

        # need to increase dataset for each class
        with tf.device('/device:GPU:0'):
            
            # instead of inputting the epochs, put a loop with fit inside,
            # then keep track of accuracy at each epoch
            
            # change this:         |||            |||              |         |            | 
            h = model.fit(X_train[:100], Y_train[:100], batch_size=2, epochs=1, verbose = 1, validation_split=0.2)
            
            #############################################
            # Plotting the losses and save the model
            
            Y = h.history['val_loss']
            Z = h.history['loss']
            X = [i for i in range(1, len(Y)+1)]
            
            plt.plot(X, Y, label="Validation Loss")
            plt.plot(X, Z, label="Training Loss")
            plt.title("Validation Loss Vs. Training Loss")
            
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            
            model_name = "cifar100_" + loss_function + "_" + str(layers) + "layers"

            plt.savefig(model_name)
            
            # save
            model.save(model_name + ".h5")
            

