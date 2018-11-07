import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import preproc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint



################################################
# check point to save model once trained
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

###############################################
# loading dataset

names = preproc.load_class_names()
X_train, Y_train = preproc.load_training_data()
X_test,  Y_test  = preproc.load_test_data()

###############################################
# building the model



model = Sequential()


# 1st
#                units  window_size  input_shape
model.add(Conv2D(128,    (3,3),       input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd
model.add(Conv2D(128,    (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd
model.add(Conv2D(64,    (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


# 5th (fully connected nn)
model.add(Flatten()) #3d feature maps to 1d feature vectors
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(20))
model.add(Activation("relu"))
model.add(Dropout(0.25))

# output layer
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


# Initialize model

#############################################
# training the model

with tf.device('/device:GPU:0'):
    #model.fit(X_train, Y_train, batch_size=32, epochs=40, validation_split=0.1, callbacks = [cp_callback])
    model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.1)

model.summary()
model.save('cifar100_model.h5')


