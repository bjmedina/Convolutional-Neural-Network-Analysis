import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import preproc
from keras import metrics
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


# 4th (fully connected nn)
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

#Need to compare MSE and cross entropy
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy', metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)])


#############################################
# training the model

with tf.device('/device:GPU:0'):
    # instead of inputting the epochs, put a loop with fit inside,
    # then keep track of accuracy at each epoch

    #model.fit returns a History object... what is that
    h = model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose = 1, validation_split=0.2)

#############################################
# Plotting the losses

Y = h.history['val_loss']
Z = h.history['loss']
X = [i for i in range(1, len(Y)+1)]

plt.plot(X, Y, label="Validation Loss")
plt.plot(X, Z, label="Training Loss")
plt.title("Validation Loss Vs. Training Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()

# save
#model.summary()
model.save('cifar100_model.h5')


