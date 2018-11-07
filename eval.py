import preproc
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

###############################################
# loading dataset

X_test,  Y_test  = preproc.load_test_data()

###############################################
# building the model

model = load_model('cifar100_model.h5')


loss, acc = model.evaluate(X_test,  Y_test)
print("Restored model accuracy: {:5.2f}%".format(100*acc))
