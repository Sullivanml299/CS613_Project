import numpy as np
from tensorflow import keras
from Convolutional2DTranpose import *

#
# X = np.array([[1, 2, 3, 4],
#               [2, 2, 3, 2],
#               [1, 3, 3, 3],
#               [4, 4, 4, 4]])

# X = np.array([[1, 2, 3, 4, 5],
#               [2, 2, 3, 2, 5],
#               [1, 3, 3, 3, 5],
#               [4, 4, 4, 4, 5]])

# K = np.array([[1, 2, 3],
#               [2, 2, 3],
#               [1, 3, 3]])
#




X = np.array([[0.0, 1.0], [2.0, 3.0]])
X_reshape = X.reshape(1, 2, 2, 1)
K = np.array([[0.0, 1.0], [2.0, 3.0]])

model_Conv2D_Transpose = keras.models.Sequential()
model_Conv2D_Transpose.add(keras.layers.Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='valid', input_shape=(2, 2, 1)))
w = model_Conv2D_Transpose.layers[0].get_weights()[0].reshape(2,2)
# w = model_Conv2D_Transpose.layers[0].get_weights()[0].reshape(2,2,2)
print("Keras Weights: \n {}".format(w))
keras_output = model_Conv2D_Transpose.predict(X_reshape)
keras_output = keras_output.reshape(3, 3)
# keras_output = keras_output.reshape(2, 2, 2)
print("Keras Conv2D: \n {}".format(keras_output))


W = w
model = Conv2DTranspose(1, W, padding="valid", strides=(1, 1))
my_output = model.forwardPropagate(X)
print("My Conv2D: \n {}".format(my_output))
print("\n")

