import numpy as np
from tensorflow import keras
from Convolutional2DTranpose import *

# X = np.arange(64).reshape(8, 8)
# X_reshape = X.reshape(1, 8, 8, 1)
# kernelSize = (5,5)
# filters = 1


# model_Conv2D_Transpose = keras.models.Sequential()
# model_Conv2D_Transpose.add(
#     keras.layers.Conv2DTranspose(filters, (2, 2), strides=(1, 1), padding='valid', input_shape=(2, 2, 1)))
# # w = model_Conv2D_Transpose.layers[0].get_weights()[0].reshape(2, 2)
# w = model_Conv2D_Transpose.layers[0].get_weights()[0]
# w = w.reshape(2,2)
# # w = w.reshape(2,2,2)
# print("Keras Weights: \n {}".format(w))
# print("\n")
#
# keras_output = model_Conv2D_Transpose.predict(X_reshape)
# # keras_output = keras_output.reshape(3, 3)
# # print(keras_output)
# keras_output = keras_output.reshape(3, 3)
# print("Keras Conv2D: \n {}".format(keras_output))
# print("\n")
#
# W = w
# model = Conv2DTranspose(filters, W, padding="valid", strides=(1, 1))
# my_output = model.forwardPropagate(X)
# print("My Conv2DTranspose: \n {}".format(my_output))
# print("\n")

X = np.arange(4).reshape(2, 2)
X_reshape = X.reshape(1, 2, 2, 1)
kernelSize = (2, 2)
strides = (2, 2)
inputShape = (2, 2, 1)
filters = 1

model_Conv2D_Transpose = keras.models.Sequential()
model_Conv2D_Transpose.add(
    keras.layers.Conv2DTranspose(filters, kernelSize, strides=strides, padding='same', input_shape=inputShape))
print(model_Conv2D_Transpose.output_shape)
# w = model_Conv2D_Transpose.layers[0].get_weights()[0].reshape(2, 2)
w = model_Conv2D_Transpose.layers[0].get_weights()[0]
if filters > 1:
    w = w.reshape(kernelSize[0], kernelSize[1], filters)
else:
    w = w.reshape(kernelSize)
# w = w.reshape(2,2,2)
# print("Keras Weights: \n {}".format(w))
# print("\n")

W = w
model = Conv2DTranspose(filters, W, padding="same", strides=strides)
model.forwardPropagate(X)
