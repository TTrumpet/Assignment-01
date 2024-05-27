import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar100
from keras import layers
from keras.applications import EfficientNetV2L
from keras import backend as K

batch_size = 128
num_classes = 100
epochs = 3

# input image dimensions
img_rows, img_cols, img_channels = 32, 32, 3

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
input_shape = (img_rows, img_cols, img_channels)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Tensorflow Backend
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# change data to float32, scale inputs to range [0-1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# modify label vectors to one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = EfficientNetV2L(
    include_top=True,
    weights=None,
    classes=num_classes,
    input_shape=(img_rows, img_cols, img_channels),
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
