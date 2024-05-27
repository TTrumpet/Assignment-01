'''
This code implements the best structure for MNIST for CIFAR100 with no changes to the architecture
and minimal changes to parameters.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import CSVLogger

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

str_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

plt.rcParams['figure.figsize'] = (7, 7)

batch_size = 256
num_classes = 100
epochs = 15
run_name = 'ep20_b256'

# input image dimensions
img_rows, img_cols, img_channels = 32, 32, 3

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
input_shape = (img_rows, img_cols, img_channels)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# examples of the training data
'''
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray', interpolation='none')
    plt.title('Class {}'.format(y_train[i]))
'''

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

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model: loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train and test model
csv_logger = CSVLogger(run_name + '_training.log', separator=',', append=False)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[csv_logger])
model.save_weights('./checkpoints/' + run_name + '.ckpt')

model.load_weights('./checkpoints/' + run_name + '.ckpt')
log_data = pd.read_csv(run_name + '_training.log', sep=',', engine='python')
#print(log_data.head())

# plotting training accuracy and loss
fig, axs = plt.subplots(ncols=2, nrows=2)
axs[0, 0].plot(log_data['accuracy'])
axs[0, 0].set_title('Training accuracy')
axs[0, 0].set_xlabel('No. epochs')
axs[0, 0].set_ylabel('Accuracy Value (%)')
axs[0, 0].set_box_aspect(1)
axs[0, 1].plot(log_data['loss'])
axs[0, 1].set_title('Training loss')
axs[0, 1].set_xlabel('No. epochs')
axs[0, 1].set_ylabel('Loss Value')
axs[0, 1].set_box_aspect(1)
axs[1, 0].plot(log_data['val_accuracy'])
axs[1, 0].set_title('Validation accuracy')
axs[1, 0].set_xlabel('No. epochs')
axs[1, 0].set_ylabel('Accuracy Value (%)')
axs[1, 0].set_box_aspect(1)
axs[1, 1].plot(log_data['val_loss'])
axs[1, 1].set_title('Validation loss')
axs[1, 1].set_xlabel('No. epochs')
axs[1, 1].set_ylabel('Loss Value')
axs[1, 1].set_box_aspect(1)
plt.tight_layout()
plt.savefig('./Images/' + run_name + '_training.png')

# evaluate performance
score = model.evaluate(x_test, y_test, verbose=0)
f = open(run_name + '_performance.txt', 'w')
print('Test loss:' + str(score[0]), file=f)
print('Test accuracy:' + str(score[1]), file=f)
f.close()

# inspect output
# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = np.argmax(model.predict(x_test), axis=1)
y_test_num = np.argmax(y_test, axis=1)

# Check which items we got right / wrong
correct_indices = np.where(predicted_classes == y_test_num, 1, 0)
incorrect_indices = np.where(predicted_classes != y_test_num, 1, 0)
#print(correct_indices)
#print(incorrect_indices)

plt.figure()
plt.axis('on')
plt.tight_layout()
counter = 0
for i, correct in enumerate(correct_indices):
    if correct == 1:
        plt.subplot(3, 3, counter + 1)
        plt.imshow(x_test[i].reshape(32, 32, 3), interpolation='none')
        plt.title("{}".format(str_labels[y_test_num[i]]))
        counter += 1
    if counter == 9:
        counter = 0
        break
plt.savefig('./Images/' + run_name + '_correct.png')

plt.figure()
plt.axis('on')
plt.tight_layout()
for i, incorrect in enumerate(incorrect_indices):
    if incorrect == 1:
        plt.subplot(3, 3, counter + 1)
        plt.imshow(x_test[i].reshape(32, 32, 3), interpolation='none')
        plt.title("Predicted {}, Actual {}".format(str_labels[predicted_classes[i]], str_labels[y_test_num[i]]), fontsize= 7)
        counter += 1
    if counter == 9:
        break
plt.savefig('./Images/' + run_name + '_incorrect.png')

plt.show()
