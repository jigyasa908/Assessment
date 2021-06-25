"""
Author - Jigyasa Singh

Take Out 1 - Facial Emotion/Expression Detection

This program focuses on building a Convolution Neural Network (CNN) Model
The program is divided into 4 major steps - 
1. Data Preprocessing
2. Building the CNN Model
3. Training the CNN on the Training set and Evaluating it on the Test set
4. Visualisation

"""
import numpy as np 
import pandas as pd  
import os
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape, \
BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#STEP 1 - DATA PREPROCESSING
x = pd.read_csv('fer2013.csv')
print('Original Dataset:', x.values.shape)

# Cleaning - removing duplicate pixel data
x.drop_duplicates(inplace=True)
print('Cleaned Dataset:', x.values.shape)
data = x.values
y = data[:, 0]  
pixels = data[:, 1]
X = np.zeros((pixels.shape[0], 48 * 48)) 
for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])

# Rescaling/Normalising value of X between 0 and 1
X = X / 255

# Splitting data into Training set and Testing Set
X_train = X[0:28710, :]
Y_train = y[0:28710]
# print (X_train.shape, Y_train.shape)
X_test = X[28710:32300, :]
Y_test = y[28710:32300]
# print (X_test.shape, Y_test.shape)

X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
X_test = X_test.reshape((X_test.shape[0], 48, 48, 1))
X_train.shape
# print (y.shape)
y_ = np_utils.to_categorical(y, 7)
# print (y_.shape)
Y_train = y_[:28710]
Y_test = y_[28710:32300]
# print (X_test.shape, Y_test.shape)
print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)
print('X_test: ', X_test.shape)
print('Y_test: ', Y_test.shape)

"""
STEP 2 - BUILDING THE CNN
The lines below are the layers present in a CNN
"""
model_cnn = tf.keras.models.Sequential()
model_cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[48, 48, 1]))
model_cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model_cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model_cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
model_cnn.add(tf.keras.layers.Dense(units=7, activation='softmax'))

"""
STEP 3 - TRAINING AND TESTING THE CNN
This phase is mainly divided into two steps -
i. Compiling the CNN
ii. Training the CNN on the Training set and evaluating it on the Test set
"""
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_cnn.fit(X_train, Y_train, batch_size=24, epochs=30, validation_data=(X_test, Y_test))

#STEP 4 - VISUALISATION
train_loss = history.history['loss']
test_loss = history.history['val_loss']
train_acc = history.history['acc']
test_acc = history.history['val_acc']
epochs = range(len(train_acc))

plt.plot(epochs, train_loss, 'r', label='train_loss')
plt.plot(epochs, test_loss, 'b', label='test_loss')
plt.title('train_loss vs test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, 'r', label='train_acc')
plt.plot(epochs, test_acc, 'b', label='test_acc')
plt.title('train_acc vs test_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.figure()
plt.show()