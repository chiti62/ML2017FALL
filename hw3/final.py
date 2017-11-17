#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:26:35 2017

@author: tingting
"""

import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
#matplotlib inline
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import sys
inputfilepath = sys.argv[1]
x = pd.read_csv(inputfilepath)

#print(x.values.shape)

data = x.values
#print(data.shape)
#print(data[0, 0])

Y = data[:, 0]

pixels = data[:, 1]
#print(type(pixels))
#print(len(pixels[0]))
#print(pixels[10][10])

X = np.zeros((pixels.shape[0], 48*48))

for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:15:25 2017

@author: tingting
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:14:45 2017

@author: tingting
"""

import numpy as np
#from matplotlib import pyplot as plt

x = X
y = Y


#for ix in range(10):
#    plt.figure(ix)
#    plt.imshow(x[ix].reshape((48, 48)), interpolation='none', cmap='gray')
#plt.show()

#x -= np.mean(x, axis=0)
#x /= np.std(x, axis=0)
x /= 255
#for ix in range(10):
#    plt.figure(ix)
#    plt.imshow(x[ix].reshape((48, 48)), interpolation='none', cmap='gray')
#plt.show()

#import theano
import os
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
#from sklearn.model_selection import train_test_split

X_train = x[0:28510,:]
Y_train = y[0:28510]
#X_train, X_crossval, Y_train, Y_crossval = train_test_split(X_train, Y_train, test_size=0.025)
#print(X_train.shape , Y_train.shape)
X_crossval = x[28510:28710,:]
Y_crossval = y[28510:28710]
#X_crossval = x[28710:32300,:]
#Y_crossval = y[28710:32300]
#print (X_crossval.shape , Y_crossval.shape)
X_train = X_train.reshape((X_train.shape[0], 48, 48, 1))
X_crossval = X_crossval.reshape((X_crossval.shape[0], 48, 48, 1))

import tensorflow as tf

#tf.python.control_flow_ops = tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
import numpy
import csv
#import scipy.misc
#import scipy
#from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization

img_rows, img_cols = 48, 48
model = Sequential()
model.add(Convolution2D(64, (5, 5),input_shape=(img_rows, img_cols,1),padding = 'valid'))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(7))
model.add(Activation('softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


#print(y.shape)
y_ = np_utils.to_categorical(y, num_classes=7)

#print(y_.shape)
Y_train = y_[:28510]
Y_crossval = y_[28510:28710]
#print(X_crossval.shape, model.input_shape, Y_crossval.shape)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
#checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
log_filename = 'abc.csv'
ModelCheckPoint_filename='c_model.{epoch:02d}-{val_acc:.4f}.h5'

cllbks = [
            #CSVLogger(log_filename, append=True, separator=';'),
            #EarlyStopping(monitor='val_loss', patience=100, verbose=0),
            TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
            ModelCheckpoint(ModelCheckPoint_filename, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=10),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=1)]

model.fit_generator(datagen.flow(X_train, Y_train,
                    batch_size=128),
                    nb_epoch=1000,
                    validation_data=(X_crossval, Y_crossval),
                    samples_per_epoch=X_train.shape[0],
                    callbacks=cllbks)
                    

model.save("11163.h5")
#model.load_weights("Model.59-0.6053.hdf5")

#for ix in range(10):
#    plt.figure(ix)
#    plt.imshow(X_train[ix].reshape(48, 48), cmap='gray')
#    print(np.argmax(pred[ix]), np.argmax(y_[ix]))
#plt.show()

from keras import backend as K

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[4].output])
layer_output =get_layer_output([X_train[4,:].reshape(1,1,48,48),0])[0] 

print(layer_output[0,1,:,:].shape)

#plt.figure(1)
#plt.imshow(layer_output[0,1,:,:], cmap='gray')
#plt.show()

#model.layers

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-4].output])
layer_output =get_layer_output([X_train[1,:].reshape(1,1,48,48),0])[0] 

#print(layer_output)
#print(layer_output.shape)

layer_output =get_layer_output([X_m.reshape(X_m.shape[0],1,48,48),0])[0] 

#print(layer_output.shape)

import os

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-4].output])
