#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:30:12 2018

@author: tingting
"""

import csv
import sys
import pandas as pd
import keras
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

encoding_dim = 64
imagefile = sys.argv[1]
modelname = 'model.860-0.1676.h5'
# test_data_path = 'test_case.csv'
test_data_path = sys.argv[2]
# predictfile = 'predict1676-3.csv'
predictfile = sys.argv[3]

autoencoder = load_model(modelname)

input_img = Input(shape=(784,))
# retrieve the last layer of the autoencoder model
#encoded = autoencoder.layers[0](input_img)
encoded = autoencoder.layers[1](input_img)
encoded = autoencoder.layers[2](encoded)
encoded = autoencoder.layers[3](encoded)
#encoded = autoencoder.layers[4](encoded)
# create the decoder model
encoder = Model(input_img, encoded)

x = np.load(imagefile)
encoded_imgs = encoder.predict(x)

kmeans = KMeans(n_clusters=2)
kmeans.fit(encoded_imgs)
labels = kmeans.labels_

#for i in range(10):
#    img = np.reshape(x[i,:], (28, 28))
#    plt.imshow(img)
#    plt.show()
#    
test = pd.read_csv(test_data_path, sep=',', header=0)
test = np.array(test.values)


ans = []
for i in range(np.size(test,0)):
    ans.append([str(i)])
    f = lambda x: 1 if labels[test[i,1]] == labels[test[i,2]] else 0
    ans[i].append(f(i))

with open(predictfile, 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['ID','Ans'])
    for i in range(len(ans)):
        writer.writerow(ans[i])
        

