#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:02:55 2017

@author: tingting
"""
import pandas as pd
import numpy as np
import csv
import os, sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical
#from sklearn.preprocessing import normalize

model = load_model('123334.h5')
testpath = sys.argv[1]
data = pd.read_csv(testpath).values

pixels = data[:, 1]
X = np.zeros((pixels.shape[0], 48*48))

for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])

test = np.array(X)
#
#test -= np.mean(test, axis=0)
#test /= np.std(test, axis=0)
test /= 255

test_predict = test.reshape((test.shape[0], 48, 48, 1))

pred = model.predict(test_predict)

#for row in pred:
#    if row[0] > row[1]:
#        row[0] = 0
#    else:
#        row[0] = 1



ans = np.zeros([7178,1])

for idx in range(pred.shape[0]):
    ans[idx,0] = np.argmax(pred[idx,0:7])
    
ans2 = []
for i in range(pred.shape[0]):
    ans2.append([str(i)])
    a = ans[i][0]
    ans2[i].append(int(a))
filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans2[i])
text.close()
