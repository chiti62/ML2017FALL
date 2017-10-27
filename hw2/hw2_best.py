#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:59:17 2017

@author: tingting
"""
import pandas
import numpy as np
import csv
import os, sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import normalize

model = load_model('my_model.h5')
test_raw = pandas.read_csv(sys.argv[5]).values
test = np.array(test_raw)
test_normed = normalize(test, axis=0, norm='max')
pred = model.predict(test_normed)
for row in pred:
    if row[0] > row[1]:
        row[0] = 0
    else:
        row[0] = 1
        
ans = []
for i in range(pred.shape[0]):
    ans.append([str(i+1)])
    a = pred[i][0]
    ans[i].append(int(a))
filename = sys.argv[6]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()