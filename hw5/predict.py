import csv
import sys
import pickle
import numpy as np
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dense, Dropout, BatchNormalization
from keras.layers import Concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

modelname = './0.8463_017_240.h5'
testfile = sys.argv[1]
predictfile = sys.argv[2]
RATING_MEAN = 3.58171208604

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true) ** 2))

reader = csv.reader(open(testfile))
test_data = list(reader)
test_data = np.array(test_data[1:], dtype = np.dtype('float64'))
model = load_model(modelname, custom_objects = {'rmse':rmse})
ans = model.predict(np.hsplit(test_data[:,1:], 2), batch_size = 512)

count = 0
for i in ans:
    if np.isnan(i):
        count += 1
ans[np.isnan(ans)] = RATING_MEAN
with open(predictfile, 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['TestDataID','Rating'])
    for i in range(len(ans)):
        writer.writerow([i+1, ans[i][0]])
