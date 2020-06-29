print("+---------------------------------+")
print("| Team Maia's LSTM Model - DMC 20 |")
print("+---------------------------------+")

print('\n'+"Importing Libraries..."+'\n')
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
#import shap


# --- Disable Keras Warnings ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# ------------------------------

seed = 1234
np.random.seed(seed)

print("Importing Data..."+'\n')
train = pd.read_csv('data/train2weeks.csv')
infos = pd.read_csv('data/infos.csv', sep="|")

x_test = train[train["weekNumber"] == 12]
x_train = train[train["weekNumber"] < 12]

aux = []
for i in range(1,10464):
    aux.append(i)
items = pd.DataFrame()
items["itemID"] = aux

x_test = x_test.merge(items, how="right", on="itemID")
x_test = x_test.sort_values(by=["itemID"])
x_test.reset_index(inplace=True)
x_test = x_test.drop("index", axis=1)
x_test['weekNumber'].fillna(12.0, inplace=True)
x_test['brand'].fillna(-1.0, inplace=True)
x_test['manufacturer'].fillna(-1.0, inplace=True)
x_test['customerRating'].fillna(-1.0, inplace=True)
x_test['recommendedRetailPrice'].fillna(-1.0, inplace=True)
x_test.fillna(0.0, inplace=True)

x_train = x_train.sort_values(by=["weekNumber", "itemID"])

y_train = x_train.pop("order")
y_train = np.log1p(y_train)
y_test = x_test.pop("order")
y_test = np.log1p(y_test)

w = infos.pop("simulationPrice")

print("Instantiating Model..."+'\n')
hidden_nodes = 100
output_labels = 1
model_lstm = Sequential()
model_lstm.add(LSTM(hidden_nodes, return_sequences=True, input_shape=(1,len(x_train.columns))))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(hidden_nodes))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=output_labels, activation='relu'))
opt = Adam(learning_rate=0.001)
model_lstm.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse'])
# Reshape the data between -1 and 1 and to 3D
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))

preds = pd.DataFrame()
score = preds.copy()

print("TRAINING START"+'\n')
n_epochs = 100
b_size = 2048

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_test)
x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
x_val_reshaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))

y_train_reshaped = y_train.values.reshape(-1, 1)
y_val_reshaped = y_test.values.reshape(-1, 1)

w_train_reshaped = w.values.reshape(-1, 1)
w_val_reshaped = w.values.reshape(-1, 1)

model_lstm.fit(x_train_reshaped, y_train_reshaped, validation_data=(x_val_reshaped, y_val_reshaped),epochs=n_epochs, batch_size=b_size, verbose=2, shuffle=False)

y_pre = model_lstm.predict(x_val_reshaped)
preds = np.concatenate(y_pre, axis=0)
preds = np.expm1(preds)
preds = preds.astype(int)

print(pd.DataFrame(preds).describe())
print('\n')
print(y_test.describe())

'''Final Score'''
score = preds * w
score[(y_test - preds) < 0] = 0.6 * w[(y_test - preds) < 0] * (y_test[(y_test - preds) < 0] - preds[(y_test - preds) < 0])
print('\n'+'Final Score: '+str(score.sum()))
'''Exact Predictions'''
equals = preds[preds.astype(int) == y_test.astype(int)]
print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds))+'\n')

print("Saving Results..."+'\n')
#preds.to_csv("out/lstm2.csv")

print("   - THE END -   "+'\n')

print("+----------------+")
print("| Copyright 2020 |")
print("+----------------+")