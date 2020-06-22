from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

print('Setting random seed...')
seed = 1234
np.random.seed(seed)
train = pd.read_csv("data/trainAllDays.csv")
train["time"] = pd.to_datetime(train["time"])
train.sort_values(by=["time"])
train["weekDay"] = train["time"].dt.day_name()   
train = pd.get_dummies(train, columns=["weekDay"]) 

del train["promotion"]

X_test = train[train["time"] > pd.to_datetime("2018-06-01")]
X_train = train[train["time"] <= pd.to_datetime("2018-06-01")]


del X_train["time"], X_test["time"]

'''Popping order and simulationPrice columns'''
y_train = X_train.pop('order')
w_train = X_train.pop('simulationPrice')
y_test = X_test.pop('order')
w_test = X_test.pop('simulationPrice')

model_lstm = Sequential()
model_lstm.add(LSTM(32, input_shape=(1,10)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

