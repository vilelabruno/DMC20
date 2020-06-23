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
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam

print('Setting random seed...')
seed = 1234
np.random.seed(seed)
train = pd.read_csv("data/trainAllDays.csv")
train["date"] = pd.to_datetime(train["date"])
train.sort_values(by=["date"])
train["weekDay"] = train["date"].dt.day_name()   
train = pd.get_dummies(train, columns=["weekDay"]) 
train = train[train["itemID"] != 10464]

'''Feature Engineering'''
train['category1_2'] = 0.0
train['category1_2'][train['category1'] == 1.0 & train['category2'] == 1.0] = 1.0
train['category1_3'] = 0.0
train['category1_3'][train['category1'] == 1.0 & train['category3'] == 1.0] = 1.0
train['category2_3'] = 0.0
train['category2_3'][train['category3'] == 1.0 & train['category2'] == 1.0] = 1.0
train['category1_2_3'] = 0.0
train['category1_2_3'][train['category1'] == 1.0 & train['category2'] == 1.0 & train['category3'] == 1.0] = 1.0

del train["promotion"]

x_test = train[train["date"] == pd.to_datetime("2018-06-01")]
x_train = train[train["date"] < pd.to_datetime("2018-06-01")]


del x_train["date"], x_test["date"]

'''Fill NaN'''
#x_train = x_train.fillna(0)
#x_test = x_test.fillna(0)

'''Popping order and simulationPrice columns'''
y_train = x_train.pop('order')
w_train = x_train.pop('simulationPrice')
y_test = x_test.pop('order')
w_test = x_test.pop('simulationPrice')

hidden_nodes = 100
output_labels = 1
model_lstm = Sequential()
model_lstm.add(LSTM(hidden_nodes, return_sequences=True, input_shape=(1,16)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(hidden_nodes))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=output_labels))
model_lstm.add(Activation('linear'))

opt = Adam(learning_rate=0.2)
model_lstm.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

# Reshape the data between -1 and 1 and to 3D
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_test)
x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
x_val_reshaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))

y_train_reshaped = y_train.values.reshape(-1, 1)
y_val_reshaped = y_test.values.reshape(-1, 1)

w_train_reshaped = w_train.values.reshape(-1, 1)
w_val_reshaped = w_test.values.reshape(-1, 1)

history = model_lstm.fit(x_train_reshaped, y_train_reshaped, validation_data=(x_val_reshaped, y_val_reshaped),epochs=15, batch_size=2048, verbose=2, shuffle=False)
y_pre = model_lstm.predict(x_val_reshaped)

'''Final Score'''
preds = pd.DataFrame(y_pre)
y_test = y_test.reset_index(drop=True)
w_test = w_test.reset_index(drop=True)
score = preds.copy()
score = w_test * preds[0]
score[(y_test - preds[0]) < 0] = 0.6 * w_test[(y_test - preds[0]) < 0] * (y_test[(y_test - preds[0]) < 0] - preds[0][(y_test - preds[0]) < 0])
print('Final Score: '+str(score.sum()))

'''Exact Predictions'''
equals = preds[0][preds[0].astype(int) == y_test.astype(int)]
print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds)))