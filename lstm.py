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
import shap


# --- Disable Keras Warnings ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
# ------------------------------

seed = 1234
np.random.seed(seed)

print("Importing Data..."+'\n')
train = pd.read_csv("data/trainAllDays.csv")
train = train[train["itemID"] != 10464]
train["date"] = pd.to_datetime(train["date"])
train.sort_values(by=["date"])
train["weekDay"] = train["date"].dt.day_name()   
#train = pd.get_dummies(train, columns=["weekDay"])

print("Feature Engineering..."+'\n')
train['category1_2'] = train['category1'] * train['category2']
train['category1_3'] = train['category1'] * train['category3']
train['category2_3'] = train['category2'] * train['category3']
train['category1_2_3'] = train['category1'] * train['category2'] * train['category3']

#train["order"][train["order"] == 0] = 0 + 1e-6
train["diffSimRec"] = train["recommendedRetailPrice"] - train["simulationPrice"]

del train["promotion"]

train["date"] = pd.to_datetime(train["date"])
train["day"] = train["date"].dt.day
train["weekNumber"] = train["date"].dt.week
train["weekDay"] = train["date"].dt.weekday

train["month"] = train["date"].dt.month
train.sort_values(by=["date"])

x_test = train[train["date"] == pd.to_datetime("2018-06-01")]
x_train = train[train["date"] < pd.to_datetime("2018-06-01")]
X_TEST = train[train["date"] >= pd.to_datetime("2018-06-01")]
X_TEST = X_TEST[train["date"] < pd.to_datetime("2018-06-15")]
Y_TEST = pd.DataFrame()
Y_TEST['order'] = X_TEST.pop('order')
Y_TEST['date'] = X_TEST['date']
Y_TEST['itemID'] = X_TEST['itemID']
TARGETS = pd.DataFrame()
TARGETS[0] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-01")].reset_index().drop('index', axis=1)['order']
TARGETS[1] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-02")].reset_index().drop('index', axis=1)['order']
TARGETS[2] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-03")].reset_index().drop('index', axis=1)['order']
TARGETS[3] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-04")].reset_index().drop('index', axis=1)['order']
TARGETS[4] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-05")].reset_index().drop('index', axis=1)['order']
TARGETS[5] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-06")].reset_index().drop('index', axis=1)['order']
TARGETS[6] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-07")].reset_index().drop('index', axis=1)['order']
TARGETS[7] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-08")].reset_index().drop('index', axis=1)['order']
TARGETS[8] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-09")].reset_index().drop('index', axis=1)['order']
TARGETS[9] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-10")].reset_index().drop('index', axis=1)['order']
TARGETS[10] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-11")].reset_index().drop('index', axis=1)['order']
TARGETS[11] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-12")].reset_index().drop('index', axis=1)['order']
TARGETS[12] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-13")].reset_index().drop('index', axis=1)['order']
TARGETS[13] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-14")].reset_index().drop('index', axis=1)['order']
del Y_TEST
Y_TEST = TARGETS
del TARGETS

del X_TEST
del x_train["date"], x_test["date"], x_train['salesPrice'], x_test['salesPrice']

'''Fill NaN'''
#x_train = x_train.fillna(0)
#x_test = x_test.fillna(0)

'''Popping order and simulationPrice columns'''
y_train = x_train.pop('order')
w_train = x_train.pop('simulationPrice')
y_test = x_test.pop('order')
w_test = x_test.pop('simulationPrice')

print("Instantiating Model..."+'\n')
hidden_nodes = 100
output_labels = 1
model_lstm = Sequential()
model_lstm.add(LSTM(hidden_nodes, return_sequences=True, input_shape=(1,len(x_train.columns))))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(hidden_nodes))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=output_labels))
model_lstm.add(Activation('linear'))
opt = Adam(learning_rate=0.001)
model_lstm.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
# Reshape the data between -1 and 1 and to 3D
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))

preds = pd.DataFrame()
score = preds.copy()

print("TRAINING START"+'\n')
days = 14
n_epochs = 1
for i in range(0, days):
    print("---- DAY "+str(i)+" ----")
    x_test["day"] = x_test["day"]+1
    if x_test["weekDay"].iloc[0] == 6:
        x_test["weekNumber"] = x_test["weekNumber"] + 1 
        x_test["weekDay"] = 0
    else:
        x_test["weekDay"] = x_test["weekDay"]+1

    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.fit_transform(x_test)
    x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
    x_val_reshaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))

    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_val_reshaped = y_test.values.reshape(-1, 1)

    w_train_reshaped = w_train.values.reshape(-1, 1)
    w_val_reshaped = w_test.values.reshape(-1, 1)

    model_lstm.fit(x_train_reshaped, y_train_reshaped, validation_data=(x_val_reshaped, y_val_reshaped),epochs=n_epochs, batch_size=2048, verbose=2, shuffle=False)
    #DE = shap.DeepExplainer(model_lstm, x_train_reshaped) # X_train is 3d numpy.ndarray
    #shap_values = DE.shap_values(x_val_reshaped, check_additivity=False) # X_validate is 3d numpy.ndarray

    #shap.initjs()
    #shap.summary_plot(
    #    shap_values[0], 
    #    x_val_reshaped,
    #    feature_names=x_train.columns,
    #    max_display=50,
    #    plot_type='bar')

    y_pre = model_lstm.predict(x_val_reshaped)
    print(np.concatenate(y_pre, axis=0))
    #y_pre = pd.DataFrame(y_pre)
    preds[i] = np.concatenate(y_pre, axis=0)

    '''Daily Score'''
    y_test = y_test.reset_index(drop=True)
    w_test = w_test.reset_index(drop=True)
    
    score[i] = w_test * preds[i]
    score[i][(Y_TEST[i] - preds[i]) < 0] = 0.6 * w_test[(Y_TEST[i] - preds[i]) < 0] * (Y_TEST[i][(Y_TEST[i] - preds[i]) < 0] - preds[i][(Y_TEST[i] - preds[i]) < 0])
    print('\n'+'Day '+str(i)+' Score: '+str(score[i].sum()))

    '''Daily Exact Predictions'''
    equals = preds[i][preds[i].astype(int) == Y_TEST[i].astype(int)]
    print('Day '+str(i)+' Exact Predictions: '+str(len(equals))+' of '+str(len(preds)))

    x_train["order"] = y_train
    x_test["order"] = np.concatenate(y_pre, axis=0)
    preds[preds < 0 ] = 0
    preds = preds.astype(int)

    x_train = pd.concat([x_train, x_test])
    
    y_train = x_train.pop('order')
    y_test = x_test.pop('order')
    print("---------------"+'\n')
print("END OF TRAINING")
print(preds)
print(Y_TEST)
print(preds.describe())
print(Y_TEST.describe())
print("\n")

'''Final Score'''
print('Final Score: '+str(score.values.sum()))

'''Exact Predictions'''
equals = preds[preds.astype(int) == Y_TEST[i].astype(int)]
equals = equals.dropna()
print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds))+'\n')
print(score.describe())

print("Score with multiplier:"+'\n')
preds = preds * 1.3

print(preds)
print(Y_TEST)
print(preds.describe())
print(Y_TEST.describe())
print("\n")

for i in range(0, 14): 
    score[i] = w_test * preds[i] 
    score[i][(Y_TEST[i] - preds[i]) < 0] = 0.6 * w_test[(Y_TEST[i] - preds[i]) < 0] * (Y_TEST[i][(Y_TEST[i] - preds[i]) < 0] - preds[i][(Y_TEST[i] - preds[i]) < 0]) 
    print('Day '+str(i)+' Score: '+str(score[i].sum()))

print('\n'+'Final Score: '+str(score.values.sum()))
equals = preds[preds.astype(int) == Y_TEST[i].astype(int)]
equals = equals.dropna()
print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds))+'\n')
print(score.describe())

print("Saving Results..."+'\n')
#preds.to_csv("out/lstm.csv")

print("   - THE END -   "+'\n')

print("+----------------+")
print("| Copyright 2020 |")
print("+----------------+")