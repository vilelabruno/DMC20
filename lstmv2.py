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
train = pd.read_csv("data/trainNew.csv")
'''Feature eng'''
train.fillna(0, inplace=True)
train["order"][train["order"] == 0] = 0 + 1e-6
train["diffSimRec"] = train["recommendedRetailPrice"] - train["simulationPrice"]
#plt.plot(train["diffSimRec"])
#plt.show()
#
'''Deleting promotion column'''
del train["promotion"]

#'''Promotion times Price'''
#train["weekPromotion"] = train["weekPromotion"] * train["simulationPrice"]

'''test this without converting to datetime before'''
# train.sort_values(by=["date"])
# X_test = train[train["date"] == "2018-06-30"]
# X_train = train[train["date"] != "2018-06-30"]
# X_train = pd.to_datetime(X_train["date"])
# X_train["day"] = X_train
# X_train["week"] = X_train
# X_train["month"] = X_train

train["date"] = pd.to_datetime(train["date"])
train["day"] = train["date"].dt.day
train["weekNumber"] = train["date"].dt.week
train["month"] = train["date"].dt.month
train.sort_values(by=["date"])
X_test = train[train["date"] == pd.to_datetime("2018-06-17")]
X_train = train[train["date"] < pd.to_datetime("2018-06-17")]
#X_test = train[train["date"] >= pd.to_datetime("2018-07-01")]
#X_train = train[train["date"] < pd.to_datetime("2018-07-01")]

del X_train["date"], X_test["date"]

'''Popping order and simulationPrice columns'''

w_train = X_train.pop('simulationPrice')
w_test = X_test.pop('simulationPrice')
y_train = X_train.pop('order')
y_test = X_test.pop('order')

model_lstm = Sequential()
model_lstm.add(LSTM(18, input_shape=(1,13)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Reshape the data between -1 and 1 and to 3D
from sklearn.preprocessing import StandardScaler,MinMaxScaler

sumPreds = np.zeros(10464)
for i in range(0,1):   
    scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train_scaled = scaler.fit_transform(X_train)
    x_valid_scaled = scaler.fit_transform(X_test)
    x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
    x_val_resaped = x_valid_scaled.reshape((x_valid_scaled.shape[0], 1, x_valid_scaled.shape[1]))
    history = model_lstm.fit(x_train_reshaped, y_train, validation_data=(x_val_resaped, y_test),epochs=5, batch_size=100, verbose=2, shuffle=False)
    preds = model_lstm.predict(x_val_resaped)
    #= xgb_model.predict(X_test)
    X_train["order"] = y_train
    X_test["order"] = preds
    X_test["order"][X_test["order"] < 0] = 0
    X_test["order"] = X_test["order"].astype(int)

    predList = []

    for ar in preds:
        predList.append(ar[0])

    sumPreds = sumPreds + np.array(predList)
    dif = pd.DataFrame(predList - y_test) 
    print(dif.describe())
    X_train = pd.concat([X_train, X_test])
    X_test["day"] = X_test["day"]+1
    y_train = X_train.pop('order')
    y_test = X_test.pop('order')

future = train[train["date"] == pd.to_datetime("2018-06-18")]
future = future.groupby("itemID").agg({"order": "sum"}) # todo set <0 to zero on sumPreds...
dif = pd.DataFrame(sumPreds - future["order"]) 
print(dif.describe())