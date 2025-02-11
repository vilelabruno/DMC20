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
train = pd.read_csv("data/trainNew.csv")
orders = pd.read_csv("data/orders.csv")
limiar = pd.read_csv("limiar.csv")
sp = pd.read_csv("salesPrice.csv")
del limiar["Unnamed: 0"], sp["Unnamed: 0"]
limiar["limiarDate"] =  pd.to_datetime(limiar["time"])
del limiar["time"]
train = train[train["itemID"] != 10464]

train = train.merge(limiar, on="itemID", how="left")
del train["salesPrice"]
train = train.merge(pd.DataFrame(orders.groupby("itemID")["salesPrice"].mean()).rename(columns={"salesPrice": "salesPriceMean"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby("itemID")["salesPrice"].std()).rename(columns={"salesPrice": "salesPriceStd"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby("itemID")["salesPrice"].min()).rename(columns={"salesPrice": "salesPriceMin"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby("itemID")["salesPrice"].max()).rename(columns={"salesPrice": "salesPriceMax"}) , how="left", on="itemID")

train["brandNA"] = 0
train["brandNA"][train["brand"] == 0] = 1
train["brandManu"] = train["brand"] * train["manufacturer"]

train["customerRatingCat"] = train["customerRating"].astype(int)
train["customerRatingNA"] = 0
train["customerRatingNA"][train["customerRating"] == 0] = 1
train = pd.get_dummies(train, columns=["customerRatingCat"])
'''Deleting promotion column'''
del train["promotion"]
#'''Promotion times Price'''
orders["time"] = pd.to_datetime(orders["time"])
orders["date"] = orders["time"].dt.date
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].count()).rename(columns={"transactID": "transactIDcount"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].mean()).rename(columns={"transactID": "transactIDmean"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].min()).rename(columns={"transactID": "transactIDmin"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].max()).rename(columns={"transactID": "transactIDmax"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].std()).rename(columns={"transactID": "transactIDstd"}) , how="left", on="itemID")

train["date"] = pd.to_datetime(train["date"])
train["daysToLimiar"] = train["limiarDate"] - train["date"]
train['daysToLimiar'] = pd.to_numeric(train['daysToLimiar'], errors='coerce')  

train.fillna(0, inplace=True)  
#train["daysToLimiar"] = train["daysToLimiar"].astype(int)
del train["limiarDate"]
train["day"] = train["date"].dt.day
train["classDay"] = train["day"]/6
train["classDay"] = train["classDay"].astype(int)
train = pd.get_dummies(train, columns=["classDay"])
train["weekNumber"] = train["date"].dt.week
train = train.merge(sp, on=["itemID", "weekNumber"], how="left")
train["weekDay"] = train["date"].dt.weekday

train["month"] = train["date"].dt.month
train.sort_values(by=["date"])
#train = pd.get_dummies(train, columns=["weekDay"])

#print("Feature Engineering..."+'\n')
#train['category1_2'] = train['category1'] * train['category2']
#train['category1_3'] = train['category1'] * train['category3']
#train['category2_3'] = train['category2'] * train['category3']
#train['category1_2_3'] = train['category1'] * train['category2'] * train['category3']

#train["order"][train["order"] == 0] = 0 + 1e-6
#train["diffSimRec"] = train["recommendedRetailPrice"] - train["simulationPrice"]

#del train["promotion"]

x_test = train[train["date"] == pd.to_datetime("2018-06-16")]
x_train = train[train["date"] <= pd.to_datetime("2018-06-16")]
w = x_test["recommendedRetailPrice"].fillna(1)

X_TEST = train[train["date"] >= pd.to_datetime("2018-06-17")]
Y_TEST = pd.DataFrame()
Y_TEST['order'] = X_TEST.pop('order')
Y_TEST['date'] = X_TEST['date']
Y_TEST['itemID'] = X_TEST['itemID']
TARGETS = pd.DataFrame()
TARGETS[0] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-17")].reset_index().drop('index', axis=1)['order']
TARGETS[1] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-18")].reset_index().drop('index', axis=1)['order']
TARGETS[2] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-19")].reset_index().drop('index', axis=1)['order']
TARGETS[3] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-20")].reset_index().drop('index', axis=1)['order']
TARGETS[4] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-21")].reset_index().drop('index', axis=1)['order']
TARGETS[5] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-22")].reset_index().drop('index', axis=1)['order']
TARGETS[6] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-23")].reset_index().drop('index', axis=1)['order']
TARGETS[7] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-24")].reset_index().drop('index', axis=1)['order']
TARGETS[8] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-25")].reset_index().drop('index', axis=1)['order']
TARGETS[9] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-26")].reset_index().drop('index', axis=1)['order']
TARGETS[10] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-27")].reset_index().drop('index', axis=1)['order']
TARGETS[11] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-28")].reset_index().drop('index', axis=1)['order']
TARGETS[12] = Y_TEST[Y_TEST['date'] == pd.to_datetime("2018-06-29")].reset_index().drop('index', axis=1)['order']
del Y_TEST
Y_TEST = TARGETS
del TARGETS

del X_TEST
del x_train["date"], x_test["date"], x_train['salesPrice'], x_test['salesPrice']

'''Fill NaN'''
#x_train = x_train.fillna(0)
#x_test = x_test.fillna(0)

'''Popping order and simulationPrice columns'''

x_train = x_train.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].mean()).rename(columns={"order": "orderMean"}), how="left", on="itemID")
x_train = x_train.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].std()).rename(columns={"order": "orderStd"}), how="left", on="itemID")
x_train = x_train.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].min()).rename(columns={"order": "orderMin"}), how="left", on="itemID")
x_train = x_train.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].max()).rename(columns={"order": "orderMax"}), how="left", on="itemID")

x_test = x_test.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].mean()).rename(columns={"order": "orderMean"}), how="left", on="itemID")
x_test = x_test.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].std()).rename(columns={"order": "orderStd"}), how="left", on="itemID")
x_test = x_test.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].min()).rename(columns={"order": "orderMin"}), how="left", on="itemID")
x_test = x_test.merge(pd.DataFrame(x_train.groupby(["itemID"])["order"].max()).rename(columns={"order": "orderMax"}), how="left", on="itemID")

train_day = x_train[x_train["weekDay"] == x_test["weekDay"].iloc[0]]
y_train_day = train_day.pop('order')
w_train = x_train.pop('simulationPrice')
w_train_day = train_day.pop('simulationPrice')
w_test = x_test.pop('simulationPrice') # qdo for prever colocar salesPrice= simulationPrice
#x_test["salesPrice"] = w_test
y_train = x_train.pop('order')
y_test = x_test.pop('order')

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
days = 13
n_epochs = 1
b_size = 2048
for i in range(0, days):
    print("---- DAY "+str(i)+" ----")
    if x_test["day"].iloc[0] == 30:
        x_test["day"] = 1
        x_test["month"] = x_test["month"]+1 
    else:
        x_test["day"] = x_test["day"]+1
    x_test["daysToLimiar"] = x_test["daysToLimiar"]+1
    if x_test["weekDay"].iloc[0] == 7:
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

    model_lstm.fit(x_train_reshaped, y_train_reshaped, validation_data=(x_val_reshaped, y_val_reshaped),epochs=n_epochs, batch_size=b_size, verbose=2, shuffle=False)
    #DE = shap.DeepExplainer(model_lstm, x_train_reshaped) # x_train is 3d numpy.ndarray
    #shap_values = DE.shap_values(x_val_reshaped, check_additivity=False) # x_validate is 3d numpy.ndarray

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
    x_test["order"][x_test["order"] < 0] = 0
    x_test["order"] = x_test["order"].astype(int)

    x_train = pd.concat([x_train, x_test])
    
    if x_test["weekDay"].iloc[0] == 7:
        train_day = x_train[x_train["weekDay"] == 0]
    else:
        train_day = x_train[x_train["weekDay"] == x_test["weekDay"].iloc[0]]
    
    y_train_day = train_day.pop('order')
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
preds = (preds * 0.5).astype(int)
#Y_TEST.std().mean() == 26.88784712221067
#preds = (preds * 27).astype(int)

print(preds)
print(Y_TEST)
print(preds.describe())
print(Y_TEST.describe())
print("\n")

for i in range(0, days): 
    score[i] = w_test * preds[i] 
    score[i][(Y_TEST[i] - preds[i]) < 0] = 0.6 * w_test[(Y_TEST[i] - preds[i]) < 0] * (Y_TEST[i][(Y_TEST[i] - preds[i]) < 0] - preds[i][(Y_TEST[i] - preds[i]) < 0]) 
    print('Day '+str(i)+' Score: '+str(score[i].sum()))

print('\n'+'Final Score: '+str(score.values.sum()))
equals = preds[preds.astype(int) == Y_TEST[i].astype(int)]
equals = equals.dropna()
print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds))+'\n')
print(score.describe())

print("Bruno Style Score:")
w = pd.DataFrame(w)
w = np.array(w["recommendedRetailPrice"])

x_train["order"] = 0
aux = x_train[['order', 'itemID']].groupby('itemID').agg({'order':'sum'}).rename(columns={"order": "orderSum"}).astype(int)
for i in range(0, days): 
    aux['orderSum'] = aux['orderSum'] + preds[i].astype(int)
aux = aux.fillna(0)
x_train = x_train.merge(aux, how="left", on="itemID")
x_train['order'] = x_train.pop('orderSum').astype(int)

future = train[(train["date"] > pd.to_datetime("2018-06-16")) & (train["date"] <= pd.to_datetime("2018-06-29"))]
future = future.groupby("itemID")["order"].sum()

preds = x_train[((x_train["day"] > 16) & (x_train["month"] == 6))]
preds = preds.groupby("itemID")["order"].sum()

#dif = pd.DataFrame(sumPreds - future) 
preds = np.array(preds)
preds[preds < 0 ] = 0
#preds[preds > 5 ] = preds[preds > 5 ] * 4
future = np.array(future)
score = preds * w
score[(future - preds) < 0] = (future[(future - preds) < 0] - preds[(future - preds) < 0]) * (0.6 * w[(future - preds) < 0])
print(sum(score))

print("Saving Results..."+'\n')
#pd.DataFrame(preds).to_csv("out/lstm.csv")

print("   - THE END -   "+'\n')

print("+----------------+")
print("| Copyright 2020 |")
print("+----------------+")