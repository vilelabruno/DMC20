import pandas as pd
import numpy as np

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
Y_TEST = TARGETS.copy()
#del TARGETS

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
x_test["salesPrice"] = w_test
y_train = x_train.pop('order')
y_test = x_test.pop('order')

lstm = pd.read_csv("out/lstm.csv")
xgb = pd.read_csv("out/xgb_day2day.csv")
infos = pd.read_csv('data/infos.csv', sep="|")

w = infos.pop("simulationPrice")

lstm = lstm.drop("index", axis=1)
xgb = xgb.drop("index", axis=1)

Y_TEST['sum'] = 0
for i in range(1, 13):
    Y_TEST["sum"] = Y_TEST["sum"] + Y_TEST[int(i)]
    Y_TEST = Y_TEST.drop(int(i), axis=1)

print("Correlation:")
print(xgb['0'].corr(lstm['0']))
print('\n')

#result = ((lstm["0"] + xgb["0"]) / len(xgb))
result = ((lstm["0"] + xgb["0"]) / len(xgb)) * 50
#result = ((lstm["0"] + xgb["0"]) / 2)
#result = (lstm["0"] + xgb["0"]) * xgb["0"]
#result = lstm["0"] + xgb["0"]
#result = lstm["0"] * xgb["0"]

print(lstm)
print(lstm.describe())
print(xgb)
print(xgb.describe())

print(result)
print(result.describe())

print('\n'+"+----- ENSEMBLE ----")
score = pd.DataFrame()
score = w * result
score[(Y_TEST["sum"] - result) < 0] = 0.6 * w[(Y_TEST["sum"] - result) < 0] * (Y_TEST["sum"][(Y_TEST["sum"] - result) < 0] - result[(Y_TEST["sum"] - result) < 0]) 
print('| Final Score: '+str(score.sum()))
equals = result[result.astype(int) == Y_TEST["sum"].astype(int)]
equals = equals.dropna()
print('| Exact Predictions: '+str(len(equals))+' of '+str(len(result)))
print("+-------------------"+'\n')

print('\n'+"+----- XGBOOST DAY2DAY ----")
scoreX = pd.DataFrame()
scoreX = w * xgb['0']
scoreX[(Y_TEST["sum"] - xgb['0']) < 0] = 0.6 * w[(Y_TEST["sum"] - xgb['0']) < 0] * (Y_TEST["sum"][(Y_TEST["sum"] - xgb['0']) < 0] - xgb['0'][(Y_TEST["sum"] - xgb['0']) < 0]) 
print('| Final Score: '+str(scoreX.sum()))
equals = xgb['0'][xgb['0'].astype(int) == Y_TEST["sum"].astype(int)]
equals = equals.dropna()
print('| Exact Predictions: '+str(len(equals))+' of '+str(len(xgb['0'])))
print("+--------------------------"+'\n')

print("Which one is the best?"+'\n')
if (scoreX.sum() > score.sum()):
    print("XGBOOST MODEL IS THE BEST!!!")
else:
    print("ENSEMBLE IS THE BEST!!!")

print("\n"+"Bruno Style Score:")
w = pd.DataFrame(w)
w = np.array(w["simulationPrice"])

x_train["order"] = y_train
future = train[(train["date"] > pd.to_datetime("2018-06-16")) & (train["date"] <= pd.to_datetime("2018-06-29"))]
future = future.groupby("itemID")["order"].sum()
future = np.array(future)

score = pd.DataFrame()
score = result * w
score[(future - result) < 0] = (future[(future - result) < 0] - result[(future - result) < 0]) * (0.6 * w[(future - result) < 0])
print("  ENSEMBLE: "+str(sum(score)))

scoreX = pd.DataFrame()
scoreX = xgb['0'] * w
scoreX[(future - xgb['0']) < 0] = (future[(future - xgb['0']) < 0] - xgb['0'][(future - xgb['0']) < 0]) * (0.6 * w[(future - xgb['0']) < 0])
print("  XGBOOST: "+str(sum(scoreX)))

print("\n"+"  Which one is the best?"+'\n')
if (scoreX.sum() > score.sum()):
    print("  XGBOOST MODEL IS THE BEST!!!")
else:
    print("  ENSEMBLE IS THE BEST!!!")

#result.to_csv("out/ensemble.csv")