import pandas as pd

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

lstm = pd.read_csv("out/lstm.csv")
xgb = pd.read_csv("out/xgb_day2day.csv")
infos = pd.read_csv('data/infos.csv', sep="|")

w = infos.pop("simulationPrice")

lstm["sum"] = 0
Y_TEST["sum"] = 0
for i in range(0,14): 
    lstm["sum"] = lstm["sum"] + lstm[str(i)]
    Y_TEST["sum"] = Y_TEST["sum"] + Y_TEST[int(i)]
    lstm = lstm.drop(str(i), axis=1)
    Y_TEST = Y_TEST.drop(int(i), axis=1)

xgb = xgb.drop("index", axis=1)

result = lstm["sum"] * xgb["0"]

print(lstm)
print(lstm.describe())
print(xgb)
print(xgb.describe())

print(result)
print(result.describe())

score = pd.DataFrame()

score = w * result 
score[(Y_TEST["sum"] - result) < 0] = 0.6 * w[(Y_TEST["sum"] - result) < 0] * (Y_TEST["sum"][(Y_TEST["sum"] - result) < 0] - result[(Y_TEST["sum"] - result) < 0]) 
print('Final Score: '+str(score.sum()))
equals = result[result.astype(int) == Y_TEST["sum"].astype(int)]
equals = equals.dropna()
print('Exact Predictions: '+str(len(equals))+' of '+str(len(result))+'\n')
print(score.describe())