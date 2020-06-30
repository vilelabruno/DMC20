import pandas as pd

print("Importing Data..."+'\n')
train = pd.read_csv("data/trainNew.csv")
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

x_test = train[train["date"] == pd.to_datetime("2018-06-17")]
x_train = train[train["date"] < pd.to_datetime("2018-06-17")]

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

lstm = pd.read_csv("out/lstm.csv")
xgb = pd.read_csv("out/xgb_day2day.csv")
infos = pd.read_csv('data/infos.csv', sep="|")

w = infos.pop("simulationPrice")

lstm["sum"] = 0
Y_TEST["sum"] = 0
for i in range(0, 13): 
    lstm["sum"] = lstm["sum"] + lstm[str(i)]
    lstm = lstm.drop(str(i), axis=1)

for i in range(1, 13):
    Y_TEST["sum"] = Y_TEST["sum"] + Y_TEST[int(i)]
    Y_TEST = Y_TEST.drop(int(i), axis=1)

xgb = xgb.drop("index", axis=1)

print("Correlation:")
print(xgb['0'].corr(lstm['sum']))
print('\n')

result = ((lstm["sum"] + xgb["0"]) / len(xgb))

print(lstm)
print(lstm.describe())
print(xgb)
print(xgb.describe())

print(result)
print(result.describe())

print('\n'+"+ ---- ENSEMBLE ----")
score = pd.DataFrame()
score = w * result
score[(Y_TEST["sum"] - result) < 0] = 0.6 * w[(Y_TEST["sum"] - result) < 0] * (Y_TEST["sum"][(Y_TEST["sum"] - result) < 0] - result[(Y_TEST["sum"] - result) < 0]) 
print('| Final Score: '+str(score.sum()))
equals = result[result.astype(int) == Y_TEST["sum"].astype(int)]
equals = equals.dropna()
print('| Exact Predictions: '+str(len(equals))+' of '+str(len(result)))
print("+ ---- ENSEMBLE ----"+'\n')

print('\n'+"+ ---- XGBOOST DAY2DAY ----")
scoreX = pd.DataFrame()
scoreX = w * xgb['0']
scoreX[(Y_TEST["sum"] - xgb['0']) < 0] = 0.6 * w[(Y_TEST["sum"] - xgb['0']) < 0] * (Y_TEST["sum"][(Y_TEST["sum"] - xgb['0']) < 0] - xgb['0'][(Y_TEST["sum"] - xgb['0']) < 0]) 
print('| Final Score: '+str(scoreX.sum()))
equals = xgb['0'][xgb['0'].astype(int) == Y_TEST["sum"].astype(int)]
equals = equals.dropna()
print('| Exact Predictions: '+str(len(equals))+' of '+str(len(xgb['0'])))
print("+ ---- XGBOOST DAY2DAY ----"+'\n')

print("Which one is the best?"+'\n')
if (scoreX.sum() > score.sum()):
    print("XGBOOST MODEL IS THE BEST!!!")
else:
    print("ENSEMBLE IS THE BEST!!!")