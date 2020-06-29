import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.grid_search import GridSearchCV
print('Setting random seed...')
seed = 1234
np.random.seed(seed)

print('Reading csv...')
train = pd.read_csv('data/trainNew.csv')
orders = pd.read_csv("data/orders.csv")
limiar = pd.read_csv("limiar.csv")
sp = pd.read_csv("salesPrice.csv")
del limiar["Unnamed: 0"], sp["Unnamed: 0"]
limiar["limiarDate"] =  pd.to_datetime(limiar["time"])
del limiar["time"]
train = train[train["itemID"] != 10464]
#test = pd.read_csv('data/test.csv', sep='|')

'''Feature eng'''
#train["order"][train["order"] == 0] = 0 + 1e-6
#train["priceXCR"] = train["customerRating"] * train["simulationPrice"]
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
#train = pd.get_dummies(train, columns=["category1"]) 
#train = pd.get_dummies(train, columns=["category3"]) 


##plt.plot(train["diffSimRec"])
#plt.show()
#
'''Deleting promotion column'''
del train["promotion"]
#'''Promotion times Price'''
#train["weekPromotion"] = train["weekPromotion"] * train["simulationPrice"]
orders["time"] = pd.to_datetime(orders["time"])
orders["date"] = orders["time"].dt.date
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].count()).rename(columns={"transactID": "transactIDcount"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].mean()).rename(columns={"transactID": "transactIDmean"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].min()).rename(columns={"transactID": "transactIDmin"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].max()).rename(columns={"transactID": "transactIDmax"}) , how="left", on="itemID")
train = train.merge(pd.DataFrame(orders.groupby(["itemID"])["transactID"].std()).rename(columns={"transactID": "transactIDstd"}) , how="left", on="itemID")
'''test this without converting to datetime before'''
# train.sort_values(by=["date"])
# X_test = train[train["date"] == "2018-06-30"]
# X_train = train[train["date"] != "2018-06-30"]
# X_train = pd.to_datetime(X_train["date"])
# X_train["day"] = X_train
# X_train["week"] = X_train
# X_train["month"] = X_train+

train["date"] = pd.to_datetime(train["date"])

train["daysToLimiar"] = train["limiarDate"] - train["date"]
train['daysToLimiar'] = pd.to_numeric(train['daysToLimiar'], errors='coerce')  

train.fillna(0, inplace=True)  
#train["daysToLimiar"] = train["daysToLimiar"].astype(int)
del train["limiarDate"]
train["day"] = train["date"].dt.day
train["classDay"] = train["day"]/10
train["classDay"] = train["classDay"].astype(int)
train["weekNumber"] = train["date"].dt.week
train = train.merge(sp, on=["itemID", "weekNumber"], how="left")
train["weekDay"] = train["date"].dt.weekday

train["month"] = train["date"].dt.month
train.sort_values(by=["date"])
X_test = train[train["date"] == pd.to_datetime("2018-06-16")]
X_train = train[train["date"] <= pd.to_datetime("2018-06-16")]
#X_test = train[train["date"] >= pd.to_datetime("2018-07-01")]
#X_train = train[train["date"] < pd.to_datetime("2018-07-01")]

X_train_aux = X_train.copy()

w = X_test["recommendedRetailPrice"].fillna(1)
del X_train["date"], X_test["date"]

'''Popping order and simulationPrice columns'''

X_train = X_train.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].mean()).rename(columns={"order": "orderMean"}), how="left", on="itemID")
X_train = X_train.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].std()).rename(columns={"order": "orderStd"}), how="left", on="itemID")
X_train = X_train.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].min()).rename(columns={"order": "orderMin"}), how="left", on="itemID")
X_train = X_train.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].max()).rename(columns={"order": "orderMax"}), how="left", on="itemID")

X_test = X_test.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].mean()).rename(columns={"order": "orderMean"}), how="left", on="itemID")
X_test = X_test.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].std()).rename(columns={"order": "orderStd"}), how="left", on="itemID")
X_test = X_test.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].min()).rename(columns={"order": "orderMin"}), how="left", on="itemID")
X_test = X_test.merge(pd.DataFrame(X_train.groupby(["itemID"])["order"].max()).rename(columns={"order": "orderMax"}), how="left", on="itemID")

train_day = X_train[X_train["weekDay"] == X_test["weekDay"].iloc[0]]
y_train_day = train_day.pop('order')
w_train = X_train.pop('simulationPrice')
w_train_day = train_day.pop('simulationPrice')
w_test = X_test.pop('simulationPrice') # qdo for prever colocar salesPrice= simulationPrice
X_test["salesPrice"] = w_test
y_train = X_train.pop('order')
y_test = X_test.pop('order')

print("\nSetting up data for XGBoost ...")
'''XGBoost parameters'''
params = {'tree_method': 'exact', 
          'seed': 1994, 
          'eta': 0.1}
#for tuning parameters
#parameters_for_testing = {
#    'colsample_bytree':[0.4,0.6,0.8],
#    'gamma':[0,0.03,0.1,0.3],
#    'min_child_weight':[1.5,6,10],
#    'learning_rate':[0.1,0.07],
#    'max_depth':[3,5],
#    'n_estimators':[10000],
#    'reg_alpha':[1e-5, 1e-2,  0.75],
#    'reg_lambda':[1e-5, 1e-2, 0.45],
#    'subsample':[0.6,0.95]  
#}
sumPreds = pd.DataFrame(np.zeros(10464))
#del  X_test["salesPrice"], X_test["recommendedRetailPrice"],  X_train["salesPrice"], X_train["recommendedRetailPrice"]
xgb_model = xgb.XGBRegressor(objective="reg:squaredlogerror", base_score=0.7, colsample_bylevel=0.6, colsample_bytree=0.6,
       gamma=0.1, learning_rate=0.01, max_delta_step=0, max_depth=5,
       min_child_weight=6, n_estimators=200, nthread=7, reg_alpha=0.75, reg_lambda=0.45,
       scale_pos_weight=1, seed=42, subsample=0.8)
w = pd.DataFrame(w)
w = np.array(w["recommendedRetailPrice"])
xgb_model.fit(X_train,y_train)
for i in range(0,14):    
    #dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    #dvalid = xgb.DMatrix(X_test, label=y_test, weight=w_test)     #todo 
    if X_test["day"].iloc[0] == 30:
        X_test["day"] = 1
        X_test["month"] = X_test["month"]+1 
    else:
        X_test["day"] = X_test["day"]+1
    X_test["daysToLimiar"] = X_test["daysToLimiar"]+1
    if X_test["weekDay"].iloc[0] == 7:
        X_test["weekNumber"] = X_test["weekNumber"] + 1 
        X_test["weekDay"] = 0
    else:
        X_test["weekDay"] = X_test["weekDay"]+1
    #gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
    #gsearch1.fit(X_train,y_train)
    #print (gsearch1.grid_scores_)
    #print('best params')
    #print (gsearch1.best_params_)
    #print('best score')
    #print (gsearch1.best_score_)
    #dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    #dvalid = xgb.DMatrix(X_test, label=y_test, weight=w_test)
    #results={}
    #
    #bst = xgb.train(params, dtrain=dtrain, num_boost_round=100, evals=[(dtrain, 'dtrain'), (dvalid, 'dvalid')], evals_result=results)
    #
    #'''Prediction'''
    preds = xgb_model.predict(X_test)
    
    X_train["order"] = y_train
    X_test["order"] = preds

    preds[preds < 0 ] = 0
    preds = preds.astype(int)
    y_test = train["order"][train["date"] == pd.to_datetime("2018-06-"+str(17+i))]

    y_test = np.array(y_test)
    preds = np.array(preds)
    score = preds * w
    score[(y_test - preds) < 0] = (y_test[(y_test - preds) < 0] - preds[(y_test - preds) < 0]) * (0.6 * w[(y_test - preds) < 0])
    X_test["order"][X_test["order"] < 0] = 0
    X_test["order"] = X_test["order"].astype(int)
    print(pd.DataFrame(score).describe())
    print(pd.DataFrame(y_test - preds).describe())
    print(sum(score))
    print(sum(y_test*w))
    
    #sumPreds = sumPreds + preds
    X_train = pd.concat([X_train, X_test])

    

    train_day = X_train[X_train["weekDay"] == X_test["weekDay"].iloc[0]]
    y_train_day = train_day.pop('order')
    y_train = X_train.pop('order')
    y_test = X_test.pop('order')
    feature_important = xgb_model.get_booster().get_score(importance_type='gain')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.plot(kind='barh')
    #fsdf
    #plt.show()

#sumPreds.to_csv("out1.csv")
X_train["order"] = y_train
future = train[(train["date"] > pd.to_datetime("2018-06-16")) & (train["date"] <= pd.to_datetime("2018-06-29"))]
future = future.groupby("itemID")["order"].sum()

preds = X_train[((X_train["day"] > 16) & (X_train["month"] == 6)) & ((X_train["day"] <= 29) & (X_train["month"] == 6))]
preds = preds.groupby("itemID")["order"].sum()

#dif = pd.DataFrame(sumPreds - future) 
preds = np.array(preds)
future = np.array(future)
score = preds * w
score[(future - preds) < 0] = (future[(future - preds) < 0] - preds[(future - preds) < 0]) * (0.6 * w[(future - preds) < 0])
print(sum(score))
#
#print(dif.describe())
#
#'''Final Score'''
#score = preds.copy()
#score = dvalid.get_weight() * preds
#score[(dvalid.get_label() - preds) < 0] = 0.6 * dvalid.get_weight()[(dvalid.get_label() - preds) < 0] * (dvalid.get_label()[(dvalid.get_label() - preds) < 0] - preds[(dvalid.get_label() - preds) < 0])
#print('Final Score: '+str(score.sum()))
#
#'''Exact Predictions'''
#equals = preds[preds.astype(int) == dvalid.get_label().astype(int)]
#print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds)))