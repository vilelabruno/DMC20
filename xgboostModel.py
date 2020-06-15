import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
print('Setting random seed...')
seed = 1234
np.random.seed(seed)

print('Reading csv...')
train = pd.read_csv('data/trainNew.csv')
#test = pd.read_csv('data/test.csv', sep='|')

#'''Definition of Condition Weighted Squared Log Error (CWSLE)'''
#def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#    '''Compute the Gradient CWSLE.'''
#    y = dtrain.get_label()
#    price = dtrain.get_weight()
#    aux = predt.copy()
#    aux = (1/(y+1))-(2*(y-predt)*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
#    aux[(predt-y) < 0] = (1/(y+1))-(8*(y-predt)*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
#    
#    return aux
#
#def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#    '''Compute the Hessian for CWSLE.'''
#    y = dtrain.get_label()
#    price = dtrain.get_weight()
#    print(price)
#    aux = predt.copy()
#    aux = (-1/np.power(y+1, 2))-(2*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
#    aux[(predt-y) < 0] = (-1/np.power(y+1, 2))-(8*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
#
#    return aux
#
#def conditional_weighted_squared_log(predt: np.ndarray, dtrain: xgb.DMatrix) -> [np.ndarray, np.ndarray]:
#    '''Conditional Weighted Squared Log Error (CWSLE) objective. A version of Squared Log Error (SLE) with
#    condition and weights used as objective function.
#    '''
#    predt[predt < -1] = -1 + 1e-6
#    grad = gradient(predt, dtrain)
#    hess = hessian(predt, dtrain)
#    return grad, hess
#def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#    '''Compute the gradient squared log error.'''
#    y = dtrain.get_label()
#    price = dtrain.get_weight()
#    return (np.log1p(predt) - np.log1p(y)) / ((predt + 1) + (np.log1p(price)*(y-predt))) # with log is best, maybe apply log in all bot part of equation
#
#def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
#    '''Compute the hessian for squared log error.'''
#    y = dtrain.get_label()
#    price = dtrain.get_weight()
#    return (((np.log1p(price)*(y-predt)+predt+1)/predt+1)+(np.log1p(price)-1)*np.log1p(predt)-np.log1p(y))/(np.log1p(price)*(y-predt)+predt+1)**2
#    #return (((-price*predt+price*y+predt+1)/predt+1)+(price-1)*np.log1p(predt)-np.log1p(y))/(-price*predt+price*y+predt+1)**2
#
#def squared_log(predt: np.ndarray, dtrain: xgb.DMatrix) -> [np.ndarray, np.ndarray]:
#    '''Squared Log Error objective. A simplified version for RMSLE used as
#    objective function.
#    '''
#    predt[predt < -1] = -1 + 1e-6
#    grad = gradient(predt, dtrain)
#    hess = hessian(predt, dtrain)
#    return grad, hess
def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))

def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> (np.ndarray, np.ndarray):
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> [str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / (len(y))))

def rmcwsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> [str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power((np.log1p(predt) - np.log1p(y)), 2)
    return 'RMCWSLE', float(np.sqrt(np.sum(elements) / ((predt + 1)+(np.log1p(price)*(y - predt)))))

'''Feature eng'''
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
train["weekDay"] = train["date"].dt.weekday

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
sumPreds = np.zeros(10464)

xgb_model = xgb.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.07, max_delta_step=0, max_depth=3,
       min_child_weight=1.5, n_estimators=100, nthread=-1, reg_alpha=0.75, reg_lambda=0.45,
       scale_pos_weight=1, seed=42, subsample=0.6)
for i in range(0,1):    
    #dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    #dvalid = xgb.DMatrix(X_test, label=y_test, weight=w_test)       

    xgb_model.fit(X_train,y_train)       
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
    score = preds * X_test["recommendedRetailPrice"]
    score[(y_test - preds) < 0] = (y_test[(y_test - preds) < 0] - preds[(y_test - preds) < 0]) * (0.6 * X_test["recommendedRetailPrice"][(y_test - preds) < 0])
    X_test["order"][X_test["order"] < 0] = 0
    X_test["order"] = X_test["order"].astype(int)
    print(pd.DataFrame(score).describe())
    print(pd.DataFrame(y_test - preds).describe())
    
    sumPreds = sumPreds + preds
    X_train = pd.concat([X_train, X_test])
    X_test["day"] = X_test["day"]+1
    y_train = X_train.pop('order')
    y_test = X_test.pop('order')

future = train[train["date"] > pd.to_datetime("2018-06-17")]
future = future.groupby("itemID").agg({"order": "sum"})
dif = pd.DataFrame(sumPreds - future["order"]) 
print(dif.describe())
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