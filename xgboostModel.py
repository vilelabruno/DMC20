import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
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
def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()
    return (np.log1p(predt) - np.log1p(y)) / ((predt + 1) + (np.log1p(price)*(y-predt))) # with log is best, maybe apply log in all bot part of equation

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()
    return (((np.log1p(price)*(y-predt)+predt+1)/predt+1)+(np.log1p(price)-1)*np.log1p(predt)-np.log1p(y))/(np.log1p(price)*(y-predt)+predt+1)**2
    #return (((-price*predt+price*y+predt+1)/predt+1)+(price-1)*np.log1p(predt)-np.log1p(y))/(-price*predt+price*y+predt+1)**2

def squared_log(predt: np.ndarray, dtrain: xgb.DMatrix) -> [np.ndarray, np.ndarray]:
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
train["weekPromotion"] = train["weekPromotion"].astype(int)
train["order"][train["order"] == 0] = 0 + 1e-6
train["diffSimRec"] = train["recommendedRetailPrice"] - train["simulationPrice"]
#plt.plot(train["diffSimRec"])
#plt.show()
#
'''Deleting promotion column'''
del train["promotion"], train["weekPromotion"]

#'''Promotion times Price'''
#train["weekPromotion"] = train["weekPromotion"] * train["simulationPrice"]

'''Ordering by weekNumber'''
train.sort_values(by=["weekNumber"])
X_test = train[train["weekNumber"] == 12]
X_train = train[train["weekNumber"] != 12]

'''Target Encoding'''
#train['order_bins'] = pd.cut(train.order, [0, 1000, 2000, 3000, 4000, 5000])
X_train['order_0'] = 0
X_train['order_1'] = 0
X_train['order_2'] = 0
X_train['order_3'] = 0
X_train['order_4'] = 0
X_train['order_5'] = 0

X_train['order_0'][X_train['order'] == 0] = 1
X_train['order_1'][(X_train['order'] > 0) & (X_train['order'] <= 100)] = 1
X_train['order_2'][(X_train['order'] > 100) & (X_train['order'] <= 300)] = 1
X_train['order_3'][(X_train['order'] > 300) & (X_train['order'] <= 500)] = 1
X_train['order_4'][(X_train['order'] > 500) & (X_train['order'] <= 1000)] = 1
X_train['order_5'][X_train['order'] > 1000] = 1

X_test = X_test.merge(X_train.groupby("itemID").agg({ 'order_0': 'max', 'order_1': 'max', 'order_2': 'max', 'order_3': 'max', 'order_4': 'max', 'order_5': 'max' }), how='left', on='itemID')
#X_test = X_test.merge(X_train.groupby("itemID").agg({ 'order_0': 'std', 'order_1': 'std', 'order_2': 'std', 'order_3': 'std', 'order_4': 'std', 'order_5': 'std' }), how='left', on='itemID')
#
#del X_train["order_0"], X_train["order_1"], X_train["order_2"], X_train["order_3"], X_train["order_4"], X_train["order_5"]

'''Popping order and simulationPrice columns'''
y_train = X_train.pop('order')
w_train = X_train.pop('simulationPrice')
y_test = X_test.pop('order')
w_test = X_test.pop('simulationPrice')


print("\nSetting up data for XGBoost ...")
'''XGBoost parameters'''
params = {'tree_method': 'exact', 
          'seed': 1994, 
          'eta': 0.1}

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dvalid = xgb.DMatrix(X_test, label=y_test, weight=w_test)
results={}

bst = xgb.train(params, dtrain=dtrain, num_boost_round=1000, obj=squared_log,
                        feval=rmsle, evals=[(dtrain, 'dtrain'), (dvalid, 'dvalid')], evals_result=results)

'''Prediction'''
preds = bst.predict(dvalid)

'''Final Score'''
score = preds.copy()
score = dvalid.get_weight() * preds
score[(dvalid.get_label() - preds) < 0] = 0.6 * dvalid.get_weight()[(dvalid.get_label() - preds) < 0] * (dvalid.get_label()[(dvalid.get_label() - preds) < 0] - preds[(dvalid.get_label() - preds) < 0])
print('Final Score: '+str(score.sum()))

'''Exact Predictions'''
equals = preds[preds.astype(int) == dvalid.get_label().astype(int)]
print('Exact Predictions: '+str(len(equals))+' of '+str(len(preds)))