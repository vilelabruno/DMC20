import numpy as np
import xgboost as xgb
import pandas as pd
print('Setting random seed...')
seed = 1234
np.random.seed(seed)

print('Reading csv...')
train = pd.read_csv('data/trainNew.csv')
#test = pd.read_csv('data/test.csv', sep='|')

'''Definition of Condition Weighted Squared Log Error (CWSLE)'''
def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the Gradient CWSLE.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()
    aux = predt.copy()
    aux = (1/(y+1))-(2*(y-predt)*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    aux[(predt-y) < 0] = (1/(y+1))-(8*(y-predt)*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    print('grad: ' + aux)
    
    return aux

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the Hessian for CWSLE.'''
    y = dtrain.get_label()
    pricedtest' is not defined = dtrain.get_weight()

    aux = predt.copy()

    aux = (-1/np.power(y+1, 2))-(2*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    aux[(predt-y) < 0] = (-1/np.power(y+1, 2))-(8*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    print('hess: ' + aux)

    return aux

def conditional_weighted_squared_log(predt: np.ndarray, dtrain: xgb.DMatrix):
    '''Conditional Weighted Squared Log Error (CWSLE) objective. A version of Squared Log Error (SLE) with
    condition and weights used as objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

'''Treatment of promotion column'''

train["promotion"] = 1
train["promotion"][train["promotion"].isnull()] = 0

'''Ordering by weekNumber'''
train.sort_values(by=["weekNumber"])
X_test = train[train["weekNumber"] == 12]
X_train = train[train["weekNumber"] != 12]

'''Popping order and simulationPrice columns'''
y_train = X_train.pop('order|sum')
w_train = X_train.pop('simulationPrice')
y_test = X_test.pop('order|sum')
w_test = X_test.pop('simulationPrice')


print("\nSetting up data for XGBoost ...")
'''XGBoost parameters'''


dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dvalid = xgb.DMatrix(X_test, label=y_test, weight=w_test)

bst = xgb.train({'tree_method': 'hist', 'seed': 1994}, dtrain=dtrain, num_boost_round=10, obj=conditional_weighted_squared_log)

'''Prediction'''
preds = bst.predict(dvalid)