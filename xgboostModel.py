import numpy as np
import xgboost as xgb

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()
    
    predt[(predt-y) < 0] = (1/(y+1))-(8*(y-predt)*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    predt[(predt-y) > 0] = (1/(y+1))-(2*(y-predt)*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    
    return predt

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    price = dtrain.get_weight()

    predt[(predt-y) < 0] = (-1/np.power(y+1, 2))-(8*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))
    predt[(predt-y) > 0] = (-1/np.power(y+1, 2))-(2*np.power(np.log1p(predt),2)*np.power(np.log1p(price), 2))

    return predt

def weighted_squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess