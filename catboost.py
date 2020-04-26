import pandas as pd
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GroupKFold
print('Setting random seed...')
seed = 1234
np.random.seed(seed)

print('Reading csv...')
train = pd.read_csv('data/trainNew.csv')
#test = pd.read_csv('data/test.csv', sep='|')

class DMC20_Metric(object):
    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0
        for i in range(0, len(target)):
            x = target[i] -approx[i] 
            w = 1.0 if weight is None else weight[i]
            if x <= 0:
                error_sum += approx[i] #(5.8*(10**(-19)) - 10*x - 15*(x**2)) 
            else:
                error_sum += .6 * x #(5.8*(10**(-19)) - 10*x - 15*(x**2)) 
            weight_sum += w
            
        return error_sum, weight_sum
    
    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        return error/(weight + 1e-38)

#del useless columns
del train["Unnamed: 0"], train["salesPrice|mean"]
del train["salesPrice|std"], train["salesPrice|sum"]

#trat promotion col
train["promotion"][train["promotion"].isnull()] = 0
train["promotion"][train["promotion"] != 0] = 1

train.sort_values(by=["weekNumber"])
X_test = train[train["weekNumber"] == 12]
X_train = train[train["weekNumber"] != 12]
y_train = X_train.pop('order|sum')
w_train = X_train.pop('simulationPrice')

y_test = X_test.pop('order|sum')
w_test = X_test.pop('simulationPrice')

model = cat.CatBoostRegressor(eval_metric=DMC20_Metric())

print('Instantiating catboost model...')
print('Training catboost model...')
model.fit(X_train, y=y_train, eval_set=[(X_test, y_test)])
resp = model.predict(X_test)
