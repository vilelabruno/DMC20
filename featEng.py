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

train = train.merge(pd.DataFrame(train.groupby(["brand"])["manufacturer"].mean()).rename(columns={"manufacturer": "manufacturerMean"}), how="left", on="brand")
train = train.merge(pd.DataFrame(train.groupby(["brand"])["manufacturer"].std()).rename(columns={"manufacturer": "manufacturerStd"}), how="left", on="brand")
train = train.merge(pd.DataFrame(train.groupby(["brand"])["manufacturer"].min()).rename(columns={"manufacturer": "manufacturerMin"}), how="left", on="brand")
train = train.merge(pd.DataFrame(train.groupby(["brand"])["manufacturer"].max()).rename(columns={"manufacturer": "manufacturerMax"}), how="left", on="brand")

train.to_csv("data/trainNewAux.csv")