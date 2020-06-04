import pandas as pd
from datetime import timedelta

orders = pd.read_csv("data/orders.csv")
items = pd.read_csv("data/items.csv", sep="|")
infos = pd.read_csv("data/infos.csv", sep="|")

orders["time"] = pd.to_datetime(orders["time"])

for j in range(1, 10464):
    if len(orders[((orders["time"] >= (pd.to_datetime("2018-01-01") + timedelta(days=(i)))) & (orders["time"] < (pd.to_datetime("2018-01-01") + timedelta(days=(i+1))))) & (orders["itemID"] == j)]) == 0:
        orders.loc[len(orders)] = [(pd.to_datetime("2018-01-01") + timedelta(days=(i))), -1, j, 0, 0]
        #print(len(orders))

orderInfos = orders.merge(infos, how='left', on='itemID')

fullData = orderInfos.merge(items, how='left', on='itemID')

fullData.to_csv('data/trainAllDays.csv', index=False)