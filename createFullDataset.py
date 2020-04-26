import pandas as pd

orders = pd.read_csv("data/train2weeksWithOrder.csv", sep=",")
items = pd.read_csv("data/items.csv", sep="|")
infos = pd.read_csv("data/infos.csv", sep="|")

orderInfos = orders.merge(infos, how='left', on='itemID')

fullData = orderInfos.merge(items, how='left', on='itemID')

fullData.to_csv('data/trainNew.csv', index=False)