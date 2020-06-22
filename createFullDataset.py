import pandas as pd

orders = pd.read_csv("data/train2weeksWithOrder.csv", sep=",")
items = pd.read_csv("data/items.csv", sep="|")
infos = pd.read_csv("data/infos.csv", sep="|")

for i in range(0,13):
    for j in range(1, 10464):
        if len(orders[(orders["weekNumber"] == i) & (orders["itemID"] == j)]) == 0:
            orders.loc[len(orders)] = [j, i, 0, 0]
            print(len(orders))

orderInfos = orders.merge(infos, how='left', on='itemID')

fullData = orderInfos.merge(items, how='left', on='itemID')

fullData.to_csv('data/train2weeks.csv', index=False)