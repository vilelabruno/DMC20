import pandas as pd
from datetime import timedelta

orders = pd.read_csv("data/orders.csv")
items = pd.read_csv("data/items.csv", sep="|")
infos = pd.read_csv("data/infos.csv", sep="|")

orders["time"] = pd.to_datetime(orders["time"])
orders["date"] = orders["time"].dt.date
orders = orders.groupby(["itemID", "date"]).agg({"order": "sum", "salesPrice": "sum"})
orders.reset_index(inplace=True)
aux = pd.DataFrame(columns=["a", "b", "c", "d"])
for j in range(7848, 9100):
    for i in range(0, 182):
        if len(orders[(orders["date"] == (pd.to_datetime("2018-01-01") + timedelta(days=(i)))) & (orders["itemID"] == j)]) == 0:
            aux.loc[len(aux)] = [j, (pd.to_datetime("2018-01-01") + timedelta(days=(i))), 0, 0]
    print(len(aux))
    
#orderInfos = orders.merge(infos, how='left', on='itemID')
#
#fullData = orderInfos.merge(items, how='left', on='itemID')

aux.to_csv('data/6.csv', index=False)

for i in range(0,9):
    if i == 6:
        continue
    aux = pd.read_csv("data/"+str(i)+".csv")
    aux["a"] = aux["a"] + 1
    aux = aux.rename(columns={"a": "itemID", "b": "date", "c": "order", "d": "salesPrice"})
    orders = pd.concat([orders, aux])
