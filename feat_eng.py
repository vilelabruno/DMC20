import numpy as np
import pandas as pd
import featuretools as ft

print("Importing Data..."+'\n')
train = pd.read_csv("data/trainAllDays.csv")
train = train[train["itemID"] != 10464]
train["date"] = pd.to_datetime(train["date"])
train.sort_values(by=["date"])
train["weekDay"] = train["date"].dt.day_name()   
#train = pd.get_dummies(train, columns=["weekDay"]) 

items = pd.DataFrame()
items = train[["brand","manufacturer","customerRating","category1","category2","category3","recommendedRetailPrice"]]
items["itemID"] = train["itemID"]

orders = pd.DataFrame()
orders = train[["time","transactID","itemID","order","salesPrice"]]

del train["brand"]
del train["manufacturer"]
del train["customerRating"]
del train["category1"]
del train["category2"]
del train["category3"]
del train["recommendedRetailPrice"]
train.reset_index(inplace=True)

es = ft.EntitySet(id="data")

es = es.entity_from_dataframe(entity_id="orders",
                              dataframe=train,
                              index="index",
                              time_index="date",
                              variable_types={"itemID": ft.variable_types.Categorical})

es