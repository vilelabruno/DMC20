import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
#items = pd.read_csv("data/items.csv")
infos = pd.read_csv("data/infos.csv", sep="|")
orders = pd.read_csv("data/orders.csv", sep="|")

orders["time"] = pd.to_datetime(orders["time"])
orders["weekNumber"] = -1
orders["weekPromotion"] = 0
aux = infos.promotion.str.split(",", expand=True).reset_index() 
aux["index"] = aux.index + 1  
orders = orders.merge(aux, how="left", left_on="itemID", right_on="index")

orders[0] = pd.to_datetime(orders[0])
orders[1] = pd.to_datetime(orders[1])
orders[2] = pd.to_datetime(orders[2])
for i in range(0,13):
    orders["weekNumber"][(orders["time"] > (pd.to_datetime("2018-01-01") + timedelta(days=(14*i)))) & (orders["time"] < (pd.to_datetime("2018-01-01") + timedelta(days=(14*(i+1)))))] = i
    orders["weekPromotion"][((orders[0] > (pd.to_datetime("2018-01-01") + timedelta(days=(14*i)))) & (orders[0] < (pd.to_datetime("2018-01-01") + timedelta(days=(14*(i+1)))))) & (orders["weekNumber"] == i)] = 1
    orders["weekPromotion"][((orders[1] > (pd.to_datetime("2018-01-01") + timedelta(days=(14*i)))) & (orders[1] < (pd.to_datetime("2018-01-01") + timedelta(days=(14*(i+1)))))) & (orders["weekNumber"] == i)] = 1
    orders["weekPromotion"][((orders[2] > (pd.to_datetime("2018-01-01") + timedelta(days=(14*i)))) & (orders[2] < (pd.to_datetime("2018-01-01") + timedelta(days=(14*(i+1)))))) & (orders["weekNumber"] == i)] = 1

gp = orders.groupby(["itemID", "weekNumber"]).agg({"order": "sum", "weekPromotion": "sum"})
gp["weekPromotion"] =  gp["weekPromotion"]/gp["weekPromotion"]
gp.fillna(0, inplace=True)
#gp = orders.groupby(["weekNumber"]).agg({"order": sum})#this lines contains the plot for orders                                                  
#gp.reset_index(inplace=True)                           #this lines contains the plot for orders                      
#gp = gp.sort_values(by=["weekNumber"])                 #this lines contains the plot for orders                              
#plt.plot(gp["weekNumber"], gp["order"], '--')          #this lines contains the plot for orders                                      
#plt.show()                                             #this lines contains the plot for orders  
gp.reset_index(inplace=True)
gp.to_csv("data/train2weeksWithOrder.csv", index=False)