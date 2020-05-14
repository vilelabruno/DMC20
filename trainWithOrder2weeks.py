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

orders["dayDate"] = orders["time"].dt.strftime("%Y-%m-%d")

orders = orders.merge(aux, how="left", left_on=["itemID", "dayDate"], right_on=["index", 0])
orders = orders.merge(aux, how="left", left_on=["itemID", "dayDate"], right_on=["index", 1])
orders = orders.merge(aux, how="left", left_on=["itemID", "dayDate"], right_on=["index", 2])

#%%
gp = orders.groupby(["itemID", "dayDate"]).agg({"order": "sum", "weekPromotion": "sum"})
#gp = orders.groupby(["weekNumber"]).agg({"order": sum})#this lines contains the plot for orders                                                  
#gp.reset_index(inplace=True)                           #this lines contains the plot for orders                      
#gp = gp.sort_values(by=["weekNumber"])                 #this lines contains the plot for orders                              
#plt.plot(gp["weekNumber"], gp["order"], '--')          #this lines contains the plot for orders                                      
#plt.show()                                             #this lines contains the plot for orders  
gp.reset_index(inplace=True)
gp.to_csv("data/train2weeksWithOrder.csv", index=False)