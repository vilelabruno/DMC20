import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
#items = pd.read_csv("data/items.csv")
#infos = pd.read_csv("data/infos.csv")
orders = pd.read_csv("data/orders.csv", sep="|")

orders["time"] = pd.to_datetime(orders["time"])
orders["weekNumber"] = -1
for i in range(0,13):
    orders["weekNumber"][(orders["time"] > (pd.to_datetime("2018-01-01") + timedelta(days=(14*i)))) & (orders["time"] < (pd.to_datetime("2018-01-01") + timedelta(days=(14*(i+1)))))] = i


gp = orders.groupby(["itemID", "weekNumber"]).agg({"order": "sum", "salesPrice": ["sum", "mean", "std"]})
#gp = orders.groupby(["weekNumber"]).agg({"order": sum})#this lines contains the plot for orders                                                  
#gp.reset_index(inplace=True)                           #this lines contains the plot for orders                      
#gp = gp.sort_values(by=["weekNumber"])                 #this lines contains the plot for orders                              
#plt.plot(gp["weekNumber"], gp["order"], '--')          #this lines contains the plot for orders                                      
#plt.show()                                             #this lines contains the plot for orders  
gp.reset_index(inplace=True)
gp.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in gp.columns]
gp.to_csv("data/train2weeksWithOrder.csv", index=False)