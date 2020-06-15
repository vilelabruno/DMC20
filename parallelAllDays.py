import pandas as pd
import numpy as np
from datetime import timedelta
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

orders = pd.read_csv("data/orders.csv")
items = pd.read_csv("data/items.csv", sep="|")
infos = pd.read_csv("data/infos.csv", sep="|")

orders["time"] = pd.to_datetime(orders["time"])
orders["date"] = orders["time"].dt.date
orders = orders.groupby(["itemID", "date"]).agg({"order": "sum", "salesPrice": "sum"})
orders.reset_index(inplace=True)

def func(df):
    for j in range(1, 10464):
        for i in range(0, 182):
            if len(df[(df["date"] == (pd.to_datetime("2018-01-01") + timedelta(days=(i)))) & (df["itemID"] == j)]) == 0:
                df.loc[len(df)] = [j, (pd.to_datetime("2018-01-01") + timedelta(days=(i))), 0, 0]
        print(len(df))

cores = mp.cpu_count()

cores = mp.cpu_count()

df_split = np.array_split(orders, cores, axis=0)

# create the multiprocessing pool
pool = Pool(cores)

# process the DataFrame by mapping function to each df across the pool
df_out = np.vstack(pool.map(func, df_split))

# close down the pool and join
pool.close()
pool.join()
pool.clear()
    
orderInfos = df_out.merge(infos, how='left', on='itemID')

fullData = orderInfos.merge(items, how='left', on='itemID')

fullData.to_csv('data/trainAllDays.csv', index=False)