import pandas as pd

df = pd.read_csv("data/orders.csv")

df["time"]= pd.to_datetime(df["time"])

df["weekNumber"] = df["time"].dt.week
df = pd.DataFrame(df.groupby(["itemID", "weekNumber"])["salesPrice"].mean())
df.reset_index(inplace=True)
for i in range(0,13):
    for j in range(0,10464):
        if len(df[(df["weekNumber"] == i) & (df["itemID"] == j)]) == 0:
            try:
                df.loc[len(df)] = [j, i, df["salesPrice"][df["itemID"] == j].iloc[0]]
            except:
                pass
            print(len(df))
df.to_csv("salesPrice.csv")