import pandas as pd
df = pd.read_csv("data/orders.csv")

df.reset_index(inplace=True)
dfAux = df.groupby("itemID")["index"].max()
dfAux = pd.DataFrame(dfAux)
dfAux.reset_index(inplace=True)
dfAux = dfAux.merge(df, how="left", on="index")