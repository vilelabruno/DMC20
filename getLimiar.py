import pandas as pd
df = pd.read_csv("data/orders.csv")
i = 2
if i == 1:
    df.reset_index(inplace=True)
    dfAux = df.groupby("itemID")["index"].max()
    dfAux = pd.DataFrame(dfAux)
    dfAux.reset_index(inplace=True)
    dfAux = dfAux.merge(df, how="left", on="index")
else:
    df.reset_index(inplace=True)
    dfAux = df.groupby("itemID")["index"].min()
    dfAux = pd.DataFrame(dfAux)
    dfAux.reset_index(inplace=True)
    dfAux = dfAux.merge(df, how="left", on="index")

dfAux["itemID"] = dfAux["itemID_x"]
del dfAux["itemID_x"], dfAux["index"], dfAux["itemID_y"], dfAux["transactID"], dfAux["order"],  dfAux["time"]
dfAux = dfAux.rename(columns={"salesPrice": "salesLimiar"+str(i)})
dfAux.to_csv("limiar"+str(i)+".csv")