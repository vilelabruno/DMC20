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