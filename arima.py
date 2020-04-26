from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

series = pd.read_csv("data/timeseries1week.csv") # change for timeseries 
model = ARIMA(series["order"], order=(2,0,1))
model_fit = model.fit(disp=0)
#print(model_fit.summary())
print(model_fit.predict(13,14))