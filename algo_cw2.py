import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


def timeSeriesGeneration(t = 3000, phi = 0.5, d = 0.02, theta = -0.3, mean = 0, variance = 1, p0 = 1000, p1 = 1000):
    series = [p0, p1]
    change = p1 - p0
    eps =   np.random.normal(mean, variance, t)

    for i in range(1, t-1):
        change_prev = change
        change = phi * (change_prev - d) + eps[i] + theta * eps[i-1] + d
        series.append(series[-1] + change)

    return pd.Series(data=series)
    
"""
    TREND FOLLOWING 
"""
def strategy_trend_following(S, dt = 0.01):
    
    time_window = 3*int(1/dt)
    cumsum = [0]

    ma = np.zeros(np.shape(S))

    w = np.zeros(np.shape(S))
    cash = np.zeros(np.shape(S))

    cash[0] = 100

    for i, x in enumerate(S[:-1], 0):
        cumsum.append(cumsum[i] + x)
        ma[i] = x
        if i>=time_window:
            moving_ave = (cumsum[i] - cumsum[i-time_window])/(time_window)
            ma[i] = moving_ave
        
        if ma[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
        
        if ma[i] < x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
            
        if ma[i] > x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0

    ma[i+1] = S[len(S)-1]
            
    tf_strategy = [a*b for a,b in zip(w,S)]+ cash
    return tf_strategy


def strategy_mean_reversion(S, dt = 0.01):
    #TRADING: MEAN REVERSION 
    time_window = 3*int(1/dt)
    cumsum = [0]

    ma = np.zeros(np.shape(S))

    w = np.zeros(np.shape(S))
    cash = np.zeros(np.shape(S))

    cash[0] = 100

    for i, x in enumerate(S[:-1], 0):
        cumsum.append(cumsum[i] + x)
        ma[i] = x
        if i>=time_window:
            moving_ave = (cumsum[i] - cumsum[i-time_window])/(time_window)
            ma[i] = moving_ave
        
        if ma[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
        
        if ma[i] > x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
            
        if ma[i] < x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0

    ma[i+1] = S[len(S)-1]

    mr_strategy = [a*b for a,b in zip(w,S)]+ cash
    return mr_strategy

def strategy_AR(S, dt = 0.01):    
    time_window = 3*int(1/dt)
    cumsum = [0]

    ar_prediction = np.zeros(np.shape(S))
    
    ma = np.zeros(np.shape(S))

    w = np.zeros(np.shape(S))
    cash = np.zeros(np.shape(S))

    cash[0] = 100

    for i, x in enumerate(S[:-1], 0):
        cumsum.append(cumsum[i] + x)
        ar_prediction[i] = x
        if i>=time_window:
            X = S[0:i]
            train = X
            # train autoregression
            model = AR(train, 1)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(train), end=len(train), dynamic=False)    
            ar_prediction[i] = predictions
        
        if ar_prediction[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
        
        if ar_prediction[i] > x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
            
        if ar_prediction[i] < x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0

    ma[i+1] = S[len(S)-1]

    ar_strategy = [a*b for a,b in zip(w,S)]+ cash
    return ar_strategy

def strategy_ARIMA(S, dt = 0.01):    
    time_window = 3*int(1/dt)
    time_window = 2250
    cumsum = [0]

    arima_prediction = np.zeros(np.shape(S))
    
    ma = np.zeros(np.shape(S))

    w = np.zeros(np.shape(S))
    cash = np.zeros(np.shape(S))

    cash[0] = 100

    for i, x in enumerate(S[:-1], 0):
        cumsum.append(cumsum[i] + x)
        arima_prediction[i] = x
        if i >= time_window:
            X = S[0:i]
            train = X
            
            # train ARIMA
            model = ARIMA(train, order=(1,1,1))
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(train), end=len(train), dynamic=False)
            
            #X = col.values
            #model_fit = pm.auto_arima(train, start_p=1, start_q=1, max_p=20, max_q=20, seasonal=False, trace=False)  
            #predictions = model_fit.predict()
            #self.computed_df[newCol] = pd.Series(data = model_fit.predict_in_sample(), index = col.index) 
            
            #model = AR(train, 1)
            #model_fit = model.fit()
            #predictions = model_fit.predict(start=len(train), end=len(train), dynamic=False)    
            arima_prediction[i] = predictions
        
        if arima_prediction[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
        
        if arima_prediction[i] > x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
            
        if arima_prediction[i] < x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0

    ma[i+1] = S[len(S)-1]

    arima_strategy = [a*b for a,b in zip(w,S)]+ cash
    return arima_strategy





cash0 = 100
x = np.linspace(0, 1, 3000)
df = timeSeriesGeneration()
train, test = train_test_split(df, test_size=0.3)

trend_following = strategy_trend_following(df)
mean_reversion = strategy_mean_reversion(df)
arima = strategy_ARIMA(df)

plt.plot(cash0/df[0]*df)
plt.plot(trend_following)
plt.plot(mean_reversion)
plt.plot(arima)
plt.show()
