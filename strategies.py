import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import scipy.stats
import math
import statsmodels
import arch
import warnings

# filter some warnings
warnings.filterwarnings('ignore')

"""
    Class to create random Synthetic Time Series.
    The parameters are preset to meet the requirements, but can be adjusted if needed
    Seed can be specified to get the same Time Series
"""
class SyntheticTimeSeries():
    # Create the Time Series
    def __init__(self, t = 3000, phi = 0.5, d = 0.02, theta = -0.3, mean = 0, variance = 1, p0 = 1000, p1 = 1000, seed = 1, train_test_split = 0.7):
        np.random.seed(seed)
        series = [p0, p1]
        change = p1 - p0
        eps =   np.random.normal(mean, variance, t)
        for i in range(1, t-1):
            change_prev = change
            change = phi * (change_prev - d) + eps[i] + theta * eps[i-1] + d                
            series.append(series[-1] + change)
            
        self.df = pd.DataFrame()
        self.df["Prices"] = pd.Series(data = series)
        self.train_test_split = train_test_split
    
    def get_df_col(self, col):
        return self.df[col]
    
    # Get the prices
    def get_prices(self):
        return self.df["Prices"]
    
    # Get the returns
    def get_returns(self):
        return (self.get_prices().shift(-1) / self.get_prices() - 1).dropna()
        
    # Choose which data to use - train, test or both
    def split_data(self, prices, use_set = "all"):
        if (use_set == "all"):
            return prices
        if (use_set == "train"):
            return prices[: int(len(prices) * self.train_test_split)]
        elif (use_set == "test"):
            return prices[int(len(prices) * self.train_test_split) :]
        
        raise Exception("Invalid split of dataaset")

    # Central plotting function to keep the plots consistent and save code repetition
    def plot(plotDataList, title, xlabel, ylabel, legendLoc = "upper left"):
        for i in range(len(plotDataList)):
            plt.plot(plotDataList[i][1], label=plotDataList[i][0])
            
        plt.title(title)
        plt.legend(loc=legendLoc, fontsize=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    # Computes SMA
    def get_simple_moving_average(prices, period):
        ma = prices.rolling(window=period).mean()
        for i in range(period):
            pd_index = i + prices.first_valid_index()
            ma[pd_index] = prices[pd_index]
        
        return ma

    # Computes EMA
    def get_exponential_moving_average(prices, alpha):
        return prices.ewm(alpha=alpha, adjust=False).mean()

    # ADF test
    def adf_test(data):
        print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(data, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
            
        print (dfoutput)

    # KPSS test
    def kpss_test(data):
        print ('Results of KPSS Test:')
        kpsstest = kpss(data, regression='c', nlags="auto")
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
            
        print (kpss_output) 

"""
    Class that encapsulates the Trend Following Strategy
    Both with SMA and EMA was impelemnted for comparison purposes
    The df object needs to be of type SyntheticTimeSeries in order to reuse the code above
"""
class TrendFollowing():
    def __init__(self, df, cash_start, train_test_split = 0.7):
        if not isinstance(df, SyntheticTimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.df = df
        self.cash_start = cash_start
        self.train_test_split = train_test_split
    
    # Trend Following with simple moving average
    def TF_sma(self, sme_period = 20, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start
        ma = SyntheticTimeSeries.get_simple_moving_average(prices, sme_period)
        strategy_returns = TrendFollowing.position(prices, ma, w, cash)
        return strategy_returns
        
    # Trend Following with exponential moving average
    def TF_ema(self, alpha = 0.5, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start
        ma = SyntheticTimeSeries.get_exponential_moving_average(prices, alpha)
        strategy_returns = TrendFollowing.position(prices, ma, w, cash)
        return strategy_returns
        
    # Decide whether to buy, hold or sell. Update cash and volume appropriately
    def position(prices, ma, w, cash):
        for i in range(len(prices) - 1):
            pd_index = i + prices.first_valid_index()
            x = prices[pd_index]
            if ma[pd_index] == x:
                w[i+1] = w[i]
                cash[i+1] = cash[i]
            elif ma[pd_index] < x: 
                w[i+1] = cash[i]/x  + w[i]
                cash[i+1] = 0  
            else:
                cash[i+1] = w[i]*x + cash[i]
                w[i+1] = 0
                
        return [a*b for a,b in zip(w,prices)]+ cash

"""
    Class that encapsulates the Mean Reversion Strategy
    Implemented both SMA and Bands+RSI
"""
class MeanReversion():
    def __init__(self, df, cash_start, train_test_split = 0.7):
        if not isinstance(df, SyntheticTimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.df = df
        self.cash_start = cash_start
        self.train_test_split = train_test_split
        
    # Mean Reversion with SMA
    def MR_sma(self, period = 20, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        ma = SyntheticTimeSeries.get_simple_moving_average(prices, period)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start   
        
        for i, x in enumerate(prices[:-1], 0):            
            if ma[i] == x:
                w[i+1] = w[i]
                cash[i+1] = cash[i]
            
            if ma[i] > x: 
                w[i+1] = cash[i]/x  + w[i]
                cash[i+1] = 0
                
            if ma[i] < x:
                cash[i+1] = w[i]*x + cash[i]
                w[i+1] = 0

        mr_strategy = [a*b for a,b in zip(w,prices)]+ cash
        return mr_strategy
        
    # Mean Reversion with Bands and RSI
    def MR_bb_rsi(self, bb_period = 20, bb_std = 2, rsi_period = 6, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start
        
        bb = MeanReversion.get_bollinger_bands(prices, bb_period, bb_std)
        rsi = MeanReversion.get_rsi(prices, rsi_period)

        for i in range(len(prices) - 1):
            pd_index = i + prices.first_valid_index()
            x = prices[pd_index]
            if rsi[pd_index] < 10 and x < bb[1][pd_index]:
                w[i+1] = cash[i]/x  + w[i]
                cash[i+1] = 0
            elif rsi[pd_index] > 90 and x > bb[0][pd_index]:
                cash[i+1] = w[i]*x + cash[i]
                w[i+1] = 0
            else:
                w[i+1] = w[i]
                cash[i+1] = cash[i]
            
        mr_strategy = [a*b for a,b in zip(w,prices)]+ cash
        return mr_strategy

    # Compute the Bands
    def get_bollinger_bands(prices, period = 20, num_std = 2):
        ma = SyntheticTimeSeries.get_simple_moving_average(prices, period)
        std = prices.rolling(period).std() 
        upper = ma + num_std * std
        lower = ma - num_std * std
        for i in range(period):
            pd_index = i + prices.first_valid_index()
            upper[pd_index] = prices[pd_index]
            lower[pd_index] = prices[pd_index]
           
        return upper, lower

    # Compute RSI
    def get_rsi(prices, period):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        sma_gain = gain.rolling(period).mean()
        sma_loss = loss.rolling(period).mean()
        rs = sma_gain / sma_loss
        rsi = 100 - (100 / (rs + 1))
        rsi = rsi.fillna(50)
        return rsi

"""
    Class that encapsulates ARIMA+GARCH trading strategy
"""
class ForecastArimaGarch():    
    def __init__(self, df, cash_start, train_test_split = 0.7):
        if not isinstance(df, SyntheticTimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.df = df
        self.cash_start = cash_start
        self.train_test_split = train_test_split
        self.prices = self.df.get_df_col("Prices")
        self.returns = self.df.get_returns()
                
    # Make one ARIMA+GARCH prediction
    def ARIMA_predict(data, p, d, q):
        # fit ARIMA on returns
        arima_model = ARIMA(data, order=(p,d,q))
        model_fit = arima_model.fit()
        arima_residuals = model_fit.resid

        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit(disp=False)

        # Use ARIMA to predict mu
        predicted_mu = model_fit.forecast()
        
        # Use GARCH to predict the residual
        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]

        # Combine both models' output: yt = mu + et
        return predicted_mu + predicted_et

    # Buy, hold or sell given the arima prediciton
    def ARIMA_position(arima_prediction, cash, w, price):
        if arima_prediction == 0:
            next_w = w
            next_cash = cash
        elif arima_prediction > 0: 
            next_w = cash / price  + w
            next_cash = 0  
        else:
            next_cash = w * price + cash
            next_w = 0
            
        return next_cash, next_w
    
    # ARIMA+GARCH only on the test set
    def ARIMA_GARCH_test(self):
        period = int(len(self.returns) * self.train_test_split)
        arima_prediction = np.zeros(np.shape(self.returns))
        w = np.zeros(np.shape(self.returns))
        cash = np.zeros(np.shape(self.returns))
        cash[0] = self.cash_start
            
        auto_arima_model = pm.auto_arima(self.returns[:period], start_p=1, start_q=1, max_p=20, max_q=20, trace = False)        
        p, d, q = auto_arima_model.order
        print(auto_arima_model.summary())
        
        for i in range(len(self.returns) - 1):
            arima_prediction[i] = 0            
            if i >= period:
                X = self.returns[0:i]
                train = X
                arima_prediction[i] = ForecastArimaGarch.ARIMA_predict(train, p, d, q)       
            
            cash[i+1], w[i+1] = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.prices[i])
                
        arima_strategy = [a*b for a,b in zip(w, self.prices)]+ cash
        return arima_strategy, arima_prediction
    
    # ARIMA+GARCH on the whole dataset
    def ARIMA_GARCH_all(self):
        period = int(len(self.returns) * self.train_test_split)
        arima_prediction = np.zeros(np.shape(self.returns))
        w = np.zeros(np.shape(self.returns))
        cash = np.zeros(np.shape(self.returns))
        cash[0] = self.cash_start
        
        auto_arima_model = pm.auto_arima(self.returns[:period], start_p=1, start_q=1, max_p=20, max_q=20, trace = False)
        predictions_in_sample = auto_arima_model.predict_in_sample()
        p, d, q = auto_arima_model.order
        
        for i in range(len(self.returns) - 1):
            if i >= period:
                X = self.returns[0:i]
                train = X
                arima_prediction[i] = ForecastArimaGarch.ARIMA_predict(train, p, d, q)       
            else:
                arima_prediction[i] = predictions_in_sample[i]
            
            cash[i+1], w[i+1] = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.prices[i])
                
        arima_strategy = [a*b for a,b in zip(self.w, self.prices)]+ self.cash   
        return arima_strategy, arima_prediction
    
    # ARIMA+GARCH only on the train set
    def ARIMA_GARCH_train(self):
        period = int(len(self.returns) * self.train_test_split)
        returns = self.returns[:period]
        arima_prediction = np.zeros(np.shape(returns))
        w = np.zeros(np.shape(returns))
        cash = np.zeros(np.shape(returns))
        cash[0] = self.cash_start
        
        auto_arima_model = pm.auto_arima(returns, start_p=1, start_q=1, max_p=20, max_q=20, trace = False)
        arima_prediction = auto_arima_model.predict_in_sample()
        for i in range(len(arima_prediction) - 1):
            cash[i+1], w[i+1] = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.prices[i])
                
        arima_strategy = [a*b for a,b in zip(w, self.prices)]+ cash
        return arima_strategy, arima_prediction
