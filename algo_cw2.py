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


class SyntheticTimeSeries():
    def __init__(self, t = 3000, phi = 0.5, d = 0.02, theta = -0.3, mean = 0, variance = 1, p0 = 1000, p1 = 1000, seed = 1, train_test_split = 0.7):
        np.random.seed(seed)
        series = [p0, p1]
        change = p1 - p0
        eps =   np.random.normal(mean, variance, t)
        eps2 =   np.random.normal(mean, 4, t)
        eps3 =   np.random.normal(mean, 0.4, t)
        for i in range(1, t-1):
            change_prev = change
            if (i < 1000):
                change = phi * (change_prev - d) + eps[i] + theta * eps[i-1] + d
            elif (i > 1000 and i < 1500):
                change = phi * (change_prev - d) + eps2[i] + theta * eps2[i-1] + d
            elif (i > 1500 and i < 2000):
                change = phi * (change_prev - d) + eps3[i] + theta * eps3[i-1] + d
            elif (i > 2000 and i < 2200):
                change = phi * (change_prev - d) + eps[i] + theta * eps[i-1] + d
            else:
                change = phi * (change_prev - d) + eps3[i] + theta * eps3[i-1] + d
                
            series.append(series[-1] + change)
            
        self.df = pd.DataFrame()
        self.df["Prices"] = pd.Series(data = series)
        self.train_test_split = train_test_split
    
    def get_df_col(self, col):
        return self.df[col]
    
    def get_prices(self):
        return self.df["Prices"]
    
    def get_returns(self):
        if ("Returns" not in self.df):
            self.df["Returns"] = (self.get_df_col("Prices").shift(-1) / self.get_df_col("Prices") - 1).dropna()
            
        return self.get_df_col("Returns")
    
    def split_data(self, prices, use_set = "all"):
        if (use_set == "all"):
            return prices
        if (use_set == "train"):
            return prices[: len(prices) * self.train_test_split]
        elif (use_set == "test"):
            return prices[len(prices) * self.train_test_split :]
        
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
    
    def get_simple_moving_average(prices, period):
        ma = prices.rolling(window=period).mean()
        for i in range(period):
            ma[i] = prices[i]
        
        return ma

    def get_exponential_moving_average(prices, alpha):
        return prices.ewm(alpha=alpha, adjust=False).mean()

    # ADF test
    def adf_test(prices):
        print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(prices, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
            
        print (dfoutput)

    # KPSS test
    def kpss_test(prices):
        print ('Results of KPSS Test:')
        kpsstest = kpss(prices, regression='c', nlags="auto")
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
            
        print (kpss_output) 



class TrendFollowing():
    def __init__(self, df, cash_start, train_test_split = 0.7):
        if not isinstance(df, SyntheticTimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.df = df
        self.cash_start = cash_start
        self.train_test_split = train_test_split
    
    def TF_sma(self, sme_period = 20, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start
        ma = SyntheticTimeSeries.get_simple_moving_average(prices, sme_period)
        strategy_returns = TrendFollowing.position(prices, ma, w, cash)
        return strategy_returns
        
    def TF_ema(self, alpha = 0.5, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start
        ma = SyntheticTimeSeries.get_exponential_moving_average(prices, alpha)
        strategy_returns = TrendFollowing.position(prices, ma, w, cash)
        return strategy_returns
        
    def position(prices, ma, w, cash):
        for i, x in enumerate(prices[:-1], 0):    
            if ma[i] == x:
                w[i+1] = w[i]
                cash[i+1] = cash[i]
            elif ma[i] < x: 
                w[i+1] = cash[i]/x  + w[i]
                cash[i+1] = 0  
            else:
                cash[i+1] = w[i]*x + cash[i]
                w[i+1] = 0
                
        return [a*b for a,b in zip(w,prices)]+ cash


class MeanReversion():
    def __init__(self, df, cash_start, train_test_split = 0.7):
        if not isinstance(df, SyntheticTimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.df = df
        self.cash_start = cash_start
        self.train_test_split = train_test_split
        
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
        

    def MR_bb_rsi(self, bb_period = 20, bb_std = 2, rsi_period = 6, use_set = "all"):
        prices = self.df.split_data(self.df.get_prices(), use_set = use_set)
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        cash[0] = self.cash_start
        
        bb = MeanReversion.get_bollinger_bands(prices, bb_period, bb_std)
        rsi = MeanReversion.get_rsi(prices, rsi_period)

        for i, x in enumerate(prices[:-1], 0):
            if rsi[i] < 10 and x < bb[1][i]:
                w[i+1] = cash[i]/x  + w[i]
                cash[i+1] = 0
            elif rsi[i] > 90 and x > bb[0][i]:
                cash[i+1] = w[i]*x + cash[i]
                w[i+1] = 0
            else:
                w[i+1] = w[i]
                cash[i+1] = cash[i]
            
        mr_strategy = [a*b for a,b in zip(w,prices)]+ cash
        return mr_strategy

    def get_bollinger_bands(prices, period = 20, num_std = 2):
        ma = SyntheticTimeSeries.get_simple_moving_average(prices, period)
        std = prices.rolling(period).std() 
        upper = ma + num_std * std
        lower = ma - num_std * std
        for i in range(period):
            upper[i] = prices[i]
            lower[i] = prices[i]
           
        return upper, lower

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


class ForecastArimaGarch():
    def __init__(self, df, cash_start, train_test_split = 0.7):
        if not isinstance(df, SyntheticTimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.df = df
        self.cash_start = cash_start
        self.train_test_split = train_test_split
            
    def ARIMA_GARCH(self):
        prices = self.df.get_df_col("Prices")
        returns = self.df.get_returns()
        period = int(len(returns) * self.train_test_split)
        cumsum = [0]
        arima_prediction = np.zeros(np.shape(returns))
        w = np.zeros(np.shape(returns))
        cash = np.zeros(np.shape(returns))
        cash[0] = self.cash_start
        
        arima_prediction2 = np.zeros(np.shape(returns))
        w2 = np.zeros(np.shape(returns))
        cash2 = np.zeros(np.shape(returns))
        cash2[0] = self.cash_start
    
        auto_arima_model = pm.auto_arima(returns[:period])
        p, d, q = auto_arima_model.order
        
        for i in range(len(returns) - 1):
            x = returns[i]
            cumsum.append(cumsum[i] + x)
            arima_prediction[i] = 0
            arima_prediction2[i] = 0
            
            if i >= period:
                X = returns[0:i]
                train = X
            
                # fit ARIMA on returns
                arima_model = ARIMA(train, order=(p,d,q))
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
                arima_prediction[i] = predicted_mu + predicted_et
                arima_prediction2[i] = predicted_mu

            if arima_prediction[i] == 0:
                w[i+1] = w[i]
                cash[i+1] = cash[i]
            elif arima_prediction[i] > 0: 
                w[i+1] = cash[i]/prices[i]  + w[i]
                cash[i+1] = 0  
            else:
                cash[i+1] = w[i]*prices[i] + cash[i]
                w[i+1] = 0
                
            if arima_prediction[i] == 0:
                w2[i+1] = w2[i]
                cash2[i+1] = cash2[i]
            elif arima_prediction[i] > 0: 
                w2[i+1] = cash2[i]/prices[i]  + w2[i]
                cash2[i+1] = 0  
            else:
                cash2[i+1] = w2[i]*prices[i] + cash2[i]
                w2[i+1] = 0
                
        arima_strategy = [a*b for a,b in zip(w,prices)]+ cash        
        arima_strategy2 = [a*b for a,b in zip(w2,prices)]+ cash2
                
        return arima_strategy, arima_strategy2


cash_start = 10000
data = SyntheticTimeSeries()

# Diagnostics
#model = ARIMA(strategy_return(df), order=(1,0,1))
#model_fit = model.fit()
#model_fit.plot_diagnostics()
#plt.show()

# Stationarity
#adf_test((df.shift(-1) / df - 1).dropna())
#kpss_test((df.shift(-1) / df - 1).dropna())

# BUY AND HOLD
prices = data.get_prices()
buy_hold = cash_start / prices[0] * prices

# TREND FOLLOWING
#tf = TrendFollowing(data, cash_start)
#tf_sma = tf.TF_sma(10)
#tf_ema = tf.TF_ema(0.5)

#SyntheticTimeSeries.plot([("Buy and Hold", buy_hold), ("Simple MA (10 period)", tf_sma), ("Exponential MA (0.5 alpha)", tf_ema)], 
#                         "Trend Following", "Days", "Cash")

SyntheticTimeSeries.plot([("Time-series", prices), ("MA (20)", SyntheticTimeSeries.get_simple_moving_average(prices, 20))], 
                         "Moving Average", "Days", "Price")

# MEAN REVERSION
#mr = MeanReversion(data, cash_start)
#mr_sma = mr.MR_sma(20)
#mr_bb_rsi = mr.MR_bb_rsi(20, 2, 6)

#SyntheticTimeSeries.plot([("Buy and Hold", buy_hold), ("Simple MA (20 period)", mr_sma), ("BB+RSI (20p, 2std, 6p)", mr_bb_rsi)], 
#                         "Mean Reversion", "Days", "Cash")

# Bands
#plt.plot(df)
#plt.fill_between([i for i in range(len(bands[0]))], bands[0], bands[1], label = 'Bollinger Bands', color='lightgrey')
#plt.show()

# RSI
#plt.plot(rsi)
#plt.fill_between([i for i in range(len(rsi))], [90 for i in range(len(rsi))], [10 for i in range(len(rsi))], label = 'Bollinger Bands', color='lightgrey')
#plt.show()

# ARIMA
arima_garch_model = ForecastArimaGarch(data, cash_start)
arima_garch, arima = arima_garch_model.ARIMA_GARCH()

SyntheticTimeSeries.plot([("Buy and Hold", buy_hold), ("ARIMA", arima), ("ARIMA+GARCH", arima_garch)], 
                         "Forecast ARIMA+GARCH", "Days", "Cash")















def sharpe_ratio_daily(strategy_returns, risk_free_return = 0):
    # the returns are daily so we can get the mean and standard deviation normally
    strategy_expected_return = np.mean(strategy_returns)
    strategy_std = np.std(strategy_returns)
    daily_sharpe_ratio = (strategy_expected_return - risk_free_return) / strategy_std
    return daily_sharpe_ratio
    
def sharpe_ratio_annual(strategy_returns, risk_free_return = 0):
    # if we want to get the annual sharpe ratio we need to annualise the returns and the standard deviation
    # we cannot just multiply by 252 (num trading days) because 1% increase every day for a year won't give 252% yearly return
    # instead we need to compound the returns and take the power of total days over trading days
    trading_days = 252
    total_returns = (1 + strategy_returns).prod()
    annualised_returns = total_returns**(252 / len(strategy_returns)) - 1
    
    # standard deviation scales with the square root of time so we only multiply by the square root of 252 days
    annualised_std = np.std(strategy_returns) * trading_days**(1/2)
    
    # computing the annual sharpe ratio now uses the annual values. We also assume that the risk_free rate is the same
    annualised_sharpe_ratio = (annualised_returns - risk_free_return) / annualised_std
    
    return annualised_sharpe_ratio

def sharpe_ratio_annual_log(strategy_returns, risk_free_return = 0):
    # convert simple returns to log returns. Other option is to compute the log returns from the normal prices log(S1 - S0)
    log_returns = np.log(strategy_returns + 1)
    
    # log returns are additive so we can add all of them together 
    # (e.g. 100% increase then 50% decrease is 0% profit -> 
    # this equals log(1) - log(0.5) = 0.69 - 0.69 = 0
    total_log_returns = np.sum(log_returns)
    
    # annualise the log returns over the 3000 days
    annualised_log_returns = total_log_returns * 252 / len(strategy_returns)
    
    # log to simple returns
    annualised_returns = np.exp(annualised_log_returns) - 1
    
    # standard deviation scales with the square root of time so we only multiply by the square root of 252 days
    annualised_std = np.std(strategy_returns) * 252**(1/2)
    
    annualised_sharpe_ratio = (annualised_returns - risk_free_return) / annualised_std
    return annualised_sharpe_ratio
    
    #data['Daily Return'] = data['Adj Close'].pct_change()   
    #return data.dropna()

def sortino_ratio_daily(strategy_returns, risk_free_return = 0):
    strategy_expected_return = strategy_returns.mean()
    strategy_std_neg = strategy_returns[strategy_returns<0].std()
    return (strategy_expected_return - risk_free_return) / strategy_std_neg

def sortino_ratio_annual(strategy_returns, risk_free_return = 0):
    trading_days = 252
    
    # annualise returns
    total_returns = (1 + strategy_returns).prod()
    annualised_returns = total_returns**(trading_days / len(strategy_returns)) - 1

    # annualise std
    annual_std_neg = strategy_returns[strategy_returns<0].std() * trading_days**(1/2)
    
    return (annualised_returns - risk_free_return) / annual_std_neg

# Computes the drawdown per time interval
def get_drawdown(data):
    roll_max = data.cummax()
    return data / roll_max - 1.0
    
# Computes the maximum drawdown for the whole data
def get_maximum_drawdown(data):
    return get_drawdown(data).cummin()

# Central plotting function to keep the plots consistent and save code repetition
def plot(plotDataList, title, xlabel, ylabel, legendLoc = "upper left"):
    for i in range(len(plotDataList)):
        plt.plot(plotDataList[i], label=i)
        
    plt.title(title)
    plt.legend(loc=legendLoc, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

"""
    Adjust Sharpe Ratio considering the number of strategies tested
    SR_values - list of all annual Sharpe Ratio values, one for each strategy
    N - number of strategies tested
    T - number of observations or degrees of freedom
    period - the period of each observation that equals one year (365 for daily periods, 12 for monthly)
    correction_method - the method used to control for FWER and adjust the Sharpe Ratios (e.g. bonferroni, sidak, holm-sidak, holm)
"""
def adjust_SR(SR_values, N, T, period, correction_method = "bonferroni"):
    #N = 200 # Number of strategies tested
    #T = 240 # Number of observations or degrees of freedom
    #SR = 0.75 # Annualised Sharpe ratio
    #period = 12
    
    t_stat = SR_values * math.sqrt(T / period)
    p_single = scipy.stats.t.sf(t_stat, df=T) # do I need abs? and do I need to mul by 2 when the strategy return is negative
    #p_single2 = (1 - scipy.stats.norm.cdf(t_stat)) # can use sf?, need to do for all strat?

    # FWER
    p_multiple = 1 - (1 - p_single)**N
    
    #p_multiple_adf = p_multiple / 2
    p_multiple_adf = statsmodels.stats.multitest.multipletests(pvals = p_multiple, method = correction_method)

    p_single_adj = scipy.stats.t.isf(p_multiple_adf, df=T)
    SR_adj = p_single_adj / math.sqrt(T) * math.sqrt(period)
    
    #z_stat = scipy.stats.t.ppf(1 - p_multiple / 2, N - 1)
    #HSR = z_stat / math.sqrt(T) * math.sqrt(period)
    
    return SR_adj




# Returns
trend_following_returns = strategy_return(trend_following)
mean_reversion_returns = strategy_return(mean_reversion)
#arima_returns = strategy_return(arima_auto)

# Sharpe Ratio
print(sharpe_ratio_daily(trend_following_returns))
tf_sr_annual = sharpe_ratio_annual(trend_following_returns)
mr_sr_annual = sharpe_ratio_annual(mean_reversion_returns)
#arima_sr_annual = sharpe_ratio_annual(arima_returns)
print(sharpe_ratio_annual_log(trend_following_returns))
print("SHARPE RATIOS")
print(tf_sr_annual)
print(mr_sr_annual)

# Sortino Ratio
print(sortino_ratio_daily(trend_following_returns))
print(sortino_ratio_annual(trend_following_returns))

# Maximum Drawdown
drawdown = get_drawdown(pd.DataFrame(data=trend_following))
max_drawdown = drawdown.cummin()
plot([drawdown, max_drawdown], "Trend Following Maximum Drawdown", "Date", "Drawdown %", "upper right")

# 
all_strat_SR = [tf_sr_annual, mr_sr_annual, arima_sr_annual]
print(adjust_SR(all_strat_SR, len(trend_following_returns), len(all_strat_SR), 365))

