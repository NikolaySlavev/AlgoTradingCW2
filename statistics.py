import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import inf

import scipy.stats
import math
import statsmodels
import warnings

# filter some warnings
warnings.filterwarnings('ignore')

# Computes the daily Sharpe Ratio because the strategy returns are daily
def sharpe_ratio_daily(strategy_returns, risk_free_return = 0):
    # the returns are daily so we can get the mean and standard deviation normally
    strategy_expected_return = np.mean(strategy_returns)
    strategy_std = np.std(strategy_returns)
    daily_sharpe_ratio = (strategy_expected_return - risk_free_return) / strategy_std
    return daily_sharpe_ratio
    
# Computes the annual Sharpe Ratio from daily returns
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

# Same annual Sharpe Ratio but with log returns implementation
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

# Daily Sortino Ratio
def sortino_ratio_daily(strategy_returns, risk_free_return = 0):
    strategy_expected_return = strategy_returns.mean()
    strategy_std_neg = strategy_returns[strategy_returns<0].std()
    return (strategy_expected_return - risk_free_return) / strategy_std_neg

# Annual Sortino Ratio
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
    T - number of observations or degrees of freedom
    period - the period of each observation that equals one year (252 for daily periods, 12 for monthly)
    correction_method - the method used to control for FWER and adjust the Sharpe Ratios (e.g. bonferroni, sidak, holm-sidak, holm)
"""
def adjust_SR(SR_values, T, period, correction_method = "bonferroni", alpha = 0.05):
    # Negative Sharpe ratios are adjusted to 0
    for i in range(len(SR_values)):
        if SR_values[i] < 0:
            SR_values[i] = 0
    
    # Sharpe ratios to t stats
    t_stat = SR_values * math.sqrt(T / period)
    
    # T stat to p values (no need to adjust negatives)
    p_single = 2 * scipy.stats.t.sf(t_stat, df=T-1)
    
    # Adjust p values with chosen correction method
    p_single_adj = statsmodels.stats.multitest.multipletests(pvals = p_single, method = correction_method, alpha = alpha)
    
    # Revert back adjusted p values to adjusted t stats
    t_stat_adj = scipy.stats.t.isf(p_single_adj[1]/2, df=T-1)
    
    # Revert back t stats to Sharpe ratios
    SR_adj = t_stat_adj / math.sqrt(T) * math.sqrt(period)
    SR_adj[SR_adj == -inf] = 0
    
    # Check which strategies meet the old threshold    
    SR_accepted = check_hypothesis(p_single, alpha)
    
    # Retrieve the correct adjusted alpha values per the docs of multipletests library
    if (correction_method == "bonferroni" or correction_method == "holm"):
        alpha_adj = p_single_adj[3]
    elif (correction_method == "sidak" or correction_method == "holm-sidak"):
        alpha_adj = p_single_adj[2]
    else:
        raise Exception("Invalid correction method")
    
    # Check which strategies meet the new threshold
    SR_adj_accepted = check_hypothesis(p_single_adj[1], alpha_adj)
    
    return ((SR_values, SR_accepted, alpha), (SR_adj, SR_adj_accepted, alpha_adj))

# Check whether to accept or reject the strategy
def check_hypothesis(p_values, alpha):
    checks = []
    for p in p_values:
        if (p < alpha):
            checks.append(True)
        else:
            checks.append(False)
            
    return checks
    
# Computes the Family-wise error rate
def get_FWER(alpha, N):
    return 1 - (1 - alpha)**N
