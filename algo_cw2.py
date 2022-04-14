import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os
from statsmodels.tsa.arima.model import ARIMA

# Add the file's path in order to import the files when compiled with visual studio code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import strategies
import statistics

# filter some warnings
warnings.filterwarnings('ignore')


# TIME SERIES
cash_start = 10000
data = strategies.SyntheticTimeSeries(seed=1)
prices = data.get_prices()
prices_test = prices[int(len(prices)*0.7):].reset_index(drop = True)
returns = data.get_returns()
returns_test = returns[int(len(returns)*0.7):].reset_index(drop = True)
returns_train = returns[:int(len(returns)*0.7)].reset_index(drop = True)

# Custom plotting function to avoid code duplication
strategies.SyntheticTimeSeries.plot([("Prices", prices)], 
                         "Time Series Data", "Days", "Price")

# Diagnostics
model = ARIMA(returns, order=(1,0,1))
model_fit = model.fit()
model_fit.plot_diagnostics()
plt.show()

# Stationarity
strategies.SyntheticTimeSeries.adf_test(returns)
strategies.SyntheticTimeSeries.kpss_test(returns)

# BUY AND HOLD
buy_hold = cash_start / prices[0] * prices
buy_hold_train = buy_hold[:int(len(returns)*0.7)].reset_index(drop = True)
buy_hold_test = cash_start / prices_test[0] * prices_test

# TREND FOLLOWING
tf = strategies.TrendFollowing(data, cash_start)
tf_sma = tf.TF_sma(10)
tf_ema = tf.TF_ema(0.5)

strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold), ("Simple MA (10 period)", tf_sma), ("Exponential MA (0.5 alpha)", tf_ema)], 
                         "Trend Following", "Days", "Cash")

strategies.SyntheticTimeSeries.plot([("Time-series", prices), ("MA (20)", strategies.SyntheticTimeSeries.get_simple_moving_average(prices, 20))], 
                         "Moving Average", "Days", "Price")

# MEAN REVERSION
mr = strategies.MeanReversion(data, cash_start)
mr_sma = mr.MR_sma(20)
mr_bb_rsi = mr.MR_bb_rsi(20, 2, 6)

strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold), ("Simple MA (20 period)", mr_sma), ("BB+RSI (20p, 2std, 6p)", mr_bb_rsi)], 
                         "Mean Reversion", "Days", "Cash")

# Bands
#plt.plot(prices)
#plt.fill_between([i for i in range(len(bands[0]))], bands[0], bands[1], label = 'Bollinger Bands', color='lightgrey')
#plt.show()

# RSI
#plt.plot(rsi)
#plt.fill_between([i for i in range(len(rsi))], [90 for i in range(len(rsi))], [10 for i in range(len(rsi))], label = 'Bollinger Bands', color='lightgrey')
#plt.show()

# ARIMA TRAIN
arima_garch_model = strategies.ForecastArimaGarch(data, cash_start)
arima_garch_train, predictions_train = arima_garch_model.ARIMA_GARCH_train()

strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_train), ("ARIMA+GARCH", arima_garch_train)], 
                          "Forecast ARIMA+GARCH Train", "Days", "Cash")

strategies.SyntheticTimeSeries.plot([("Returns Time-Series", returns_train), ("ARIMA+GARCH Fit", predictions_train)],
                         "ARIMA+GARCH fit on train set", "Days", "Returns")

# ARIMA TEST
#arima_garch_test, predictions_test = arima_garch_model.ARIMA_GARCH_test()

# strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_test), ("ARIMA+GARCH", arima_garch_test)], 
#                          "Forecast ARIMA+GARCH Test", "Days", "Cash")

# strategies.SyntheticTimeSeries.plot([("Returns Time-Series", returns_test), ("ARIMA+GARCH Fit", predictions_test)],
#                          "ARIMA+GARCH fit on test set", "Days", "Returns")


# STRATEGIES PARAMETER TUNING
ema_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
bb_periods = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]
bb_stds = [0.5, 1, 1.5, 2, 2.5, 3]
rsi_periods = [4, 6, 10, 14, 20, 30, 50]

best_ema = 0
best_tf = 0
tf_ema_train_best = None
for i in range(len(ema_alphas)):
    tf_ema_train = tf.TF_ema(ema_alphas[i], "train")
    if tf_ema_train[-1] > best_tf and tf_ema_train[-1] != cash_start:
        best_tf = tf_ema_train[-1]
        best_ema = ema_alphas[i]
        tf_ema_train_best = tf_ema_train

print("Best Trend Following EMA alpha is", best_ema)
    
best_mr_params = [0,0,0]
best_mr = 0
mr_bb_rsi_train_best = None
for i in range(len(bb_periods)):
    for j in range(len(bb_stds)):
        for k in range(len(rsi_periods)):        
            mr_bb_rsi_train = mr.MR_bb_rsi(bb_periods[i], bb_stds[j], rsi_periods[k], "train")
            if mr_bb_rsi_train[-1] > best_mr and mr_bb_rsi_train[-1] != cash_start and mr_bb_rsi_train.tolist().count(cash_start) < len(mr_bb_rsi_train)*0.7:
                best_mr = mr_bb_rsi_train[-1]
                best_mr_params = [bb_periods[i], bb_stds[j] , rsi_periods[k]]
                mr_bb_rsi_train_best = mr_bb_rsi_train

print("Best Mean Reversion BB period, BB std and RSI period are", best_mr_params)

arima_garch_train = arima_garch_model.ARIMA_GARCH_train()

strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_train), 
                                              ("TF EMA [" + str(best_ema) + "]", tf_ema_train_best), 
                                              ("MR BB RSI " + str(best_mr_params), mr_bb_rsi_train_best),
                                              ("ARIMA+GARCH", arima_garch_train[0])], 
                                             "Strategies Train Set", "Days", "Cash")


#prices_test = prices[int(len(buy_hold)*0.7):].reset_index(drop = True)
#buy_hold_test = cash_start / prices_test[0] * prices_test
tf_ema_test = tf.TF_ema(best_ema, "test")
mr_bb_rsi_test = mr.MR_bb_rsi(best_mr_params[0], best_mr_params[1], best_mr_params[2], "test")
arima_garch_test = arima_garch_model.ARIMA_GARCH_test()
arima_garch_test = arima_garch_test[0][len(buy_hold_train):]

strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_test), 
                                              ("TF EMA [" + str(best_ema) + "]" , tf_ema_test), 
                                              ("MR BB RSI " + str(best_mr_params), mr_bb_rsi_test),
                                              ("ARIMA+GARCH", arima_garch_test)], 
                                             "Strategies Test Set", "Days", "Cash")

# Returns
# predict in sample for arima and do the positions and then compare with the others
# + other 3 returns for the tests

buy_hold_train_returns = np.diff(buy_hold_train) / buy_hold_train[: -1]
buy_hold_test_returns = np.diff(buy_hold_test) / buy_hold_test[: -1]
strategies.SyntheticTimeSeries.plot([("Buy Hold", buy_hold_train_returns)], "Buy Hold Train Returns", "Days", "Returns")
strategies.SyntheticTimeSeries.plot([("Buy Hold", buy_hold_test_returns)], "Buy Hold Test Returns", "Days", "Returns")

tf_train_returns = np.diff(tf_ema_train_best) / tf_ema_train_best[: -1]
tf_test_returns = np.diff(tf_ema_test) / tf_ema_test[: -1]
strategies.SyntheticTimeSeries.plot([("Trend Following", tf_train_returns)], "Trend Following Train Returns", "Days", "Returns")
strategies.SyntheticTimeSeries.plot([("Trend Following", tf_test_returns)], "Trend Following Test Returns", "Days", "Returns")

mr_train_returns = np.diff(mr_bb_rsi_train_best) / mr_bb_rsi_train_best[: -1]
mr_test_returns = np.diff(mr_bb_rsi_test) / mr_bb_rsi_test[: -1]

strategies.SyntheticTimeSeries.plot([("Mean Reversion", mr_train_returns)], "Mean Reversion Train Returns", "Days", "Returns")
strategies.SyntheticTimeSeries.plot([("Mean Reversion", mr_test_returns)], "Mean Reversion Test Returns", "Days", "Returns")

arima_train_returns = np.diff(arima_garch_train[0]) / arima_garch_train[0][: -1]
arima_test_returns = np.diff(arima_garch_test) / arima_garch_test[: -1]
strategies.SyntheticTimeSeries.plot([("ARIMA+GARCH", arima_train_returns)], "ARIMA+GARCH Train Returns", "Days", "Returns")
strategies.SyntheticTimeSeries.plot([("ARIMA+GARCH", arima_test_returns)], "ARIMA+GARCH Test Returns", "Days", "Returns")

print("Buy Hold Train {:.2f}% return".format((buy_hold_train.iloc[-1] / cash_start - 1) * 100))
print("TF Train {:.2f}% return".format((tf_ema_train_best[-1] / cash_start - 1) * 100))
print("MR Train {:.2f}% return".format((mr_bb_rsi_train_best[-1] / cash_start - 1) * 100))
print("ARIMA+GARCH Train {:.2f}% return".format((arima_garch_train[0][-1] / cash_start - 1) * 100))

print("Buy Hold Test {:.2f}% return".format((buy_hold_test.iloc[-1] / cash_start - 1) * 100))
print("TF Test {:.2f}% return".format((tf_ema_test[-1] / cash_start - 1) * 100))
print("MR Test {:.2f}% return".format((mr_bb_rsi_test[-1] / cash_start - 1) * 100))
print("ARIMA+GARCH Test {:.2f}% return".format((arima_garch_test[-1] / cash_start - 1) * 100))

# Sharpe Ratio
print("Buy Hold SR Train Daily", statistics.sharpe_ratio_daily(buy_hold_train_returns))
print("TF SR Train Daily", statistics.sharpe_ratio_daily(tf_train_returns))
print("MR SR Train Daily", statistics.sharpe_ratio_daily(mr_train_returns))
print("ARIMA SR Train Daily", statistics.sharpe_ratio_daily(arima_train_returns))

print("TF SR Train Annual Log", statistics.sharpe_ratio_annual_log(tf_train_returns))

print("Buy Hold SR Train Annual", statistics.sharpe_ratio_annual(buy_hold_train_returns))
print("TF SR Train Annual", statistics.sharpe_ratio_annual(tf_train_returns))
print("MR SR Train Annual", statistics.sharpe_ratio_annual(mr_train_returns))
print("ARIMA SR Train Annual", statistics.sharpe_ratio_annual(arima_train_returns))

print("Buy Hold SR Test Daily", statistics.sharpe_ratio_daily(buy_hold_test_returns))
print("TF SR Test Daily", statistics.sharpe_ratio_daily(tf_test_returns))
print("MR SR Test Daily", statistics.sharpe_ratio_daily(mr_test_returns))
print("ARIMA SR Test Daily", statistics.sharpe_ratio_daily(arima_test_returns))

print("Buy Hold SR Test Annual", statistics.sharpe_ratio_annual(buy_hold_test_returns))
tf_sr_test = statistics.sharpe_ratio_annual(tf_test_returns)
print("TF SR Test Annual", tf_sr_test)
mr_sr_test = statistics.sharpe_ratio_annual(mr_test_returns)
print("MR SR Test Annual", mr_sr_test)
arima_sr_test = statistics.sharpe_ratio_annual(arima_test_returns)
print("ARIMA SR Test Annual", arima_sr_test)

# Sortino Ratio
print("TF Sortino Train Daily", statistics.sortino_ratio_daily(tf_train_returns))

print("Buy Hold Sortino Train Annual", statistics.sortino_ratio_annual(buy_hold_train_returns))
print("TF Sortino Train Annual", statistics.sortino_ratio_annual(tf_train_returns))
print("MR Sortino Train Annual", statistics.sortino_ratio_annual(mr_train_returns))
print("ARIMA Sortino Train Annual", statistics.sortino_ratio_annual(arima_train_returns))

print("Buy Hold Sortino Test Annual", statistics.sortino_ratio_annual(buy_hold_test_returns))
print("TF Sortino Test Annual", statistics.sortino_ratio_annual(tf_test_returns))
print("MR Sortino Test Annual", statistics.sortino_ratio_annual(mr_test_returns))
print("ARIMA Sortino Test Annual", statistics.sortino_ratio_annual(arima_test_returns))

# Maximum Drawdown
for strat in [buy_hold_train, tf_ema_train_best, mr_bb_rsi_train_best, arima_garch_train[0], 
                  buy_hold_test, tf_ema_test, mr_bb_rsi_test, arima_garch_test]:
    drawdown = statistics.get_drawdown(pd.DataFrame(data=strat))    
    max_drawdown = drawdown.cummin()
    print(max_drawdown.iloc[-1])
    strategies.SyntheticTimeSeries.plot([("Drawdown", drawdown), ("Max Drawdown", max_drawdown)], "Maximum Drawdown", "Days", "Drawdown", "upper right")
    
tf_train_drawdown = statistics.get_drawdown(pd.DataFrame(data=tf_ema_train_best))
tf_train_max_drawdown = tf_train_drawdown.cummin()
print(tf_train_max_drawdown.iloc[-1])

strategies.SyntheticTimeSeries.plot([tf_train_drawdown, tf_train_max_drawdown], "Trend Following Maximum Drawdown", "Days", "Drawdown %", "upper right")

# Adjust Sharpe Ratio
all_strat_SR = np.array([tf_sr_test, mr_sr_test, arima_sr_test])
SR_old_info, SR_new_info = statistics.adjust_SR(all_strat_SR, len(tf_test_returns), 252)

# FWER
FWER_old = statistics.get_FWER(SR_old_info[2], len(all_strat_SR))
FWER_new = statistics.get_FWER(SR_new_info[2], len(all_strat_SR))
