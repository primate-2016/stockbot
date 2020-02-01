import os
# import pandas as pd
# import datetime
import pandas_datareader.data as web
import pandas as pd
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot, register_matplotlib_converters  # for autocorrelation plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # for different type of autocorrelation plot
from statsmodels.tsa.arima_model import ARIMA
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime, timedelta
# from sklearn import preprocessing, model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import make_regression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from dateutil.parser import parse
from pmdarima.arima.utils import ndiffs
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
from pmdarima.model_selection import train_test_split
#%matplotlib inline
register_matplotlib_converters()
# from pandas import DataFrame
#from matplotlib import style

start = datetime(2008, 1, 1)
end = datetime(2020, 1, 17)
av_api_key = os.getenv('ALPHAVANTAGE_API_KEY')

df = web.DataReader("LON:BARC", 'av-daily', start=start, end=end, api_key=av_api_key)
# print(df)
# close_px = df['close']
# hundred_day_moving_avg = close_px.rolling(window=100).mean()
# ten_day_moving_avg = close_px.rolling(window=10).mean()

# give the date column a name
# first column is index
# df.rename(columns={list(df)[0]:'date'}, inplace=True)

# print(df.index)

df['datetime'] = pd.to_datetime(df.index)

df = df.set_index('datetime')

#print(df.head())

# create a graph of price trends per year to see if there are any notable patterns
# Multiplicative Decomposition
 
# result_mul = seasonal_decompose(df['close'], model='multiplicative', extrapolate_trend='freq', freq=260)

# Additive Decomposition
# result_add = seasonal_decompose(df['close'], model='additive', extrapolate_trend='freq', freq=260)

# # Plot
# plt.rcParams.update({'figure.figsize': (10,10)})
# result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
# result_add.plot().suptitle('Additive Decompose', fontsize=22)
# plt.show()

# df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
# df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']

# print(df_reconstructed.head())




# KPSS Test
result = kpss(df.close.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

############################################################
# Use an ARIMA model
# try: https://www.youtube.com/watch?v=o7Ux5jKEbcw
# looks like this might be more accurate for an individual
# stock accorrding to research - but linear regression looks
# much easier to do multi-variate analysis so I can use
# all stocks in a sector in the training model for example more easily
##########################################################

# for ARIMA needs to determine p, d, q values
# d is the order of differencing - how many times the 
# time series needs to be differenced to make it
# stationary

# ADF Test
result = adfuller(df.close.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critical Values:')
    print(f'   {key}, {value}')



# create test and train datasets
train_len = int(df.shape[0] * 0.8)
train_data, test_data = df[:train_len], df[train_len:]
 
y_train = train_data['close'].values
y_test = test_data['close'].values

print(f"{train_len} train samples")
print(f"{df.shape[0] - train_len} test samples")

# Above for barclays gives a very low p value
# so you infer that the time series is stationary
# and doesn't need any differencing

# below gives the number of differences (d value)
# required to make a time series stationary
# ADF Test
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf')

# KPSS Test
kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss')

# PP Test
# pp = ndiffs(df.close.values, test='pp')

# here we're taking the max of the two differencing
# value above to use in the model
#  - maybe we should just run the model 
# with both and see which is most accurate?
# in the example with Barc prices the adf test 
# was very definitely 0 differences, but we ended up with
# 1 because KPSS was 1
n_diffs = max(adf_diffs, kpss_diffs)
print(f"Estimated differencing term: {n_diffs}")

# allow auto_arima to determine values for hyper-parameters p and q

auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                     suppress_warnings=True, error_action="ignore", max_p=6,
                     max_order=None, trace=True)

print(f"Estimated auto_arima hyper-params: {auto.order}")

model = auto

# create train test split

# y_train, y_test = train_test_split(df.close.values, test_size=0.1)

# does the below forecast one period ahead - yes?
# can i just use it to predict the next value beyond
# the time series (which is what I want)
# if not refer back to the machine learnimg mastery
# page - we can use the original ARIMA library you were
# looking at and use the autoarima hyper params below
# for that library - this does allow you to predict
# beyond the time series https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])


forecasts = []
confidence_intervals = []

for new_ob in y_test:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
    model.update(new_ob)

# lower values are better, 0 is best
print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
print(f"SMAPE: {smape(y_test, forecasts)}")


# plot the output of ARIMA

# switch data lists to numpy arrays
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --------------------- Actual vs. Predicted --------------------------
axes[0].plot(y_train_np, color='blue', label='Training Data')
axes[0].plot(test_data.index, forecasts, color='green', marker='o',
             label='Predicted Price')

axes[0].plot(test_data.index, y_test_np, color='red', label='Actual Price')
axes[0].set_title('Barclays Prices Prediction')
axes[0].set_xlabel('Dates')
axes[0].set_ylabel('Prices')

axes[0].set_xticks(np.arange(0, 7982, 1300).tolist(), df['Date'][0:7982:1300].tolist())
axes[0].legend()


# ------------------ Predicted with confidence intervals ----------------
axes[1].plot(y_train_np, color='blue', label='Training Data')
axes[1].plot(test_data.index, forecasts, color='green',
             label='Predicted Price')

axes[1].set_title('Prices Predictions & Confidence Intervals')
axes[1].set_xlabel('Dates')
axes[1].set_ylabel('Prices')

conf_int = np.asarray(confidence_intervals)
axes[1].fill_between(test_data.index,
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.9, color='orange',
                     label="Confidence Intervals")

axes[1].set_xticks(np.arange(0, 7982, 1300).tolist(), df['Date'][0:7982:1300].tolist())
axes[1].legend()


# print('adf value is', adf)
# print('kpss value is', kpss)
# print('pp value is', pp)

# determine if the model needs any AR terms
# can find out the required number of AR terms
# by inspecting the Partial Autocorrelation 
# (PACF) plot - this gives you the p value
# the order of the Auto-Regressive (AR) term - the numner
# of lags to be used as predictors



# using auto-arima https://alkaline-ml.com/pmdarima/develop/usecases/stocks.html





# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

#The right order of differencing is the minimum differencing required to get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.

# Original Series
# fig, axes = plt.subplots(3, 2, sharex=True)
# axes[0, 0].plot(df.close); axes[0, 0].set_title('Original Series')
# plot_acf(df.close, ax=axes[0, 1])

# 1st order Differencing
# axes[1, 0].plot(df.close.diff()); axes[1, 0].set_title('1st Order Differencing')
# plot_acf(df.close.diff().dropna(), ax=axes[1, 1])

# # 2nd order Differencing
# axes[2, 0].plot(df.close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
# plot_acf(df.close.diff().diff().dropna(), ax=axes[2, 1])

# plt.show()


# # Adjusting the size of matplotlib
# mpl.rc('figure', figsize=(8, 7))
# mpl.__version__

# # Adjusting the style of matplotlib
# style.use('ggplot')

# close_px.plot(label='BARC')
# hundred_day_moving_avg.plot(label='100d_moving_avg')
# ten_day_moving_avg.plot(label='10d_moving_avg')
# plt.show()

'''process the data so we can put into processing models
we create a mapping in the table between the close price
for a given day and the close price for some time in the future
based on the size of the data set which will be used to map a 
relationship between the two over the whole dataset and then be used
to make predictions'''

# # create a new table called dfreg
# dfreg = df.loc[:, ['close', 'volume']]
# # create a new column showing the diff between high and low as % of overall price (measure of volatility?)
# dfreg['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
# # create a new column showing the % change on the day
# dfreg['PCT_CHANGE'] = (df['close'] - df['open']) / df['open'] * 100.0

# # print(dfreg)

# # Drop missing value
# dfreg.fillna(value=-99999, inplace=True)

# # figure out how big 1% of the data is and use this difference to forecast
# forecast_out = int(math.ceil(0.01 * len(dfreg)))
# # print(forecast_out)
# # i.e. if the length of the table is 1000 this will return 10 (i.e. 0.01 or 1%)

# # we want to predict the AdjClose, so use the 'close' column as the data source
# forecast_col = 'close'
# # shift the data 'forecast_out' (or 1%) periods backwards (i.e. the close price for 10th Jan gets shifted to 1st Jan if our forecast_out is 10)
# dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
# print(dfreg)

# # this just appears to be dropping the label column we just added?
# X = np.array(dfreg.drop(['label', forecast_col], 1))

# # print(X)

# # Scale the X so that everyone can have the same distribution for linear regression
# # .scale() standardizes a dataset along an axis - looks like it smooths out the data
# # so models don't get flipped out by weird variances
# X = preprocessing.scale(X)

# # print(X)

# # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
# X_end = X[-forecast_out:]

# # print(X_end)
# X_early = X[:-forecast_out]
# # print(X_early)
# # Separate label and identify it as y

# dfreg.dropna(inplace=True)
# y = np.array(dfreg['label'])
# # y_early = y[:-forecast_out]
# # print(y_early)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X_early, y, test_size=0.2)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)

# confidence = clf.score(X_test, y_test)

# forecast_set = clf.predict(X_end)
# df['Forecast'] = np.nan

# last_date = df.iloc[-1].name
# last_unix = last_date
# next_unix = last_unix + str(timedelta(days=1))

# for i in forecast_set:
#     next_date = next_unix
#     next_unix += str(timedelta(days=1))
#     df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


# df['close'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()


# # \******candlesticks are showing diff beteern max and min and open and close - need to test on this to work out direction
# print(df.tail())

# print(df.describe())

# df = df.fillna(method='ffill')

# ##############################################################################
# # the below used a linear regression against Barclays stock with
# # high accuracy - but not clear whether this is appropriate to forecast beyond
# # the end of the time series??
# # https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# #####################################################

# # what we're mapping relationships to the thing we want to predict
# x = df[['open', 'high', 'low', 'volume']].values

# # the thing we want to predict
# y = df['close'].values


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# regressor = LinearRegression()
# regressor.fit(x_train, y_train)

# # print(regressor.coef_)


# # coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])
# # print(coeff_df)

# y_pred = regressor.predict(x_test)
# # so how do I turn y_pred into actual future data?

# # create a new table showing test vs predicted
# df_2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(df_2.head(50))

# # df_2.plot(kind='bar', figsize=(10, 8))
# # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# # plt.show()

# # print(df_2.mean())

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('Linear Regression Score:', regressor.score(x_test, y_test))


# x_new, _ = make_regression(n_samples=5, n_features=4, noise=0, random_state=0)
# # make a prediction

# y_new = regressor.predict(x_new)
# for i in range(len(x_new)):
#     print("X=%s, Predicted=%s" % (x_new[i], y_new[i]))

# next step - go through the prediction tutorial
# use it to predict the next 5 close prices
# do that across all of FTSE 500 and check for 
# biggest changes since the previous day - then
# see if these are actually the biggest movers next
# week



# autocorrelation_plot(df)  # autocorrelation plot with pandas
# plot_acf(df)  # autocorrelation plot with statsmodels
# pyplot.show()

# fit model - can't cope with multi-variate data.....
# model = ARIMA(df, order=(5,1,0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())