import os
# import pandas as pd
# import datetime
import pandas_datareader.data as web
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime, timedelta
# from sklearn import preprocessing, model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline

# from pandas import DataFrame
#from matplotlib import style

start = datetime(2008, 1, 1)
end = datetime(2020, 1, 1)
av_api_key = os.getenv('ALPHAVANTAGE_API_KEY')

df = web.DataReader("LON:BARC", 'av-daily', start=start, end=end, api_key=av_api_key)
# print(df)
close_px = df['close']
hundred_day_moving_avg = close_px.rolling(window=100).mean()
ten_day_moving_avg = close_px.rolling(window=10).mean()

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


# map between sma and change in price? above was just mapping bbetween one point and another arbitrary point in the future

# \******candlesticks are showing diff beteern max and min and open and close - need to test on this to work out direction
print(df)

print(df.describe())

df = df.fillna(method='ffill')

# what we're mapping relationships to the thing we want to predict
x = df[['open', 'high', 'low', 'volume']].values

# the thing we want to predict
y = df['close'].values

# plt.figure(figsize=(15,10))
# plt.tight_layout()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# print(regressor.coef_)

# coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])
# print(coeff_df)

y_pred = regressor.predict(x_test)
# so how do I turn y_pred into actual future data?

df_2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_2.head(50))

# df_2.plot(kind='bar', figsize=(10, 8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

# print(df_2.mean())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))