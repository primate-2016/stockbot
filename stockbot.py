import os
# import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import preprocessing
# from pandas import DataFrame
from matplotlib import style

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)
av_api_key = os.getenv('ALPHAVANTAGE_API_KEY')

df = web.DataReader("LON:BARC", 'av-daily', start=start, end=end, api_key=av_api_key)
print(df)
close_px = df['close']
hundred_day_moving_avg = close_px.rolling(window=100).mean()
ten_day_moving_avg = close_px.rolling(window=10).mean()

# Adjusting the size of matplotlib
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

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

# create a new table called dfreg
dfreg = df.loc[:, ['close', 'volume']]
# create a new column showing the diff between high and low as % of overall price (measure of volatility?)
dfreg['HL_PCT'] = (df['high'] - df['low']) / df['close'] * 100.0
# create a new column showing the % change on the day
dfreg['PCT_CHANGE'] = (df['close'] - df['open']) / df['open'] * 100.0

print(dfreg)

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# figure out how big 1% of the data is and use this difference to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
print(forecast_out)
# i.e. if the length of the table is 1000 this will return 10 (i.e. 0.01 or 1%)

# we want to predict the AdjClose, so use the 'close' column as the data source
forecast_col = 'close'
# shift the data 'forecast_out' (or 1%) periods backwards (i.e. the close price for 10th Jan gets shifted to 1st Jan if our forecast_out is 10)
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
print(dfreg)

# this just appears to be dropping the label column we just added?
X = np.array(dfreg.drop(['label'], 1))

print(X)

# Scale the X so that everyone can have the same distribution for linear regression
# .scale() standardizes a dataset along an axis - looks like it smooths out the data
# so models don't get flipped out by weird variances
X = preprocessing.scale(X)

print(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
