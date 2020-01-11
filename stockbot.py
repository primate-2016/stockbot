import os
import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
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

close_px.plot(label='BARC')
hundred_day_moving_avg.plot(label='100d_moving_avg')
ten_day_moving_avg.plot(label='10d_moving_avg')
plt.show()