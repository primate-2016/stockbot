import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

import bs4 as bs
# import pickle
import requests

def get_tickers(webpage, ticker_position):
    resp = requests.get(webpage)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    # ignore first row in table and take second (ticker_position) column
    # rsrtip() tickers as some have trailing /n
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[ticker_position].text
        tickers.append(ticker.rstrip())
        
    #with open("sp500tickers.pickle","wb") as f:
    #    pickle.dump(tickers,f)

    return tickers

stock_indices = [
    'https://en.wikipedia.org/wiki/FTSE_100_Index',
    'https://en.wikipedia.org/wiki/FTSE_250_Index'

]

tickers = []
for stock_index in stock_indices:
    tickers.extend(get_tickers(stock_index, 1))

print(tickers)

# style.use('ggplot')

# start = dt.datetime(2000,1,1)
# end = dt.datetime(2016,12,31)

# df = web.DataReader('TSLA', 'yahoo', start, end)
# print(df.tail(6))

# # dump to csv
# df.to_csv('tsla.csv')

# read from csv
# df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
# print(df[['Open', 'High']].head())

# # create a graph based on the Adj Close column in the csv
# df['Adj Close'].plot()
# plt.show()


# get ftse 100 and ftse 250 list from wikipedia (ftse 250 is companies 101 to 250) 


