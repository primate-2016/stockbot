import bs4 as bs
# import pickle
import requests


def get_tickers(webpage, ticker_position):
    resp = requests.get(webpage)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    # ignore first row in table and take second (ticker_position) column
    # rstrip() tickers as some have trailing /n
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[ticker_position].text
        tickers.append(ticker.rstrip())
        
    # with open("sp500tickers.pickle","wb") as f:
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
