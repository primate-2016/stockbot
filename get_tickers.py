import bs4 as bs
# import pickle
import requests


def get_tickers(webpage, ticker_position, sector_position):
    resp = requests.get(webpage)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = {}
    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[ticker_position].text.rstrip()
        sector = row.findAll('td')[sector_position].text.rstrip()

        tickers[symbol] = sector
        print(tickers)
        
#     # with open("sp500tickers.pickle","wb") as f:
#     #    pickle.dump(tickers,f)

    return tickers


stock_indices = [
    'https://en.wikipedia.org/wiki/FTSE_100_Index',
    # 'https://en.wikipedia.org/wiki/FTSE_250_Index' doesn't have sectors
]

if __name__ == "__main__":

    tickers = {}
    for stock_index in stock_indices:
        tickers.update(get_tickers(stock_index, 1, 2))

    print(tickers)
