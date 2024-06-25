import yfinance as yf
from sys import argv
import pandas as pd

PERIOD = "5y"
INTERVAL = "1d"

def get_stock_data(stock_names):
    return yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = stock_names,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = PERIOD,

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = INTERVAL,

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,
    )

if __name__ == "__main__":
    stock_names = argv[1: ]
    for name in stock_names:
        data = get_stock_data(name)
        data['stock'] = name
        data.to_csv(f'{name}.csv')

    
    