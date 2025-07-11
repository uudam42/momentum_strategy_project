import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time
import os

ALPHA_VANTAGE_API_KEY = "BY82OPQKDXAWTHVQ"

def fetch_stock_data(symbol):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

def download_batch(symbols, save_path="raw_data"):
    os.makedirs(save_path, exist_ok=True)
    for symbol in symbols:
        try:
            print(f"Fetching {symbol}...")
            df = fetch_stock_data(symbol)
            df.to_csv(f"{save_path}/{symbol}.csv")
            print(f"{symbol} saved.")
            time.sleep(15)  # avoid API limitation
        except Exception as e:
            print(f"Failed for {symbol}: {e}")

if __name__ == "__main__":
    stock_list = ["AAPL", "MSFT", "GOOG", "NVDA", "TSLA"]
    download_batch(stock_list)
