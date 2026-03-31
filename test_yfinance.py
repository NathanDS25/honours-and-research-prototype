import yfinance as yf
import pandas as pd

def test_fetch():
    ticker = "^BSESN"
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start="2014-01-01", end="2026-03-31")
    print("Columns:", data.columns)
    print("Head:\n", data.head())
    print("Shape:", data.shape)

if __name__ == "__main__":
    test_fetch()
