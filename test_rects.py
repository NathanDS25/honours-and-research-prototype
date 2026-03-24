import pandas as pd, numpy as np, yfinance as yf
data = yf.download('^BSESN', start='2014-01-01', end='2024-01-01')
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
df = data[['Close']].copy()
df['Returns'] = df['Close'].pct_change()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()
c1 = (df['SMA_20'] > df['SMA_50']) & (df['Returns'].rolling(10).mean() > 0)
c2 = (df['SMA_20'] < df['SMA_50']) & (df['Returns'].rolling(10).mean() < 0)
df['Regime'] = np.select([c1, c2], [0, 1], default=2)
df.dropna(inplace=True)
groups = (df['Regime'] != df['Regime'].shift()).cumsum()
print("Number of rectangles:", len(df.groupby(groups)))
