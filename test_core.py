import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TICKER = "^BSESN"

def load_and_engineer_data(ticker=TICKER):
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start="2014-01-01", end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    df = data[['Close']].copy()
    
    df['Returns'] = df['Close'].pct_change()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Macro proxies (simplified for test)
    print("Downloading macro data...")
    vix = yf.download("^VIX", start="2014-01-01", end=end_date, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})
    
    df = df.join(vix, how='left')
    df['VIX'] = df['VIX'].ffill()
    df['VIX'] = (df['VIX'] - df['VIX'].mean()) / df['VIX'].std()
    
    # Labeling
    conditions = [
        (df['SMA_20'] > df['SMA_50']) & (df['Returns'].rolling(10).mean() > 0), # Bull
        (df['SMA_20'] < df['SMA_50']) & (df['Returns'].rolling(10).mean() < 0), # Bear
    ]
    choices = [0, 1]
    df['Regime'] = np.select(conditions, choices, default=2)
    df['Regime'] = df['Regime'].rolling(window=5, min_periods=1, center=True).median().round().astype(int)
    
    df.dropna(inplace=True)
    return df

def test_train(df):
    features = ['Close', 'RSI', 'SMA_20', 'SMA_50', 'VIX']
    X = df[features]
    y = df['Regime']
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training GradientBoosting...")
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    df = load_and_engineer_data()
    print(f"Data shape: {df.shape}")
    test_train(df)
