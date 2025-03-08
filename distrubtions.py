import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib

# Load NASDAQ tickers
df = pd.read_csv("nasdaq.csv")
tickers = df["Symbol"].dropna().unique()

# Take a random sample of 100 tickers
tickers = np.random.choice(tickers, size=30, replace=False)

def download_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="max")
        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def compute_indicators(df):
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    macd, signal, _ = talib.MACD(df['Close'])
    df['MACD'] = macd - signal
    
    # Fix: Correctly unpack the tuple returned by STOCHRSI
    fastk, fastd = talib.STOCHRSI(df['Close'])
    df['StochRSI'] = fastk
    
    return df.dropna()

def find_best_trade_opportunities(df):
    df['Next High'] = df['High'].shift(-1)
    df['Next Low'] = df['Low'].shift(-1)
    df['Profit Potential'] = df['Next High'] - df['Next Low']
    best_trades = df.nlargest(10, 'Profit Potential')
    return best_trades

def plot_combined_distributions(all_best_trades, indicators):
    fig, axes = plt.subplots(len(indicators), 1, figsize=(10, 5 * len(indicators)))
    fig.suptitle('Combined Indicator Distributions Across All Tickers')
    
    for ax, indicator in zip(axes, indicators):
        all_values = pd.concat([df[indicator].dropna() for df in all_best_trades], ignore_index=True)
        if all_values.empty:
            continue
        
        ax.hist(all_values, bins=20, alpha=0.6, color='b', density=True)
        
        # Probability distribution function
        mu, sigma = np.mean(all_values), np.std(all_values)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2), 'r')
        
        ax.set_title(f'{indicator} Distribution')
        ax.set_xlabel(indicator)
        ax.set_ylabel("Probability Density")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("combined_distributions.png")
    plt.show()

# Process each stock
all_best_trades = []
for ticker in tickers[:100]:  # Limiting to 10 for demonstration
    print(f"Processing {ticker}...")
    data = download_stock_data(ticker)
    if data is not None and not data.empty:
        data = compute_indicators(data)
        best_trades = find_best_trade_opportunities(data)
        all_best_trades.append(best_trades)

# Plot combined distributions
if all_best_trades:
    plot_combined_distributions(all_best_trades, ['ROC', 'MOM', 'ATR', 'MACD', 'StochRSI'])
