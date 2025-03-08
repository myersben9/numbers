import yfinance as yf
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import talib
import seaborn as sns

# Grab the 'Symbols' column from nasdaq.csv
nasdaq = pd.read_csv('nasdaq.csv')
symbols = nasdaq['Symbol'].tolist()
# Grab a random sample of 100 symbols
sample_symbols = random.sample(symbols, 10)

# Get the max available data for each symbol
max_data = {}
try:
    for symbol in sample_symbols:
        try:
            ticker = yf.Ticker(symbol)
            ticker_data = ticker.history(period="1mo", interval="5m")
            
            # Skip tickers with no data
            if ticker_data.empty:
                print(f"No data available for {symbol}, skipping")
                continue
                
            max_data[symbol] = ticker_data
        except Exception as e:
            print(f"Error getting data for {symbol}: {str(e)}")
except Exception as e:
    print(f"Error in ticker processing: {str(e)}")

# For each day of each symbol, store the times of day where there was the most profit from a buy and sell 
# Dataframe for trading days
trading_days = pd.DataFrame(columns=['symbol', 'date', 'buy_time', 'sell_time', 'profit'])

# Function to compute indicators for a single row
def compute_indicators(df, time, full_data):
    """ 
    Computes RSI, MACD, and ATR for the given timestamp using historical data.
    
    Args:
        df: DataFrame containing single day's data
        time: Timestamp to compute indicators for
        full_data: Full historical data for the symbol (not just current day)
    """
    if time not in df.index:
        return None, None, None  # Return zeros if time doesn't exist
    
    try:
        # Get all data up to this timestamp from the full historical dataset
        historical_data = full_data[full_data.index <= time]
        
        # We need sufficient data for calculation
        if len(historical_data) < 30:  # Minimum data points needed (adjust as needed)
            return None, None, None
            
        close_prices = historical_data['Close'].values
        high_prices = historical_data['High'].values
        low_prices = historical_data['Low'].values
        
        # Compute indicators using all available historical data
        rsi = talib.RSI(close_prices, timeperiod=14)[-1]
        macd, macdsignal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        return rsi, macd[-1], atr
    except Exception as e:
        print(f"Error calculating indicators for {time}: {str(e)}")
        return None, None, None  # Return zeros on any error


# Function to calculate max profit for a day with O(n) time complexity
def calculate_max_profit(day_data):
    if len(day_data) < 2:
        return None, None, 0  # Not enough data to trade

    min_price = day_data.iloc[0]['Close']
    max_profit = 0
    best_buy_time = day_data.index[0]
    best_sell_time = day_data.index[0]
    potential_buy_time = day_data.index[0]  # Initialize here to avoid reference before assignment

    for i in range(1, len(day_data)):
        current_price = day_data.iloc[i]['Close']
        current_time = day_data.index[i]

        # Update the minimum price encountered so far
        if current_price < min_price:
            min_price = current_price
            potential_buy_time = current_time

        # Calculate the profit if we sell at the current price
        profit = current_price - min_price

        # Update the maximum profit and best buy/sell times
        if profit > max_profit:
            max_profit = profit
            best_buy_time = potential_buy_time
            best_sell_time = current_time

    return best_buy_time, best_sell_time, max_profit

# Function to plot and save the buy/sell signals on top of the price graph
def plot_buy_sell_signals(symbol, date, day_data, buy_time, sell_time, profit):
    plt.figure(figsize=(14, 7))
    plt.plot(day_data.index, day_data['Close'], label='Close Price')
    plt.axvline(x=buy_time, color='g', linestyle='--', label=f'Buy Time: {buy_time.time()}')
    plt.axvline(x=sell_time, color='r', linestyle='--', label=f'Sell Time: {sell_time.time()}')
    plt.title(f'{symbol} on {date} - Max Profit: {profit:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Create a directory to save the plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the plot
    plt.savefig(f'plots/{symbol}_{date}.png')
    plt.close()

# For each symbol, for each day, store the times of day where there was max profit for that day from a buy and sell 
trading_days = []
try:
    for symbol in sample_symbols:
        if symbol not in max_data:
            continue  # Skip if data for this symbol is not available

        # Get the full data for this symbol
        data = max_data[symbol]
        
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Warning: Index for {symbol} is not a DatetimeIndex. Skipping.")
            continue
            
        # Create a Date column from the index
        data = data.copy()  # Create a copy to avoid SettingWithCopyWarning
        data['Date'] = data.index.to_series().dt.date  # Use dt.date instead of date property
        
        # Now group by the Date column
        grouped_by_day = data.groupby('Date')

        for date, day_data in grouped_by_day:
            if len(day_data) < 2:
                continue  # Skip days with insufficient data

            # Calculate the best buy and sell times for the day
            buy_time, sell_time, profit = calculate_max_profit(day_data)

            if buy_time and sell_time:
                # Use try/except to safely access time attributes
                try:
                    buy_time_value = buy_time.time()
                    sell_time_value = sell_time.time()
                except AttributeError:
                    print(f"Warning: Time attribute issue for {symbol} on {date}")
                    continue
                    
                # Compute indicators using full historical data up to these times
                buy_rsi, buy_macd, buy_atr = compute_indicators(day_data, buy_time, data)
                sell_rsi, sell_macd, sell_atr = compute_indicators(day_data, sell_time, data)
                
                # Skip rows where indicators are not available
                if buy_rsi is None or sell_rsi is None or buy_macd is None or sell_macd is None or buy_atr is None or sell_atr is None:
                    print(f"Skipping {symbol} on {date}: insufficient data for indicators")
                    continue

                trading_days.append({
                    'symbol': symbol,
                    'date': date,
                    'buy_time': buy_time_value,
                    'sell_time': sell_time_value,
                    'profit': profit,
                    'buy_RSI': buy_rsi,
                    'sell_RSI': sell_rsi,
                    'buy_MACD': buy_macd,
                    'sell_MACD': sell_macd,
                    'buy_ATR': buy_atr,
                    'sell_ATR': sell_atr
                })

    # Only create the DataFrame if we have data
    if trading_days:
        trading_days_df = pd.DataFrame(trading_days)
        # trading_days_df.to_csv('trading_days.csv', index=False)
        # Ensure the plots directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')

        # List of indicators to plot
        indicators = ['RSI', 'MACD', 'ATR']

        # Create a single figure with subplots
        fig, axes = plt.subplots(1, len(indicators), figsize=(18, 6))

        # Generate and save plots for each indicator
        for i, indicator in enumerate(indicators):
            bins = max(10, len(trading_days_df) // 2)  # Dynamic bins based on data size
            sns.histplot(trading_days_df[f'buy_{indicator}'], color='blue', label=f'Buy {indicator}', kde=True, bins=bins, alpha=0.6, ax=axes[i])
            sns.histplot(trading_days_df[f'sell_{indicator}'], color='red', label=f'Sell {indicator}', kde=True, bins=bins, alpha=0.6, ax=axes[i])
            axes[i].set_xlabel(indicator)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of Buy & Sell {indicator}')
            axes[i].legend()
            axes[i].grid()

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig('plots/indicator_distributions.png')
        plt.close()
    else:
        print("No trading opportunities found!")
        
except Exception as e:
    print(f"Error in trading days processing: {str(e)}")