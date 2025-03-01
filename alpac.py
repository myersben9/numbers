import os
import time
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import ta
from yfetch import Yfetch
from dotenv import load_dotenv

load_dotenv()

class Bot:
    def __init__(self, ticker):
        self.ticker = ticker
        self.period = "1d"      # one day of 1-minute data
        self.interval = "1m"
        self.data = self.fetch_data()
        self.indicatordata = self.calculate_indicators()

    def fetch_data(self):
        # Fetch 1-minute data for one day using Yfetch
        data = Yfetch(self.ticker, self.period, self.interval)
        df = data.get_chart_dataframe()
        return df

    def calculate_indicators(self):
        indicatordata = {}
        close_series = pd.Series(self.data['Close'].values.flatten())

        # Calculate RSI
        rsi_indicator = ta.momentum.RSIIndicator(close_series)
        indicatordata['RSI'] = rsi_indicator.rsi()

        # Calculate MACD and its signal line
        macd_indicator = ta.trend.MACD(close_series)
        indicatordata['MACD'] = macd_indicator.macd()
        indicatordata['MACD_signal'] = macd_indicator.macd_signal()

        # Create a simple MACD buy signal (True when MACD crosses above MACD_signal)
        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']

        return indicatordata

    def backtest_strategy(self, initial_balance=10000):
        """
        Backtests the 1-minute data for a single day.
        The strategy enters on the first instance where:
           • MACD crosses above MACD_signal (current True and previous False)
           • RSI is below 40.
        After entry, it exits on the first instance where:
           • MACD reverses (current False, previous True)
           • RSI is above 70.
        If no sell signal is found, it liquidates at the end of the day.
        Returns:
           (entry_price, exit_price, entry_time, exit_time)
        """
        data = self.data
        indicatordata = self.indicatordata
        entry_index = None
        exit_index = None

        # Find first buy signal
        for i in range(1, len(data)):
            if (indicatordata['MACD_buy_signal'].iloc[i] and not indicatordata['MACD_buy_signal'].iloc[i-1]
                and indicatordata['RSI'].iloc[i] < 40):
                entry_index = i
                break

        if entry_index is None:
            return None

        # Find first sell signal after entry
        for i in range(entry_index + 1, len(data)):
            if (not indicatordata['MACD_buy_signal'].iloc[i] and indicatordata['MACD_buy_signal'].iloc[i-1]
                and indicatordata['RSI'].iloc[i] > 70):
                exit_index = i
                break

        if exit_index is None:
            exit_index = len(data) - 1  # Liquidate at end of day

        entry_price = data['Close'].iloc[entry_index]
        exit_price = data['Close'].iloc[exit_index]
        entry_time = data.index[entry_index]
        exit_time = data.index[exit_index]

        return entry_price, exit_price, entry_time, exit_time

def main():
    # Load Alpaca credentials from environment variables
    API_KEY = os.getenv("APCA_API_KEY_ID")
    SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    BASE_URL = os.getenv("APCA_API_BASE_URL")  # Should be "https://paper-api.alpaca.markets" for paper trading

    # Initialize the Alpaca API
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    # Choose a ticker to trade (example: MSFT)
    symbol = "MSFT"
    bot = Bot(symbol)
    signal = bot.backtest_strategy()

    if signal is None:
        print(f"No trade signal found for {symbol} today.")
        return

    entry_price, exit_price, entry_time, exit_time = signal
    print(f"Trade signal for {symbol}:")
    print(f"  Buy at {entry_price} (time: {entry_time})")
    print(f"  Sell at {exit_price} (time: {exit_time})")

    # Derive bracket order parameters.
    # We'll assume that a valid trade signal has exit_price > entry_price.
    if exit_price > entry_price:
        take_profit_price = exit_price
    else:
        # In case of a loss signal, set a modest profit target (e.g., 2% above entry)
        take_profit_price = entry_price * 1.02

    # Set a stop loss, e.g. 1% below entry.
    stop_loss_price = entry_price * 0.99

    # Define quantity to trade (for example, 1 share)
    qty = 1

    try:
        # Submit a bracket order to buy and simultaneously attach stop loss and take profit orders.
        print("Submitting bracket order...")
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day',
            order_class='bracket',
            stop_loss={
                'stop_price': round(stop_loss_price, 2),
                'limit_price': round(stop_loss_price * 0.995, 2)  # slight buffer
            },
            take_profit={
                'limit_price': round(take_profit_price, 2)
            }
        )
        print("Bracket order submitted:")
        print(order)
    except Exception as e:
        print("Error placing order:", e)

if __name__ == "__main__":
    main()
