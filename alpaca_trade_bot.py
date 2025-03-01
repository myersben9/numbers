import os
import time
import math
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import ta
from yfetch import Yfetch
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

class Bot:
    def __init__(self, ticker):
        self.ticker = ticker
        self.period = "1d"      # one day of data
        self.interval = "1m"    # 1-minute resolution
        self.data = self.fetch_data()
        self.indicatordata = self.calculate_indicators()

    def fetch_data(self):
        # Fetch 1-minute data for one day using Yfetch
        data = Yfetch(self.ticker, self.period, self.interval)
        df = data.get_chart_dataframe()
        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    # Signal functions
    def rsi_buy_signal(self, indicatordata, i):
        # Buy if RSI is below 40
        return indicatordata['RSI'][i] < 40

    def rsi_sell_signal(self, indicatordata, i):
        # Sell if RSI is above 70
        return indicatordata['RSI'][i] > 70

    def macd_buy_signal(self, indicatordata, i):
        # Check for a MACD crossover: current bar is bullish and previous bar was not.
        # Here, we also check that both MACD and MACD_signal are negative.
        if indicatordata['MACD_signal'][i] < 0 and indicatordata['MACD'][i] < 0:
            if indicatordata['MACD_buy_signal'][i] and not indicatordata['MACD_buy_signal'][i-1]:
                return True
        return False

    def macd_sell_signal(self, indicatordata, i):
        # Sell when MACD_buy_signal turns False (i.e. a reversal)
        if not indicatordata['MACD_buy_signal'][i] and indicatordata['MACD_buy_signal'][i-1]:
            return True
        return False

    def calculate_indicators(self):
        indicatordata = {}
        # Use the Close prices
        close_series = pd.Series(self.data['Close'].values.flatten())

        # Calculate RSI
        rsi_indicator = ta.momentum.RSIIndicator(close_series)
        indicatordata['RSI'] = rsi_indicator.rsi()

        # Calculate MACD and its signal
        macd_indicator = ta.trend.MACD(close_series)
        indicatordata['MACD'] = macd_indicator.macd()
        indicatordata['MACD_signal'] = macd_indicator.macd_signal()

        # Bollinger Bands for reference (optional)
        bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        indicatordata['Bollinger_hband'] = bb.bollinger_hband()
        indicatordata['Bollinger_lband'] = bb.bollinger_lband()
        indicatordata['Bollinger_mavg'] = bb.bollinger_mavg()

        # Create a basic MACD buy signal: True when MACD is above its signal line
        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']

        return indicatordata

    def backtest_strategy(self, initial_balance=10000):
        """
        Backtests the strategy for one day using 1-minute data.
        It enters on the first instance where:
          • MACD crosses above its signal (current True, previous False) AND RSI < 40.
        It exits on the first instance after entry where:
          • MACD reverses (current False, previous True) AND RSI > 70.
        If no sell signal is found, it liquidates at the end of the day.
        Returns:
           (entry_price, exit_price, entry_time, exit_time)
        """
        data = self.data
        indicatordata = self.indicatordata
        entry_index = None
        exit_index = None

        # Find the first buy signal
        for i in range(1, len(data)):
            if self.macd_buy_signal(indicatordata, i) and self.rsi_buy_signal(indicatordata, i):
                entry_index = i
                break

        if entry_index is None:
            # No trade signal found for the day
            return None

        # Find the first sell signal after entry
        for i in range(entry_index + 1, len(data)):
            if self.macd_sell_signal(indicatordata, i) and self.rsi_sell_signal(indicatordata, i):
                exit_index = i
                break

        # If no sell signal, liquidate at the end of the day
        if exit_index is None:
            exit_index = len(data) - 1

        entry_price = data['Close'].iloc[entry_index]
        exit_price = data['Close'].iloc[exit_index]
        entry_time = data.index[entry_index]
        exit_time = data.index[exit_index]

        return entry_price, exit_price, entry_time, exit_time

def main():
    # Define starting capital
    INITIAL_BALANCE = 1000  # $1000
    RISK_PERCENT = 0.01     # Risk 1% of capital ($10)
    
    # Load Alpaca credentials from environment variables
    API_KEY = os.getenv("APCA_API_KEY_ID")
    SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    BASE_URL = os.getenv("APCA_API_BASE_URL")  # Use "https://paper-api.alpaca.markets" for paper trading

    # Initialize the Alpaca API
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    # Choose the ticker to trade (example: MSFT)
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

    # Determine stop loss and take profit based on backtest.
    # Here we set a stop loss 1% below entry.
    stop_loss_price = entry_price * 0.99
    # Use the backtest exit price if it represents a profit; otherwise, set a modest target (e.g., 2% above entry).
    if exit_price > entry_price:
        take_profit_price = exit_price
    else:
        take_profit_price = entry_price * 1.02

    # Calculate risk per share (difference between entry and stop loss)
    risk_per_share = entry_price - stop_loss_price  # This is 1% of entry_price
    # Determine the number of shares to trade such that risk is RISK_PERCENT * INITIAL_BALANCE
    risk_amount = INITIAL_BALANCE * RISK_PERCENT
    qty = math.floor(risk_amount / risk_per_share)
    
    # Ensure you do not buy more shares than your account can afford:
    affordable_qty = math.floor(INITIAL_BALANCE / entry_price)
    qty = min(qty, affordable_qty)
    
    if qty < 1:
        print(f"Not enough capital to risk. Calculated quantity: {qty}")
        return

    print(f"Calculated quantity to trade: {qty} shares")

    try:
        # Submit a bracket order: this submits a primary market order with attached stop loss and take profit orders.
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
                'limit_price': round(stop_loss_price * 0.995, 2)  # slight buffer below stop
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
