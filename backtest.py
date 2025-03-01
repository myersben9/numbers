from yfetch import Yfetch
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from movers import TopMovers

class Bot:
    def __init__(self, ticker):
        self.ticker = ticker
        self.period = "1d"      # one day of data
        self.interval = "1m"    # 1-minute resolution
        self.data = self.fetch_data()
        self.indicatordata = self.calculate_indicators()

    def fetch_data(self):
        # Fetch data using Yfetch (assumed similar to yfinance)
        data = Yfetch(self.ticker, self.period, self.interval)
        df = data.get_chart_dataframe()
        return df

    # Signal functions
    def rsi_buy_signal(self, indicatordata, i):
        # Buy if RSI is below 40
        return indicatordata['RSI'][i] < 40

    def rsi_sell_signal(self, indicatordata, i):
        # Sell if RSI is above 70
        return indicatordata['RSI'][i] > 70

    def macd_buy_signal(self, indicatordata, i):
        # Check for a MACD crossover: current bar is bullish and previous bar was not
        # Optionally, check that both MACD and MACD_signal are negative (or within a certain range)
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

        # Bollinger Bands for plotting/reference
        bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        indicatordata['Bollinger_hband'] = bb.bollinger_hband()
        indicatordata['Bollinger_lband'] = bb.bollinger_lband()
        indicatordata['Bollinger_mavg'] = bb.bollinger_mavg()

        # Create a basic MACD buy signal: True when MACD is above its signal line
        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']

        return indicatordata

    def backtest_strategy(self, initial_balance=10000):
        """
        Backtest the strategy for one day (1-minute data for 1 day).
        The strategy makes only one buy and one sell signal:
          - It enters on the first instance where both MACD and RSI conditions are met.
          - It then looks for the first sell signal after entry.
          - If no sell signal is found, the position is liquidated at the end of the day.
        Returns a trade log, final balance, and trade return.
        """
        data = self.data
        indicatordata = self.indicatordata
        balance = initial_balance
        trade_log = []
        entry_index = None
        exit_index = None

        # Find all buy signals
        buy_signals = [i for i in range(1, len(data)) if self.macd_buy_signal(indicatordata, i) and self.rsi_buy_signal(indicatordata, i)]

        if not buy_signals:
            # No buy signals found, no trade executed
            return None, balance, 0

        # Take the last buy signal
        entry_index = buy_signals[-1]

        entry_price = data['Close'].iloc[entry_index]
        trade_log.append(("BUY", entry_price, data.index[entry_index]))

        # Find the first sell signal after the buy signal
        for i in range(entry_index + 1, len(data)):
            if self.macd_sell_signal(indicatordata, i) and self.rsi_sell_signal(indicatordata, i):
                exit_index = i
                break

        # If no sell signal is found, liquidate at the end of the day
        if exit_index is None:
            exit_index = len(data) - 1

        exit_price = data['Close'].iloc[exit_index]
        trade_log.append(("SELL", exit_price, data.index[exit_index]))

        # Calculate profit and performance
        profit = exit_price - entry_price
        trade_return = profit / entry_price
        final_balance = balance * (1 + trade_return)

        return trade_log, final_balance, trade_return


# Main execution: backtest the strategy on multiple tickers and save performance summary to CSV.
if __name__ == "__main__":
    tickers = TopMovers().symbols
    print(tickers)
    results = []
    for ticker in tickers:
        try:
            bot = Bot(ticker)
            trade_log, final_balance, trade_return = bot.backtest_strategy(initial_balance=10000)
            if trade_log is None:
                results.append({
                    "Ticker": ticker,
                    "Trade Executed": False,
                    "Final Balance": 10000,
                    "Trade Return": 0
                })
            else:
                results.append({
                    "Ticker": ticker,
                    "Trade Executed": True,
                    "Final Balance": final_balance,
                    "Trade Return": trade_return
                })
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Trade Executed": "Error",
                "Final Balance": None,
                "Trade Return": None,
                "Error": str(e)
            })
    
    df_results = pd.DataFrame(results)
    csv_filename = "backtest_summary.csv"
    df_results.to_csv(csv_filename, index=False)
    print("Backtest summary saved as", csv_filename)
