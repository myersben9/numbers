from yfetch import Yfetch
import ta
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import time
from movers import TopMovers

class LiveTradingBot:
    def __init__(self, ticker, alpaca_api, allocation_per_ticker):
        self.ticker = ticker
        self.alpaca_api = alpaca_api
        self.allocation_per_ticker = allocation_per_ticker
        self.period = "1d"
        self.interval = "1m"
        self.data = self.fetch_data()
        self.indicatordata = self.calculate_indicators()
        self.position = None
        self.entry_price = None
        self.received_buy_signal = False

    def fetch_data(self):
        data = Yfetch(self.ticker, self.period, self.interval)
        df = data.get_chart_dataframe()
        return df

    def calculate_indicators(self):
        indicatordata = {}
        close_series = pd.Series(self.data['Close'].values.flatten())

        rsi_indicator = ta.momentum.RSIIndicator(close_series)
        indicatordata['RSI'] = rsi_indicator.rsi()

        macd_indicator = ta.trend.MACD(close_series)
        indicatordata['MACD'] = macd_indicator.macd()
        indicatordata['MACD_signal'] = macd_indicator.macd_signal()

        bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        indicatordata['Bollinger_hband'] = bb.bollinger_hband()
        indicatordata['Bollinger_lband'] = bb.bollinger_lband()
        indicatordata['Bollinger_mavg'] = bb.bollinger_mavg()

        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']

        return indicatordata

    def rsi_buy_signal(self, i):
        return self.indicatordata['RSI'][i] < 40

    def rsi_sell_signal(self, i):
        return self.indicatordata['RSI'][i] > 70

    def macd_buy_signal(self, i):
        if self.indicatordata['MACD_signal'][i] < 0 and self.indicatordata['MACD'][i] < 0:
            if self.indicatordata['MACD_buy_signal'][i] and not self.indicatordata['MACD_buy_signal'][i-1]:
                return True
        return False

    def macd_sell_signal(self, i):
        if not self.indicatordata['MACD_buy_signal'][i] and self.indicatordata['MACD_buy_signal'][i-1]:
            return True
        return False

    def execute_trade(self, action, price, qty):
        print(f"Executing {action} trade for {self.ticker} at ${price} with quantity {qty}")
        if action == "BUY":
            self.alpaca_api.submit_order(
                symbol=self.ticker,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            self.entry_price = price
            self.position = True
        elif action == "SELL":
            self.alpaca_api.submit_order(
                symbol=self.ticker,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            self.position = False

    def run(self):
        buy_signals = [i for i in range(1, len(self.data)) if self.macd_buy_signal(i) and self.rsi_buy_signal(i)]
        if buy_signals:
            self.received_buy_signal = True
            entry_index = buy_signals[0]  # Take the first buy signal
            entry_price = self.data['Close'].iloc[entry_index]
            qty = max(1, int(self.allocation_per_ticker / entry_price))  # Determine the number of shares to buy
            self.execute_trade("BUY", entry_price, qty)
            stop_loss = entry_price * 0.99

            for i in range(entry_index + 1, len(self.data)):
                if self.macd_sell_signal(i) and self.rsi_sell_signal(i):
                    self.execute_trade("SELL", self.data['Close'].iloc[i], qty)
                    break
                elif self.data['Close'].iloc[i] <= stop_loss:
                    print(f"Stop loss triggered at ${self.data['Close'].iloc[i]}")
                    self.execute_trade("SELL", self.data['Close'].iloc[i], qty)
                    break

if __name__ == "__main__":
    alpaca_api = tradeapi.REST("API_KEY", "API_SECRET", "https://paper-api.alpaca.markets", api_version='v2')
    tickers = TopMovers().symbols
    total_fund = 1000
    tradable_tickers = []
    buy_signal_tickers = []

    for ticker in tickers:
        try:
            asset = alpaca_api.get_asset(ticker)
            if asset.tradable:
                bot = LiveTradingBot(ticker, alpaca_api, 0)  # Temporary allocation
                bot.run()
                if bot.received_buy_signal:
                    buy_signal_tickers.append(ticker)
                    tradable_tickers.append(bot)
        except Exception as e:
            print(f"Error checking {ticker}: {e}")
    
    if tradable_tickers:
        allocation_per_ticker = total_fund / len(tradable_tickers)
        for bot in tradable_tickers:
            bot.allocation_per_ticker = allocation_per_ticker
            bot.run()
    else:
        print("No tradable tickers with buy signals found.")
