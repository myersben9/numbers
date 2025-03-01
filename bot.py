

from yfetch import Yfetch
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Bot:
    def __init__(self, ticker):
        self.ticker = ticker
        self.period = "1d"
        self.interval = "1m"
        self.data = self.fetch_data()
        self.indicatordata = self.calculate_indicators()

    def fetch_data(self):
        # Get the data from yfinance
        data = Yfetch(self.ticker, self.period, self.interval)
        df = data.get_chart_dataframe()
        return df
    
    def rsi_buy_signal(self, indicatordata, i):
        if indicatordata['RSI'][i] < 40:
            return True
        
    
    def rsi_sell_signal(self, indicatordata, i):
        if indicatordata['RSI'][i] > 70:
            return True
    
    # Write a function that returns true or false
    def macd_buy_signal(self, indicatordata, i):
        # First check if the the macd signal and macd are both less than 0.01
        if indicatordata['MACD_signal'][i] < 0 and indicatordata['MACD'][i] < 0:
            if indicatordata['MACD_buy_signal'][i] == True and indicatordata['MACD_buy_signal'][i-1] == False:
                return True
            
    def macd_sell_signal(self, indicatordata, i):
        # First check if there was already a buy signal at or before this index
        if indicatordata['MACD_buy_signal'][i] == False and indicatordata['MACD_buy_signal'][i-1] == True:
            # Check if there is a buy signal at or before this index
            return True
        
    def calculate_indicators(self):

        indicatordata = {}

        rsi = ta.momentum.RSIIndicator(pd.Series(self.data['Close'].values.flatten())).rsi()
        indicatordata['RSI'] = rsi

        macd = ta.trend.MACD(pd.Series(self.data['Close'].values.flatten())).macd()
        indicatordata['MACD'] = macd
        macd_signal = ta.trend.MACD(pd.Series(self.data['Close'].values.flatten())).macd_signal()
        indicatordata['MACD_signal'] = macd_signal
        close_prices = pd.Series(self.data['Close'].values.flatten())
        bollinger = ta.volatility.BollingerBands(close_prices, window=20, window_dev=2)

        indicatordata['Bollinger_hband'] = bollinger.bollinger_hband()
        indicatordata['Bollinger_lband'] = bollinger.bollinger_lband()
        indicatordata['Bollinger_mavg'] = bollinger.bollinger_mavg()

        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']
        # indicatordata['MACD_sell_signal'] = indicatordata['MACD'] < indicatordata['MACD_signal']
        # Calculate the relative minimum or maximum of the RSI at a particular index

        indicatordata['signal'] = [0] * len(indicatordata['MACD_buy_signal'])   

        buy_coords = []
        sell_coords = []
        for i in range(1, len(indicatordata['MACD_buy_signal'])):
            if self.macd_buy_signal(indicatordata, i) and self.rsi_buy_signal(indicatordata, i):
                buy_coords.append((self.data.index[i], self.data['Close'].iloc[i]))
            elif self.macd_sell_signal(indicatordata, i) and self.rsi_sell_signal(indicatordata, i):
                sell_coords.append((self.data.index[i], self.data['Close'].iloc[i]))

        indicatordata['buy_coords'] = buy_coords
        indicatordata['sell_coords'] = sell_coords
        
        return indicatordata

    

    def plot_graphs(self, ticker):
        fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        
        axs[0].plot(self.data.index, self.data['Close'], label='Price')
        # Plot the buy and sell coords
        for coord in self.indicatordata['buy_coords']:
            axs[0].scatter(coord[0], coord[1], color='green', marker='^')
        for coord in self.indicatordata['sell_coords']:
            axs[0].scatter(coord[0], coord[1], color='red', marker='v')

        axs[0].plot(self.data.index, self.indicatordata['Bollinger_hband'], label='Bollinger_hband', color='grey', linestyle='--')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_lband'], label='Bollinger_lband', color='grey', linestyle='--')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_mavg'], label='Bollinger_mavg', color='grey', linestyle='-.')

        axs[0].fill_between(self.data.index, self.indicatordata['Bollinger_hband'], self.indicatordata['Bollinger_lband'], color='grey', alpha=0.2)
        axs[0].legend()
        axs[1].plot(self.data.index, self.indicatordata['RSI'], label='RSI')
        axs[1].legend()
        axs[2].plot(self.data.index, self.indicatordata['MACD'], label='MACD')
        axs[2].plot(self.data.index, self.indicatordata['MACD_signal'], label='MACD_signal')
        axs[2].legend()
        n = len(self.data.index)
        step = max(1, n // 12)  # Show about 12 ticks on the x-axis
        plt.xticks(self.data.index[::step], rotation=45)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Download as high dpi png
        plt.savefig('stockgraphs/' + ticker + '_graph.png', dpi=300)
        # plt.show()


    
# Test the class
if __name__ == "__main__":

    tickers = ["PRGO"
    ]

    # Test on all the tickers
    for ticker in tickers:
        bot = Bot(ticker)
        bot.plot_graphs(ticker)

