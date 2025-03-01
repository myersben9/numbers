

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

    def calculate_relative_minima(self, rsis):
        """Calculate the relative minima of the RSI"""

        if 'RSI' not in rsis:
            return None
            
        # Invert RSI to find minima (find_peaks looks for maxima by default)
        inverted_rsi = -1 * rsis
        
        # Find peaks of the inverted RSI (which are minima of the original RSI)
        # Adjust the prominence and distance parameters to control sensitivity
        peaks, _ = find_peaks(inverted_rsi, prominence=5, distance=5)
        
        relative_minima = []
        for peak in peaks:
            relative_minima.append(rsis[peak])
        
        # Find the minimum of the relative minima
        minimum = min(relative_minima)
        return minimum

    def fetch_data(self):
        # Get the data from yfinance
        data = Yfetch(self.ticker, self.period, self.interval)
        df = data.get_chart_dataframe()
        print(df)   
        return df
    
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
            # Find the relative minimum of the RSI at this index
            rsi_min = self.calculate_relative_minima(indicatordata['RSI'])
            print(rsi_min)
            if indicatordata['MACD_buy_signal'][i] == True and indicatordata['MACD_buy_signal'][i-1] == False:
                buy_coords.append((self.data.index[i], self.data['Close'].iloc[i]))
            elif indicatordata['MACD_buy_signal'][i] == False and indicatordata['MACD_buy_signal'][i-1] == True:
                sell_coords.append((self.data.index[i], self.data['Close'].iloc[i]))

        indicatordata['buy_coords'] = buy_coords
        indicatordata['sell_coords'] = sell_coords
        
        return indicatordata

    

    def plot_graphs(self):
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
        plt.show()


    
# Test the class
if __name__ == "__main__":
    bot = Bot("NVDA")
    bot.plot_graphs()

