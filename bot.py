from yfetch import Yfetch
from movers import TopMovers
from measures import Measures

from signals import Signal
from matplotlib.dates import DateFormatter
from datetime import time, datetime
import ta
from ta.trend import MACD, IchimokuIndicator, ADXIndicator, EMAIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator
from ta.trend import CCIIndicator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytz
import numpy as np
from typing import Dict, List, Tuple, Any

class Bot:
    def __init__(self: 'Bot', 
                 ticker: str, 
                 interval: str = "5m",
                 pre_post: bool = False,
                 start_date: str = None,
                 range: str = None,
                 end_date: str = None,
                 time_zone: str = 'America/Los_Angeles') -> None:
        self.ticker: str = ticker # type: str
        self.range: str = range # type: str
        self.interval: str = interval # type: str
        self.pre_post: bool = pre_post # type: bool
        self.start_date: str = start_date # type: str
        self.end_date: str = end_date # type: str
        self.time_zone: str = time_zone # type: str
        self.data: pd.DataFrame = self._fetch_df() # type: pd.DataFrame 
        self.indicatordata: Dict[str, Any] = self._calculate_indicators() # type: Dict[str, Any] | None

    def _fetch_df(self: 'Bot') -> pd.DataFrame:
        # Get the data from yfinance
        df = Yfetch(self.ticker, 
                    interval = self.interval,  
                    pre_post = self.pre_post,
                    start_date = self.start_date,
                    end_date = self.end_date,
                    range = self.range,
                    timezone = self.time_zone).df
        return df
    
    def _calculate_indicators(self: 'Bot') -> Dict[str, Any]:
        """
            Calculate the indicators for the data.
        """
        try:
            indicatordata = {}
            close_prices = pd.Series(self.data['Close'].values.flatten())
            high_prices = pd.Series(self.data['High'].values.flatten())
            low_prices = pd.Series(self.data['Low'].values.flatten())
            volume = pd.Series(self.data['Volume'].values.flatten())

            # If there are less than 14 rows in the dataframe, return None
            if len(self.data) < 14:
                raise Exception("Not enough data to calculate indicators")

            # Momentum Indicators
            rsi = RSIIndicator(close_prices).rsi()
            indicatordata['RSI'] = rsi

            # Stochastic Oscillator
            stochastic = StochasticOscillator(high_prices,
                                                    low_prices, 
                                                    close_prices, 
                                                    window=14, 
                                                    smooth_window=3, 
                                                    fillna=False)
            indicatordata['Stochastic_K'] = stochastic.stoch()
            indicatordata['Stochastic_D'] = stochastic.stoch_signal()
        
            cci = CCIIndicator(high=high_prices, low=low_prices, close=close_prices, window=20)
            indicatordata['CCI'] = cci.cci()
        

            # Trend Indicators
            indicatordata['MACD'] = MACD(close_prices).macd()
            indicatordata['MACD_signal'] = MACD(close_prices).macd_signal()
            indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']

            # Volatility Indicators
            bollinger = BollingerBands(close_prices, window=20, window_dev=2)
            indicatordata['Bollinger_hband'] = bollinger.bollinger_hband()
            indicatordata['Bollinger_lband'] = bollinger.bollinger_lband()
            indicatordata['Bollinger_mavg'] = bollinger.bollinger_mavg()
            indicatordata['Parkinsons_Volatility'] = Measures._get_parkinsons_volatility(
                pd.Series(self.data['High'].values.flatten()), 
                pd.Series(self.data['Low'].values.flatten()), 
                window=20
            )

            buy_coords, sell_coords = self.process_bars(indicatordata)
                    
            indicatordata['buy_coords'] = buy_coords
            indicatordata['sell_coords'] = sell_coords

            return indicatordata
        
        except Exception as e:
            print(e)
            return None

    def process_bars(self, indicatordata: pd.DataFrame) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
            """
                Process each bar sequentially (as in real time).
                When not in a trade, check for an entry condition.
                When in a trade, on each new bar, check if an exit condition is met—
                either by your indicator-based sell signals OR if the current price hits the stop loss or take profit.
            """
            buy_coords = []
            sell_coords = []

            signal = Signal(self.data, indicatordata)

            for i in range(1, len(indicatordata['MACD_buy_signal'])):

                current_time = self.data.index[i]
                current_price = self.data['Close'].iloc[i]

                if not len(buy_coords) > 0 and signal.is_entry_condition(i, self.interval):
                    buy_coords.append((current_time, current_price))

                if len(buy_coords) > 0 and signal.is_exit_condition(i, self.interval):
                    sell_coords.append((current_time, current_price))

            return buy_coords, sell_coords

    
    def plot_graphs(self: 'Bot', ticker: str) -> None:
        # Create subplots with a shared x-axis.
        fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        axs: List[Axes] = axs
        axs[0].set_title(f'{ticker} Chart')
        axs[0].plot(self.data.index, self.data['Close'], label='Price', color='blue')
        for coord in self.indicatordata['buy_coords']:
            axs[0].scatter(coord[0], coord[1], color='green', marker='^', s=100, label='Buy Signal')
        for coord in self.indicatordata['sell_coords']:
            axs[0].scatter(coord[0], coord[1], color='red', marker='v', s=100, label='Sell Signal')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_hband'], label='Bollinger_hband', color='green', linestyle='--')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_lband'], label='Bollinger_lband', color='red', linestyle='--')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_mavg'], label='Bollinger_mavg', color='grey', linestyle='-.')
        axs[0].fill_between(self.data.index, self.indicatordata['Bollinger_hband'], self.indicatordata['Bollinger_lband'], color='grey', alpha=0.2)
        
        # Subplot 1: RSI
        axs[1].plot(self.data.index, self.indicatordata['RSI'], label='RSI', color='purple')
        axs[1].legend()

        # Subplot 2: MACD
        axs[2].plot(self.data.index, self.indicatordata['MACD'], label='MACD', color='blue')
        axs[2].plot(self.data.index, self.indicatordata['MACD_signal'], label='MACD_signal', color='orange')
        axs[2].legend()

        # Subplot 3: ATR
        # axs[3].plot(self.data.index, self.indicatordata['ATR'], label='ATR', color='purple')
        # axs[3].legend()

        # Subplot 4: Parkinson's Volatility
        axs[3].plot(self.data.index, self.indicatordata['Parkinsons_Volatility'], label="Parkinson's Volatility", color='orange')
        axs[3].legend()
        axs[3].set_xlabel('Time')

        # Create a DateFormatter that shows only hours and minutes.
        time_formatter = DateFormatter("%H:%M", tz=pytz.timezone(self.time_zone))
        # Apply the formatter to the shared x-axis by setting it on the last subplot.
        axs[-1].xaxis.set_major_formatter(time_formatter)
        # Auto-format the x-axis tick labels for better readability.
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig('plots/' + ticker + '_graph.png', dpi=300)


if __name__ == "__main__":
    bot = Bot(ticker="AAPL", pre_post=True, range="1d", interval="5m")
    bot.plot_graphs("AAPL")


