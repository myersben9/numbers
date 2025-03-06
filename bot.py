from yfetch import Yfetch
from movers import TopMovers
from signals import Signal
from matplotlib.dates import DateFormatter
from datetime import time, datetime

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

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
                 range: str = "1d",
                 interval: str = "5m",
                 pre_post: bool = False,
                 start_date: str = None,
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
                    timezone = self.time_zone).df
        return df
    
    def _calculate_indicators(self: 'Bot') -> Dict[str, Any]:
        indicatordata = {}
        rsi = RSIIndicator(pd.Series(self.data['Close'].values.flatten())).rsi()
        indicatordata['RSI'] = rsi

        # Trend Indicators
        macd = MACD(pd.Series(self.data['Close'].values.flatten())).macd()
        indicatordata['MACD'] = macd
        macd_signal = MACD(pd.Series(self.data['Close'].values.flatten())).macd_signal()
        indicatordata['MACD_signal'] = macd_signal
        close_prices = pd.Series(self.data['Close'].values.flatten())

        # Volatility Indicators
        bollinger = BollingerBands(close_prices, window=20, window_dev=2)
        indicatordata['Bollinger_hband'] = bollinger.bollinger_hband()
        indicatordata['Bollinger_lband'] = bollinger.bollinger_lband()
        indicatordata['Bollinger_mavg'] = bollinger.bollinger_mavg()
        indicatordata['Parkinsons_Volatility'] = self.parkinsons_volatility(
            pd.Series(self.data['High'].values.flatten()), 
            pd.Series(self.data['Low'].values.flatten()), 
            window=20
        )

        # Check if there are atleast 14 rows in the dataframe
        if len(self.data) < 14:
            indicatordata['ATR'] = pd.Series(np.nan, index=self.data.index)
        else:
            indicatordata['ATR'] = AverageTrueRange(
                high=pd.Series(self.data['High'].values.flatten()), 
                low=pd.Series(self.data['Low'].values.flatten()), 
                close=pd.Series(self.data['Close'].values.flatten()),
                window=14,
                fillna=False
        ).average_true_range()
        # Volume Indicators


        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']
        indicatordata['signal'] = [0] * len(indicatordata['MACD_buy_signal']) 

        # Calculate the 20-day SMA
        sma_20 = self.data['Close'].rolling(window=20).mean()
        # Create a new column for the 20-day SMA
        indicatordata['SMA20'] = sma_20

        buy_coords, sell_coords = self.process_bars(indicatordata)
                
        indicatordata['buy_coords'] = buy_coords
        indicatordata['sell_coords'] = sell_coords

        return indicatordata

    def process_bars(self, indicatordata: pd.DataFrame) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
        """
            Process each bar sequentially (as in real time).
            When not in a trade, check for an entry condition.
            When in a trade, on each new bar, check if an exit condition is met—
            either by your indicator-based sell signals OR if the current price hits the stop loss or take profit.
        """
        buy_coords = []
        sell_coords = []
        in_trade = False
        entry_index = None
        entry_price = None

        signal = Signal(self.data, indicatordata)

        for i in range(1, len(indicatordata['MACD_buy_signal'])):
            current_time = self.data.index[i]
            if not self.is_market_open(current_time):
                continue

            # If not in a trade, check for entry.
            if not in_trade:
                if signal.is_entry_condition(i):
                    entry_index = i
                    entry_price = self.data['Close'].iloc[i]
                    buy_coords.append((current_time, entry_price))
                    in_trade = True
            else:
                # In a trade, calculate stop loss and take profit based on entry.
                atr_value = indicatordata['ATR'].iloc[entry_index]
                stop_loss = signal._get_stop_loss(entry_price, atr_value, multiplier=1.5)
                take_profit = signal._get_take_profit(entry_price, atr_value, multiplier=2.0)
                current_price = self.data['Close'].iloc[i]

                # Check indicator-based exit
                indicator_exit = (signal._get_macd_sell_signal(i) or
                                  signal._get_rsi_sell_signal(i) or
                                  signal._get_bollinger_sell_signal(i) or
                                  signal._get_extra_sell_filter(i))
                
                if indicator_exit or (current_price <= stop_loss) or (current_price >= take_profit):
                    sell_coords.append((current_time, current_price))
                    in_trade = False  # Exit the trade

            # Optionally, if market close is reached and you're in a trade, force exit.
            if in_trade and current_time == self._get_last_market_close_index():
                forced_exit_price = self.data['Close'].iloc[i]
                sell_coords.append((current_time, forced_exit_price))
                in_trade = False

        return buy_coords, sell_coords
    
    def is_market_open(self: 'Bot', dt: datetime) -> bool:
        """
        Check if a UTC datetime is within U.S. market hours for UTC time.
        """
        dt_utc = pd.Timestamp(dt).tz_convert('UTC')
        market_open = time(14, 30)
        market_close = time(21, 0)
        return market_open <= dt_utc.time() <= market_close
    
    def _get_last_market_close_index(self: 'Bot') -> datetime:
        market_times = [t for t in self.data.index if self.is_market_open(t)]
        return market_times[-1] if market_times else None
    
    def plot_graphs(self: 'Bot', ticker: str) -> None:
        # Create subplots with a shared x-axis.
        fig, axs = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
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
        axs[3].plot(self.data.index, self.indicatordata['ATR'], label='ATR', color='purple')
        axs[3].legend()

        # Subplot 4: Parkinson's Volatility
        axs[4].plot(self.data.index, self.indicatordata['Parkinsons_Volatility'], label="Parkinson's Volatility", color='orange')
        axs[4].legend()
        axs[4].set_xlabel('Time')

        # Create a DateFormatter that shows only hours and minutes.
        time_formatter = DateFormatter("%H:%M", tz=pytz.timezone(self.time_zone))
        # Apply the formatter to the shared x-axis by setting it on the last subplot.
        axs[-1].xaxis.set_major_formatter(time_formatter)
        # Auto-format the x-axis tick labels for better readability.
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig('stockgraphs/' + ticker + '_graph.png', dpi=300)



if __name__ == "__main__":
    bot = Bot(ticker="AAPL", pre_post=False, range="1d", interval="5m")
    bot.plot_graphs("AAPL")


