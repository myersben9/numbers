import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Tuple
from datetime import datetime, time, timedelta, timezone
class Signal:
    def __init__(self, 
                 data: pd.DataFrame, 
                 indicatordata: Dict[str, Union[pd.Series, pd.DataFrame, Any]]) -> None: 
        self.data = data
        self.indicatordata = indicatordata

    def rsi_buy_signal(self, i: int) -> bool: 
        """
        Adjusted RSI buy condition based on historical profitable trades.
        """
        if 38 <= self.indicatordata['RSI'][i] <= 58:
            return True
        return False

    def macd_buy_signal(self, i: int) -> bool:
        """
        Adjusted MACD buy condition based on statistics.
        """
        if self.indicatordata['MACD'][i] > self.indicatordata['MACD_signal'][i] and self.indicatordata['MACD'][i] > -0.1:
            return True
        return False
    
    def atr_buy_signal(self, i: int) -> bool:
        """
        ATR-based buy signal to filter low volatility.
        """
        if self.indicatordata['ATR'][i] > 0.14:
            return True
        return False

    def stoch_buy_signal(self, i: int) -> bool:
        """
        Stochastic K buy signal.
        """
        if 40 <= self.indicatordata['Stoch_K'][i] <= 45:
            return True
        return False

    def macd_sell_signal(self, i: int) -> bool:
        """
        Adjusted MACD sell condition based on historical profitable trades.
        """
        if self.indicatordata['MACD'][i] < self.indicatordata['MACD_signal'][i] and self.indicatordata['MACD'][i] > 0.1:
            return True
        return False
    
    def _is_in_trade(self, i: int, buys: List[Tuple[pd.Timestamp, float]], sells: List[Tuple[pd.Timestamp, float]]) -> bool:
        """
        Returns True if there is an active trade
        Returns false if there is no active trade
        """
        if len(buys) == len(sells) + 1:
            return True
        elif len(buys) == len(sells):
            return False
        else:
            raise ValueError("There is an error in the number of buys and sells")

    
    def is_entry_condition(self, i: int, interval: str, buys: List[Tuple[pd.Timestamp, float]], sells: List[Tuple[pd.Timestamp, float]]) -> bool:
        """
        Return True if multiple high-quality entry conditions are met.
        """
        if self._is_in_trade(i, buys, sells):
            return False
        
        if len(buys) == len(sells) and len(buys) > 0:
            # Check if the last sell trade was made on the current index's day  
            if sells[-1][0].date() == self.data.index[i].date():
                return False

        signals = [
            self.rsi_buy_signal(i),
            self.macd_buy_signal(i),
            self.atr_buy_signal(i),
            self.stoch_buy_signal(i)
        ]

        if sum(signals) >= 2 and self.is_market_open(i, interval):
            return True
        return False
    
    def is_market_open(self: 'Signal', i: int, interval: str) -> bool:
        """
        Check if a UTC datetime is within U.S. market hours for UTC time.
        """
        dt_utc = pd.Timestamp(self.data.index[i]).tz_convert('UTC')

        market_open = time(14, 30)
        market_close = time(21, 0)

        minuteint = int(interval.replace('m', ''))
        
        today = datetime.now().date()
        market_close_dt = datetime.combine(today, market_close)
        adjusted_close_dt = market_close_dt - timedelta(minutes=minuteint)
        adjusted_close_time = adjusted_close_dt.time()
        
        if market_open <= dt_utc.time() <= adjusted_close_time:
            return True
        return False
    
    def _get_last_market_close_index(self: 'Signal', i: int, interval: str) -> int:
        # Get the index and check if the market is open
        if not self.is_market_open(i, interval):
            return i

    def _is_last_daytrade(self, i: int, interval: str) -> bool:
        """
        Return True if the current time is the last daytrade.
        """
        if i == self._get_last_market_close_index(i, interval):
            return True
        return False
    
    
    def is_exit_condition(self, i: int, interval: str, buys: List[Tuple[pd.Timestamp, float]], sells: List[Tuple[pd.Timestamp, float]]) -> bool:
        """
            Improved exit condition with multiple factors
        """
        # Check if there is already a buy signal
        if not self._is_in_trade(i, buys, sells):
            return False

        if self.macd_sell_signal(i):
            return True
        return False

