import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Tuple
from datetime import datetime, time, timedelta, timezone
class Signal:
    def __init__(self, 
                 data: pd.DataFrame, 
                 indicatordata: Dict[str, Union[pd.Series, pd.DataFrame, Any]],
                 buy_coords: List[Tuple[pd.Timestamp, float]],
                 sell_coords: List[Tuple[pd.Timestamp, float]],
                 recent_only: bool = False) -> None: 
        self.data = data
        self.indicatordata = indicatordata
        self.buy_coords = buy_coords
        self.sell_coords = sell_coords
        self.recent_only = recent_only

    def rsi_buy_signal(self, i: int) -> bool: 
        if self.indicatordata['RSI'][i] < 40:
            return True
        return False
    
    def rsi_sell_signal(self, i: int) -> bool:
        if self.indicatordata['RSI'][i] > 70:
            return True
    
    # Write a function that returns true or false
    def macd_buy_signal(self, i: int) -> bool:

        if self.indicatordata['MACD_buy_signal'][i] == True and self.indicatordata['MACD_buy_signal'][i-1] == False:
            return True
        return False
            
    def macd_sell_signal(self, i: int) -> bool:
        # First check if there was already a buy signal at or before this index
        if self.indicatordata['MACD_buy_signal'][i] == False and self.indicatordata['MACD_buy_signal'][i-1] == True:
            return True
        return False
    
    def _is_daily_enter(self, i: int) -> List[Tuple[pd.Timestamp, float]]:
        """
        Function to check if there is already a buy signal at or before this index on the same day
        Returns True if there is a buy signal at or before this index on the same day
        """
        # Get the date of the current index
        # Type index as datetime
        current_date : datetime = self.data.index[i]
        # Get the buy coordinates for the current date
        buy_coords = [coord for coord in self.buy_coords if coord[0].date() == current_date.date()]
        # Return the buy coordinates
        if len(buy_coords) > 0:
            return True
        return False

    def is_entry_condition(self, i: int, interval: str) -> bool:
        """
        Return True if multiple high-quality entry conditions are met.
        """

        # Check if there is already a buy signal
        if self._is_daily_enter(i):
            return False

        # Check if the current time is recent only
        if self._is_recent_only(i):
            return False

        signals = [
            self.rsi_buy_signal(i),
            self.macd_buy_signal(i)
        ]

        conditions = [
            self.is_market_open(i, interval),
            self._is_daily_enter(i)
        ]
        if sum(signals) > 0 and sum(conditions) > 0:
            return True
        return False
    
    def _get_take_profit_signal(self, i: int) -> bool:
        """
        Return True if the take profit signal is met.
        """
        if self.data['Close'].iloc[i] > self.data['High'].iloc[i-1] * 1.05:
            return True
        return False
    
    def _get_stop_loss_signal(self, i: int) -> bool:
        """
        Return True if the stop loss signal is met.
        """
        if self.data['Close'].iloc[i] < self.data['Low'].iloc[i-1] * 0.95:
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
        
        return market_open <= dt_utc.time() <= adjusted_close_time
    
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
    
    def _is_recent_only(self, i: int) -> bool:
        """
        Return True if the current time minus 30 minutes is greater than the current time
        """
        current_time : datetime = self.data.index[i]

        # Subtract 30 minutes from the current time
        thirty_minutes_ago : datetime = current_time - timedelta(minutes=30)

        if self.recent_only:
            if thirty_minutes_ago < datetime.now(timezone.utc):
                return True
        return False
    
    def _is_daily_exit(self, i: int) -> bool:
        """
        Function to check if there is already a sell signal at or before this index on the same day
        Returns True if there is a sell signal at or before this index on the same day
        """
        # Get the date of the current index
        current_date : datetime = self.data.index[i]
        # Get the sell coordinates for the current date
        sell_coords = [coord for coord in self.sell_coords if coord[0].date() == current_date.date()]
        # Return the sell coordinates
        if len(sell_coords) > 0:
            return True
        return False
    
    def is_exit_condition(self, i: int, interval: str) -> bool:
        """
            Improved exit condition with multiple factors
        """
        # Check if there is already a sell signal

        # Check if we are already in a trade

        if self._is_daily_enter(i):
            return False

        if self._is_daily_exit(i):
            return False

        signals = [
            self._get_take_profit_signal(i),
            self._get_stop_loss_signal(i),
        ]
        conditions = [
            self._is_last_daytrade(i, interval),
        ]
        if sum(signals) > 0 or sum(conditions) > 0:
            return True
        return False
