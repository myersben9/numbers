import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from datetime import datetime, time, timedelta
class Signal:
    def __init__(self, 
                 data: pd.DataFrame, 
                 indicatordata: Dict[str, Union[pd.Series, pd.DataFrame, Any]],
                 atr_multiplier: float = 1.5,
                 base_tolerance: float = 0.15) -> None: 
        self.data = data
        self.indicatordata = indicatordata
        self.atr_multiplier = atr_multiplier
        self.base_tolerance = base_tolerance
    
    def rsi_buy_signal(self, i: int) -> bool: 
        if self.indicatordata['RSI'][i] < 40:
            return True
        return False
    
    def rsi_sell_signal(self, i: int) -> bool:
        if self.indicatordata['RSI'][i] > 70:
            return True
    
    # Write a function that returns true or false
    def macd_buy_signal(self, i: int) -> bool:
        # First check if the the macd signal and macd are both less than 0.01
        if self.indicatordata['MACD_signal'][i] < 0 and self.indicatordata['MACD'][i] < 0:
            if self.indicatordata['MACD_buy_signal'][i] == True and self.indicatordata['MACD_buy_signal'][i-1] == False:
                return True
        return False
            
    def macd_sell_signal(self, i: int) -> bool:
        # First check if there was already a buy signal at or before this index
        if self.indicatordata['MACD_buy_signal'][i] == False and self.indicatordata['MACD_buy_signal'][i-1] == True:
            # Check if there is a buy signal at or before this index
            return True
        return False
    
    def is_low_volatility(self, i: int) -> bool:
        """
        Get the Bollinger Band width and ATR
        """
        bb_width = (self.indicatordata['Bollinger_hband'].iloc[i] - self.indicatordata['Bollinger_lband'].iloc[i]) / self.data['Close'].iloc[i]
        print(bb_width)
        if bb_width < 0.009:
            return True
        return False

    def is_high_volatility(self, i: int) -> bool:
        """
        Get the Bollinger Band width and ATR
        """
        bb_width = (self.indicatordata['Bollinger_hband'].iloc[i] - self.indicatordata['Bollinger_lband'].iloc[i]) / self.data['Close'].iloc[i]    
        # print(bb_width)
        if bb_width > 0.1:
            return True
        return False

    def is_entry_condition(self, i: int, interval: str) -> bool:
        """
        Return True if multiple high-quality entry conditions are met.
        """
        signals = [
            self.rsi_buy_signal(i),
            self.macd_buy_signal(i)
        ]

        conditions = [
            self.is_low_volatility(i),
            self.is_market_open(i, interval),
        ]
        print(signals, conditions)
        if sum(signals) > 0 and sum(conditions) > 1:
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
    
    def is_exit_condition(self, i: int, interval: str) -> bool:
        """
            Improved exit condition with multiple factors
        """
        signals = [
            self._get_take_profit_signal(i),
            self._get_stop_loss_signal(i),
        ]
        conditions = [
            self._is_last_daytrade(i, interval),
            self.is_high_volatility(i)
        ]
        if sum(signals) > 0 or sum(conditions) > 0:
            return True
        return False
