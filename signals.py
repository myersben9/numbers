import pandas as pd
import numpy as np
from typing import Dict, Any, Union

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

    def rsi_buy_signal(self, i):
        if self.indicatordata['RSI'][i] < 40:
            return True
        
    
    def rsi_sell_signal(self,i):
        if self.indicatordata['RSI'][i] > 70:
            return True
    
    # Write a function that returns true or false
    def macd_buy_signal(self, i):
        # First check if the the macd signal and macd are both less than 0.01
        if self.indicatordata['MACD_signal'][i] < 0 and self.indicatordata['MACD'][i] < 0:
            if self.indicatordata['MACD_buy_signal'][i] == True and self.indicatordata['MACD_buy_signal'][i-1] == False:
                return True
        return False
            
    def macd_sell_signal(self, i):
        # First check if there was already a buy signal at or before this index
        if self.indicatordata['MACD_buy_signal'][i] == False and self.indicatordata['MACD_buy_signal'][i-1] == True:
            # Check if there is a buy signal at or before this index
            return True
        return False

    def is_entry_condition(self, i: int) -> bool:
        """
        Return True if multiple high-quality entry conditions are met.
        """
        signals = (self.rsi_buy_signal(i) or self.macd_buy_signal(i))
        if signals:
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
    
    def is_exit_condition(self, i: int) -> bool:
        """
            Improved exit condition with multiple factors
        """
        signals = (self._get_take_profit_signal(i) or self._get_stop_loss_signal(i))
        if signals:
            return True
        return False
