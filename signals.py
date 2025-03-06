
import pandas as pd
import numpy as np


class Signal:
    def __init__(self, 
                 data: pd.DataFrame, 
                 indicatordata: pd.DataFrame) -> None: 
        self.data = data
        self.indicatordata = indicatordata

    def _get_stop_loss(self, 
                       entry_price: float, 
                       atr_value: float, 
                       multiplier: float = 1.5) -> float:
        """
        Calculate a dynamic stop loss based on the entry price and a multiple of the ATR.
        """
        return entry_price - atr_value * multiplier
    
    def _get_take_profit(self, entry_price: float, atr_value: float, multiplier: float = 1.5) -> float:
        """
        Calculate a dynamic take profit based on the entry price and a multiple of the ATR.
        """
        return entry_price + atr_value * multiplier
    
    def is_exit_condition(self, i: int) -> bool:
        """
        Return True if atleast 2 of the exit conditions are met.
        """

        # Get the stop loss and take profit
        entry_price = self.data['Close'].iloc[i]
        atr_value = self.indicatordata['ATR'].iloc[i]

        is_stop_loss = entry_price <= self._get_stop_loss(entry_price, atr_value, multiplier=1.5)
        is_take_profit = entry_price >= self._get_take_profit(entry_price, atr_value, multiplier=2.0)

        if is_stop_loss or is_take_profit:
            return True

        exit_signals = [
            self.macd_sell_signal(self.indicatordata, i),
            self.rsi_sell_signal(self.indicatordata, i),
            self.bollinger_sell_signal(self.indicatordata, i),
            self.extra_sell_filter(self.indicatordata, i)
        ]

        # Count the number of True values in exit_signals   
        true_count = sum(exit_signals)

        return true_count >= 2
    
    def _get_parkinsons_volatility(self, high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        log_hl = np.log(high / low)  # Log of High/Low ratio
        squared_term : pd.Series = log_hl ** 2
        k = 1 / (4 * np.log(2) * window)
        return np.sqrt(k * squared_term.rolling(window=window).sum())

    def _get_dynamic_tolerance(self, i: int, base_tolerance: float = 0.15) -> float:
        """
        Calculate a dynamic tolerance for Bollinger signals based on current ATR relative to its average.
        """
        avg_atr = self.indicatordata['ATR'].mean()
        current_atr = self.indicatordata['ATR'].iloc[i]
        # Increase tolerance when current ATR is higher than average (i.e., market is more volatile)
        return base_tolerance * (current_atr / avg_atr)

    def _get_bollinger_buy_signal(self, i: int, base_tolerance: float = 0.15) -> bool:
        """
            Returns True if the current close is near (within a dynamic tolerance) the lower Bollinger band.
        """
        tol = self.dynamic_tolerance(i, base_tolerance)
        price = self.data['Close'].iloc[i]
        lower_band = self.indicatordata['Bollinger_lband'].iloc[i]
        return price <= lower_band * (1 + tol)

    def _get_bollinger_sell_signal(self, i: int, base_tolerance: float = 0.15) -> bool:
        """
        Returns True if the current close is near (within a dynamic tolerance) the upper Bollinger band.
        """
        tol = self._get_dynamic_tolerance(i, base_tolerance)
        price = self.data['Close'].iloc[i]
        upper_band = self.indicatordata['Bollinger_hband'].iloc[i]
        return price >= upper_band * (1 - tol)

    def _get_atr_buy_signal(self, i: int) -> bool:
        """
        Returns True if the current ATR is below a dynamic threshold (indicating relatively calm conditions).
        """
        avg_atr = self.indicatordata['ATR'].mean()
        current_atr = self.indicatordata['ATR'].iloc[i]
        # For example, if the current ATR is less than 80% of the average, it indicates a lower volatility environment.
        return current_atr < avg_atr * 0.8

    def _get_parkinsons_buy_signal(self, i: int) -> bool:
        """
        Returns True if the current Parkinson's volatility is below a dynamic threshold.
        """ 
        avg_parkinson = self.indicatordata['Parkinsons_Volatility'].mean()
        current_parkinson = self.indicatordata['Parkinsons_Volatility'].iloc[i]
        # Signal buy if current Parkinson's volatility is less than 80% of its average.
        return current_parkinson < avg_parkinson * 0.8

    def _get_rsi_buy_signal(self, i: int) -> bool:
        """
        Returns True if the current RSI is below a dynamic threshold.
        """
        current_atr = self.indicatordata['ATR'].iloc[i]
        avg_atr = self.indicatordata['ATR'].mean()
        threshold = 40 if current_atr <= avg_atr else 45
        return self.indicatordata['RSI'].iloc[i] < threshold

    def _get_rsi_sell_signal(self, i: int) -> bool:
        """
        Returns True if the current RSI is above a dynamic threshold.
        """
        current_atr = self.indicatordata['ATR'].iloc[i]
        avg_atr = self.indicatordata['ATR'].mean()
        threshold = 60 if current_atr <= avg_atr else 55
        return self.indicatordata['RSI'].iloc[i] > threshold

    def _get_macd_buy_signal(self, i: int) -> bool:
        """
        Returns True if the current MACD is below a dynamic threshold.
        """
        current_atr = self.indicatordata['ATR'].iloc[i]
        avg_atr = self.indicatordata['ATR'].mean()
        if current_atr > avg_atr * 1.5:
            return False
        if self.indicatordata['MACD'].iloc[i] < 0 and self.indicatordata['MACD_signal'].iloc[i] < 0.01:
            if self.indicatordata['MACD_buy_signal'][i] and not self.indicatordata['MACD_buy_signal'][i-1]:
                return True
        return False

    def _get_macd_sell_signal(self, i: int) -> bool:
        """
        Returns True if the current MACD is above a dynamic threshold.
        """
        current_atr = self.indicatordata['ATR'].iloc[i]
        avg_atr = self.indicatordata['ATR'].mean()
        if current_atr > avg_atr * 1.5:
            return False
        if not self.indicatordata['MACD_buy_signal'][i] and self.indicatordata['MACD_buy_signal'][i-1]:
            return True
        return False
    
    def _get_extra_buy_filter(self, i: int) -> bool:
        """
            Extra buy filter: Require that the current price is above the 20-period SMA.
        """
        # Ensure SMA value is not NaN
        sma_value = self.indicatordata['SMA20'].iloc[i]
        if pd.isna(sma_value):
            return False
        return self.data['Close'].iloc[i] > sma_value

    def _get_extra_sell_filter(self, i: int) -> bool:
        """
            Extra sell filter: Require that the current price is below the 20-period SMA.
        """
        sma_value = self.indicatordata['SMA20'].iloc[i]
        if pd.isna(sma_value):
            return False
        return self.data['Close'].iloc[i] < sma_value
    
    def is_entry_condition(self, i: int) -> bool:
        """
            Return True if 2 of the entry conditions are met.
        """
        entry_signals = [
            self._get_macd_buy_signal(i),
            self._get_rsi_buy_signal(i),
            self._get_bollinger_buy_signal(i),
            self._get_atr_buy_signal(i),
            self._get_parkinsons_buy_signal(i),
            self._get_extra_buy_filter(i)
        ]
        true_count = sum(entry_signals)
        return true_count >= 2