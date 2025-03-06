import numpy as np
import pandas as pd
from typing import Tuple

class Measures:
    def _get_parkinsons_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        log_hl = np.log(high / low)  # Log of High/Low ratio
        squared_term : pd.Series = log_hl ** 2
        k = 1 / (4 * np.log(2) * window)
        return np.sqrt(k * squared_term.rolling(window=window).sum())

    def _get_dynamic_tolerance(i: int, indicatordata: pd.DataFrame, base_tolerance: float = 0.15) -> float:
        """
        Calculate a dynamic tolerance for Bollinger signals based on current ATR relative to its average.
        """
        avg_atr = indicatordata['ATR'].mean()
        current_atr = indicatordata['ATR'].iloc[i]
        # Increase tolerance when current ATR is higher than average (i.e., market is more volatile)
        return base_tolerance * (current_atr / avg_atr)
    
    @staticmethod
    def _get_stoch_k_d(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K
        """
        # Validate inputs
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("Input series must have the same length")
        
        if len(close) < k_period + d_period - 1:
            # Not enough data, return series of NaNs
            return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
        
        try:
            # Initialize Series for %K and %D
            stoch_k = pd.Series(np.nan, index=close.index)
            
            # Calculate %K
            for i in range(k_period - 1, len(close)):
                highest_high = high.iloc[i-k_period+1:i+1].max()
                lowest_low = low.iloc[i-k_period+1:i+1].min()
                
                if highest_high == lowest_low:  # Avoid division by zero
                    stoch_k.iloc[i] = 50.0
                else:
                    stoch_k.iloc[i] = ((close.iloc[i] - lowest_low) / (highest_high - lowest_low)) * 100
            
            # Calculate %D (3-day SMA of %K)
            stoch_d = stoch_k.rolling(window=d_period).mean()
            
            return stoch_k, stoch_d
            
        except Exception as e:
            print(f"Error calculating Stochastic Oscillator: {e}")
            # Return series of NaNs in case of error
            return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    

