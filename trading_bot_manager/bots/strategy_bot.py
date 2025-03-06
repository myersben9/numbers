from bots.base_bot import BaseBot
from yfetch import Yfetch
from datetime import datetime
import pandas as pd
import numpy as np
import pytz
import logging
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

class StrategyBot(BaseBot):
    def __init__(self, ticker, bot_id, pre_post=False, 
                 rsi_buy_threshold=40, rsi_sell_threshold=60,
                 bollinger_tolerance=0.15, atr_threshold=3,
                 parkinson_threshold=0.15, interval=60):
        """
        Bot that implements a strategy with RSI, MACD, Bollinger Bands, and volatility indicators.
        
        Args:
            ticker: The stock ticker to trade
            bot_id: Database ID of the bot configuration
            pre_post: Whether to include pre/post market data
            rsi_buy_threshold: RSI level below which to consider buying
            rsi_sell_threshold: RSI level above which to consider selling
            bollinger_tolerance: Tolerance for Bollinger band signals
            atr_threshold: ATR threshold for volatility
            parkinson_threshold: Parkinson's volatility threshold
            interval: How often to check for trades (in seconds)
        """
        super().__init__(ticker, bot_id, interval)
        self.pre_post = pre_post
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.bollinger_tolerance = bollinger_tolerance
        self.atr_threshold = atr_threshold
        self.parkinson_threshold = parkinson_threshold
        self.position = False  # Flag to track if we're in a position
        self.position_price = None  # Price at which position was entered
        self.position_quantity = 0  # Quantity of shares held
    
    def fetch_data(self):
        """Fetch latest market data for the ticker"""
        try:
            data = Yfetch(self.ticker, "1d", "1m", self.pre_post)
            df = data.get_chart_dataframe()
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {str(e)}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators from price data"""
        if data is None or len(data) < 20:  # Need at least 20 bars for some indicators
            return None

        indicators = {}
        
        # Calculate RSI
        rsi = RSIIndicator(pd.Series(data['Close'].values.flatten())).rsi()
        indicators['RSI'] = rsi
        
        # Calculate MACD
        macd = MACD(pd.Series(data['Close'].values.flatten()))
        indicators['MACD'] = macd.macd()
        indicators['MACD_signal'] = macd.macd_signal()
        
        # Calculate Bollinger Bands
        bollinger = BollingerBands(
            pd.Series(data['Close'].values.flatten()), 
            window=20, 
            window_dev=2
        )
        indicators['Bollinger_hband'] = bollinger.bollinger_hband()
        indicators['Bollinger_lband'] = bollinger.bollinger_lband()
        indicators['Bollinger_mavg'] = bollinger.bollinger_mavg()
        
        # Calculate ATR
        if len(data) >= 14:
            indicators['ATR'] = AverageTrueRange(
                high=pd.Series(data['High'].values.flatten()),
                low=pd.Series(data['Low'].values.flatten()),
                close=pd.Series(data['Close'].values.flatten()),
                window=14,
                fillna=False
            ).average_true_range()
        else:
            indicators['ATR'] = pd.Series(np.nan, index=data.index)
        
        # Calculate Parkinson's Volatility
        indicators['Parkinsons_Volatility'] = self.parkinsons_volatility(
            pd.Series(data['High'].values.flatten()),
            pd.Series(data['Low'].values.flatten()),
            window=20
        )
        
        # MACD buy signal (True when MACD >= MACD signal)
        indicators['MACD_buy_signal'] = indicators['MACD'] >= indicators['MACD_signal']
        
        return indicators
    
    def parkinsons_volatility(self, high, low, window=20):
        """Calculate Parkinson's volatility from high and low prices"""
        log_hl = np.log(high / low)
        squared_term = log_hl ** 2
        k = 1 / (4 * np.log(2) * window)
        return np.sqrt(k * squared_term.rolling(window=window).sum())
    
    def check_for_trades(self):
        """Check for trading opportunities based on the strategy"""
        # Fetch latest data
        data = self.fetch_data()
        if data is None:
            return
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        if indicators is None:
            return
        
        # Get the latest close price
        latest_price = data['Close'].iloc[-1]
        latest_time = data.index[-1]
        
        # Get the latest indicator values
        latest_rsi = indicators['RSI'].iloc[-1]
        latest_macd = indicators['MACD'].iloc[-1]
        latest_macd_signal = indicators['MACD_signal'].iloc[-1]
        latest_macd_buy_signal = indicators['MACD_buy_signal'].iloc[-1]
        prev_macd_buy_signal = indicators['MACD_buy_signal'].iloc[-2] if len(indicators['MACD_buy_signal']) > 1 else False
        latest_bollinger_lower = indicators['Bollinger_lband'].iloc[-1]
        latest_bollinger_upper = indicators['Bollinger_hband'].iloc[-1]
        latest_atr = indicators['ATR'].iloc[-1]
        latest_parkinsons = indicators['Parkinsons_Volatility'].iloc[-1]
        
        # Check if we should buy
        if not self.position:
            # Buy if all conditions are met
            buy_signal = (
                # MACD buy signal (current True, previous False)
                latest_macd_buy_signal and not prev_macd_buy_signal and
                # RSI below threshold
                latest_rsi < self.rsi_buy_threshold and
                # Price near lower Bollinger band
                latest_price <= latest_bollinger_lower * (1 + self.bollinger_tolerance) and
                # ATR below threshold
                latest_atr < self.atr_threshold and
                # Parkinson's volatility below threshold
                latest_parkinsons < self.parkinson_threshold
            )
            
            if buy_signal:
                # Calculate position size (for demo, just use 10 shares)
                # In a real system, this would use proper position sizing
                quantity = 10
                
                # Record the trade
                self.record_trade(
                    action="BUY",
                    price=latest_price,
                    quantity=quantity,
                    timestamp=latest_time,
                    rsi=latest_rsi,
                    macd=latest_macd,
                    macd_signal=latest_macd_signal,
                    bollinger_upper=latest_bollinger_upper,
                    bollinger_lower=latest_bollinger_lower,
                    atr=latest_atr,
                    parkinsons_volatility=latest_parkinsons
                )
                
                # Update position status
                self.position = True
                self.position_price = latest_price
                self.position_quantity = quantity
                
                logger.info(f"BUY signal for {self.ticker} at {latest_price}")
        
        # Check if we should sell
        elif self.position:
            # Sell if conditions are met
            sell_signal = (
                # MACD sell signal (current False, previous True)
                not latest_macd_buy_signal and prev_macd_buy_signal and
                # RSI above threshold
                latest_rsi > self.rsi_sell_threshold and
                # Price near upper Bollinger band
                latest_price >= latest_bollinger_upper * (1 - self.bollinger_tolerance)
            )
            
            if sell_signal:
                # Calculate profit/loss
                profit_loss = (latest_price - self.position_price) * self.position_quantity
                profit_loss_percent = (latest_price / self.position_price - 1) * 100
                
                # Record the trade
                self.record_trade(
                    action="SELL",
                    price=latest_price,
                    quantity=self.position_quantity,
                    timestamp=latest_time,
                    profit_loss=profit_loss,
                    profit_loss_percent=profit_loss_percent,
                    rsi=latest_rsi,
                    macd=latest_macd,
                    macd_signal=latest_macd_signal,
                    bollinger_upper=latest_bollinger_upper,
                    bollinger_lower=latest_bollinger_lower,
                    atr=latest_atr,
                    parkinsons_volatility=latest_parkinsons
                )
                
                # Reset position status
                self.position = False
                self.position_price = None
                self.position_quantity = 0
                
                logger.info(f"SELL signal for {self.ticker} at {latest_price} (P/L: {profit_loss_percent:.2f}%)")