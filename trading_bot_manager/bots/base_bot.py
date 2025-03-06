import threading
import time
from datetime import datetime
from abc import ABC, abstractmethod
from services.trade_service import TradeService
import logging

logger = logging.getLogger(__name__)

class BaseBot(ABC, threading.Thread):
    def __init__(self, ticker, bot_id, interval=60):
        """
        Base class for all bots
        
        Args:
            ticker: The stock ticker to trade
            bot_id: Database ID of the bot configuration
            interval: How often to check for trades (in seconds)
        """
        threading.Thread.__init__(self)
        self.ticker = ticker
        self.bot_id = bot_id
        self.interval = interval
        self.running = False
        self.start_time = None
        self.trade_service = TradeService()
        self.daemon = True  # Thread will exit when the main program exits
    
    def run(self):
        """Main bot loop, runs when the thread is started"""
        self.running = True
        self.start_time = datetime.now()
        logger.info(f"Bot {self.bot_id} started for {self.ticker}")
        
        while self.running:
            try:
                # Check for trading opportunities
                self.check_for_trades()
                
                # Sleep for the specified interval
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in bot {self.bot_id}: {str(e)}")
                time.sleep(self.interval)  # Sleep even on error
    
    def stop(self):
        """Stop the bot thread"""
        self.running = False
        logger.info(f"Bot {self.bot_id} stopped")
    
    def get_uptime(self):
        """Get the uptime of the bot in seconds"""
        if not self.start_time:
            return 0
        return (datetime.now() - self.start_time).total_seconds()
    
    @abstractmethod
    def check_for_trades(self):
        """
        Abstract method to check for trading opportunities.
        This should be implemented by each bot strategy.
        """
        pass
    
    def record_trade(self, action, price, quantity, timestamp=None, **kwargs):
        """Record a trade in the database"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return self.trade_service.create_trade(
            bot_id=self.bot_id,
            symbol=self.ticker,
            action=action,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
            **kwargs
        )