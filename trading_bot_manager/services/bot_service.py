from models.bot import Bot
from models.trade import Trade
from bots.strategy_bot import StrategyBot
from bots.alpaca_bot import AlpacaBot
from database.db import db
from datetime import datetime
import traceback
import logging

logger = logging.getLogger(__name__)

class BotService:
    def __init__(self):
        self.active_bots = {}  # Store running bot instances
    
    def get_all_bots(self):
        """Get all bot configurations from the database"""
        return Bot.query.all()
    
    def get_bot_by_id(self, bot_id):
        """Get bot configuration by ID"""
        return Bot.query.get(bot_id)
    
    def create_bot(self, name, ticker, strategy, **kwargs):
        """Create a new bot configuration"""
        bot = Bot(
            name=name,
            ticker=ticker,
            strategy=strategy,
            **kwargs
        )
        db.session.add(bot)
        db.session.commit()
        return bot
    
    def update_bot(self, bot_id, **kwargs):
        """Update bot configuration"""
        bot = self.get_bot_by_id(bot_id)
        if not bot:
            return None
        
        # If the bot is active, we need to restart it
        was_active = bot.active
        if was_active:
            self.stop_bot(bot_id)
        
        # Update bot attributes
        for key, value in kwargs.items():
            if hasattr(bot, key):
                setattr(bot, key, value)
        
        db.session.commit()
        
        # Restart the bot if it was active
        if was_active:
            self.start_bot(bot_id)
        
        return bot
    
    def delete_bot(self, bot_id):
        """Delete a bot configuration"""
        bot = self.get_bot_by_id(bot_id)
        if not bot:
            return False
        
        # Stop the bot if it's running
        if bot.active:
            self.stop_bot(bot_id)
        
        db.session.delete(bot)
        db.session.commit()
        return True
    
    def start_bot(self, bot_id):
        """Start a bot running with the given configuration"""
        bot_config = self.get_bot_by_id(bot_id)
        if not bot_config:
            return False
        
        # Don't start if already running
        if bot_id in self.active_bots:
            return False
        
        try:
            # Create the appropriate bot instance based on strategy
            if bot_config.strategy == "strategy_bot":
                bot_instance = StrategyBot(
                    ticker=bot_config.ticker,
                    pre_post=bot_config.pre_post,
                    rsi_buy_threshold=bot_config.rsi_buy_threshold,
                    rsi_sell_threshold=bot_config.rsi_sell_threshold,
                    bollinger_tolerance=bot_config.bollinger_tolerance,
                    atr_threshold=bot_config.atr_threshold,
                    parkinson_threshold=bot_config.parkinson_threshold,
                    bot_id=bot_id
                )
            elif bot_config.strategy == "alpaca_bot":
                bot_instance = AlpacaBot(
                    ticker=bot_config.ticker,
                    initial_balance=bot_config.initial_balance,
                    risk_per_trade=bot_config.risk_per_trade,
                    bot_id=bot_id
                )
            else:
                logger.error(f"Unknown strategy {bot_config.strategy}")
                return False
            
            # Start the bot in a separate thread
            bot_instance.start()
            
            # Update the active status
            bot_config.active = True
            db.session.commit()
            
            # Add to active bots
            self.active_bots[bot_id] = bot_instance
            
            return True
        except Exception as e:
            logger.error(f"Error starting bot {bot_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def stop_bot(self, bot_id):
        """Stop a running bot"""
        if bot_id not in self.active_bots:
            return False
        
        try:
            # Stop the bot
            self.active_bots[bot_id].stop()
            
            # Remove from active bots
            del self.active_bots[bot_id]
            
            # Update the active status
            bot = self.get_bot_by_id(bot_id)
            if bot:
                bot.active = False
                db.session.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error stopping bot {bot_id}: {str(e)}")
            return False
    
    def get_bot_status(self, bot_id):
        """Get the current status of a bot"""
        bot = self.get_bot_by_id(bot_id)
        if not bot:
            return None
        
        is_active = bot_id in self.active_bots
        
        # Get latest trade
        latest_trade = Trade.query.filter_by(bot_id=bot_id).order_by(Trade.timestamp.desc()).first()
        
        return {
            "id": bot.id,
            "name": bot.name,
            "ticker": bot.ticker,
            "strategy": bot.strategy,
            "active": is_active,
            "latest_trade": latest_trade,
            "uptime": self.active_bots[bot_id].get_uptime() if is_active else None
        }