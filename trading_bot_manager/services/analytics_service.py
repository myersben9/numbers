from models.trade import Trade
from models.performance import Performance
from models.bot import Bot
from database.db import db
from sqlalchemy import func
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class AnalyticsService:
    def get_bot_performance(self, bot_id, period='daily'):
        """Get performance metrics for a bot over time"""
        performances = Performance.query.filter_by(
            bot_id=bot_id,
            period=period
        ).order_by(Performance.date).all()
        
        return performances
    
    def get_trade_performance(self, bot_id, start_date=None, end_date=None):
        """Get trade performance metrics for a bot"""
        query = Trade.query.filter_by(bot_id=bot_id)
        
        if start_date:
            query = query.filter(Trade.timestamp >= start_date)
        if end_date:
            query = query.filter(Trade.timestamp <= end_date)
        
        trades = query.order_by(Trade.timestamp).all()
        
        # Group trades into pairs (buy/sell)
        trade_pairs = []
        current_pair = {}
        
        for trade in trades:
            if trade.action == "BUY" and not current_pair.get("buy"):
                current_pair["buy"] = trade
            elif trade.action == "SELL" and current_pair.get("buy") and not current_pair.get("sell"):
                current_pair["sell"] = trade
                trade_pairs.append(current_pair)
                current_pair = {}
        
        # Calculate metrics
        total_trades = len(trade_pairs)
        winning_trades = sum(1 for pair in trade_pairs if pair["sell"].profit_loss > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        total_profit = sum(pair["sell"].profit_loss for pair in trade_pairs)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        avg_win = sum(pair["sell"].profit_loss for pair in trade_pairs if pair["sell"].profit_loss > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(pair["sell"].profit_loss for pair in trade_pairs if pair["sell"].profit_loss <= 0) / losing_trades if losing_trades > 0 else 0
        
        # Calculate max drawdown
        balances = []
        balance = 1000  # Starting balance
        for pair in trade_pairs:
            balance += pair["sell"].profit_loss
            balances.append(balance)
        
        max_drawdown = 0
        peak = balances[0]
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / peak
            max_drawdown = max(max_drawdown, dd)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_profit": avg_profit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "trade_pairs": trade_pairs
        }
    
    def calculate_daily_performance(self, bot_id=None):
        """Calculate and store daily performance for all bots"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # Get all active bots or a specific bot
        if bot_id:
            bots = [Bot.query.get(bot_id)]
        else:
            bots = Bot.query.filter_by(active=True).all()
        
        for bot in bots:
            # Get yesterday's trades
            yesterday_trades = Trade.query.filter(
                Trade.bot_id == bot.id,
                func.date(Trade.timestamp) == yesterday
            ).all()
            
            # Calculate metrics
            num_trades = len([t for t in yesterday_trades if t.action == "SELL"])
            profit_loss = sum(t.profit_loss or 0 for t in yesterday_trades if t.profit_loss)
            
            # Get previous day's performance to calculate balance
            prev_performance = Performance.query.filter_by(
                bot_id=bot.id,
                period='daily'
            ).order_by(Performance.date.desc()).first()
            
            if prev_performance:
                balance = prev_performance.balance + profit_loss
            else:
                balance = bot.initial_balance + profit_loss
            
            # Calculate other metrics
            winning_trades = len([t for t in yesterday_trades if t.action == "SELL" and t.profit_loss > 0])
            losing_trades = num_trades - winning_trades
            win_rate = (winning_trades / num_trades) if num_trades > 0 else 0
            profit_loss_percent = (profit_loss / (balance - profit_loss)) if balance != profit_loss else 0
            
            # Create or update performance record
            performance = Performance.query.filter_by(
                bot_id=bot.id,
                date=yesterday,
                period='daily'
            ).first()
            
            if performance:
                # Update existing record
                performance.balance = balance
                performance.profit_loss = profit_loss
                performance.profit_loss_percent = profit_loss_percent
                performance.num_trades = num_trades
                performance.winning_trades = winning_trades
                performance.losing_trades = losing_trades
                performance.win_rate = win_rate
            else:
                # Create new record
                performance = Performance(
                    bot_id=bot.id,
                    date=yesterday,
                    period='daily',
                    balance=balance,
                    profit_loss=profit_loss,
                    profit_loss_percent=profit_loss_percent,
                    num_trades=num_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    win_rate=win_rate
                )
                db.session.add(performance)
            
            db.session.commit()