from datetime import datetime
from database.db import db

class Performance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer, db.ForeignKey('bot.id'), nullable=False)
    
    # Time period
    date = db.Column(db.Date, nullable=False)
    period = db.Column(db.String(10), nullable=False)  # 'daily', 'weekly', 'monthly'
    
    # Performance metrics
    balance = db.Column(db.Float, nullable=False)
    deposits = db.Column(db.Float, default=0.0)
    withdrawals = db.Column(db.Float, default=0.0)
    profit_loss = db.Column(db.Float, nullable=False)
    profit_loss_percent = db.Column(db.Float, nullable=False)
    
    # Trade metrics
    num_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    losing_trades = db.Column(db.Integer, default=0)
    win_rate = db.Column(db.Float, default=0.0)
    
    # Risk metrics
    max_drawdown = db.Column(db.Float, default=0.0)
    sharpe_ratio = db.Column(db.Float, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Performance {self.bot_id} {self.date} {self.period}>"