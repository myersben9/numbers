from datetime import datetime
from database.db import db

class Bot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)
    strategy = db.Column(db.String(50), nullable=False)
    active = db.Column(db.Boolean, default=False)
    
    # Bot configuration
    period = db.Column(db.String(5), default="1d")
    interval = db.Column(db.String(5), default="1m")
    pre_post = db.Column(db.Boolean, default=False)
    
    # Strategy parameters
    rsi_buy_threshold = db.Column(db.Float, default=40.0)
    
    rsi_sell_threshold = db.Column(db.Float, default=60.0)
    bollinger_tolerance = db.Column(db.Float, default=0.15)
    atr_threshold = db.Column(db.Float, default=3.0)
    parkinson_threshold = db.Column(db.Float, default=0.15)
    
    # Risk management
    initial_balance = db.Column(db.Float, default=1000.0)
    risk_per_trade = db.Column(db.Float, default=0.01)  # Percentage of balance
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trades = db.relationship('Trade', backref='bot', lazy=True)
    performances = db.relationship('Performance', backref='bot', lazy=True)

    def __repr__(self):
        return f"<Bot {self.name}>"
