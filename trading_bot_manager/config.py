import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///trading_bot.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Alpaca API configuration
    ALPACA_API_KEY = os.environ.get('APCA_API_KEY_ID')
    ALPACA_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
    ALPACA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL') or 'https://paper-api.alpaca.markets'
    
    # Application settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    DEFAULT_TICKER = os.environ.get('DEFAULT_TICKER') or 'AAPL'
    RISK_PERCENT = float(os.environ.get('RISK_PERCENT') or 0.01)  # Default risk of 1%
    INITIAL_BALANCE = float(os.environ.get('INITIAL_BALANCE') or 10000.0)  # Default $10,000
    
    # Background job settings
    SCHEDULER_API_ENABLED = True