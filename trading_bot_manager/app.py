from flask import Flask, render_template
from database.db import db
from routes.dashboard import dashboard_bp
from routes.bot_management import bot_management_bp
from routes.trades import trades_bp
from routes.backtest import backtest_bp
import logging
import os
from config import Config
from services.analytics_service import AnalyticsService
from apscheduler.schedulers.background import BackgroundScheduler

def create_app(config_class=Config):
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    db.init_app(app)
    
    # Register blueprints
    app.register_blueprint(dashboard_bp, url_prefix='/')
    app.register_blueprint(bot_management_bp, url_prefix='/bots')
    app.register_blueprint(trades_bp, url_prefix='/trades')
    app.register_blueprint(backtest_bp, url_prefix='/backtest')
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Set up scheduler for performance calculations
    scheduler = BackgroundScheduler()
    analytics_service = AnalyticsService()
    
    # Schedule daily performance calculation at midnight
    scheduler.add_job(
        analytics_service.calculate_daily_performance,
        'cron',
        hour=0,
        minute=1,
        id='daily_performance_calculation'
    )
    
    # Start the scheduler
    scheduler.start()
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('errors/500.html'), 500
    
    return app

# Create the routes for bot management
def create_bot_management_routes():
    from routes.bot_management import create_bot, update_bot, delete_bot, start_bot, stop_bot
    return {
        '/bots/create': create_bot,
        '/bots/update/<int:bot_id>': update_bot,
        '/bots/delete/<int:bot_id>': delete_bot,
        '/bots/start/<int:bot_id>': start_bot,
        '/bots/stop/<int:bot_id>': stop_bot,
    }

# Create the routes for trade management
def create_trade_routes():
    from routes.trades import get_trades, get_trade_details
    return {
        '/trades/<int:bot_id>': get_trades,
        '/trades/details/<int:trade_id>': get_trade_details,
    }

# Create the routes for backtesting
def create_backtest_routes():
    from routes.backtest import run_backtest, get_backtest_results
    return {
        '/backtest/run': run_backtest,
        '/backtest/results/<int:backtest_id>': get_backtest_results,
    }