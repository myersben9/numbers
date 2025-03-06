from flask import Blueprint, render_template, jsonify, request
from services.bot_service import BotService
from services.analytics_service import AnalyticsService
from datetime import datetime, timedelta

dashboard_bp = Blueprint('dashboard', __name__)
bot_service = BotService()
analytics_service = AnalyticsService()

@dashboard_bp.route('/')
def index():
    """Render the main dashboard page"""
    # Get all bots
    bots = bot_service.get_all_bots()
    
    # Get active bots
    active_bots = [bot for bot in bots if bot.active]
    
    # Get recent trades (last 24 hours)
    recent_trades = []
    for bot in bots:
        status = bot_service.get_bot_status(bot.id)
        if status and status.get('latest_trade'):
            recent_trades.append({
                'bot_name': bot.name,
                'trade': status['latest_trade']
            })
    
    # Sort recent trades by timestamp
    recent_trades.sort(key=lambda x: x['trade'].timestamp, reverse=True)
    
    # Get performance summary for active bots
    performance_summary = []
    for bot in active_bots:
        performance = analytics_service.get_bot_performance(bot.id, period='daily')
        if performance:
            performance_summary.append({
                'bot_name': bot.name,
                'performance': performance[-1] if performance else None
            })
    
    return render_template('dashboard.html',
                          bots=bots,
                          active_bots=active_bots,
                          recent_trades=recent_trades,
                          performance_summary=performance_summary)

@dashboard_bp.route('/api/bot/<int:bot_id>/status')
def get_bot_status(bot_id):
    """API endpoint to get the current status of a bot"""
    status = bot_service.get_bot_status(bot_id)
    if not status:
        return jsonify({"error": "Bot not found"}), 404
    
    # Convert datetime to string for JSON serialization
    if status.get('latest_trade'):
        status['latest_trade'] = {
            'action': status['latest_trade'].action,
            'price': status['latest_trade'].price,
            'timestamp': status['latest_trade'].timestamp.isoformat(),
            'profit_loss': status['latest_trade'].profit_loss,
            'profit_loss_percent': status['latest_trade'].profit_loss_percent
        }
    
    return jsonify(status)

@dashboard_bp.route('/api/bot/<int:bot_id>/performance/<period>')
def get_bot_performance(bot_id, period):
    """API endpoint to get performance metrics for a bot"""
    if period not in ['daily', 'weekly', 'monthly']:
        return jsonify({"error": "Invalid period"}), 400
    
    performances = analytics_service.get_bot_performance(bot_id, period)
    
    # Convert to list of dicts for JSON serialization
    performance_data = []
    for p in performances:
        performance_data.append({
            'date': p.date.isoformat(),
            'balance': p.balance,
            'profit_loss': p.profit_loss,
            'profit_loss_percent': p.profit_loss_percent,
            'num_trades': p.num_trades,
            'win_rate': p.win_rate
        })
    
    return jsonify(performance_data)

@dashboard_bp.route('/api/bot/<int:bot_id>/trades')
def get_bot_trades(bot_id):
    """API endpoint to get trade history for a bot"""
    # Get query parameters for date filtering
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str) if end_date_str else None
    
    # Get trade performance
    performance = analytics_service.get_trade_performance(bot_id, start_date, end_date)
    
    # Format for JSON
    trade_pairs = []
    for pair in performance['trade_pairs']:
        trade_pairs.append({
            'buy': {
                'price': pair['buy'].price,
                'quantity': pair['buy'].quantity,
                'timestamp': pair['buy'].timestamp.isoformat()
            },
            'sell': {
                'price': pair['sell'].price,
                'quantity': pair['sell'].quantity,
                'timestamp': pair['sell'].timestamp.isoformat(),
                'profit_loss': pair['sell'].profit_loss,
                'profit_loss_percent': pair['sell'].profit_loss_percent
            }
        })
    
    performance['trade_pairs'] = trade_pairs
    
    return jsonify(performance)