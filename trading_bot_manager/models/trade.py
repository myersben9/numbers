import os

def create_files_and_dirs(base_dir, structure):
    for path in structure:
        full_path = os.path.join(base_dir, path)
        if path.endswith("/"):  # Create directories
            os.makedirs(full_path, exist_ok=True)
        else:  # Create empty files
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write("")

project_structure = [
    "app.py",
    "config.py",
    "requirements.txt",
    "run.py",
    "static/css/styles.css",
    "static/js/dashboard.js",
    "templates/base.html",
    "templates/dashboard.html",
    "templates/bots.html",
    "templates/trades.html",
    "templates/backtest.html",
    "templates/settings.html",
    "routes/__init__.py",
    "routes/dashboard.py",
    "routes/bot_management.py",
    "routes/trades.py",
    "routes/backtest.py",
    "models/__init__.py",
    "models/bot.py",
    "models/trade.py",
    "models/performance.py",
    "services/__init__.py",
    "services/bot_service.py",
    "services/trade_service.py",
    "services/analytics_service.py",
    "services/data_service.py",
    "database/__init__.py",
    "database/db.py",
    "database/migrations/",  # This is a directory
    "bots/__init__.py",
    "bots/base_bot.py",
    "bots/strategy_bot.py",
    "bots/alpaca_bot.py",
]

if __name__ == "__main__":
    base_directory = "trading_bot_manager"
    os.makedirs(base_directory, exist_ok=True)
    create_files_and_dirs(base_directory, project_structure)
    print(f"Project structure created inside {base_directory}/")