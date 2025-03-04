from yfetch import Yfetch
from movers import TopMovers
from matplotlib.dates import DateFormatter
import pytz
from datetime import time, datetime

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytz
import numpy as np
from typing import Dict, List, Tuple, Any

class Bot:
    def __init__(self: 'Bot', 
                 ticker: str, 
                 pre_post: bool = False,
                 start_date: str = None,
                 end_date: str = None,
                 time_zone: str = 'America/Los_Angeles') -> None:
        self.ticker: str = ticker # type: str
        self.period: str = "1d" # type: str
        self.interval: str = "1m" # type: str
        self.pre_post: bool = pre_post # type: bool
        self.start_date: str = start_date # type: str
        self.end_date: str = end_date # type: str
        self.time_zone: str = time_zone # type: str
        self.data: pd.DataFrame = self.fetch_data() # type: pd.DataFrame | None
        self.indicatordata: Dict[str, Any] = self.calculate_indicators() # type: Dict[str, Any] | None

    def fetch_data(self: 'Bot') -> pd.DataFrame:
        # Get the data from yfinance
        if self.start_date and self.end_date:
            data = Yfetch(self.ticker, 
                          interval = self.interval, 
                          pre_post = self.pre_post,
                          start_date = self.start_date,
                          end_date = self.end_date,
                          timezone = self.time_zone)
        elif self.period:
            data = Yfetch(self.ticker, 
                          period = self.period, 
                          interval = self.interval, 
                          timezone = self.time_zone)
        else:
            raise ValueError("Invalid parameters")
        df = data.get_chart_dataframe()
        return df
    
    def calculate_stop_loss(self, entry_price: float, atr_value: float, multiplier: float = 1.5) -> float:
        """
        Calculate a stop loss price based on the entry price and a multiple of the ATR.
        For a long trade, the stop loss is placed below the entry.
        """
        return entry_price - atr_value * multiplier
    
    def calculate_take_profit(self, entry_price: float, atr_value: float, multiplier: float = 1.5) -> float:
        """
        Calculate a take profit price based on the entry price and a multiple of the ATR.
        For a long trade, the take profit is placed above the entry.
        """
        return entry_price + atr_value * multiplier
    
    def is_exit_condition(self, indicatordata: pd.DataFrame, i: int) -> bool:
        """
        Return True if any of the exit conditions are met.
        (Using OR logic for increased sensitivity.)
        """
        return (
            self.macd_sell_signal(indicatordata, i) or
            self.rsi_sell_signal(indicatordata, i) or
            self.bollinger_sell_signal(indicatordata, i) or
            self.extra_sell_filter(indicatordata, i)
        )
    
    def simulate_trade_exit_combined(self, entry_index: int, sl_multiplier: float = 1.5, tp_multiplier: float = 2.0) -> Tuple[pd.Timestamp, float]:
        """
        Scan forward from the entry_index and exit the trade at the first occurrence of either:
          - any indicator-based sell signal (MACD, RSI, Bollinger, or extra sell filter), or
          - if the price reaches the stop loss or take profit levels.
        If neither condition is met, exit at the last available price.
        """
        entry_price = self.data['Close'].iloc[entry_index]
        atr_value = self.indicatordata['ATR'].iloc[entry_index]
        stop_loss = self.calculate_stop_loss(entry_price, atr_value, sl_multiplier)
        take_profit = self.calculate_take_profit(entry_price, atr_value, tp_multiplier)
        
        # Loop over subsequent minutes
        for i in range(entry_index + 1, len(self.data)):
            current_time = self.data.index[i]
            current_low = self.data['Low'].iloc[i]
            current_high = self.data['High'].iloc[i]
            
            # Check indicator-based sell signal (using OR logic)
            indicator_sell = self.is_exit_condition(self.indicatordata, i)
            if indicator_sell:
                return current_time, self.data['Close'].iloc[i]
            if current_low <= stop_loss:
                return current_time, stop_loss
            if current_high >= take_profit:
                return current_time, take_profit
        return self.data.index[-1], self.data['Close'].iloc[-1]
    
    def parkinsons_volatility(self: 'Bot', high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        log_hl = np.log(high / low)  # Log of High/Low ratio
        squared_term : pd.Series = log_hl ** 2
        k = 1 / (4 * np.log(2) * window)
        return np.sqrt(k * squared_term.rolling(window=window).sum())

    def dynamic_tolerance(self, indicatordata: pd.DataFrame, i: int, base_tolerance: float = 0.15) -> float:
        """
        Calculate a dynamic tolerance for Bollinger signals based on current ATR relative to its average.
        """
        avg_atr = indicatordata['ATR'].mean()
        current_atr = indicatordata['ATR'].iloc[i]
        # Increase tolerance when current ATR is higher than average (i.e., market is more volatile)
        return base_tolerance * (current_atr / avg_atr)

    def bollinger_buy_signal(self, indicatordata: pd.DataFrame, i: int, base_tolerance: float = 0.15) -> bool:
        """
        Returns True if the current close is near (within a dynamic tolerance) the lower Bollinger band.
        """
        tol = self.dynamic_tolerance(indicatordata, i, base_tolerance)
        price = self.data['Close'].iloc[i]
        lower_band = indicatordata['Bollinger_lband'].iloc[i]
        return price <= lower_band * (1 + tol)

    def bollinger_sell_signal(self, indicatordata: pd.DataFrame, i: int, base_tolerance: float = 0.15) -> bool:
        """
        Returns True if the current close is near (within a dynamic tolerance) the upper Bollinger band.
        """
        tol = self.dynamic_tolerance(indicatordata, i, base_tolerance)
        price = self.data['Close'].iloc[i]
        upper_band = indicatordata['Bollinger_hband'].iloc[i]
        return price >= upper_band * (1 - tol)

    def atr_buy_signal(self, indicatordata: pd.DataFrame, i: int) -> bool:
        """
        Returns True if the current ATR is below a dynamic threshold (indicating relatively calm conditions).
        """
        avg_atr = indicatordata['ATR'].mean()
        current_atr = indicatordata['ATR'].iloc[i]
        # For example, if the current ATR is less than 80% of the average, it indicates a lower volatility environment.
        return current_atr < avg_atr * 0.8

    def parkinsons_buy_signal(self, indicatordata: pd.DataFrame, i: int) -> bool:
        """
        Returns True if the current Parkinson's volatility is below a dynamic threshold.
        """
        avg_parkinson = indicatordata['Parkinsons_Volatility'].mean()
        current_parkinson = indicatordata['Parkinsons_Volatility'].iloc[i]
        # Signal buy if current Parkinson's volatility is less than 80% of its average.
        return current_parkinson < avg_parkinson * 0.8

    def rsi_buy_signal(self, indicatordata: pd.DataFrame, i: int) -> bool:
    # Adjusted RSI threshold for buy (e.g., allow RSI up to 45 instead of 40)
        current_atr = indicatordata['ATR'].iloc[i]
        avg_atr = indicatordata['ATR'].mean()
        threshold = 40 if current_atr <= avg_atr else 45
        return indicatordata['RSI'].iloc[i] < threshold

    def rsi_sell_signal(self, indicatordata: pd.DataFrame, i: int) -> bool:
        # Adjusted RSI threshold for sell (e.g., allow RSI above 55 instead of 60)
        current_atr = indicatordata['ATR'].iloc[i]
        avg_atr = indicatordata['ATR'].mean()
        threshold = 60 if current_atr <= avg_atr else 55
        return indicatordata['RSI'].iloc[i] > threshold

    def macd_buy_signal(self, indicatordata: pd.DataFrame, i: int) -> bool:
        # Using the crossover approach (ensure only one definition exists)
        current_atr = indicatordata['ATR'].iloc[i]
        avg_atr = indicatordata['ATR'].mean()
        if current_atr > avg_atr * 1.5:
            return False
        if indicatordata['MACD'].iloc[i] < 0 and indicatordata['MACD_signal'].iloc[i] < 0.01:
            if indicatordata['MACD_buy_signal'][i] and not indicatordata['MACD_buy_signal'][i-1]:
                return True
        return False

    def macd_sell_signal(self, indicatordata: pd.DataFrame, i: int) -> bool:
        current_atr = indicatordata['ATR'].iloc[i]
        avg_atr = indicatordata['ATR'].mean()
        if current_atr > avg_atr * 1.5:
            return False
        if not indicatordata['MACD_buy_signal'][i] and indicatordata['MACD_buy_signal'][i-1]:
            return True
        return False
    
    def extra_buy_filter(self, indicatordata: pd.DataFrame, i: int) -> bool:
        """
        Extra buy filter: Require that the current price is above the 20-period SMA.
        You can adjust this logic to include other trend conditions.
        """
        # Ensure SMA value is not NaN
        sma_value = indicatordata['SMA20'].iloc[i]
        if pd.isna(sma_value):
            return False
        return self.data['Close'].iloc[i] > sma_value

    def extra_sell_filter(self, indicatordata: pd.DataFrame, i: int) -> bool:
        """
        Extra sell filter: Require that the current price is below the 20-period SMA.
        Modify as needed for additional exit criteria.
        """
        sma_value = indicatordata['SMA20'].iloc[i]
        if pd.isna(sma_value):
            return False
        return self.data['Close'].iloc[i] < sma_value
    
    def is_entry_condition(self, indicatordata: pd.DataFrame, i: int) -> bool:
        """
        Return True if any of the entry conditions are met.
        (Using OR logic for increased sensitivity.)
        """
        return (
            self.macd_buy_signal(indicatordata, i) or
            self.rsi_buy_signal(indicatordata, i) or
            self.bollinger_buy_signal(indicatordata, i) or
            self.atr_buy_signal(indicatordata, i) or
            self.parkinsons_buy_signal(indicatordata, i) or
            self.extra_buy_filter(indicatordata, i)
        )
    
    def process_bars(self, indicatordata: pd.DataFrame) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
        """
        Process each bar sequentially (as in real time).
        When not in a trade, check for an entry condition.
        When in a trade, on each new bar, check if an exit condition is met—
        either by your indicator-based sell signals OR if the current price hits the stop loss or take profit.
        """
        buy_coords = []
        sell_coords = []
        in_trade = False
        entry_index = None
        entry_price = None

        for i in range(1, len(indicatordata['MACD_buy_signal'])):
            current_time = self.data.index[i]
            if not self.is_market_open(current_time):
                continue

            # If not in a trade, check for entry.
            if not in_trade:
                if self.is_entry_condition(indicatordata, i):
                    entry_index = i
                    entry_price = self.data['Close'].iloc[i]
                    print(f"Buy signal at {current_time}: Price={entry_price:.2f}")
                    buy_coords.append((current_time, entry_price))
                    in_trade = True
            else:
                # In a trade, calculate stop loss and take profit based on entry.
                atr_value = indicatordata['ATR'].iloc[entry_index]
                stop_loss = self.calculate_stop_loss(entry_price, atr_value, multiplier=1.5)
                take_profit = self.calculate_take_profit(entry_price, atr_value, multiplier=2.0)
                current_price = self.data['Close'].iloc[i]

                # Check indicator-based exit
                indicator_exit = (self.macd_sell_signal(indicatordata, i) or
                                  self.rsi_sell_signal(indicatordata, i) or
                                  self.bollinger_sell_signal(indicatordata, i) or
                                  self.extra_sell_filter(indicatordata, i))
                
                if indicator_exit or (current_price <= stop_loss) or (current_price >= take_profit):
                    print(f"Exit signal at {current_time}: Price={current_price:.2f}")
                    sell_coords.append((current_time, current_price))
                    in_trade = False  # Exit the trade

            # Optionally, if market close is reached and you're in a trade, force exit.
            if in_trade and current_time == self.get_last_market_close_index():
                forced_exit_price = self.data['Close'].iloc[i]
                print(f"Forced exit at market close {current_time}: Price={forced_exit_price:.2f}")
                sell_coords.append((current_time, forced_exit_price))
                in_trade = False

        return buy_coords, sell_coords
    def is_market_open(self: 'Bot', dt_utc: datetime) -> bool:
        """
        Check if a UTC datetime is within U.S. market hours (9:30-16:00 ET).
        """
        # localize to utc
        dt_utc = pytz.utc.localize(dt_utc)
        eastern = pytz.timezone('US/Eastern')
        dt_est = dt_utc.astimezone(eastern)  # Convert UTC datetime to Eastern Time.
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= dt_est.time() <= market_close
    
    def get_last_market_close_index(self: 'Bot') -> datetime | None:
        market_times = [t for t in self.data.index if self.is_market_open(t)]
        return market_times[-1] if market_times else None

    def calculate_indicators(self: 'Bot') -> Dict[str, Any]:
        indicatordata = {}
        rsi = RSIIndicator(pd.Series(self.data['Close'].values.flatten())).rsi()
        indicatordata['RSI'] = rsi

        # Trend Indicators
        macd = MACD(pd.Series(self.data['Close'].values.flatten())).macd()
        indicatordata['MACD'] = macd
        macd_signal = MACD(pd.Series(self.data['Close'].values.flatten())).macd_signal()
        indicatordata['MACD_signal'] = macd_signal
        close_prices = pd.Series(self.data['Close'].values.flatten())

        # Volatility Indicators
        bollinger = BollingerBands(close_prices, window=20, window_dev=2)
        indicatordata['Bollinger_hband'] = bollinger.bollinger_hband()
        indicatordata['Bollinger_lband'] = bollinger.bollinger_lband()
        indicatordata['Bollinger_mavg'] = bollinger.bollinger_mavg()
        indicatordata['Parkinsons_Volatility'] = self.parkinsons_volatility(
            pd.Series(self.data['High'].values.flatten()), 
            pd.Series(self.data['Low'].values.flatten()), 
            window=20
        )

        # Check if there are atleast 14 rows in the dataframe
        if len(self.data) < 14:
            indicatordata['ATR'] = pd.Series(np.nan, index=self.data.index)
        else:
            indicatordata['ATR'] = AverageTrueRange(
                high=pd.Series(self.data['High'].values.flatten()), 
            low=pd.Series(self.data['Low'].values.flatten()), 
            close=pd.Series(self.data['Close'].values.flatten()),
            window=14,
            fillna=False
        ).average_true_range()
        # Volume Indicators


        indicatordata['MACD_buy_signal'] = indicatordata['MACD'] >= indicatordata['MACD_signal']
        indicatordata['signal'] = [0] * len(indicatordata['MACD_buy_signal']) 

        # Calculate the 20-day SMA
        sma_20 = self.data['Close'].rolling(window=20).mean()
        # Create a new column for the 20-day SMA
        indicatordata['SMA20'] = sma_20

        buy_coords, sell_coords = self.process_bars(indicatordata)
                
        indicatordata['buy_coords'] = buy_coords
        indicatordata['sell_coords'] = sell_coords


        
        return indicatordata
    

    def plot_graphs(self: 'Bot', ticker: str) -> None:
        # Create subplots with a shared x-axis.
        fig, axs = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
        axs: List[Axes] = axs

        # Subplot 0: Price chart with Bollinger Bands and Signals
        # Title axs[0] with the ticker and Chart Title
        axs[0].set_title(f'{ticker} Chart')
        axs[0].plot(self.data.index, self.data['Close'], label='Price', color='blue')
        for coord in self.indicatordata['buy_coords']:
            axs[0].scatter(coord[0], coord[1], color='green', marker='^', s=100, label='Buy Signal')
        for coord in self.indicatordata['sell_coords']:
            axs[0].scatter(coord[0], coord[1], color='red', marker='v', s=100, label='Sell Signal')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_hband'], label='Bollinger_hband', color='green', linestyle='--')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_lband'], label='Bollinger_lband', color='red', linestyle='--')
        axs[0].plot(self.data.index, self.indicatordata['Bollinger_mavg'], label='Bollinger_mavg', color='grey', linestyle='-.')
        axs[0].fill_between(self.data.index, self.indicatordata['Bollinger_hband'], self.indicatordata['Bollinger_lband'], color='grey', alpha=0.2)

        # Subplot 1: RSI
        axs[1].plot(self.data.index, self.indicatordata['RSI'], label='RSI', color='purple')
        axs[1].legend()

        # Subplot 2: MACD
        axs[2].plot(self.data.index, self.indicatordata['MACD'], label='MACD', color='blue')
        axs[2].plot(self.data.index, self.indicatordata['MACD_signal'], label='MACD_signal', color='orange')
        axs[2].legend()

        # Subplot 3: ATR
        axs[3].plot(self.data.index, self.indicatordata['ATR'], label='ATR', color='purple')
        axs[3].legend()

        # Subplot 4: Parkinson's Volatility
        axs[4].plot(self.data.index, self.indicatordata['Parkinsons_Volatility'], label="Parkinson's Volatility", color='orange')
        axs[4].legend()
        axs[4].set_xlabel('Time')

        # Create a DateFormatter that shows only hours and minutes.
        time_formatter = DateFormatter("%H:%M", tz=pytz.timezone(self.time_zone))
        # Apply the formatter to the shared x-axis by setting it on the last subplot.
        axs[-1].xaxis.set_major_formatter(time_formatter)
        # Auto-format the x-axis tick labels for better readability.
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig('stockgraphs/' + ticker + '_graph.png', dpi=300)

    def backtest(self: 'Bot', ticker: str) -> None:
        # Get the trade signals from your Bot's calculated indicators.
        buy_coords = self.indicatordata.get("buy_coords", [])
        sell_coords = self.indicatordata.get("sell_coords", [])

def evaluate_trades():
    # Read the NASDAQ symbols from a CSV file.
    nasdaq_df = pd.read_csv("nasdaq.csv")
    # Randomaly sample 200 symbols from the NASDAQ symbols
    # symbols = nasdaq_df["Symbol"].sample(n=30).tolist()
    # Get top movers
    symbols = TopMovers(40).symbols
    trade_results = []   # List to store trade details for each symbol
    
    for symbol in symbols:
        try:
            # Instantiate your Bot (using pre_post=True if desired)
            bot = Bot(symbol, pre_post=True, start_date="2025-02-28", end_date="2025-02-28")
            # Plot the graph
            bot.plot_graphs(symbol)
            # Get the trade signals from your Bot's calculated indicators.
            buy_coords = bot.indicatordata.get("buy_coords", [])
            sell_coords = bot.indicatordata.get("sell_coords", [])
            
            # If we have at least one buy and one sell signal, use the first pair.
            if buy_coords and sell_coords:
                buy_time, buy_price = buy_coords[0]
                sell_time, sell_price = sell_coords[0]
                profit_per_share = sell_price - buy_price
                roc = (profit_per_share / buy_price) * 100  # expressed as a percentage
                trade_results.append({
                    "Symbol": symbol,
                    "Buy Time": buy_time,
                    "Buy Price": buy_price,
                    "Sell Time": sell_time,
                    "Sell Price": sell_price,
                    "Profit per Share": profit_per_share,
                    "ROC (%)": roc
                })
            else:
                # If no trade signals were generated, record empty trade info.
                trade_results.append({
                    "Symbol": symbol,
                    "Buy Time": None,
                    "Buy Price": None,
                    "Sell Time": None,
                    "Sell Price": None,
                    "Profit per Share": None,
                    "ROC (%)": None
                })
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            trade_results.append({
                "Symbol": symbol,
                "Buy Time": None,
                "Buy Price": None,
                "Sell Time": None,
                "Sell Price": None,
                "Profit per Share": None,
                "ROC (%)": None
            })
    
    # Convert results into a DataFrame and write to CSV.
    summary_df = pd.DataFrame(trade_results)
    summary_df.to_csv("trade_summary.csv", index=False)
    print("Trade summary saved to trade_summary.csv")
    
    return summary_df

def generate_performance_stats(trade_df: pd.DataFrame) -> pd.DataFrame:
    # Filter out trades where trade data is missing.
    valid_trades = trade_df.dropna(subset=["Buy Price", "Sell Price"])
    total_trades = len(valid_trades)
    
    # Identify winning and losing trades.
    winning_trades = valid_trades[valid_trades["Profit per Share"] > 0]
    losing_trades = valid_trades[valid_trades["Profit per Share"] <= 0]
    
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    total_profit = valid_trades["Profit per Share"].sum()
    avg_profit = valid_trades["Profit per Share"].mean() if total_trades > 0 else 0
    avg_ROC = valid_trades["ROC (%)"].mean() if total_trades > 0 else 0
    
    stats = {
         "Total Trades": total_trades,
         "Winning Trades": len(winning_trades),
         "Losing Trades": len(losing_trades),
         "Win Rate (%)": win_rate,
         "Total Profit": total_profit,
         "Average Profit per Trade": avg_profit,
         "Average ROC (%)": avg_ROC
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv("performance_stats.csv", index=False)
    print("Performance statistics saved to performance_stats.csv")
    
    return stats_df

def export_detailed_summary_csv(all_results: dict, filename: str = 'detailed_backtest_summary.csv') -> None:
    """
    Export a detailed summary of trades from the backtest results to a CSV file.
    Each row represents an individual trade.
    """
    rows = []
    for symbol, data in all_results.items():
        trades = data.get('trades', [])
        if trades:
            for trade in trades:
                rows.append({
                    'Symbol': symbol,
                    'Date': trade.get('date'),
                    'Buy Price': trade.get('buy_price'),
                    'Sell Price': trade.get('sell_price'),
                    'Profit': trade.get('profit'),
                    'Profit (%)': trade.get('profit_percent')
                })
        else:
            rows.append({
                'Symbol': symbol,
                'Date': None,
                'Buy Price': None,
                'Sell Price': None,
                'Profit': None,
                'Profit (%)': None
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(filename, index=False)
    print(f"Detailed trade summary saved to {filename}")

def export_overall_performance_csv(overall_stats: dict, filename: str = 'overall_performance_summary.csv') -> None:
    """
    Export the overall performance statistics of the strategy to a CSV file.
    """
    overall_df = pd.DataFrame([overall_stats])
    overall_df.to_csv(filename, index=False)
    print(f"Overall performance summary saved to {filename}")

def backtest(symbols: list, start_date: str, end_date: str) -> Tuple[dict, dict]:
    """
    Backtest the trading strategy across multiple symbols between start_date and end_date.
    Returns:
        Tuple containing:
          - all_results: A dictionary with detailed trade results per symbol.
          - overall_stats: A dictionary with aggregated performance metrics.
    """
    # Generate list of business days (B) between the dates.
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    all_results = {}
    overall_stats = {
        'total_trades': 0,
        'profitable_trades': 0,
        'total_profit': 0,
        'best_trade_symbol': '',
        'best_trade_date': '',
        'best_trade_profit_percent': 0,
        'worst_trade_symbol': '',
        'worst_trade_date': '',
        'worst_trade_profit_percent': 0
    }
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol} from {start_date} to {end_date}")
        results = []
        total_trades = 0
        profitable_trades = 0
        total_profit = 0
        
        for i in range(len(date_range) - 1):
            current_date = date_range[i].strftime('%Y-%m-%d')
            try:
                bot = Bot(ticker=symbol, start_date=current_date, end_date=current_date)
                indicators = bot.calculate_indicators()
                buy_coords = indicators.get('buy_coords', [])
                sell_coords = indicators.get('sell_coords', [])
                
                if buy_coords and sell_coords:
                    buy_price = buy_coords[0][1]
                    sell_price = sell_coords[0][1]
                    profit = sell_price - buy_price
                    profit_percent = (profit / buy_price) * 100
                    
                    results.append({
                        'date': current_date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'profit': profit,
                        'profit_percent': profit_percent
                    })
                    
                    total_trades += 1
                    if profit > 0:
                        profitable_trades += 1
                    total_profit += profit
                    
                    # Update best trade if applicable.
                    if profit_percent > overall_stats['best_trade_profit_percent']:
                        overall_stats['best_trade_profit_percent'] = profit_percent
                        overall_stats['best_trade_symbol'] = symbol
                        overall_stats['best_trade_date'] = current_date
                    # Update worst trade if applicable.
                    if (profit_percent < overall_stats['worst_trade_profit_percent'] or 
                        overall_stats['worst_trade_profit_percent'] == 0):
                        overall_stats['worst_trade_profit_percent'] = profit_percent
                        overall_stats['worst_trade_symbol'] = symbol
                        overall_stats['worst_trade_date'] = current_date
                        
            except Exception as e:
                print(f"Error backtesting {symbol} for {current_date}: {e}")
        
        all_results[symbol] = {
            'trades': results,
            'summary': {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate (%)': (profitable_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_profit': total_profit,
                'average_profit': (total_profit / total_trades) if total_trades > 0 else 0
            }
        }
        
        overall_stats['total_trades'] += total_trades
        overall_stats['profitable_trades'] += profitable_trades
        overall_stats['total_profit'] += total_profit
    
    overall_stats['overall_win_rate (%)'] = ((overall_stats['profitable_trades'] / overall_stats['total_trades'] * 100)
                                             if overall_stats['total_trades'] > 0 else 0)
    overall_stats['overall_average_profit'] = ((overall_stats['total_profit'] / overall_stats['total_trades'])
                                               if overall_stats['total_trades'] > 0 else 0)
    
    
    return all_results, overall_stats

if __name__ == "__main__":
    # Randomly sample 50 symbols from NASDAQ.
    nasdaq_df = pd.read_csv("nasdaq.csv")
    symbols = nasdaq_df["Symbol"].sample(n=50).tolist()
    all_results, overall_stats = backtest(symbols, "2025-02-01", "2025-03-03")
    export_detailed_summary_csv(all_results, filename='detailed_backtest_summary.csv')
    export_overall_performance_csv(overall_stats, filename='overall_performance_summary.csv')