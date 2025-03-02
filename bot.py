

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
    def __init__(self: 'Bot', ticker: str, pre_post: bool = False, time_zone: str = 'America/Los_Angeles') -> None:
        self.ticker: str = ticker # type: str
        self.period: str = "1d" # type: str
        self.interval: str = "1m" # type: str
        self.pre_post: bool = pre_post # type: bool
        self.time_zone: str = time_zone # type: str
        self.data: pd.DataFrame = self.fetch_data() # type: pd.DataFrame | None
        self.indicatordata: Dict[str, Any] = self.calculate_indicators() # type: Dict[str, Any] | None

    def fetch_data(self: 'Bot') -> pd.DataFrame:
        # Get the data from yfinance
        
        data = Yfetch(self.ticker, 
                      self.period, 
                      self.interval, 
                      self.pre_post)
        df = data.get_chart_dataframe()
        return df
    
    def parkinsons_volatility(self: 'Bot', high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        log_hl = np.log(high / low)  # Log of High/Low ratio
        squared_term : pd.Series = log_hl ** 2
        k = 1 / (4 * np.log(2) * window)
        return np.sqrt(k * squared_term.rolling(window=window).sum())

    def bollinger_buy_signal(self, indicatordata: pd.DataFrame, i: int, tolerance: float = 0.15) -> bool:
        """
        Returns True if the current close is near (within a tolerance) the lower Bollinger band.
        """
        price = self.data['Close'].iloc[i]
        lower_band = indicatordata['Bollinger_lband'][i]
        return price <= lower_band * (1 + tolerance)

    def bollinger_sell_signal(self, indicatordata: pd.DataFrame, i: int, tolerance: float = 0.15) -> bool:
        """
        Returns True if the current close is near (within a tolerance) the upper Bollinger band.
        """
        price = self.data['Close'].iloc[i]
        upper_band = indicatordata['Bollinger_hband'][i]
        return price >= upper_band * (1 - tolerance)

    def atr_buy_signal(self, indicatordata: pd.DataFrame, i: int, atr_threshold: float = 3) -> bool:
        """
        Returns True if the ATR is below a given threshold.
        """
        return indicatordata['ATR'][i] < atr_threshold

    def parkinsons_buy_signal(self, indicatordata: pd.DataFrame, i: int, parkinson_threshold: float = 0.15) -> bool:
        """
        Returns True if Parkinson's volatility is below a given threshold.
        """
        return indicatordata['Parkinsons_Volatility'][i] < parkinson_threshold

        
    def rsi_buy_signal(self: 'Bot', indicatordata: pd.DataFrame, i: int) -> bool:
        # Buy if RSI is below 40
        if indicatordata['RSI'][i] < 40:
            return True
        return False

    def rsi_sell_signal(self: 'Bot', indicatordata: pd.DataFrame, i: int) -> bool:
        # Sell if RSI is between 60 and 70
        if indicatordata['RSI'][i] > 60: 
            return True
        return False
    
    # Write a function that returns true or false
    def macd_buy_signal(self: 'Bot', indicatordata: pd.DataFrame, i: int) -> bool:
        # First check if the the macd signal and macd are both less than 0.01
        if indicatordata['MACD_signal'][i] < 0 and indicatordata['MACD'][i] < 0:
            if indicatordata['MACD_buy_signal'][i] == True and indicatordata['MACD_buy_signal'][i-1] == False:
                return True
        return False
    
    def macd_sell_signal(self: 'Bot', indicatordata: pd.DataFrame, i: int) -> bool:
        # First check if there was already a buy signal at or before this index
        if indicatordata['MACD_buy_signal'][i] == False and indicatordata['MACD_buy_signal'][i-1] == True:
            return True
        return False
    
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

        buy_coords = []
        sell_coords = []
        found_buy = False
        found_sell = False
        # Retrieve the index of the index where the timestamp is 20:59 for pre and post market
        market_close_index = self.get_last_market_close_index()

        for i in range(1, len(indicatordata['MACD_buy_signal'])):
            index_datetime = self.data.index[i]
            if not self.is_market_open(index_datetime):
                continue

            # Buy signal: all conditions must be met.
            if (self.macd_buy_signal(indicatordata, i) and 
                self.rsi_buy_signal(indicatordata, i) and 
                self.bollinger_buy_signal(indicatordata, i) and 
                self.atr_buy_signal(indicatordata, i) and 
                self.parkinsons_buy_signal(indicatordata, i) and 
                not found_buy and not found_sell):
                buy_coords.append((self.data.index[i], self.data['Close'].iloc[i]))
                found_buy = True

            # Sell signal: here you might want to check if the price is near the upper Bollinger band.
            if (self.macd_sell_signal(indicatordata, i) and 
                self.rsi_sell_signal(indicatordata, i) and 
                self.bollinger_sell_signal(indicatordata, i) and 
                found_buy and not found_sell):
                sell_coords.append((self.data.index[i], self.data['Close'].iloc[i]))
                found_sell = True

            if found_buy and found_sell:
                break

            # If there was a buy but no sell, force a sell at the last market timestamp.
            if self.data.index[i] == market_close_index and not found_sell and found_buy:
                sell_coords.append((self.data.index[i], self.data['Close'].iloc[i]))
                found_sell = True
                

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

class SimulatedBot:
    def __init__(self: 'SimulatedBot', ticker: str, balance: float = 1000.0) -> None:
        self.ticker: str = ticker
        self.balance: float = balance
        self.position: Dict[str, Any] = {}  # empty dict means no open position
        self.trade_log: List[Dict[str, Any]] = []

    def is_market_open(self) -> bool:
        # Check if the market is open for the datetime we started running the bot
        return True
    
    def run(self) -> None:
        # Check if the market is open for the datetime we started running the bot
        while self.is_market_open():
            # Create bot instance
            bot = Bot(self.ticker)
            # Use the most recent index (latest minute data)
            i = len(bot.data) - 1
            current_time = bot.data.index[i]
            current_price = bot.data['Close'].iloc[i]
            indicatordata = bot.indicatordata

            # Check if the indicatordata has buy coords
            if 'buy_coords' in indicatordata:
                buy_coords = indicatordata['buy_coords']
                # If there is a buy coord,
                if len(buy_coords) > 1:
                    # Buy at the first buy coord
                    buy_time = buy_coords[0][0]
                    buy_price = buy_coords[0][1]
                    self.position['buy_time'] = buy_time
                    self.position['buy_price'] = buy_price
                    self.position['quantity'] = self.balance / buy_price
                    
    
        

def evaluate_trades():
    # Read the NASDAQ symbols from a CSV file.
    nasdaq_df = pd.read_csv("nasdaq.csv")
    # Randomaly sample 200 symbols from the NASDAQ symbols
    symbols = nasdaq_df["Symbol"].sample(n=600).tolist()
    
    trade_results = []  # List to store trade details for each symbol
    
    for symbol in symbols:
        try:
            # Instantiate your Bot (using pre_post=True if desired)
            bot = Bot(symbol, pre_post=True)
            
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

if __name__ == "__main__":
    # First, evaluate trades for all symbols and save the trade summary.
    trade_summary_df = evaluate_trades()
    
    # Next, generate performance statistics from the trade summary and save to CSV.
    performance_stats_df = generate_performance_stats(trade_summary_df)
# # Test the class
# if __name__ == "__main__":

#     # top_movers = TopMovers(40).symbols
#     # print(top_movers)
#     tickers = ['TRNR', 'OMI', 'ACON', 'BTAI', 'GRRR', 'PRGO', 'PBYI', 'ALHC', 'RMNI','CTM', 'IBRX']
#     # Test on all the tickers
#     for ticker in tickers:
#         bot = Bot(ticker, pre_post=True)
#         bot.plot_graphs(ticker)

