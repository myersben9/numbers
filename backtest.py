from bot import Bot 
from movers import TopMovers
import pandas as pd
from typing import List, Union
class Backtest:
    def __init__(self: 'Backtest', 
                 tickers: Union[List[str], str] = [],
                 position: float = 10000,
                 interval: str = "5m",
                 range: str = None,
                 share_size: float = 0.01,
                 start_date: str = None,   
                 end_date: str = None,) -> None:
    
        self.tickers = tickers
        self.position = position
        self.range = range
        self.share_size = share_size
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def backtest(self: 'Backtest') -> None:
        """
            Backtest the bot.
        """
        trades_coords = {}
        if isinstance(self.tickers, str):
            self.tickers = [self.tickers]

        for ticker in self.tickers:
            bot = Bot(ticker=ticker,
                  range=self.range,
                  interval=self.interval,
                  start_date=self.start_date, 
                  end_date=self.end_date)

            buy_coords = bot.process_bars()[0]
            sell_coords = bot.process_bars()[1]

            trades_coords[ticker] = {
                'buy_coords': buy_coords,
                'sell_coords': sell_coords
            }
        position = self.position 
        profit = 0
        shares = 0
        trade_count = 0
        win_count = 0 
        loss_count = 0
        win_return = 0
        loss_return = 0
        
        # Create a dictionary to store the trades
        trades = []
        # Iterate through the buy and sell coordinates
        for ticker in trades_coords:
            buy_coords = trades_coords[ticker]['buy_coords']
            sell_coords = trades_coords[ticker]['sell_coords']

            for buy_coord, sell_coord in zip(buy_coords, sell_coords):
                buy_price = buy_coord[1]
                sell_price = sell_coord[1]
                profit_per_share = sell_price - buy_price
                
                number_of_shares = self.position * self.share_size / buy_price
                profit += profit_per_share * number_of_shares
                shares += number_of_shares
                trade_count += 1
                position += profit_per_share * number_of_shares

                if profit_per_share > 0:
                    win_count += 1
                    win_return += profit_per_share * number_of_shares
                else:
                    loss_count += 1
                    loss_return += profit_per_share * number_of_shares\
                    
                # Add the trade to the trades list
                trades.append({
                    'buytime': buy_coord[0],
                    'selltime': sell_coord[0],
                    'ticker': ticker,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'profit_per_share': profit_per_share
                })
                    
        print(f"Total profit: {profit}")
        print(f"Total shares: {shares}")
        print(f"Total trades: {trade_count}")
        print(f"Win count: {win_count}")
        print(f"Loss count: {loss_count}")
        print(f"Win return: {win_return}")
        print(f"Loss return: {loss_return}")

        overall_stats = {
            'balance': position + profit,
            'total_profit': profit,
            'total_trades': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_return': win_return,
            'loss_return': loss_return
        }
        # Save the overall stats to a csv file
        df = pd.DataFrame(overall_stats, index=[0])
        df.to_csv("overall_stats.csv", index=False)
        df = pd.DataFrame(trades)
        df.to_csv("trades.csv", index=False)


        return trades

if __name__ == "__main__":
    # Get the top 10 movers
    symbols = [
        "GOOG",
        "NVDA",
        "MSFT",
        "AMZN",
        "TSLA",
        "AAPL",
        "TSM",
    ]
    backtest = Backtest(tickers=symbols, 
                        position=10000,
                        share_size= len(symbols) / 100,
                        interval="5m",
                        range="20d")
    backtest.backtest()
