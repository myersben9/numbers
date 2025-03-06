import yfinance
from typing import List


# Configuration constants
PERCENTAGE_CHANGE: float = 15
EXCHANGES: List[str] = ["NMS", "NGM", "NCM", "NYQ", "ASE", "PCX"]  # Supported exchanges

class TopMovers:
    """
    A class to fetch top movers from Yahoo Finance based on predefined screening
    criteria. Does not contain permarket movers.

    Attributes:
        percentage_change (float): The minimum percentage change to filter stocks.
        symbols (List[str]): A list of stock symbols that meet the criteria.
        return_message (str): A message to return if no stocks are found.
        info (Dict[str, Union[str,int]]): A dictionary containing the screening info.
    """

    def __init__(self, percentage_change: float = PERCENTAGE_CHANGE):
        """
        Initialize the TopMovers class.

        Args:
            percentage_change (float): The minimum percentage change to filter stocks.
        """
        self.percentage_change = percentage_change
        self.screen = self.screen_percentages()
        self.quotes = self.screen['quotes'] if self.screen and 'quotes' in self.screen else []
        self.symbols = [quote['symbol'] for quote in self.quotes]

    def screen_percentages(self) -> List[str]:
        """
        Fetch stocks from Yahoo Finance based on predefined screening criteria.

        Returns:
            List[str]: A list of stock symbols that meet the criteria.
        """
        try:
            # Define the screening criteria
            q = yfinance.EquityQuery('and', [
                yfinance.EquityQuery('gt', ['percentchange', self.percentage_change]),  # Stocks with > percentage_change
                yfinance.EquityQuery('eq', ['region', 'us']),  # US region
                yfinance.EquityQuery('is-in', ['exchange', *EXCHANGES]),  # Specific exchanges
            ])
            screen = yfinance.screen(q, sortField='percentchange', sortAsc=False, size=250)
            
            screen["quotes"] = [quote for quote in screen["quotes"]]

            return screen

        except:
            raise Exception("Error fetching stocks")

# Example usage
if __name__ == "__main__":
    try:
        top_movers = TopMovers(PERCENTAGE_CHANGE)
        print(top_movers.symbols)
    except Exception as e:
        print(e)