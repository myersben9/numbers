import requests
from constants import Constants
from chartparse import ChartParse
from typing import Dict, Optional, Any
import pandas as pd

class Yfetch:
    """
        Yfetch is a class that fetches data from the Yfinance API.
        It is used to fetch chart data

        Args:
            ticker: The ticker symbol of the stock to fetch data for.
            range: The range of data to fetch.
            interval: The interval of data to fetch.
            pre_post: Whether to include pre and post market data.
            start_date: The start date of the data to fetch.
            end_date: The end date of the data to fetch.
            timezone: The timezone of the data to fetch.
            dataframe: The dataframe to store the data in.

        Returns:
            A dataframe containing the chart data.
        """
    def __init__(self, 
                 ticker: str, 
                 interval: str = "5m",
                 pre_post: bool = False,
                 timezone: str = 'America/Los_Angeles',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 range: Optional[str] = None):

        self.ticker = ticker
        self.range = range
        self.interval = interval
        self.pre_post = pre_post
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone
        self.df = self._get_chart_df()

    def _get_request(self, 
                     endpoint: str, 
                     params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform a GET request to the specified endpoint with given parameters."""
        try:
            response = requests.get(endpoint, headers=Constants._HEADERS, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"GET request failed: {e}")
            return None

    def _build_chart_url(self) -> str:
        """Construct the URL for fetching chart data."""
        return f"{Constants._BASE_URL}{Constants._endpoints['chart'].format(ticker=self.ticker)}"

    def _get_chart_df(self) -> pd.DataFrame:
        """
            Get the chart data as a dataframe.
        """
        try: 
            chart_url = self._build_chart_url()
            params = {
                'interval': self.interval,
                'includePrePost': self.pre_post
            }
            if (
                self.start_date and 
                self.end_date and 
                not self.range
            ):
                start_date = pd.to_datetime(self.start_date).tz_localize(self.timezone).tz_convert('UTC')
                end_date = pd.to_datetime(self.end_date).tz_localize(self.timezone).tz_convert('UTC')

                if self.pre_post:
                    # Pre market start time
                    start_date = start_date + pd.Timedelta(hours=1)
                    # Post market end time
                    end_date = end_date + pd.Timedelta(hours=17)
                else:
                    # Regular Market start time
                    start_date = start_date + pd.Timedelta(hours=5.5)
                    # Regular Market end time
                    end_date = end_date + pd.Timedelta(hours=24)

                # int the timestamps
                start_date = int(start_date.timestamp())
                end_date = int(end_date.timestamp())

                if start_date == end_date:
                    end_date = start_date + 86400
                elif start_date > end_date:
                    raise ValueError("Start date must be before end date")

                params['period1'] = start_date
                params['period2'] = end_date

                response = self._get_request(chart_url, params)
                df = ChartParse(response, 
                                self.timezone, 
                                start_date, 
                                end_date,
                                self.pre_post).df
            elif (
                self.range and 
                not self.start_date and 
                not self.end_date
            ):
                params['range'] = self.range   
                response = self._get_request(chart_url, params)
                df = ChartParse(response, 
                                self.timezone, 
                                pre_post=self.pre_post).df
            else:
                print(self.start_date, self.end_date, self.range)
                raise ValueError("Invalid parameters")
            return df
        except:
            raise ValueError("Invalid parameters or request failed")
        

# Example usage
if __name__ == "__main__":
    df = Yfetch(ticker="AAPL", 
                start_date="2025-02-21", 
                end_date="2025-02-21",
                timezone="America/Los_Angeles",
                pre_post=True).df
    print(df)
