import requests
from constants import Constants
from chartparse import ChartParse
from typing import Dict, Optional, List, Any, Tuple
import pandas as pd
from datetime import datetime
import pytz
import traceback
class Yfetch:
    def __init__(self, 
                 symbol: str, 
                 period: Optional[str] = None,
                 interval: Optional[str] = '1m',
                 pre_post: bool = False,
                 start_date: str = None,
                 end_date: str = None,
                 timezone: str = 'America/Los_Angeles',
                 str_format: str = '%H:%M'):
        self.symbol = symbol
        self.period = period
        self.pre_post = pre_post
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.str_format = str_format
        self.timezone = timezone
        self._news = None

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

    def _post_request(self, 
                      endpoint: str, 
                      data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform a POST request to the specified endpoint with given data."""
        try:
            response = requests.post(endpoint, headers=Constants._HEADERS, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"POST request failed: {e}")
            return None

    def _build_chart_url(self) -> str:
        """Construct the URL for fetching chart data."""
        return f"{Constants._BASE_URL}{Constants._endpoints['chart'].format(symbol=self.symbol)}"


    def get_chart_dataframe(self) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch chart data and return it as a DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the chart data.
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames, one for the regular market and one for the pre and post market.
        """
        # Must have start,end, and interval, or just period and interval
        try: 
            chart_url = self._build_chart_url()

            if (
                self.start_date and 
                self.end_date and 
                self.interval and 
                not self.period
            ):
                chart_url = self._build_chart_url()
                # Put star tand end date in localized utz to timezone convert to string
                # Conver date to seconds
                start_date = int(pd.Timestamp(self.start_date).timestamp())
                if self.start_date == self.end_date:
                    # Make sure the end date only goes up to last minute of the start date
                    end_date = start_date + 86400
                else:
                    end_date = int(pd.Timestamp(self.end_date).timestamp())
                params = {
                    'period1': start_date,
                    'period2': end_date,
                    'interval': self.interval,
                    'includePrePost': self.pre_post
                }
                response = self._get_request(chart_url, params)
                chart_parse = ChartParse(response, start_date, end_date)
            elif (
                self.period and 
                self.interval and 
                not self.start_date and 
                not self.end_date
            ):
                
                params = {
                    'range': self.period,
                    'interval': self.interval
                }
                response = self._get_request(chart_url, params)
                chart_parse = ChartParse(response)
            else:
                raise ValueError("Invalid parameters")
            return chart_parse.get_dataframe()
        except Exception as e:
            return None
        

# Example usage
if __name__ == "__main__":
    yfetch_instance = Yfetch(symbol="AAPL", pre_post=False, start_date="2025-02-28", end_date="2025-02-28", timezone="America/Los_Angeles")
    # Get the dataframe
    df = yfetch_instance.get_chart_dataframe()
    print(df)
