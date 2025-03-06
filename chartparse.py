# Class to take a payload from payloads/apple_chart.json
# Parse a payload like payloads/apple_chart.json and return the quotes/indicates
# as a dataframe
import pandas as pd
from typing import Dict, Any, Tuple
# Make sure tz_localize is typed


class ChartParse:
    def __init__(self: 'ChartParse', 
                 payload: Dict[str, Any], 
                 timezone: str,
                 start_date: int = None, 
                 end_date: int = None,
                 pre_post: bool = False) -> None:
        self.payload = payload
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone
        self.pre_post = pre_post
        self.df = self._get_df()

    def _get_df(self: 'ChartParse') -> pd.DataFrame:
        """
        Parses the Yahoo Finance chart payload and returns the data as a DataFrame.
        Returns a DataFrame with timestamp as index and OHLCV data as columns.
        """
        # Extract the relevant data from the payload
        
        try:
            chart_data = self.payload['chart']['result'][0]
            quotes = chart_data['indicators']['quote'][0]
            timestamps = chart_data['timestamp']
        except:
            # Return an empty dataframe with the correct columns
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=pd.to_datetime([]))

        opens = quotes.get('open', [])
        highs = quotes.get('high', [])
        lows = quotes.get('low', [])
        closes = quotes.get('close', [])
        volumes = quotes.get('volume', [])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        })
        
        # Convert timestamps to datetime and set as index
        df.index = pd.to_datetime(timestamps, unit='s')
        df.index = df.index.tz_localize('utc').tz_convert(self.timezone)
        df.index.name = 'Timestamp'
        # if self.pre_post and self.start_date and self.end_date:
        #     df = df.iloc[:-1]

        df = df.dropna()
        return df

    
