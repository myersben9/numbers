import pandas as pd
from typing import Dict, Any, List


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
            chart_data : Dict[str, Any] = self.payload['chart']['result'][0]
            quotes : Dict[str, Any] = chart_data['indicators']['quote'][0]
            timestamps : List[int] = chart_data['timestamp']
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

        df = df.dropna()
        return df

    
