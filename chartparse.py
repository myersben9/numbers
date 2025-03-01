# Class to take a payload from payloads/apple_chart.json
# Parse a payload like payloads/apple_chart.json and return the quotes/indicates
# as a dataframe
import pandas as pd
import datetime

class ChartParse:
    def __init__(self, payload):
        self.payload = payload

    def get_dataframe(self) -> pd.DataFrame:
        """
        Parses the Yahoo Finance chart payload and returns the data as a DataFrame.
        Returns a DataFrame with timestamp as index and OHLCV data as columns.
        """
        # Extract the relevant data from the payload
        chart_data = self.payload['chart']['result'][0]
        
        # Get timestamps
        timestamps = chart_data['timestamp']
        
        # Get quote data
        quotes = chart_data['indicators']['quote'][0]
        
        # Extract OHLCV data
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
        # Put timestamps as pst datetime objects
        df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
        df.index = df.index.strftime('%H:%M')
        df.index.name = 'Timestamp'
        # drop rows with nan values
        df = df.dropna()
        
        # Add metadata as attributes to the DataFrame
        if 'meta' in chart_data:
            for key, value in chart_data['meta'].items():
                df.attrs[key] = value
        
        return df 
    
