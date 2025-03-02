import requests
from constants import Constants
from chartparse import ChartParse
from typing import Dict, Optional, List, Any, Tuple
import pandas as pd

class Yfetch:
    def __init__(self, 
                 symbol: str, 
                 range: str = "5m", 
                 interval: Optional[str] = None,
                 pre_post: bool = False):
        self.symbol = symbol
        self.range = range
        self.pre_post = pre_post
        self.interval = interval
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

    def _build_news_url(self, tab: str) -> str:
        """Construct the URL for fetching news data based on the specified tab."""
        return f"{Constants._ROOT_URL_}/xhr/ncp?queryRef={tab}&serviceKey=ncp_fin"

    def _get_news_payload(self, count: int) -> Dict[str, Any]:
        """Create the payload for the news request."""
        return {
            "serviceConfig": {
                "snippetCount": count,
                "s": [self.symbol]
            }
        }

    def _parse_news_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse the news response and return a list of articles."""
        news = data.get("data", {}).get("tickerStream", {}).get("stream", [])
        return [article for article in news if not article.get('ad', [])]

    def get_news(self, 
                 count: int = 10, 
                 tab: str = "news") -> List[Dict[str, Any]]:
        """Fetch news articles specific to the ticker."""
        if self._news:
            return self._news

        tab_queryrefs = {
            "all": "newsAll",
            "news": "latestNews",
            "press releases": "pressRelease",
        }

        query_ref = tab_queryrefs.get(tab.lower())
        if not query_ref:
            raise ValueError(f"Invalid tab name '{tab}'. Choose from: {', '.join(tab_queryrefs.keys())}")

        url = self._build_news_url(query_ref)
        payload = self._get_news_payload(count)

        data = self._post_request(url, payload)
        if data is None or "Will be right back" in data.get('text', ''):
            raise RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***\n"
                               "Our engineers are working quickly to resolve "
                               "the issue. Thank you for your patience.")

        self._news = self._parse_news_response(data)
        return self._news

    def get_chart_dataframe(self) -> pd.DataFrame:
        """
        Fetch chart data and return it as a DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the chart data.
        """
        chart_url = self._build_chart_url()
        params = {
            'range': self.range,
            'interval': self.interval,
            'includePrePost': self.pre_post
        }
        response = self._get_request(chart_url, params)
        chart_parse = ChartParse(response)
        return chart_parse.get_dataframe()

# Example usage
if __name__ == "__main__":
    yfetch_instance = Yfetch(symbol="AAPL", range="1d", interval="1m", pre_post=False)
    # Get the dataframe
    df = yfetch_instance.get_chart_dataframe()
    print(df)
