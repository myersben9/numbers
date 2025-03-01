

class Constants:
    _USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    _HEADERS = {
        "User-Agent": _USER_AGENT
    }
    _BASE_URL = 'https://query1.finance.yahoo.com'
    _ROOT_URL_ = 'https://finance.yahoo.com'
    _endpoints = {
    'chart': '/v8/finance/chart/{symbol}',
    'news': '/v6/finance/news',
    }