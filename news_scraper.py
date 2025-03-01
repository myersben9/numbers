import requests
from bs4 import BeautifulSoup
import json
import os
from constants import Constants

class NewsScraper:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
        self.articles = []

    def fetch_news(self):
        try:
            response = requests.get(self.url, headers=Constants._HEADERS)
            response.raise_for_status()
        except:
            print(f"Failed to fetch news articles for {self.ticker}: {e}")

    def parse_html(self, html: str):
        soup = BeautifulSoup(html, 'html.parser')
        
        # Update the selector based on the actual HTML structure
        news_items = soup.find_all('li', class_='js-stream-content')  # Adjust this if necessary

        # If the above selector doesn't work, you may need to inspect the HTML and adjust accordingly
        if not news_items:
            print("No news items found. Please check the HTML structure.")
            return

        for item in news_items:
            title_tag = item.find('h3')  # Adjust this if necessary
            title = title_tag.get_text() if title_tag else 'No Title'
            
            link_tag = item.find('a')  # Adjust this if necessary
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else 'No Link'
            
            if not link.startswith('http'):
                link = 'https://finance.yahoo.com' + link  # Ensure full URL
            
            self.articles.append({'title': title, 'link': link})

    def save_to_json(self):
        if not os.path.exists('payloads'):
            os.makedirs('payloads')
        with open(f'payloads/{self.ticker}_news.json', 'w') as f:
            json.dump(self.articles, f, indent=4)
        print(f"Saved {len(self.articles)} articles to payloads/{self.ticker}_news.json")

    def load_and_parse_html(self):
        # Load HTML from file if it exists
        html_file_path = f'payloads/{self.ticker}_news.html'
        if os.path.exists(html_file_path):
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.parse_html(html_content)
            print(f"Parsed HTML content from {html_file_path}")
        else:
            print(f"No HTML file found at {html_file_path}. Please fetch news first.")

if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
    scraper = NewsScraper(ticker)
    
    # Fetch news and save HTML
    scraper.fetch_news()
    
    # Load and parse the saved HTML
    scraper.load_and_parse_html()
    
    # Save articles to JSON
    scraper.save_to_json() 