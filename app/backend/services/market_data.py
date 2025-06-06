# Market Tools
# Handles polling of real-time & historical market data

from typing import Optional, Dict, List
from functools import lru_cache
from utils.config import Config
import yfinance as yf
import requests
import logging

ALPHAVANTAGE_API_KEY = Config.ALPHAVANTAGE_API_KEY
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
logger = logging.getLogger("finbreaker")


class MarketService:

    def search_ticker(company_name: str) -> Dict[str]:
        """
        Search for the most relevant ticker symbol for a given company name.
        
        Args:
            company_name (str): Name of the company to search for.
            api_key (str): Your Alpha Vantage API key.
        
        Returns:
            dict: The most relevant result (symbol, name, region), or None if not found.
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": company_name,
            "apikey": ALPHAVANTAGE_API_KEY
        }

        try:
            response = requests.get(ALPHAVANTAGE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            best_matches = data.get("bestMatches", [])
            if not best_matches:
                print(f"No results found for '{company_name}'.")
                return None

            # Return the most relevant match (first one)
            best = best_matches[0]
            return {
                "symbol": best["1. symbol"],
                "name": best["2. name"],
                "region": best["4. region"]
            }

        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None


    def fetch_time_series_market_data(ticker: str, period: str = "1d", interval: str = "1d") -> Dict[str]:
        """
        Fetch real-time and historical market data for a given ticker symbol
        
        Args:
            ticker (str): ticker symbol for the company
            period (str): time period
            interval (str): time interval

        Returns:
            dict : ticker's time series market data
        """
        logger.info(f"Fetching market data for {ticker} (period={period}, interval={interval}")

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": ALPHAVANTAGE_API_KEY
        }

        try:
            response = requests.get(ALPHAVANTAGE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"AlphaVantage data fetched for {ticker}")
            return data

        except requests.RequestException as e:
            logger.info("AlphaVantage failed, now trying yFinance")

            data = yf.Ticker(ticker).history(period=period, interval=interval)
            if not data.empty:
                logger.info(f"yfinance data fetched for {ticker}")
                return data.tail(1).to_dict()
            else:
                logger.warning(f"No data found for {ticker}")
                return {"error": "No data found"}

            
    def fetch_earnings(ticker: str) -> Dict:
        """
        Fetch earnings data for a given ticker symbol.
        
        Args:
            ticker (str): ticker symbol for the company

        Returns:
            dict : ticker's earnings data
        """
        logger.info(f"Fetching earnings for {ticker}")
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        if earnings is not None:
            logger.info(f"Earnings data found for {ticker}")
            return earnings.head(1).to_dict()
        logger.warning(f"No earnings data found for {ticker}")
        return {"error": "No earnings data found"}


    def fetch_company_news(ticker: str) -> Dict:
        """
        Fetch market news for a given ticker symbol (eg. GOOGL, AAPL etc)
        
        Args:
            ticker (str): topic ticker that news data is required for

        Returns:
            dict: news data from the tickers 
        """
        logger.info(f"Fetching news data for symbol: {ticker})")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": ALPHAVANTAGE_API_KEY
        }

        try:
            response = requests.get(ALPHAVANTAGE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"News data fetched for {ticker}")
            return data

        except requests.RequestException as e:
            return {"error": "No data found"}


    def fetch_topic_news(tickers: List[str]) -> Dict:
        """
        Fetch market news for given topic tickers.\n
        Supported topics (in format "topic name: ticker_name"):

        Blockchain: blockchain,
        Earnings: earnings,
        IPO: ipo,
        Mergers & Acquisitions: mergers_and_acquisitions,
        Financial Markets: financial_markets,
        Economy - Fiscal Policy (e.g., tax reform, government spending): economy_fiscal,
        Economy - Monetary Policy (e.g., interest rates, inflation): economy_monetary,
        Economy - Macro/Overall: economy_macro,
        Energy & Transportation: energy_transportation,
        Finance: finance,
        Life Sciences: life_sciences,
        Manufacturing: manufacturing,
        Real Estate & Construction: real_estate,
        Retail & Wholesale: retail_wholesale,
        Technology: technology

        Args:
            tickers (List[str]): list of topic tickers that news data is required for
        
        Returns:
            dict: news data from the tickers 

        """
        logger.info(f"Fetching news data for topic: {tickers})")

        ticker_names = [
            "blockchain",
            "earnings",
            "ipo",
            "mergers_and_acquisitions",
            "financial_markets",
            "economy_fiscal",
            "economy_monetary",
            "economy_macro",
            "energy_transportation",
            "finance",
            "life_sciences",
            "manufacturing",
            "real_estate",
            "retail_wholesale",
            "technology"
        ]
        tickers = [ticker for ticker in tickers if ticker in ticker_names] # Keep only valid symbols
        ticker_str = ','.join(tickers)

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker_str,
            "apikey": ALPHAVANTAGE_API_KEY
        }

        try:
            response = requests.get(ALPHAVANTAGE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"News data fetched for {tickers}")
            return data

        except requests.RequestException as e:
            return {"error": "No data found"}


    def fetch_stock_trends(ticker: str) -> Dict:
        """
        Fetch recommendation trends for a given ticker symbol (eg. GOOGL, AAPL etc)
        
        Args:
            ticker (str): topic ticker that trends data is required for

        Returns:
            dict: trends data from the tickers 
        """
        logger.info(f"Fetching stock trends data for {ticker})")

        try:
            response = requests.get(f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={Config.FINNHUB_API_KEY}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"News data fetched for {ticker}")
            return data

        except requests.RequestException as e:
            return {"error": "No data found"}
    

@lru_cache
def get_market_service() -> MarketService:
    return MarketService()