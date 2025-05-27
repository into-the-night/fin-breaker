# API Agent
# Handles polling of real-time & historical market data

from fastapi import APIRouter, Query
from typing import Optional
import yfinance as yf
import os
import requests
import logging

router = APIRouter(prefix="/api", tags=["API Agent"])

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
logger = logging.getLogger("finbreaker")

@router.get("/market_data")
def get_market_data(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. TSM"),
    period: str = Query("1d", description="Period for yfinance, e.g. 1d, 5d, 1mo, 1y"),
    interval: str = Query("1d", description="Interval for yfinance, e.g. 1d, 1h, 5m"),
    use_alpha: bool = Query(False, description="Use AlphaVantage instead of yfinance")
):
    logger.info(f"Fetching market data for {ticker} (period={period}, interval={interval}, use_alpha={use_alpha})")
    if use_alpha and ALPHAVANTAGE_API_KEY:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
        resp = requests.get(url)
        if resp.status_code == 200:
            logger.info(f"AlphaVantage data fetched for {ticker}")
            return resp.json()
        else:
            logger.error(f"AlphaVantage API error for {ticker}")
            return {"error": "AlphaVantage API error"}
    else:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if not data.empty:
            logger.info(f"yfinance data fetched for {ticker}")
            return data.tail(1).to_dict()
        else:
            logger.warning(f"No data found for {ticker}")
            return {"error": "No data found"}

@router.get("/earnings")
def get_earnings(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. TSM")
):
    logger.info(f"Fetching earnings for {ticker}")
    stock = yf.Ticker(ticker)
    earnings = stock.earnings_dates
    if earnings is not None:
        logger.info(f"Earnings data found for {ticker}")
        return earnings.head(1).to_dict()
    logger.warning(f"No earnings data found for {ticker}")
    return {"error": "No earnings data found"}

@router.get("/")
def root():
    return {"status": "API Agent running"}
