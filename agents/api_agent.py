# API Agent
# Handles polling of real-time & historical market data

from fastapi import APIRouter, Query
from typing import Optional
import yfinance as yf
import os
import requests

router = APIRouter(prefix="/api", tags=["API Agent"])

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

@router.get("/market_data")
def get_market_data(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. TSM"),
    period: str = Query("1d", description="Period for yfinance, e.g. 1d, 5d, 1mo, 1y"),
    interval: str = Query("1d", description="Interval for yfinance, e.g. 1d, 1h, 5m"),
    use_alpha: bool = Query(False, description="Use AlphaVantage instead of yfinance")
):
    if use_alpha and ALPHAVANTAGE_API_KEY:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": "AlphaVantage API error"}
    else:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return data.tail(1).to_dict() if not data.empty else {"error": "No data found"}

@router.get("/earnings")
def get_earnings(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. TSM")
):
    stock = yf.Ticker(ticker)
    earnings = stock.earnings_dates
    if earnings is not None:
        return earnings.head(1).to_dict()
    return {"error": "No earnings data found"}

@router.get("/")
def root():
    return {"status": "API Agent running"}
