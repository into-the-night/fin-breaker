# API Agent
# Handles polling of real-time & historical market data

from fastapi import APIRouter
from services.market_data import fetch_earnings, fetch_company_news, fetch_time_series_market_data, search_ticker, fetch_topic_news
from app.backend.api.schema import MarketDataRequest, EarningsRequest, CompanyNewsRequest, TickerSearchRequest, TopicNewsRequest

router = APIRouter(prefix="/api", tags=["Market API"])

@router.post("/market_data")
def get_market_data(request: MarketDataRequest):
    return fetch_time_series_market_data(
        ticker=request.ticker,
        period=request.period,
        interval=request.interval,
        use_alpha=request.use_alpha
    )

@router.post("/earnings")
def get_earnings(request: EarningsRequest):
    return fetch_earnings(request.ticker)

@router.post("/company_news")
def get_company_news(request: CompanyNewsRequest):
    return fetch_company_news(request.ticker)

@router.post("/search_ticker")
def search_ticker(request: TickerSearchRequest):
    return search_ticker(request.company_name)

@router.post("/topic_news")
def get_topic_news(request: TopicNewsRequest):
    return fetch_topic_news(request.tickers)
    
@router.get("/")
def root():
    return {"status": "API Agent running"}