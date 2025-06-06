# API Agent
# Handles polling of real-time & historical market data

from fastapi import APIRouter, Depends
from services.market_data import MarketDataService, get_market_data
from app.backend.api.schema import MarketDataRequest, EarningsRequest, CompanyNewsRequest, TickerSearchRequest, TopicNewsRequest

router = APIRouter(prefix="/api", tags=["Market API"])

@router.post("/market_data")
def get_market_data(
    request: MarketDataRequest,
    market_service: MarketDataService = Depends(get_market_data)
    ):
    return market_service.fetch_time_series_market_data(
        ticker=request.ticker,
        period=request.period,
        interval=request.interval,
        use_alpha=request.use_alpha
    )

@router.post("/earnings")
def get_earnings(
    request: EarningsRequest,
    market_service: MarketDataService = Depends(get_market_data)
    ):
    return market_service.fetch_earnings(request.ticker)

@router.post("/company_news")
def get_company_news(
    request: CompanyNewsRequest,
    market_service: MarketDataService = Depends(get_market_data)
    ):
    return market_service.fetch_company_news(request.ticker)

@router.post("/search_ticker")
def search_ticker(
    request: TickerSearchRequest,
    market_service: MarketDataService = Depends(get_market_data)
    ):
    return market_service.search_ticker(request.company_name)

@router.post("/topic_news")
def get_topic_news(
    request: TopicNewsRequest,
    market_service: MarketDataService = Depends(get_market_data)
    ):
    return market_service.fetch_topic_news(request.tickers)
    
@router.get("/")
def root():
    return {"status": "API Agent running"}