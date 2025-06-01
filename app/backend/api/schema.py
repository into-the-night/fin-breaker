from pydantic import BaseModel, Field
from typing import List

class MarketDataRequest(BaseModel):
    ticker: str
    period: str = "1d"
    interval: str = "1d"
    use_alpha: bool = False

class EarningsRequest(BaseModel):
    ticker: str

class CompanyNewsRequest(BaseModel):
    ticker: str

class TickerSearchRequest(BaseModel):
    company_name: str

class TopicNewsRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of topic tickers for news (e.g. ['blockchain', 'earnings'])")