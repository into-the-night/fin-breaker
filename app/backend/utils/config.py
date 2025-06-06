import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")