import yfinance as yf

# Market Data Loader
class MarketDataLoader:
    def get_market_data(self, ticker, period="1d", interval="1d"):
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return data.tail(1).to_dict() if not data.empty else {"error": "No data found"}
    def get_earnings(self, ticker):
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        if earnings is not None:
            return earnings.head(1).to_dict()
        return {"error": "No earnings data found"}
