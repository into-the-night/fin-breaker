from app.backend.services.market_data import get_market_data
from app.backend.services.retrieval import get_vector_store

market_data_service = get_market_data()
vector_store_service = get_vector_store()

def search_ticker(query: str) -> str:
    """
    Search for a ticker symbol for a given company name.
    Args:
        query: The name of the company to search for.
    Returns:
        The ticker symbol for the company.
    """
    return market_data_service.search_ticker(query)

def fetch_company_news(ticker: str) -> str:
    """
    Fetch company news for a given ticker symbol.
    Args:
        ticker: The ticker symbol of the company.
    Returns:
        The latest news about the company.
    """
    return market_data_service.fetch_company_news(ticker)

def fetch_earnings(ticker: str) -> str:
    """
    Fetch earnings data for a given ticker symbol.
    Args:
        ticker: The ticker symbol of the company.
    Returns:
        The earnings data for the company.
    """
    return market_data_service.fetch_earnings(ticker)

def fetch_topic_news(topic: str) -> str:
    """
    Fetch news on a given topic.
    Args:
        topic: The topic to search for news on.
    Returns:
        The latest news on the topic.
    """
    return market_data_service.fetch_topic_news(topic)

def fetch_time_series_market_data(ticker: str) -> str:
    """
    Fetch time series market data for a given ticker symbol.
    Args:
        ticker: The ticker symbol of the company.
    Returns:
        The time series market data for the company.
    """
    return market_data_service.fetch_time_series_market_data(ticker)

def retrieve_from_vector_store(query: str) -> str:
    """
    Retrieve relevant documents from the vector store for a given query.
    Args:
        query: The query to retrieve documents for.
    Returns:
        A list of relevant documents.
    """
    return str(vector_store_service.retrieve(query))


def get_tools():
    """Returns a list of all available tools."""
    return [
        search_ticker,
        fetch_company_news,
        fetch_earnings,
        fetch_topic_news,
        fetch_time_series_market_data,
        retrieve_from_vector_store,
    ]

# We also need a mapping from tool name to tool function to execute them.
TOOL_MAP = {
    "search_ticker": search_ticker,
    "fetch_company_news": fetch_company_news,
    "fetch_earnings": fetch_earnings,
    "fetch_topic_news": fetch_topic_news,
    "fetch_time_series_market_data": fetch_time_series_market_data,
    "retrieve_from_vector_store": retrieve_from_vector_store,
} 