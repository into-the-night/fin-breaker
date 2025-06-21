from fastapi import APIRouter, Request
import requests
from typing import List
from crewai import Crew, Agent, Task, LLM
from crewai.tools import tool
import logging
from app.backend.services.market_data import MarketDataService
from app.backend.services.retrieval import VectorStoreService
from app.backend.services.synthesis import LLMService

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])
logger = logging.getLogger("finbreaker")

vector_store = VectorStoreService()


@tool("Market Data")
def market_data_tool(ticker: str, period: str = "1d", interval: str = "1d"):
    """Fetch real-time and historical market data for a given ticker symbol."""
    return MarketDataService.fetch_time_series_market_data(
        ticker=ticker,
        period=period,
        interval=interval,
    )

@tool("Earnings")
def earnings_tool(ticker: str):
    """Fetch earnings data for a given ticker symbol."""
    return MarketDataService.fetch_earnings(ticker)

@tool("Company News")
def company_news_tool(ticker: str):
    """Fetch company news for a given ticker symbol."""
    return MarketDataService.fetch_company_news(ticker)

@tool("Ticker Search")
def ticker_search_tool(company_name: str):
    """Search for the most relevant ticker symbol for a given company name."""
    return MarketDataService.search_ticker(company_name)

@tool("Topic News")
def topic_news_tool(tickers: List[str]):
    """Fetch market news for given topic tickers."""
    return MarketDataService.fetch_topic_news(tickers)

@tool("Index Data")
def index_data(docs: List[str]):
    """Indexes the market data in a vector store to be used for querying."""
    return vector_store.index_documents(docs)

@tool("Retriever")
def retriever_tool(query: str, max_results: int = 3):
    """Retrieve relevant indexed financial documents for a given query using the vector store."""
    return vector_store.retrieve(query, max_results)


orchestrator_agent = Agent(
    role="Market Analyst Agent",
    goal="Answer complex finance questions using all available tools.",
    backstory="You are an expert financial analyst with access to market data, filings, analytics, and document retrieval.",
    tools=[ticker_search_tool, market_data_tool, earnings_tool, company_news_tool, topic_news_tool, index_data, retriever_tool]
)

tasks = [
    Task(
        agent=orchestrator_agent,
        description="Assist a financial analyst in understading the data related to the question.",
        expected_output="A comprehensive, analysis of the acquired information with the sources.",
    )
]

llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.3)

crew = Crew(
    agents=[orchestrator_agent],
    process="sequential",
    tasks=tasks,
    llm=llm
)


@router.post("/morning_brief")
async def morning_brief(request: Request):
    """Orchestrate the full multi-agent workflow: transcribe audio (if needed), run CrewAI agents for market data, filings, risk analysis, and retrieval, then synthesize a final answer using the language agent and return both text and TTS audio."""
    data = await request.json()
    question = data.get("question")
    logger.info(f"Received question for morning brief: {question}")

    results = 
    # Synthesize a final answer using the language agent
    context_chunks = [str(r) for r in results]
    answer = LLMService.synthesize_with_context(question, context_chunks)
    return {"transcript": question, "answer": answer}

@router.get("/")
def root():
    """Health check endpoint for the orchestrator service."""
    return {"status": "Orchestrator running"}
