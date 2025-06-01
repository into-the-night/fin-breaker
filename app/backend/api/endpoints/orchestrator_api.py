from fastapi import APIRouter, Request
import requests
from crewai import Crew, Agent, Task, LLM
from crewai.tools import tool
import logging

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])
logger = logging.getLogger("finbreaker")

# Define CrewAI tools for each microservice using @tool decorator
@tool("Market Data")
def market_data_tool(ticker: str):
    """Fetch real-time and historical market data for a given ticker symbol using the API agent."""
    return requests.get(f"http://localhost:8000/api/market_data?ticker={ticker}").json()

@tool("Filings Scraper")
def filings_scraper_tool(ticker: str):
    """Fetch the latest SEC or company filings for a given ticker symbol using the scraping agent."""
    return requests.get(f"http://localhost:8000/scraping/filing?ticker={ticker}").json()

@tool("Risk Analysis")
def risk_analysis_tool(region: str, sector: str, allocations: list = None):
    """Analyze risk exposure for a given region and sector using the analysis agent."""
    if allocations is None:
        allocations = []
    return requests.post(
        "http://localhost:8000/analysis/risk_exposure",
        json={"region": [region], "sector": [sector], "allocations": allocations}
    ).json()

@tool("Retriever")
def retriever_tool(query: str):
    """Retrieve relevant indexed financial documents for a given query using the retriever agent."""
    return requests.get(f"http://localhost:8000/retriever/retrieve?query={query}").json()


orchestrator_agent = Agent(
    role="Market Analyst Agent",
    goal="Answer complex finance questions using all available tools.",
    backstory="You are an expert financial analyst with access to market data, filings, analytics, and document retrieval.",
    tools=[market_data_tool, filings_scraper_tool, risk_analysis_tool, retriever_tool]
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

# Example orchestrator endpoint: voice input -> STT -> analysis -> TTS
@router.post("/morning_brief")
async def morning_brief(request: Request):
    """Orchestrate the full multi-agent workflow: transcribe audio (if needed), run CrewAI agents for market data, filings, risk analysis, and retrieval, then synthesize a final answer using the language agent and return both text and TTS audio."""
    data = await request.json()
    # Accept either text or audio (for demo, focus on text)
    question = data.get("question")
    if not question:
        # If audio, transcribe first
        audio_bytes = data.get("audio")
        stt_resp = requests.post("http://localhost:8000/voice/transcribe", files={"audio": ("audio.wav", audio_bytes, "audio/wav")})
        question = stt_resp.json().get("transcript", "")
    logger.info(f"Received question for morning brief: {question}")

    results = crew.kickoff()
    # Synthesize a final answer using the language agent
    context_chunks = [str(r) for r in results]
    synth_resp = requests.post(
        "http://localhost:8000/language/synthesize",
        json={"question": question, "context": context_chunks}
    )
    answer = synth_resp.json().get("answer", "No answer.")
    tts_resp = requests.post("http://localhost:8000/voice/speak", json={"text": answer})
    audio_out = tts_resp.json().get("audio", None)
    return {"transcript": question, "answer": answer, "audio": audio_out}

@router.get("/")
def root():
    """Health check endpoint for the orchestrator service."""
    return {"status": "Orchestrator running"}
