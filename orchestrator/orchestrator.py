from fastapi import APIRouter, Request
import requests
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
import json

from agents.api_agent import router as api_router
from agents.scraping_agent import router as scraping_router
from agents.retriever_agent import router as retriever_router
from agents.analysis_agent import router as analysis_router
from agents.language_agent import router as language_router
from agents.voice_agent import router as voice_router

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

# Define LangChain tools for each agent
market_data_tool = Tool(
    name="Market Data",
    func=lambda ticker: requests.get(f"http://localhost:8000/api/market_data?ticker={ticker}").json(),
    description="Get real-time and historical market data for a given ticker."
)

scraping_tool = Tool(
    name="Filings Scraper",
    func=lambda ticker: requests.get(f"http://localhost:8000/scraping/filing?ticker={ticker}").json(),
    description="Get latest SEC/filings for a given ticker."
)

analysis_tool = Tool(
    name="Risk Analysis",
    func=lambda params: requests.post(f"http://localhost:8000/analysis/risk_exposure", json=params).json(),
    description="Analyze risk exposure for a given region and sector."
)

retriever_tool = Tool(
    name="Retriever",
    func=lambda query: requests.get(f"http://localhost:8000/retriever/retrieve?query={query}").json(),
    description="Retrieve relevant indexed financial documents."
)

# LLM for agent orchestration (can be OpenAI or Gemini, here using OpenAI for demo)
llm = OpenAI(temperature=0.2)

# Initialize LangChain agent with tools
agent = initialize_agent(
    tools=[market_data_tool, scraping_tool, analysis_tool, retriever_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example orchestrator endpoint: voice input -> STT -> analysis -> TTS
@router.post("/morning_brief")
async def morning_brief(request: Request):
    data = await request.json()
    # Accept either text or audio (for demo, focus on text)
    question = data.get("question")
    if not question:
        # If audio, transcribe first
        audio_bytes = data.get("audio")
        stt_resp = requests.post("http://localhost:8000/voice/transcribe", files={"audio": ("audio.wav", audio_bytes, "audio/wav")})
        question = stt_resp.json().get("transcript", "")
    # Use agent to orchestrate workflow
    # Example: "What’s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    result = agent.run(question)
    # TTS for result
    tts_resp = requests.post("http://localhost:8000/voice/speak", json={"text": result})
    audio_out = tts_resp.json().get("audio", None)
    return {"transcript": question, "answer": result, "audio": audio_out}

@router.get("/")
def root():
    return {"status": "Orchestrator running"}
