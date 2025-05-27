# AI Tool Usage Log

## Agents
- **API Agent**: FastAPI, yfinance, AlphaVantage API, LangChain for future RAG integration.
- **Scraping Agent**: FastAPI, BeautifulSoup, SEC EDGAR scraping.
- **Retriever Agent**: FastAPI, LangChain, FAISS, HuggingFace Embeddings.
- **Analysis Agent**: FastAPI, NumPy, custom analytics logic.
- **Language Agent**: FastAPI, LangChain LLM, OpenAI API.
- **Voice Agent**: FastAPI, Deepgram SDK for STT, pyttsx3 for TTS (cross-platform, works on Windows).

## Orchestrator
- FastAPI APIRouter, orchestrates calls between agents.
- Example: `/morning_brief` endpoint chains STT, analysis, and TTS.

## Data Ingestion
- yfinance for market data.
- BeautifulSoup for filings.

## Streamlit App
- streamlit for UI, requests for backend calls.
- Handles both voice and text input/output.

## Model Parameters
- Deepgram: `punctuate=True`
- pyttsx3: default system voice
- yfinance: `period`, `interval` as per user query
- LangChain/OpenAI: `temperature=0.2`

## Code Generation Steps
- Scaffolded modular agents and routers.
- Refactored to single FastAPI app with routers.
- Integrated Deepgram SDK and pyttsx3 for voice.
- Built orchestrator and data ingestion classes.
- Created Streamlit UI and basic tests.

## Prompts
- See code for all API and LLM prompts.

---

*This log is auto-generated to meet documentation requirements.*
