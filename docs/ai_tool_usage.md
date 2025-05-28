# AI Tool Usage Log

## Agents
- **API Agent**: FastAPI, yfinance, AlphaVantage API. Handles real-time and historical market data.
- **Scraping Agent**: FastAPI, BeautifulSoup, SEC EDGAR scraping. Handles filings retrieval.
- **Retriever Agent**: FastAPI, LangChain, FAISS, HuggingFace Embeddings. Handles document indexing and retrieval for RAG.
- **Analysis Agent**: FastAPI, custom analytics logic. Handles risk exposure and analytics.
- **Language Agent**: FastAPI, Gemini/Google Generative AI, custom prompt engineering. Synthesizes narrative from context and question.
- **Voice Agent**: FastAPI, faster-whisper (open-source STT), pyttsx3 (cross-platform TTS, works on Windows). Fully open-source voice pipeline.

## Orchestrator
- FastAPI APIRouter, orchestrates calls between agents using CrewAI.
- CrewAI: Multi-agent workflow automation, with both single-agent (all tools) and multi-agent (specialized) patterns supported.
- Example: `/morning_brief` endpoint chains STT, market data, filings, analytics, retrieval, LLM synthesis, and TTS.

## Data Ingestion
- yfinance for market data.
- BeautifulSoup for filings.
- HuggingFace Transformers for embeddings.

## Streamlit App
- streamlit for UI, requests for backend calls.
- Handles both voice and text input/output, calls orchestrator endpoint for full workflow.

## Model Parameters
- faster-whisper: `base` model, `device=cpu`, `compute_type=int8`
- pyttsx3: default system voice
- yfinance: `period`, `interval` as per user query
- Gemini/Google Generative AI: `model='gemini-2.0-flash-001'`, custom prompt for financial analysis
- FAISS: L2 similarity, in-memory docstore

## Code Generation Steps
- Scaffolded modular agents and routers.
- Refactored to single FastAPI app with routers and orchestrator.
- Integrated CrewAI for multi-agent workflow automation.
- Integrated faster-whisper and pyttsx3 for fully open-source voice pipeline.
- Built orchestrator and data ingestion classes.
- Created Streamlit UI and basic tests.

## Prompts
- Language agent prompt: "You are a financial analyst assistant. Given the following context from market data, filings, and analytics, answer the user's question in a concise, professional, and insightful manner. Highlight risk exposure, key numbers, and any earnings surprises."
- Extraction prompt (for dynamic agent input): "Extract the following fields from the question: region, sector, allocations, ticker. Return as JSON. Question: ..."
- See code for all API and LLM prompts.

---

*This log is auto-generated to meet documentation requirements and reflects the latest project architecture and workflow.*
