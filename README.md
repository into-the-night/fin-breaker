# FinBreaker: Multi-Agent Finance Assistant

![FinBreaker Banner](docs/architecture.png)

## Overview 🔍
FinBreaker is an open-source, multi-agent finance assistant that delivers spoken and written market briefs.

---

## Features ✨
- **Morning Market Brief**: Answers questions like "What’s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?" with up-to-date data and analytics.
- **Multi-Agent Orchestration**: Modular agents for market data, filings, analytics, retrieval, and language synthesis, coordinated by CrewAI.
- **Voice & Text I/O**: Upload a voice question or type your query; get a spoken and text market brief.
- **Open-Source Pipelines**: Uses yfinance, BeautifulSoup, FAISS, HuggingFace, Whisper/faster-whisper, pyttsx3, and more.
- **RAG & LLM**: Retrieval-Augmented Generation with vector search and LLM synthesis (Gemini/OpenAI supported).

---

## Quickstart 🏃🏻‍♂️💨

1. **Clone the repo**
    ```powershell
    git clone https://github.com/into-the-night/fin-breaker.git
    cd fin-breaker
    ```

2. **Install dependencies**: Make sure you have Poetry installed
    ```
    poetry install
    ```

3. **Set up environment variables:** Copy or edit the .env file in src/ with your Google API key
    ```
    MODEL = gemini/gemini-2.0-flash
    ALPHAVANTAGE_API_KEY=your-alphavantage-api-key
    GEMINI_API_KEY=your-google-api-key
    ```

3. **Run the Backend**
    ```powershell
    poetry run uvicorn main:app --reload --port 8000
    ```

4. **Run the Streamlit App**
    ```powershell
    streamlit run streamlit_app/app.py
    ```

---

## Usage 📕
- **Voice**: Upload a WAV file with your question, or use the text box.
- **Text**: Type your market question and get a spoken/text answer.
- **Example Query**: "What’s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"

---

## Project Structure 🏗
```
fin-breaker/
  agents/           # All agent microservices (API, scraping, retriever, etc.)
  data_ingestion/   # Data loaders for market data and filings
  orchestrator/     # CrewAI orchestrator logic
  streamlit_app/    # Streamlit UI
  utils/            # Config and logging
  docs/             # Documentation and AI tool usage logs
  tests/            # Test suite
  main.py           # FastAPI app entry point
  Dockerfile        # Containerization
  requirements.txt  # Python dependencies
```

---

## Frameworks & Toolkits 🛠
- **FastAPI**: Microservices for each agent
- **CrewAI**: Multi-agent orchestration
- **LangChain**: RAG, vector search, LLM integration
- **Streamlit**: Frontend UI
- **FAISS**: Vector store for document retrieval
- **HuggingFace**: Embeddings
- **Whisper/faster-whisper**: Speech-to-text
- **pyttsx3/Coqui TTS**: Text-to-speech

---

## Deployment ⛵
- **Docker**: Build and run with the provided Dockerfile for easy deployment.
- **Streamlit Cloud**: Deploy the UI for public access.

---

## Contributing 🤗
Pull requests and issues are welcome! Please see the code comments and documentation for extension points.

---

## License 📜
This project is open-source and available under the [MIT License](LICENSE.md).

---