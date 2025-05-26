# Main FastAPI app combining all agent routers
from fastapi import FastAPI
from agents.api_agent import router as api_router
from agents.scraping_agent import router as scraping_router
from agents.retriever_agent import router as retriever_router
from agents.analysis_agent import router as analysis_router
from agents.language_agent import router as language_router
from agents.voice_agent import router as voice_router

app = FastAPI(title="Multi-Agent Finance Assistant")

# Include all agent routers
app.include_router(api_router)
app.include_router(scraping_router)
app.include_router(retriever_router)
app.include_router(analysis_router)
app.include_router(language_router)
app.include_router(voice_router)

@app.get("/")
def root():
    return {"status": "Multi-Agent Finance Assistant running"}
