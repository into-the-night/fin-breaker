from fastapi import APIRouter, Request
import requests

from agents.api_agent import router as api_router
from agents.scraping_agent import router as scraping_router
from agents.retriever_agent import router as retriever_router
from agents.analysis_agent import router as analysis_router
from agents.language_agent import router as language_router
from agents.voice_agent import router as voice_router

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

# Example orchestrator endpoint: voice input -> STT -> analysis -> TTS
@router.post("/morning_brief")
async def morning_brief(request: Request):
    data = await request.json()
    audio_bytes = data.get("audio")
    # 1. Transcribe audio (STT)
    stt_resp = requests.post("http://localhost:8000/voice/transcribe", files={"audio": ("audio.wav", audio_bytes, "audio/wav")})
    transcript = stt_resp.json().get("transcript", "")
    # 2. Call analysis agent (dummy example)
    # ...call analysis, api, retriever, language agents as needed...
    # 3. Synthesize response (TTS)
    tts_resp = requests.post("http://localhost:8000/voice/speak", data={"text": "Your brief: " + transcript})
    audio_out = tts_resp.json().get("audio", None)
    return {"transcript": transcript, "audio": audio_out}

@router.get("/")
def root():
    return {"status": "Orchestrator running"}
