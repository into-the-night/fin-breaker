from fastapi import APIRouter, File, UploadFile
import tempfile
import os
from deepgram import Deepgram
import pyttsx3
import wave

router = APIRouter(prefix="/voice", tags=["Voice Agent"])

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
dg_client = Deepgram(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None

@router.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    if not dg_client:
        return {"error": "Deepgram API key not set"}
    audio_bytes = audio.file.read()
    try:
        response = dg_client.transcription.sync_prerecorded(
            {"buffer": audio_bytes, "mimetype": "audio/wav"},
            {"punctuate": True}
        )
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return {"transcript": transcript}
    except Exception as e:
        return {"error": str(e)}

@router.post("/speak")
def speak(text: str):
    # Generate speech audio from text using pyttsx3
    out_path = tempfile.mktemp(suffix=".wav")
    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    with open(out_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(out_path)
    return {"audio": audio_bytes}

@router.get("/")
def root():
    return {"status": "Voice Agent running"}