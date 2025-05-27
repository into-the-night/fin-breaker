from fastapi import APIRouter, File, UploadFile
import tempfile
import os
import pyttsx3
import logging

logger = logging.getLogger("finbreaker")

router = APIRouter(prefix="/voice", tags=["Voice Agent"])

@router.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    logger.info("Received audio for transcription.")
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.file.read())
        tmp_path = tmp.name
    segments, info = model.transcribe(tmp_path)
    transcript = " ".join([segment.text for segment in segments])
    os.remove(tmp_path)
    logger.info(f"Transcription complete: {transcript.strip()}")
    return {"transcript": transcript.strip()}

@router.post("/speak")
def speak(text: str):
    logger.info(f"Received text for TTS: {text}")
    out_path = tempfile.mktemp(suffix=".wav")
    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    with open(out_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(out_path)
    logger.info("TTS audio generated and returned.")
    return {"audio": audio_bytes}

@router.get("/")
def root():
    return {"status": "Voice Agent running"}