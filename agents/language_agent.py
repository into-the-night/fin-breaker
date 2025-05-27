# Language Agent
# Handles LLM-based narrative synthesis

from fastapi import APIRouter, Query
from google import genai
from langchain.chains import RetrievalQA
from langchain.schema import Document

from pydantic import BaseModel
from utils.config import Config
import logging

router = APIRouter(prefix="/language", tags=["Language Agent"])

llm = genai.Client(api_key=Config.GOOGLE_API_KEY)
logger = logging.getLogger("finbreaker")

class SynthesizeRequest(BaseModel):
    question: str
    context: list


@router.post("/synthesize")
def synthesize(
    request: SynthesizeRequest
):
    question = request.question
    context = request.context

    logger.info(f"Synthesizing answer for question: {question}")
    # Construct a more instructive prompt for the LLM
    prompt = (
        "You are a financial analyst assistant. "
        "Given the following context from market data, filings, and analytics, "
        "answer the user's question in a concise, professional, and insightful manner. "
        "Highlight risk exposure, key numbers, and any earnings surprises.\n\n"
        f"Context:\n{'\n'.join(context)}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = llm.models.generate_content(
        model='gemini-2.0-flash-001', contents=prompt
    )
    answer = response.text
    logger.info(f"Answer synthesized: {answer}")
    return {"answer": answer}

@router.get("/")
def root():
    return {"status": "Language Agent running"}
