# Language Agent
# Handles LLM-based narrative synthesis

from fastapi import APIRouter, Query
from google import genai

from typing import List
from utils.config import Config
import logging


logger = logging.getLogger("finbreaker")

class LLM:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    def synthesize_with_context(
        self,
        question: str,
        context: List[str]
    )-> str:

        logger.info(f"Synthesizing answer for question: {question}")
        # Construct a more instructive prompt for the LLM
        prompt = (
            "You are a financial analyst assistant. "
            "Given the following context from market data, filings, and analytics, "
            "answer the user's question in a concise, professional, and insightful manner. "
            "Highlight risk exposure, key numbers, and any earnings surprises.\n\n"
            f"Context:\n{'\n'.join(context)}\n\nQuestion: {question}\n\nAnswer:"
        )
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001', contents=prompt
        )
        answer = response.text
        logger.info(f"Answer synthesized: {answer}")
        return answer
