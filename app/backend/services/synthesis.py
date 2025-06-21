# Language Agent
# Handles LLM-based narrative synthesis

from functools import lru_cache
from typing import List, Optional, Dict, Any
from utils.config import Config
from google import genai
import google.genai.types as gemini_types
import logging
from pydantic import BaseModel

logger = logging.getLogger("finbreaker")

class Content(BaseModel):
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    thinking: Optional[Any] = None

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class Result(BaseModel):
    content: List[Content]
    usage: Usage


class LLMService:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        thinking: Optional[bool] = False,
        thinking_budget_tokens: Optional[int] = None,
    ) -> Result:
        
        try:
        # Only Gemini 2.5 Flash and Pro models support thinking
            if model.startswith("gemini-2.5-"):
                if thinking:
                    thinking_config = gemini_types.ThinkingConfig(
                        include_thoughts=thinking,
                        thinking_budget=thinking_budget_tokens,
                    )
                else:
                    thinking_config = None
            else:
                thinking_config = None

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=messages,
                config=gemini_types.GenerateContentConfig(
                    system_instruction=[system] if system else None,
                    temperature=temperature if temperature else None,
                    max_output_tokens=max_tokens,
                    tools=[tools],
                    thinking_config=thinking_config,
                ),
            )

            contents = [
                Content(
                    type="text",
                    text=part.text,
                ) if part.text
                else Content(
                    type="tool_call",
                    id=part.function_call.id, # Gemini does not return the tool call id
                    name=part.function_call.name,
                    input=part.function_call.args,
                ) if part.function_call
                else Content(
                    type="thinking",
                    text=part.text,
                ) if part.thought else None
                for part in response.candidates[0].content.parts
            ]

            return Result(
                content=contents,
                usage=Usage(
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count
                ),
            )

        except Exception as e:
            logger.error(f"Error generating response with {model}: {str(e)}")
            raise

    async def synthesize_with_context(
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
        response = await self.generate(
            model='gemini-2.0-flash-001',
            messages=[{
                "role": "user",
                "content" : prompt
            }],
        )
        answer = response.content[0].text
        logger.info(f"Answer synthesized: {answer}")
        return answer
    

    async def generate_plan(
        self,
        question: str,
        tools: List[Dict[str, Any]]
    )-> Result:
        tool_names = [t['function_declaration']['name'] for t in tools]
        system_prompt = (
            "You are a financial analyst assistant. You have access to the following tools: {tool_names}. "
            "Given the user's question, decide which tools to call to answer it. "
            "If the question cannot be answered with the tools provided or is irrelevant to Financial Analysis, you can respond with 'I can't answer that question.' "
            "Otherwise, respond with one or more tool calls."
        ).format(tool_names=", ".join(tool_names))

        result = await self.generate(
            model='gemini-2.5-pro-latest',
            messages=[{
                "role": "user",
                "content" : question
            }],
            tools=tools,
            system=system_prompt,
        )
        return result

    async def evaluate_context(
        self,
        question: str,
        context: List[str]
    ) -> str:
        logger.info(f"Evaluating context for question: {question}")
        prompt = (
            "You are a financial analyst assistant. "
            "Given the following context and a user's question, "
            "evaluate if the context contains enough information to answer the question comprehensively. "
            "Respond with 'CONTINUE' if the context is sufficient, or 'REPLAN' if more information is needed which would require another tool call.\n\n"
            f"Context:\n{'-'*80}\n{'\n'.join(context)}\n{'-'*80}\n\nQuestion: {question}\n\n"
            "Evaluation (CONTINUE or REPLAN):"
        )
        response = await self.generate(
            model='gemini-2.5-pro-latest',
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0,
        )
        evaluation = response.content[0].text.strip()
        logger.info(f"Evaluation result: {evaluation}")
        return evaluation

@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()