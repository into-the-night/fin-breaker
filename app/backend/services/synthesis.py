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
        response = self.generate(
            model='gemini-2.0-flash-001',
            messages={
                "role": "user",
                "content" : prompt
            },
        )
        answer = response.text
        logger.info(f"Answer synthesized: {answer}")
        return answer
    

    def generate_plan(
        self,
        question: str,
        tools: List[Dict[str, Any]]
    )-> str:
        prompt = (
            "You are a financial analyst assistant. You have access to the following tools: {tool_names} "
            "Given the following question and tools, "
            "generate a plan for the user's question if it can be answered with the tools provided. "
            "If the question cannot be answered with the tools provided or is irrelevant to Financial Analysis, just answer something like 'I'm sorry, I can't answer that question.' "
            "The plan MUST consist of a list of tools to use and the arguments to pass to each tool. "
            "You must also generate a query or a list of queries that you want to infer from the tool results."
            "The queries must be in the following format: "
        )
        prompt =  prompt.format(tool_names=[tool["name"] for tool in tools])
        result = self.generate(
            model='gemini-2.0-flash-001',
            messages={
                "role": "user",
                "content" : question
            },
            tools=tools,
            system=prompt,
        )
        return result

@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()