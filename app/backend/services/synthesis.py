# Language Agent
# Handles LLM-based narrative synthesis

from functools import lru_cache
from typing import List, Optional, Dict, Any
from app.backend.utils.config import Config
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

            # Build config parameters conditionally
            config_params = {}
            
            if system:
                config_params["system_instruction"] = [system]
            
            if temperature is not None:
                config_params["temperature"] = temperature
                
            if max_tokens is not None:
                config_params["max_output_tokens"] = max_tokens
                
            if thinking_config is not None:
                config_params["thinking_config"] = thinking_config
            
            # Only include tools if they are provided and not None/empty
            if tools is not None and len(tools) > 0:
                # Filter out None values and ensure proper format
                valid_tools = []
                for tool in tools:
                    if tool is not None and isinstance(tool, dict):
                        # Basic validation for tool structure
                        if 'function_declarations' in tool or ('Tool' in tool and tool['Tool'] is not None):
                            valid_tools.append(tool)
                        else:
                            logger.warning(f"Skipping invalid tool structure: {tool}")
                    elif tool is not None:
                        logger.warning(f"Skipping non-dict tool: {type(tool)}")
                
                if valid_tools:
                    config_params["tools"] = valid_tools
                else:
                    logger.debug("No valid tools found, excluding tools from config")
            
            # Convert messages to Gemini format
            if isinstance(messages, list) and len(messages) > 0:
                # Extract content from the first message (assuming single user message for now)
                if isinstance(messages[0], dict) and 'content' in messages[0]:
                    content = messages[0]['content']
                else:
                    content = str(messages[0])
            else:
                content = str(messages)
            
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=content,
                config=gemini_types.GenerateContentConfig(**config_params),
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
            "You are a financial analyst master agent. You have access to the following subagents, each specializing in their analysis: {tool_names}. "
            "Given the user's query, break down the problem into simpler problems and decide which subagents to call for their respective analysis. "
            "If the question cannot be answered with the help of subagents or is irrelevant to Financial Analysis, you can respond with 'I can't answer that question.' "
            "You will generate a plan which MUST contain the task to be performed by the necessary subagents."
            "These subagents can share memory and progress with each other."
            "More than one same type of subagents can work at the same time, so break down any complex queries into simpler analysis if deemed necessary."
        ).format(tool_names=", ".join(tool_names))

        result = await self.generate(
            model='gemini-2.0-flash-001',
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
            model='gemini-2.0-flash-001',
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