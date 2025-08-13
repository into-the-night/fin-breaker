from fastapi import APIRouter, Request, HTTPException, status
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime

from app.backend.agent.agent import OrchestratorAgent
from app.backend.models.models import OrchestratorConfig
from app.backend.models.enums import AgentStatus
from app.backend.utils.agent_config import get_config_manager

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])
logger = logging.getLogger("finbreaker")

# Initialize orchestrator agent with default configuration
# This will be initialized on first request to avoid startup issues
_orchestrator_agent: Optional[OrchestratorAgent] = None

def get_orchestrator_agent() -> OrchestratorAgent:
    """Get or create the orchestrator agent instance."""
    global _orchestrator_agent
    if _orchestrator_agent is None:
        logger.info("Initializing OrchestratorAgent")
        config_manager = get_config_manager()
        orchestrator_config = config_manager.get_orchestrator_config()
        _orchestrator_agent = OrchestratorAgent(orchestrator_config)
        logger.info(f"OrchestratorAgent initialized with ID: {_orchestrator_agent.orchestrator_id}")
        logger.info(f"Using configuration: {config_manager.environment.value}")
    return _orchestrator_agent

@router.post("/morning_brief", 
             summary="Morning Brief Analysis",
             description="Process financial queries using multi-agent orchestration (legacy endpoint)")
async def morning_brief(request: Request) -> Dict[str, Any]:
    """
    Orchestrate the full multi-agent workflow using the new OrchestratorAgent.
    
    This endpoint maintains backward compatibility with the existing response format
    while leveraging the new multi-agent orchestration system.
    
    **Request Body:**
    ```json
    {
        "question": "What is the current stock price of AAPL and its recent performance?"
    }
    ```
    
    **Response:**
    - `transcript`: The original question
    - `answer`: Synthesized response from multiple agents
    - `confidence`: Confidence score (0.0 to 1.0)
    - `sources`: List of data sources used
    - `supporting_data`: Additional structured data
    - `execution_metadata`: Execution details and timing
    """
    start_time = time.time()
    request_id = f"morning_brief_{int(start_time)}"
    
    try:
        # Parse request data with validation
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in request {request_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format"
            )
        
        question = data.get("question")
        if not question or not isinstance(question, str) or not question.strip():
            logger.error(f"Missing or invalid question in request {request_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question is required and must be a non-empty string"
            )
        
        question = question.strip()
        logger.info(f"[{request_id}] Processing morning brief request: {question}")
        
        # Get orchestrator agent instance
        orchestrator_agent = get_orchestrator_agent()
        
        # Process query using the new orchestrator agent
        response = await orchestrator_agent.process_query(question)
        
        execution_time = time.time() - start_time
        logger.info(f"[{request_id}] Morning brief completed in {execution_time:.2f}s")
        
        # Maintain backward compatibility with existing response format
        result = {
            "transcript": question,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "supporting_data": response.supporting_data,
            "execution_metadata": {
                "query_id": response.execution_metadata.query_id,
                "orchestrator_id": response.execution_metadata.orchestrator_id,
                "total_cycles": response.execution_metadata.total_cycles,
                "execution_time_seconds": (
                    response.execution_metadata.completed_at - response.execution_metadata.started_at
                ).total_seconds() if response.execution_metadata.completed_at else None,
                "api_execution_time_seconds": execution_time
            }
        }
        
        logger.debug(f"[{request_id}] Returning response with {len(response.sources)} sources")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[{request_id}] Error in morning_brief endpoint after {execution_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request. Please try again."
        )

@router.post("/query",
             summary="Comprehensive Financial Query",
             description="Process financial queries with full multi-agent orchestration and detailed metadata")
async def process_query(request: Request) -> Dict[str, Any]:
    """
    Process a financial query using the multi-agent orchestration system.
    
    This is the comprehensive endpoint that provides full access to the
    orchestrator's capabilities and detailed execution metadata.
    
    **Request Body:**
    ```json
    {
        "question": "Analyze Tesla's Q4 earnings and compare with analyst expectations"
    }
    ```
    
    **Response:**
    - `query`: The processed query
    - `answer`: Comprehensive synthesized response
    - `confidence`: Confidence score (0.0 to 1.0)
    - `sources`: List of data sources used
    - `supporting_data`: Structured supporting data from all agents
    - `execution_metadata`: Detailed execution information including:
      - Agent coordination cycles
      - Individual agent performance
      - Timing and resource usage
    """
    start_time = time.time()
    request_id = f"query_{int(start_time)}"
    
    try:
        # Parse request data with validation
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in request {request_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format"
            )
        
        question = data.get("question")
        if not question or not isinstance(question, str) or not question.strip():
            logger.error(f"Missing or invalid question in request {request_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question is required and must be a non-empty string"
            )
        
        question = question.strip()
        logger.info(f"[{request_id}] Processing comprehensive query: {question}")
        
        # Get orchestrator agent instance
        orchestrator_agent = get_orchestrator_agent()
        
        # Process query using the orchestrator agent
        response = await orchestrator_agent.process_query(question)
        
        execution_time = time.time() - start_time
        logger.info(f"[{request_id}] Query processing completed in {execution_time:.2f}s")
        
        # Return full response with all metadata
        result = {
            "query": response.query,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "supporting_data": response.supporting_data,
            "execution_metadata": {
                "query_id": response.execution_metadata.query_id,
                "orchestrator_id": response.execution_metadata.orchestrator_id,
                "started_at": response.execution_metadata.started_at.isoformat(),
                "completed_at": response.execution_metadata.completed_at.isoformat() if response.execution_metadata.completed_at else None,
                "total_cycles": response.execution_metadata.total_cycles,
                "synthesis_model": response.execution_metadata.synthesis_model,
                "execution_time_seconds": (
                    response.execution_metadata.completed_at - response.execution_metadata.started_at
                ).total_seconds() if response.execution_metadata.completed_at else None,
                "api_execution_time_seconds": execution_time
            }
        }
        
        logger.debug(f"[{request_id}] Returning comprehensive response with {len(response.sources)} sources")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[{request_id}] Error in process_query endpoint after {execution_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request. Please try again."
        )

@router.get("/health",
           summary="Service Health Check",
           description="Check the health and status of the orchestrator service")
def health_check():
    """
    Health check endpoint for the orchestrator service.
    
    Returns the current status of the orchestrator service, including:
    - Service availability
    - Agent initialization status
    - Configuration details
    - Timestamp information
    """
    try:
        # Check if orchestrator agent is initialized
        if _orchestrator_agent is None:
            return {
                "status": "Orchestrator service ready (agent not initialized)",
                "timestamp": datetime.utcnow().isoformat(),
                "initialized": False
            }
        
        return {
            "status": "Orchestrator running",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized": True,
            "agent_id": _orchestrator_agent.orchestrator_id,
            "config": {
                "max_concurrent_agents": _orchestrator_agent.config.max_concurrent_agents,
                "max_cycles": _orchestrator_agent.config.max_cycles,
                "synthesis_model": _orchestrator_agent.config.synthesis_model,
                "context_sharing_enabled": _orchestrator_agent.config.context_sharing_enabled
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}", exc_info=True)
        return {
            "status": "Orchestrator service error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/")
def root():
    """Root endpoint - redirects to health check for backward compatibility."""
    return health_check()
