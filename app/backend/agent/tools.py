"""
Tool registry and assignment system for the multi-agent finance orchestrator.

This module provides a centralized registry for managing tool access by agent type,
implementing the tool assignment mappings and shared tool access functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache

from app.backend.models.enums import AgentType
from app.backend.services.market_data import MarketDataService, get_market_data
from app.backend.services.synthesis import LLMService, get_llm_service
from app.backend.services.retrieval import VectorStoreService, get_vector_store

logger = logging.getLogger("finbreaker")


class ToolCategory(Enum):
    """Categories of tools available in the system."""
    MARKET_DATA = "market_data"
    COMPANY_RESEARCH = "company_research"
    TOPIC_ANALYSIS = "topic_analysis"
    RISK_ANALYSIS = "risk_analysis"
    SYNTHESIS = "synthesis"
    RETRIEVAL = "retrieval"
    SHARED = "shared"


@dataclass
class Tool:
    """Represents a tool that can be used by agents."""
    name: str
    description: str
    category: ToolCategory
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    agent_types: Set[AgentType] = field(default_factory=set)
    is_shared: bool = False
    
    def __post_init__(self):
        """Validate tool configuration after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not callable(self.function):
            raise ValueError("Tool function must be callable")


class ToolRegistry:
    """
    Central registry for managing tool access by agent type.
    
    Provides functionality for:
    - Tool registration and retrieval
    - Agent-specific tool assignment
    - Shared tool access management
    - Dynamic tool registration
    """
    
    def __init__(self):
        """Initialize the tool registry with default tools."""
        self._tools: Dict[str, Tool] = {}
        self._agent_tool_mappings: Dict[AgentType, Set[str]] = {
            agent_type: set() for agent_type in AgentType
        }
        self._shared_tools: Set[str] = set()
        self._initialize_default_tools()
    
    def _initialize_default_tools(self) -> None:
        """Initialize the registry with default tools for each agent type."""
        logger.info("Initializing default tools in registry")
        
        # Market Data Tools
        self._register_market_data_tools()
        
        # Company Research Tools
        self._register_company_research_tools()
        
        # Topic Analysis Tools
        self._register_topic_analysis_tools()
        
        # Risk Analysis Tools
        self._register_risk_analysis_tools()
        
        # Shared Tools
        self._register_shared_tools()
        
        logger.info(f"Initialized {len(self._tools)} tools in registry")
    
    def _register_market_data_tools(self) -> None:
        """Register tools specific to market data agents."""
        market_service = get_market_data()
        
        tools = [
            Tool(
                name="search_ticker",
                description="Search for ticker symbol by company name",
                category=ToolCategory.MARKET_DATA,
                function=market_service.search_ticker,
                parameters={"company_name": {"type": "string", "required": True}},
                agent_types={AgentType.MARKET_DATA}
            ),
            Tool(
                name="fetch_time_series_data",
                description="Fetch historical and real-time market data for a ticker",
                category=ToolCategory.MARKET_DATA,
                function=market_service.fetch_time_series_market_data,
                parameters={
                    "ticker": {"type": "string", "required": True},
                    "period": {"type": "string", "default": "1d"},
                    "interval": {"type": "string", "default": "1d"}
                },
                agent_types={AgentType.MARKET_DATA}
            ),
            Tool(
                name="fetch_earnings",
                description="Fetch earnings data for a ticker",
                category=ToolCategory.MARKET_DATA,
                function=market_service.fetch_earnings,
                parameters={"ticker": {"type": "string", "required": True}},
                agent_types={AgentType.MARKET_DATA}
            ),
            Tool(
                name="fetch_stock_trends",
                description="Fetch recommendation trends for a ticker",
                category=ToolCategory.MARKET_DATA,
                function=market_service.fetch_stock_trends,
                parameters={"ticker": {"type": "string", "required": True}},
                agent_types={AgentType.MARKET_DATA, AgentType.RISK_ANALYSIS}
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    def _register_company_research_tools(self) -> None:
        """Register tools specific to company research agents."""
        market_service = get_market_data()
        
        # Create a wrapper for the filing function
        def get_company_filing(ticker: str, doc_type: str = "10-K"):
            """Wrapper for getting company filings."""
            from bs4 import BeautifulSoup
            import requests
            
            base_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={doc_type}&dateb=&owner=exclude&count=1"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(base_url, headers=headers)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                doc_link = soup.find('a', {'id': 'documentsbutton'})
                if doc_link:
                    return {"filing_url": f"https://www.sec.gov{doc_link['href']}"}
                return {"message": "No filing found"}
            return {"error": "Failed to fetch filing"}
        
        tools = [
            Tool(
                name="fetch_company_news",
                description="Fetch news for a specific company ticker",
                category=ToolCategory.COMPANY_RESEARCH,
                function=market_service.fetch_company_news,
                parameters={"ticker": {"type": "string", "required": True}},
                agent_types={AgentType.COMPANY_RESEARCH}
            ),
            Tool(
                name="get_company_filing",
                description="Fetch SEC filings for a company",
                category=ToolCategory.COMPANY_RESEARCH,
                function=get_company_filing,
                parameters={
                    "ticker": {"type": "string", "required": True},
                    "doc_type": {"type": "string", "default": "10-K"}
                },
                agent_types={AgentType.COMPANY_RESEARCH}
            ),
            Tool(
                name="search_ticker_for_research",
                description="Search ticker for company research purposes",
                category=ToolCategory.COMPANY_RESEARCH,
                function=get_market_data().search_ticker,
                parameters={"company_name": {"type": "string", "required": True}},
                agent_types={AgentType.COMPANY_RESEARCH}
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    def _register_topic_analysis_tools(self) -> None:
        """Register tools specific to topic analysis agents."""
        market_service = get_market_data()
        
        tools = [
            Tool(
                name="fetch_topic_news",
                description="Fetch news for financial topics and sectors",
                category=ToolCategory.TOPIC_ANALYSIS,
                function=market_service.fetch_topic_news,
                parameters={"tickers": {"type": "array", "required": True}},
                agent_types={AgentType.TOPIC_ANALYSIS}
            ),
            Tool(
                name="fetch_sector_trends",
                description="Fetch trends for specific sectors",
                category=ToolCategory.TOPIC_ANALYSIS,
                function=market_service.fetch_stock_trends,
                parameters={"ticker": {"type": "string", "required": True}},
                agent_types={AgentType.TOPIC_ANALYSIS}
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    def _register_risk_analysis_tools(self) -> None:
        """Register tools specific to risk analysis agents."""
        market_service = get_market_data()
        
        tools = [
            Tool(
                name="fetch_risk_metrics",
                description="Fetch time series data for risk analysis",
                category=ToolCategory.RISK_ANALYSIS,
                function=market_service.fetch_time_series_market_data,
                parameters={
                    "ticker": {"type": "string", "required": True},
                    "period": {"type": "string", "default": "1y"},
                    "interval": {"type": "string", "default": "1d"}
                },
                agent_types={AgentType.RISK_ANALYSIS}
            ),
            Tool(
                name="fetch_volatility_data",
                description="Fetch earnings data for volatility analysis",
                category=ToolCategory.RISK_ANALYSIS,
                function=market_service.fetch_earnings,
                parameters={"ticker": {"type": "string", "required": True}},
                agent_types={AgentType.RISK_ANALYSIS}
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    def _register_shared_tools(self) -> None:
        """Register tools that are shared across all agent types."""
        llm_service = get_llm_service()
        vector_service = get_vector_store()
        
        tools = [
            Tool(
                name="synthesize_context",
                description="Synthesize information with context using LLM",
                category=ToolCategory.SHARED,
                function=llm_service.synthesize_with_context,
                parameters={
                    "question": {"type": "string", "required": True},
                    "context": {"type": "array", "required": True}
                },
                agent_types=set(AgentType),
                is_shared=True
            ),
            Tool(
                name="evaluate_context",
                description="Evaluate if context is sufficient for answering a question",
                category=ToolCategory.SHARED,
                function=llm_service.evaluate_context,
                parameters={
                    "question": {"type": "string", "required": True},
                    "context": {"type": "array", "required": True}
                },
                agent_types=set(AgentType),
                is_shared=True
            ),
            Tool(
                name="retrieve_documents",
                description="Retrieve relevant documents from vector store",
                category=ToolCategory.SHARED,
                function=vector_service.retrieve,
                parameters={
                    "query": {"type": "string", "required": True},
                    "k": {"type": "integer", "default": 3}
                },
                agent_types=set(AgentType),
                is_shared=True
            ),
            Tool(
                name="index_documents",
                description="Index documents in vector store",
                category=ToolCategory.SHARED,
                function=vector_service.index_documents,
                parameters={"docs": {"type": "array", "required": True}},
                agent_types=set(AgentType),
                is_shared=True
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
            if tool.is_shared:
                self._shared_tools.add(tool.name)
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a new tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool name already exists or tool is invalid
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already exists in registry")
        
        self._tools[tool.name] = tool
        
        # Update agent-tool mappings
        for agent_type in tool.agent_types:
            self._agent_tool_mappings[agent_type].add(tool.name)
        
        # Update shared tools if applicable
        if tool.is_shared:
            self._shared_tools.add(tool.name)
        
        logger.debug(f"Registered tool '{tool.name}' for agents: {tool.agent_types}")
    
    def get_tools_for_agent(self, agent_type: AgentType) -> List[Tool]:
        """
        Get all tools available for a specific agent type.
        
        Args:
            agent_type: Type of agent to get tools for
            
        Returns:
            List of tools available for the agent type
        """
        if agent_type not in self._agent_tool_mappings:
            logger.warning(f"Unknown agent type: {agent_type}")
            return []
        
        tool_names = self._agent_tool_mappings[agent_type].union(self._shared_tools)
        tools = [self._tools[name] for name in tool_names if name in self._tools]
        
        logger.debug(f"Retrieved {len(tools)} tools for agent type {agent_type}")
        return tools
    
    def get_shared_tools(self) -> List[Tool]:
        """
        Get all shared tools available to all agents.
        
        Returns:
            List of shared tools
        """
        shared_tools = [self._tools[name] for name in self._shared_tools if name in self._tools]
        logger.debug(f"Retrieved {len(shared_tools)} shared tools")
        return shared_tools
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Tool category to filter by
            
        Returns:
            List of tools in the specified category
        """
        tools = [tool for tool in self._tools.values() if tool.category == category]
        logger.debug(f"Retrieved {len(tools)} tools for category {category}")
        return tools
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool '{tool_name}' not found for removal")
            return False
        
        tool = self._tools[tool_name]
        
        # Remove from agent mappings
        for agent_type in tool.agent_types:
            self._agent_tool_mappings[agent_type].discard(tool_name)
        
        # Remove from shared tools if applicable
        self._shared_tools.discard(tool_name)
        
        # Remove from main registry
        del self._tools[tool_name]
        
        logger.info(f"Unregistered tool '{tool_name}'")
        return True
    
    def list_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all registered tools.
        
        Returns:
            Dictionary with tool information
        """
        return {
            name: {
                "description": tool.description,
                "category": tool.category.value,
                "agent_types": [at.value for at in tool.agent_types],
                "is_shared": tool.is_shared,
                "parameters": tool.parameters
            }
            for name, tool in self._tools.items()
        }
    
    def get_agent_tool_mappings(self) -> Dict[AgentType, List[str]]:
        """
        Get the current agent-to-tool mappings.
        
        Returns:
            Dictionary mapping agent types to their available tools
        """
        return {
            agent_type: list(tool_names.union(self._shared_tools))
            for agent_type, tool_names in self._agent_tool_mappings.items()
        }
    
    def validate_tool_access(self, agent_type: AgentType, tool_name: str) -> bool:
        """
        Validate if an agent type has access to a specific tool.
        
        Args:
            agent_type: Type of agent requesting access
            tool_name: Name of the tool to check access for
            
        Returns:
            True if agent has access, False otherwise
        """
        if tool_name in self._shared_tools:
            return True
        
        return tool_name in self._agent_tool_mappings.get(agent_type, set())


# Global registry instance
_registry_instance: Optional[ToolRegistry] = None


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        ToolRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance


def reset_tool_registry() -> None:
    """Reset the global tool registry instance. Mainly for testing."""
    global _registry_instance
    _registry_instance = None
    get_tool_registry.cache_clear()