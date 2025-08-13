"""
Sub-agent factory and spawning logic for the multi-agent finance orchestrator.

This module provides the SubAgentFactory for dynamic agent creation, configuration
management, concurrent spawning with resource limits, and agent lifecycle tracking.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional, Type, Set
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from app.backend.models.models import QueryAnalysis, AgentConfig, FinancialEntity
from app.backend.models.enums import AgentType, AgentStatus
from app.backend.utils.agent_config import get_config_manager
from app.backend.agent.subagents import (
    BaseSubAgent, MarketDataAgent, CompanyResearchAgent, 
    TopicAnalysisAgent, RiskAnalysisAgent
)
from app.backend.services.context_store import get_context_store

# Import tools registry with fallback
try:
    from app.backend.agent.tools import get_tool_registry, ToolRegistry
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    get_tool_registry = None

logger = logging.getLogger(__name__)


@dataclass
class AgentSpawnConfig:
    """Configuration for spawning a specific agent type."""
    agent_type: AgentType
    max_instances: int = 3
    default_config: AgentConfig = field(default_factory=AgentConfig)
    required_entities: List[str] = field(default_factory=list)  # Entity types required
    optional_entities: List[str] = field(default_factory=list)  # Entity types optional
    resource_weight: float = 1.0  # Resource consumption weight
    
    def __post_init__(self):
        """Validate spawn configuration."""
        if self.max_instances <= 0:
            raise ValueError("Max instances must be positive")
        if self.resource_weight <= 0:
            raise ValueError("Resource weight must be positive")


@dataclass
class AgentInstance:
    """Represents a spawned agent instance with tracking information."""
    agent_id: str
    agent_type: AgentType
    agent: BaseSubAgent
    spawn_time: datetime
    config: AgentConfig
    entities: List[Dict[str, Any]]
    status: AgentStatus = AgentStatus.NOT_STARTED
    resource_weight: float = 1.0
    
    def __post_init__(self):
        """Initialize tracking information."""
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")


@dataclass
class ResourceLimits:
    """Resource limits for concurrent agent spawning."""
    max_concurrent_agents: int = 5
    max_total_resource_weight: float = 10.0
    max_agents_per_type: int = 3
    spawn_timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate resource limits."""
        if self.max_concurrent_agents <= 0:
            raise ValueError("Max concurrent agents must be positive")
        if self.max_total_resource_weight <= 0:
            raise ValueError("Max total resource weight must be positive")
        if self.max_agents_per_type <= 0:
            raise ValueError("Max agents per type must be positive")


class SubAgentFactory:
    """
    Factory for dynamic sub-agent creation and lifecycle management.
    
    Provides functionality for:
    - Dynamic agent creation based on query analysis
    - Configuration management for agent parameters and tool assignments
    - Concurrent agent spawning with resource limits
    - Agent instance tracking and cleanup
    - Resource management and optimization
    """
    
    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        """
        Initialize the sub-agent factory.
        
        Args:
            resource_limits: Resource limits for concurrent spawning
        """
        self.resource_limits = resource_limits or ResourceLimits()
        self.factory_id = f"factory_{uuid.uuid4().hex[:8]}"
        
        # Agent type to class mapping
        self._agent_classes: Dict[AgentType, Type[BaseSubAgent]] = {
            AgentType.MARKET_DATA: MarketDataAgent,
            AgentType.COMPANY_RESEARCH: CompanyResearchAgent,
            AgentType.TOPIC_ANALYSIS: TopicAnalysisAgent,
            AgentType.RISK_ANALYSIS: RiskAnalysisAgent
        }
        
        # Agent spawn configurations
        self._spawn_configs: Dict[AgentType, AgentSpawnConfig] = {}
        self._initialize_default_spawn_configs()
        
        # Active agent tracking
        self._active_agents: Dict[str, AgentInstance] = {}
        self._agents_by_type: Dict[AgentType, Set[str]] = {
            agent_type: set() for agent_type in AgentType
        }
        self._agents_by_query: Dict[str, Set[str]] = {}
        
        # Resource management
        self._current_resource_usage = 0.0
        self._spawn_semaphore = asyncio.Semaphore(self.resource_limits.max_concurrent_agents)
        
        # Services
        self._tool_registry = get_tool_registry() if TOOLS_AVAILABLE else None
        self._context_store = get_context_store()
        
        logger.info(f"Initialized SubAgentFactory {self.factory_id}")
    
    def _initialize_default_spawn_configs(self) -> None:
        """Initialize default spawn configurations for each agent type."""
        # Market Data Agent Configuration
        self._spawn_configs[AgentType.MARKET_DATA] = AgentSpawnConfig(
            agent_type=AgentType.MARKET_DATA,
            max_instances=3,
            default_config=AgentConfig(
                max_retries=3,
                timeout_seconds=300,
                goal_evaluation_interval=10,
                tools=["search_ticker", "fetch_time_series_data", "fetch_earnings", "fetch_stock_trends"]
            ),
            required_entities=["company", "ticker"],
            optional_entities=["sector"],
            resource_weight=1.5  # Higher weight due to API calls
        )
        
        # Company Research Agent Configuration
        self._spawn_configs[AgentType.COMPANY_RESEARCH] = AgentSpawnConfig(
            agent_type=AgentType.COMPANY_RESEARCH,
            max_instances=2,
            default_config=AgentConfig(
                max_retries=2,
                timeout_seconds=400,
                goal_evaluation_interval=15,
                tools=["fetch_company_news", "get_company_filing", "search_ticker_for_research"]
            ),
            required_entities=["company", "ticker"],
            optional_entities=[],
            resource_weight=2.0  # Higher weight due to complex research
        )
        
        # Topic Analysis Agent Configuration
        self._spawn_configs[AgentType.TOPIC_ANALYSIS] = AgentSpawnConfig(
            agent_type=AgentType.TOPIC_ANALYSIS,
            max_instances=2,
            default_config=AgentConfig(
                max_retries=2,
                timeout_seconds=350,
                goal_evaluation_interval=12,
                tools=["fetch_topic_news", "fetch_sector_trends"]
            ),
            required_entities=["topic", "sector"],
            optional_entities=["company", "ticker"],
            resource_weight=1.2
        )
        
        # Risk Analysis Agent Configuration
        self._spawn_configs[AgentType.RISK_ANALYSIS] = AgentSpawnConfig(
            agent_type=AgentType.RISK_ANALYSIS,
            max_instances=2,
            default_config=AgentConfig(
                max_retries=2,
                timeout_seconds=300,
                goal_evaluation_interval=10,
                tools=["fetch_risk_metrics", "fetch_volatility_data", "fetch_stock_trends"]
            ),
            required_entities=[],  # Can work with any entities
            optional_entities=["company", "ticker", "sector"],
            resource_weight=1.3
        )
        
        logger.debug(f"Initialized spawn configurations for {len(self._spawn_configs)} agent types")
    
    async def create_agents_for_query(
        self, 
        analysis: QueryAnalysis, 
        cycle_number: int = 1,
        custom_configs: Optional[Dict[AgentType, AgentConfig]] = None
    ) -> List[AgentInstance]:
        """
        Create agents dynamically based on query analysis.
        
        Args:
            analysis: Query analysis with entities and required agents
            cycle_number: Current cycle number for agent naming
            custom_configs: Custom configurations for specific agent types
            
        Returns:
            List of created agent instances
        """
        logger.info(f"Creating agents for query {analysis.query_id}, cycle {cycle_number}")
        
        # Check resource availability
        if not await self._check_resource_availability(analysis.required_agents):
            logger.warning("Insufficient resources for all requested agents")
            # Prioritize agents based on query analysis priority
            analysis.required_agents = self._prioritize_agents(analysis.required_agents, analysis)
        
        created_agents = []
        
        for agent_type in analysis.required_agents:
            try:
                # Check if we can spawn this agent type
                if not self._can_spawn_agent_type(agent_type):
                    logger.warning(f"Cannot spawn {agent_type.value} agent - resource limits exceeded")
                    continue
                
                # Get relevant entities for this agent type
                relevant_entities = self._get_relevant_entities_for_agent(analysis.entities, agent_type)
                
                # Skip if no relevant entities and agent requires them
                spawn_config = self._spawn_configs.get(agent_type)
                if spawn_config and spawn_config.required_entities and not relevant_entities:
                    logger.debug(f"Skipping {agent_type.value} agent - no required entities found")
                    continue
                
                # Create agent configuration
                agent_config = custom_configs.get(agent_type) if custom_configs else None
                if not agent_config:
                    agent_config = self._create_agent_config(agent_type, analysis)
                
                # Create agent instance
                agent_instance = await self._create_agent_instance(
                    agent_type=agent_type,
                    query_id=analysis.query_id,
                    entities=relevant_entities,
                    config=agent_config,
                    cycle_number=cycle_number
                )
                
                if agent_instance:
                    created_agents.append(agent_instance)
                    logger.debug(f"Created {agent_type.value} agent: {agent_instance.agent_id}")
                
            except Exception as e:
                logger.error(f"Error creating {agent_type.value} agent: {str(e)}")
                continue
        
        logger.info(f"Created {len(created_agents)} agents for query {analysis.query_id}")
        return created_agents
    
    async def spawn_agents_concurrently(
        self, 
        agent_instances: List[AgentInstance]
    ) -> List[AgentInstance]:
        """
        Spawn multiple agents concurrently with resource limits.
        
        Args:
            agent_instances: List of agent instances to spawn
            
        Returns:
            List of successfully spawned agent instances
        """
        if not agent_instances:
            return []
        
        logger.info(f"Spawning {len(agent_instances)} agents concurrently")
        
        # Create spawn tasks with semaphore control
        spawn_tasks = []
        for agent_instance in agent_instances:
            task = self._spawn_agent_with_limits(agent_instance)
            spawn_tasks.append(task)
        
        # Execute spawning with timeout
        try:
            spawned_agents = await asyncio.wait_for(
                asyncio.gather(*spawn_tasks, return_exceptions=True),
                timeout=self.resource_limits.spawn_timeout_seconds
            )
            
            # Filter successful spawns
            successful_spawns = []
            for i, result in enumerate(spawned_agents):
                if isinstance(result, AgentInstance):
                    successful_spawns.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Failed to spawn agent {agent_instances[i].agent_id}: {str(result)}")
                else:
                    logger.warning(f"Unexpected spawn result for agent {agent_instances[i].agent_id}: {type(result)}")
            
            logger.info(f"Successfully spawned {len(successful_spawns)}/{len(agent_instances)} agents")
            return successful_spawns
            
        except asyncio.TimeoutError:
            logger.error(f"Agent spawning timed out after {self.resource_limits.spawn_timeout_seconds} seconds")
            # Return any agents that were successfully spawned before timeout
            return [ai for ai in agent_instances if ai.status != AgentStatus.NOT_STARTED]
    
    @asynccontextmanager
    async def managed_agent_execution(self, agent_instances: List[AgentInstance]):
        """
        Context manager for managed agent execution with automatic cleanup.
        
        Args:
            agent_instances: List of agent instances to manage
            
        Yields:
            List of agent instances ready for execution
        """
        try:
            # Register agents for tracking
            for agent_instance in agent_instances:
                await self._register_agent_instance(agent_instance)
            
            logger.info(f"Starting managed execution of {len(agent_instances)} agents")
            yield agent_instances
            
        finally:
            # Cleanup agents
            await self._cleanup_agent_instances(agent_instances)
            logger.info(f"Completed managed execution and cleanup of {len(agent_instances)} agents")
    
    async def _spawn_agent_with_limits(self, agent_instance: AgentInstance) -> AgentInstance:
        """
        Spawn a single agent with resource limit enforcement.
        
        Args:
            agent_instance: Agent instance to spawn
            
        Returns:
            Spawned agent instance
        """
        async with self._spawn_semaphore:
            try:
                # Check resource availability before spawning
                if not self._can_allocate_resources(agent_instance.resource_weight):
                    raise RuntimeError(f"Insufficient resources to spawn agent {agent_instance.agent_id}")
                
                # Allocate resources
                self._current_resource_usage += agent_instance.resource_weight
                
                # Update agent status
                agent_instance.status = AgentStatus.IN_PROGRESS
                
                logger.debug(f"Spawned agent {agent_instance.agent_id} (resource usage: {self._current_resource_usage:.1f})")
                return agent_instance
                
            except Exception as e:
                logger.error(f"Error spawning agent {agent_instance.agent_id}: {str(e)}")
                agent_instance.status = AgentStatus.FAILED
                raise
    
    async def _create_agent_instance(
        self,
        agent_type: AgentType,
        query_id: str,
        entities: List[Dict[str, Any]],
        config: AgentConfig,
        cycle_number: int
    ) -> Optional[AgentInstance]:
        """
        Create a single agent instance.
        
        Args:
            agent_type: Type of agent to create
            query_id: Query ID for the agent
            entities: Entities for the agent to process
            config: Agent configuration
            cycle_number: Current cycle number
            
        Returns:
            Created agent instance or None if creation failed
        """
        try:
            # Get agent class
            agent_class = self._agent_classes.get(agent_type)
            if not agent_class:
                logger.error(f"No agent class found for type: {agent_type}")
                return None
            
            # Generate unique agent ID
            agent_id = f"{agent_type.value}_c{cycle_number}_{uuid.uuid4().hex[:6]}"
            
            # Create agent instance
            agent = agent_class(
                query_id=query_id,
                entities=entities,
                config=config,
                agent_id=agent_id
            )
            
            # Get resource weight from spawn config
            spawn_config = self._spawn_configs.get(agent_type)
            resource_weight = spawn_config.resource_weight if spawn_config else 1.0
            
            # Create agent instance wrapper
            agent_instance = AgentInstance(
                agent_id=agent_id,
                agent_type=agent_type,
                agent=agent,
                spawn_time=datetime.utcnow(),
                config=config,
                entities=entities,
                resource_weight=resource_weight
            )
            
            return agent_instance
            
        except Exception as e:
            logger.error(f"Error creating agent instance for {agent_type.value}: {str(e)}")
            return None
    
    def _create_agent_config(self, agent_type: AgentType, analysis: QueryAnalysis) -> AgentConfig:
        """
        Create agent configuration based on spawn config and query analysis.
        
        Args:
            agent_type: Type of agent to configure
            analysis: Query analysis for context
            
        Returns:
            Agent configuration
        """
        # Get configuration from configuration manager
        config_manager = get_config_manager()
        config = config_manager.get_agent_config(agent_type)
        
        # Create a copy to avoid modifying the original
        config = AgentConfig(
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
            goal_evaluation_interval=config.goal_evaluation_interval,
            tools=config.tools.copy()
        )
        
        # Adjust based on query priority
        if analysis.priority.value == "high":
            config.timeout_seconds = int(config.timeout_seconds * 1.5)
            config.max_retries += 1
        elif analysis.priority.value == "urgent":
            config.timeout_seconds = int(config.timeout_seconds * 2.0)
            config.max_retries += 2
        
        # Add shared tools if tool registry is available
        if self._tool_registry:
            shared_tools = self._tool_registry.get_shared_tools()
            for tool in shared_tools:
                if tool.name not in config.tools:
                    config.tools.append(tool.name)
        
        return config
    
    def _get_relevant_entities_for_agent(
        self, 
        entities: List[FinancialEntity], 
        agent_type: AgentType
    ) -> List[Dict[str, Any]]:
        """
        Get entities relevant to a specific agent type.
        
        Args:
            entities: List of all extracted entities
            agent_type: Type of agent to get entities for
            
        Returns:
            List of entity dictionaries relevant to the agent type
        """
        spawn_config = self._spawn_configs.get(agent_type)
        if not spawn_config:
            return [entity.to_dict() for entity in entities]
        
        relevant_entities = []
        
        for entity in entities:
            entity_type_str = entity.entity_type.value
            
            # Check if entity is required or optional for this agent
            if (entity_type_str in spawn_config.required_entities or 
                entity_type_str in spawn_config.optional_entities):
                relevant_entities.append(entity.to_dict())
        
        # If no specific entities found but agent can work with any, provide all
        if not relevant_entities and not spawn_config.required_entities:
            relevant_entities = [entity.to_dict() for entity in entities]
        
        return relevant_entities
    
    def _can_spawn_agent_type(self, agent_type: AgentType) -> bool:
        """
        Check if we can spawn another agent of the given type.
        
        Args:
            agent_type: Type of agent to check
            
        Returns:
            True if agent can be spawned
        """
        # Check per-type limits
        current_count = len(self._agents_by_type.get(agent_type, set()))
        spawn_config = self._spawn_configs.get(agent_type)
        max_instances = spawn_config.max_instances if spawn_config else self.resource_limits.max_agents_per_type
        
        if current_count >= max_instances:
            return False
        
        # Check global limits
        if len(self._active_agents) >= self.resource_limits.max_concurrent_agents:
            return False
        
        # Check resource weight limits
        spawn_config = self._spawn_configs.get(agent_type)
        resource_weight = spawn_config.resource_weight if spawn_config else 1.0
        
        return self._can_allocate_resources(resource_weight)
    
    def _can_allocate_resources(self, resource_weight: float) -> bool:
        """
        Check if we can allocate the specified resource weight.
        
        Args:
            resource_weight: Resource weight to allocate
            
        Returns:
            True if resources can be allocated
        """
        return (self._current_resource_usage + resource_weight) <= self.resource_limits.max_total_resource_weight
    
    async def _check_resource_availability(self, required_agents: List[AgentType]) -> bool:
        """
        Check if we have sufficient resources for all required agents.
        
        Args:
            required_agents: List of required agent types
            
        Returns:
            True if sufficient resources available
        """
        total_weight = 0.0
        agent_counts = {}
        
        for agent_type in required_agents:
            spawn_config = self._spawn_configs.get(agent_type)
            if spawn_config:
                total_weight += spawn_config.resource_weight
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        # Check total resource weight
        if (self._current_resource_usage + total_weight) > self.resource_limits.max_total_resource_weight:
            return False
        
        # Check per-type limits
        for agent_type, count in agent_counts.items():
            current_count = len(self._agents_by_type.get(agent_type, set()))
            spawn_config = self._spawn_configs.get(agent_type)
            max_instances = spawn_config.max_instances if spawn_config else self.resource_limits.max_agents_per_type
            
            if (current_count + count) > max_instances:
                return False
        
        return True
    
    def _prioritize_agents(self, required_agents: List[AgentType], analysis: QueryAnalysis) -> List[AgentType]:
        """
        Prioritize agents when resources are limited.
        
        Args:
            required_agents: List of required agent types
            analysis: Query analysis for prioritization context
            
        Returns:
            Prioritized list of agent types
        """
        # Priority order based on typical query patterns
        priority_order = {
            AgentType.MARKET_DATA: 1,      # Highest priority - fundamental data
            AgentType.COMPANY_RESEARCH: 2, # Second - specific company info
            AgentType.RISK_ANALYSIS: 3,    # Third - risk assessment
            AgentType.TOPIC_ANALYSIS: 4    # Fourth - broader topics
        }
        
        # Sort by priority and resource efficiency
        def priority_key(agent_type):
            base_priority = priority_order.get(agent_type, 5)
            spawn_config = self._spawn_configs.get(agent_type)
            resource_weight = spawn_config.resource_weight if spawn_config else 1.0
            
            # Lower resource weight = higher efficiency
            efficiency_factor = 1.0 / resource_weight
            
            return (base_priority, -efficiency_factor)
        
        prioritized = sorted(required_agents, key=priority_key)
        
        # Filter to fit within resource limits
        filtered_agents = []
        projected_usage = self._current_resource_usage
        
        for agent_type in prioritized:
            spawn_config = self._spawn_configs.get(agent_type)
            resource_weight = spawn_config.resource_weight if spawn_config else 1.0
            
            if (projected_usage + resource_weight) <= self.resource_limits.max_total_resource_weight:
                filtered_agents.append(agent_type)
                projected_usage += resource_weight
            else:
                logger.debug(f"Filtered out {agent_type.value} due to resource limits")
        
        return filtered_agents
    
    async def _register_agent_instance(self, agent_instance: AgentInstance) -> None:
        """
        Register an agent instance for tracking.
        
        Args:
            agent_instance: Agent instance to register
        """
        # Add to active agents
        self._active_agents[agent_instance.agent_id] = agent_instance
        
        # Add to type tracking
        self._agents_by_type[agent_instance.agent_type].add(agent_instance.agent_id)
        
        # Add to query tracking
        query_id = agent_instance.agent.query_id
        if query_id not in self._agents_by_query:
            self._agents_by_query[query_id] = set()
        self._agents_by_query[query_id].add(agent_instance.agent_id)
        
        logger.debug(f"Registered agent instance {agent_instance.agent_id}")
    
    async def _cleanup_agent_instances(self, agent_instances: List[AgentInstance]) -> None:
        """
        Clean up agent instances and release resources.
        
        Args:
            agent_instances: List of agent instances to clean up
        """
        for agent_instance in agent_instances:
            try:
                # Remove from tracking
                if agent_instance.agent_id in self._active_agents:
                    del self._active_agents[agent_instance.agent_id]
                
                # Remove from type tracking
                self._agents_by_type[agent_instance.agent_type].discard(agent_instance.agent_id)
                
                # Remove from query tracking
                query_id = agent_instance.agent.query_id
                if query_id in self._agents_by_query:
                    self._agents_by_query[query_id].discard(agent_instance.agent_id)
                    if not self._agents_by_query[query_id]:
                        del self._agents_by_query[query_id]
                
                # Release resources
                self._current_resource_usage -= agent_instance.resource_weight
                self._current_resource_usage = max(0.0, self._current_resource_usage)  # Prevent negative
                
                logger.debug(f"Cleaned up agent instance {agent_instance.agent_id}")
                
            except Exception as e:
                logger.error(f"Error cleaning up agent {agent_instance.agent_id}: {str(e)}")
    
    def get_active_agents(self) -> Dict[str, AgentInstance]:
        """Get all currently active agent instances."""
        return self._active_agents.copy()
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentInstance]:
        """Get all active agents of a specific type."""
        agent_ids = self._agents_by_type.get(agent_type, set())
        return [self._active_agents[aid] for aid in agent_ids if aid in self._active_agents]
    
    def get_agents_by_query(self, query_id: str) -> List[AgentInstance]:
        """Get all active agents for a specific query."""
        agent_ids = self._agents_by_query.get(query_id, set())
        return [self._active_agents[aid] for aid in agent_ids if aid in self._active_agents]
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        return {
            "current_resource_usage": self._current_resource_usage,
            "max_resource_limit": self.resource_limits.max_total_resource_weight,
            "resource_utilization": self._current_resource_usage / self.resource_limits.max_total_resource_weight,
            "active_agents": len(self._active_agents),
            "max_concurrent_agents": self.resource_limits.max_concurrent_agents,
            "agents_by_type": {
                agent_type.value: len(agent_ids) 
                for agent_type, agent_ids in self._agents_by_type.items()
            }
        }
    
    def update_spawn_config(self, agent_type: AgentType, config: AgentSpawnConfig) -> None:
        """
        Update spawn configuration for an agent type.
        
        Args:
            agent_type: Agent type to update
            config: New spawn configuration
        """
        self._spawn_configs[agent_type] = config
        logger.info(f"Updated spawn configuration for {agent_type.value}")
    
    def get_spawn_config(self, agent_type: AgentType) -> Optional[AgentSpawnConfig]:
        """
        Get spawn configuration for an agent type.
        
        Args:
            agent_type: Agent type to get configuration for
            
        Returns:
            Spawn configuration or None if not found
        """
        return self._spawn_configs.get(agent_type)


# Global factory instance
_factory_instance: Optional[SubAgentFactory] = None


def get_sub_agent_factory(resource_limits: Optional[ResourceLimits] = None) -> SubAgentFactory:
    """
    Get the global sub-agent factory instance.
    
    Args:
        resource_limits: Resource limits for the factory
        
    Returns:
        SubAgentFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = SubAgentFactory(resource_limits)
    return _factory_instance


def reset_sub_agent_factory() -> None:
    """Reset the global sub-agent factory instance. Mainly for testing."""
    global _factory_instance
    _factory_instance = None