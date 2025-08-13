"""
Base sub-agent class with cyclic execution for the multi-agent finance orchestrator.

This module provides the abstract base class for all specialized sub-agents,
implementing cyclic execution, failure handling, and context management.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from app.backend.models.models import AgentResult, AgentConfig
from app.backend.models.enums import AgentType, AgentStatus
from app.backend.services.context_store import get_context_store
from app.backend.agent.error_handling import get_error_handler, RetryConfig, TimeoutConfig

logger = logging.getLogger(__name__)


class BaseSubAgent(ABC):
    """
    Abstract base class for all specialized sub-agents.
    
    Implements cyclic execution pattern with configurable retry limits,
    timeouts, and context management integration.
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        query_id: str,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the base sub-agent.
        
        Args:
            agent_type: Type of the specialized agent
            query_id: ID of the query this agent is processing
            config: Agent configuration, uses defaults if None
            agent_id: Unique agent ID, generates one if None
        """
        self.agent_type = agent_type
        self.query_id = query_id
        self.agent_id = agent_id or f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        self.config = config or AgentConfig()
        
        # Execution state
        self._status = AgentStatus.NOT_STARTED
        self._retry_count = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._context: Dict[str, Any] = {}
        self._data: Dict[str, Any] = {}
        self._error_message: Optional[str] = None
        
        # Context store integration
        self._context_store = get_context_store()
        
        # Enhanced error handling
        self._error_handler = get_error_handler()
        
        # Configure retry and timeout based on agent config
        self._retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            base_delay=1.0,
            max_delay=min(60.0, self.config.timeout_seconds / 10),  # Max delay is 1/10 of timeout
            exponential_base=2.0,
            jitter=True
        )
        
        self._timeout_config = TimeoutConfig(
            agent_timeout=self.config.timeout_seconds,
            operation_timeout=min(60.0, self.config.timeout_seconds / 5),  # Operations get 1/5 of agent timeout
            cycle_timeout=self.config.timeout_seconds * 2,
            total_timeout=self.config.timeout_seconds * 3
        )
        
        logger.info(f"Initialized {self.agent_type.value} agent {self.agent_id} for query {self.query_id}")
    
    @property
    def status(self) -> AgentStatus:
        """Get the current agent status."""
        return self._status
    
    @property
    def execution_time(self) -> float:
        """Get the total execution time in seconds."""
        if self._start_time is None:
            return 0.0
        end_time = self._end_time or datetime.utcnow()
        return (end_time - self._start_time).total_seconds()
    
    async def run(self) -> AgentResult:
        """
        Run the agent with cyclic execution logic.
        
        This is the main entry point that implements the cyclic execution
        pattern with retry logic, timeouts, and failure handling.
        
        Returns:
            AgentResult containing the execution results
        """
        self._status = AgentStatus.IN_PROGRESS
        self._start_time = datetime.utcnow()
        self._retry_count = 0
        
        logger.info(f"Starting agent {self.agent_id} execution")
        
        # Register this agent with the context store
        await self._context_store.register_agent_for_query(self.query_id, self.agent_id)
        
        try:
            # Main execution loop with timeout
            await asyncio.wait_for(
                self._execution_loop(),
                timeout=self.config.timeout_seconds
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Agent {self.agent_id} timed out after {self.config.timeout_seconds} seconds")
            self._status = AgentStatus.TIMEOUT
            self._error_message = f"Agent execution timed out after {self.config.timeout_seconds} seconds"
            
        except Exception as e:
            logger.error(f"Unexpected error in agent {self.agent_id}: {str(e)}", exc_info=True)
            self._status = AgentStatus.FAILED
            self._error_message = f"Unexpected error: {str(e)}"
        
        finally:
            self._end_time = datetime.utcnow()
            
            # Store final context
            await self._store_context()
            
            # Create and return result
            result = self._create_result()
            logger.info(f"Agent {self.agent_id} completed with status {self._status.value} in {self.execution_time:.2f}s")
            return result
    
    async def _execution_loop(self) -> None:
        """
        Main execution loop implementing cyclic execution pattern with enhanced error handling.
        """
        while self._retry_count <= self.config.max_retries:
            try:
                logger.debug(f"Agent {self.agent_id} starting execution cycle {self._retry_count + 1}")
                
                # Execute the agent's main logic with enhanced error handling
                await self._error_handler.execute_with_retry(
                    operation=self.execute,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    operation_name="agent_execution",
                    context={"cycle": self._retry_count + 1, "query_id": self.query_id},
                    custom_retry_config=self._retry_config
                )
                
                # Evaluate if the goal has been achieved with error handling
                goal_achieved = await self._error_handler.execute_with_retry(
                    operation=self.evaluate_goal,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    operation_name="goal_evaluation",
                    context={"cycle": self._retry_count + 1, "query_id": self.query_id}
                )
                
                if goal_achieved:
                    logger.info(f"Agent {self.agent_id} achieved its goal after {self._retry_count + 1} attempts")
                    self._status = AgentStatus.COMPLETED
                    break
                
                # Goal not achieved, check if we should retry
                if self._retry_count >= self.config.max_retries:
                    logger.warning(f"Agent {self.agent_id} reached max retries ({self.config.max_retries}) without achieving goal")
                    self._status = AgentStatus.FAILED
                    self._error_message = f"Failed to achieve goal after {self.config.max_retries + 1} attempts"
                    break
                
                # Calculate adaptive delay based on retry count
                delay = self._retry_config.calculate_delay(self._retry_count + 1)
                delay = max(delay, self.config.goal_evaluation_interval)  # Ensure minimum delay
                
                logger.debug(f"Agent {self.agent_id} waiting {delay:.2f}s before next cycle")
                await asyncio.sleep(delay)
                self._retry_count += 1
                
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id} execution cycle {self._retry_count + 1}: {str(e)}")
                
                # Handle the failure with enhanced error handling
                try:
                    should_retry = await self.handle_failure(e)
                except Exception as handle_error:
                    logger.error(f"Error in failure handler for agent {self.agent_id}: {str(handle_error)}")
                    should_retry = False
                
                if not should_retry or self._retry_count >= self.config.max_retries:
                    logger.error(f"Agent {self.agent_id} failed permanently after {self._retry_count + 1} attempts")
                    self._status = AgentStatus.FAILED
                    self._error_message = str(e)
                    break
                
                # Calculate adaptive delay for retry
                delay = self._retry_config.calculate_delay(self._retry_count + 1)
                logger.info(f"Agent {self.agent_id} will retry after {delay:.2f}s delay")
                await asyncio.sleep(delay)
                self._retry_count += 1
    
    @abstractmethod
    async def execute(self) -> None:
        """
        Execute the agent's main logic.
        
        This method should be implemented by each specialized sub-agent
        to perform its specific tasks (e.g., market data retrieval,
        company research, etc.).
        
        The method should update self._data with collected information
        and self._context with relevant context for other agents.
        
        Raises:
            Exception: Any exception that occurs during execution
        """
        pass
    
    @abstractmethod
    async def evaluate_goal(self) -> bool:
        """
        Evaluate whether the agent has achieved its goal.
        
        This method should be implemented by each specialized sub-agent
        to determine if it has collected sufficient information to
        complete its task.
        
        Returns:
            True if the goal has been achieved, False otherwise
        """
        pass
    
    async def handle_failure(self, error: Exception) -> bool:
        """
        Handle execution failures and determine if retry should be attempted.
        
        This method uses the enhanced error handling system to classify errors
        and determine appropriate retry behavior.
        
        Args:
            error: The exception that caused the failure
            
        Returns:
            True if the agent should retry, False to fail permanently
        """
        from app.backend.agent.error_handling import ErrorClassifier
        
        logger.warning(f"Agent {self.agent_id} handling failure: {str(error)}")
        
        # Classify the error to determine handling strategy
        error_category, error_severity = ErrorClassifier.classify_error(
            error, 
            context={
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "query_id": self.query_id,
                "retry_count": self._retry_count
            }
        )
        
        # Use the classifier to determine if we should retry
        should_retry = ErrorClassifier.should_retry(
            error_category, error_severity, self._retry_count, self.config.max_retries
        )
        
        if should_retry:
            logger.info(
                f"Agent {self.agent_id} will retry after {error_category.value} error "
                f"(severity: {error_severity.value}, attempt {self._retry_count + 1}/{self.config.max_retries + 1})"
            )
        else:
            logger.error(
                f"Agent {self.agent_id} will not retry {error_category.value} error "
                f"(severity: {error_severity.value}): {str(error)}"
            )
        
        return should_retry
    
    async def update_context(self, context: Dict[str, Any]) -> None:
        """
        Update the agent's context with new information.
        
        This method merges new context information with existing context
        and stores it in the shared context store.
        
        Args:
            context: New context information to merge
        """
        if context:
            self._context.update(context)
            await self._store_context()
            logger.debug(f"Agent {self.agent_id} updated context with {len(context)} items")
    
    async def get_shared_context(self) -> Dict[str, Any]:
        """
        Get shared context for the current query.
        
        Returns:
            Dict containing shared context from other agents
        """
        shared_context = await self._context_store.get_shared_context(self.query_id)
        logger.debug(f"Agent {self.agent_id} retrieved shared context with {len(shared_context)} items")
        return shared_context
    
    async def update_shared_context(self, updates: Dict[str, Any]) -> None:
        """
        Update the shared context with new information.
        
        Args:
            updates: Updates to apply to the shared context
        """
        if updates:
            await self._context_store.update_shared_context(self.query_id, updates)
            logger.debug(f"Agent {self.agent_id} updated shared context with {len(updates)} items")
    
    async def _store_context(self) -> None:
        """Store the agent's current context in the context store."""
        if self._context:
            await self._context_store.store_context(self.agent_id, self._context)
    
    def _create_result(self) -> AgentResult:
        """
        Create an AgentResult from the current agent state.
        
        Returns:
            AgentResult containing execution results
        """
        # Get error summary from error handler
        error_summary = self._error_handler.get_error_summary()
        
        # Create performance metrics
        performance_metrics = {
            "execution_time": self.execution_time,
            "retry_count": self._retry_count,
            "timeout_config": {
                "agent_timeout": self._timeout_config.agent_timeout,
                "operation_timeout": self._timeout_config.operation_timeout
            },
            "retry_config": {
                "max_retries": self._retry_config.max_retries,
                "base_delay": self._retry_config.base_delay,
                "max_delay": self._retry_config.max_delay
            }
        }
        
        # Create error details if there were errors
        error_details = {}
        if self._error_message or error_summary.get("total_errors", 0) > 0:
            error_details = {
                "primary_error": self._error_message,
                "error_summary": error_summary,
                "final_status": self._status.value
            }
        
        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=self._status,
            data=self._data.copy(),
            context=self._context.copy(),
            execution_time=self.execution_time,
            retry_count=self._retry_count,
            error_message=self._error_message,
            started_at=self._start_time or datetime.utcnow(),
            completed_at=self._end_time,
            error_details=error_details,
            performance_metrics=performance_metrics
        )
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_type.value}Agent({self.agent_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"{self.__class__.__name__}("
                f"agent_id='{self.agent_id}', "
                f"agent_type={self.agent_type}, "
                f"query_id='{self.query_id}', "
                f"status={self._status})")


class MarketDataAgent(BaseSubAgent):
    """
    Specialized sub-agent for market data collection.
    
    This agent focuses on gathering market data including ticker information,
    time series data, earnings data, and company news for financial entities.
    """
    
    def __init__(
        self,
        query_id: str,
        entities: List[Dict[str, Any]],
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the market data agent.
        
        Args:
            query_id: ID of the query this agent is processing
            entities: List of financial entities to gather data for
            config: Agent configuration
            agent_id: Unique agent ID
        """
        super().__init__(AgentType.MARKET_DATA, query_id, config, agent_id)
        self.entities = entities or []
        self._required_data_types = {"ticker_info", "time_series", "earnings", "news"}
        self._collected_data_types = set()
        
        # Import market data service
        from app.backend.services.market_data import get_market_data
        self.market_service = get_market_data()
        
        logger.info(f"MarketDataAgent {self.agent_id} initialized with {len(self.entities)} entities")
    
    async def execute(self) -> None:
        """
        Execute market data collection for all entities.
        
        Collects ticker information, time series data, earnings, and news
        for each financial entity provided.
        """
        logger.info(f"MarketDataAgent {self.agent_id} starting data collection")
        
        for entity in self.entities:
            entity_value = entity.get("value", "")
            entity_type = entity.get("entity_type", "")
            
            try:
                await self._collect_entity_data(entity_value, entity_type)
            except Exception as e:
                logger.error(f"Error collecting data for entity {entity_value}: {str(e)}")
                # Continue with other entities even if one fails
                continue
        
        # Update shared context with collected data
        await self.update_shared_context({
            "market_data_collected": True,
            "entities_processed": len(self.entities),
            "data_types_collected": list(self._collected_data_types)
        })
        
        logger.info(f"MarketDataAgent {self.agent_id} completed data collection")
    
    async def _collect_entity_data(self, entity_value: str, entity_type: str) -> None:
        """
        Collect all available market data for a specific entity with enhanced error handling.
        
        Args:
            entity_value: The entity value (company name, ticker, etc.)
            entity_type: The type of entity (company, ticker, etc.)
        """
        logger.debug(f"Collecting data for {entity_type}: {entity_value}")
        
        # Initialize entity data structure
        entity_data = {
            "entity_value": entity_value,
            "entity_type": entity_type,
            "ticker_info": None,
            "time_series": None,
            "earnings": None,
            "news": None,
            "errors": []
        }
        
        # Step 1: Get ticker symbol if we have a company name
        ticker_symbol = entity_value
        if entity_type == "company":
            try:
                # Use enhanced error handling for ticker search
                ticker_info = await self._error_handler.execute_with_retry(
                    operation=lambda: asyncio.create_task(asyncio.to_thread(self.market_service.search_ticker, entity_value)),
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    operation_name="ticker_search",
                    context={"entity_value": entity_value, "entity_type": entity_type}
                )
                
                if ticker_info:
                    ticker_symbol = ticker_info["symbol"]
                    entity_data["ticker_info"] = ticker_info
                    self._collected_data_types.add("ticker_info")
                    logger.debug(f"Found ticker {ticker_symbol} for company {entity_value}")
                else:
                    entity_data["errors"].append(f"No ticker found for company: {entity_value}")
                    logger.warning(f"No ticker found for company: {entity_value}")
                    # Store the entity data even if we couldn't find a ticker
                    self._data[f"entity_{entity_value}"] = entity_data
                    return
            except Exception as e:
                error_msg = f"Error searching ticker for {entity_value}: {str(e)}"
                entity_data["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Step 2: Get time series data with graceful degradation
        try:
            time_series_data, used_fallback = await self._error_handler.graceful_degradation(
                primary_operation=lambda: asyncio.create_task(asyncio.to_thread(
                    self.market_service.fetch_time_series_market_data, ticker_symbol
                )),
                fallback_operation=lambda: asyncio.create_task(asyncio.to_thread(
                    self.market_service.fetch_time_series_market_data, ticker_symbol, "1d"  # Shorter period as fallback
                )),
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                operation_name="time_series_fetch",
                context={"ticker_symbol": ticker_symbol}
            )
            
            if time_series_data and "error" not in time_series_data:
                entity_data["time_series"] = time_series_data
                self._collected_data_types.add("time_series")
                logger.debug(f"Collected time series data for {ticker_symbol} (fallback: {used_fallback})")
            else:
                entity_data["errors"].append(f"No time series data available for {ticker_symbol}")
        except Exception as e:
            error_msg = f"Error fetching time series for {ticker_symbol}: {str(e)}"
            entity_data["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Step 3: Get earnings data with enhanced error handling
        try:
            earnings_data = await self._error_handler.execute_with_retry(
                operation=lambda: asyncio.create_task(asyncio.to_thread(self.market_service.fetch_earnings, ticker_symbol)),
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                operation_name="earnings_fetch",
                context={"ticker_symbol": ticker_symbol}
            )
            
            if earnings_data and "error" not in earnings_data:
                entity_data["earnings"] = earnings_data
                self._collected_data_types.add("earnings")
                logger.debug(f"Collected earnings data for {ticker_symbol}")
            else:
                entity_data["errors"].append(f"No earnings data available for {ticker_symbol}")
        except Exception as e:
            error_msg = f"Error fetching earnings for {ticker_symbol}: {str(e)}"
            entity_data["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Step 4: Get company news with enhanced error handling
        try:
            news_data = await self._error_handler.execute_with_retry(
                operation=lambda: asyncio.create_task(asyncio.to_thread(self.market_service.fetch_company_news, ticker_symbol)),
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                operation_name="news_fetch",
                context={"ticker_symbol": ticker_symbol}
            )
            
            if news_data and "error" not in news_data:
                entity_data["news"] = news_data
                self._collected_data_types.add("news")
                logger.debug(f"Collected news data for {ticker_symbol}")
            else:
                entity_data["errors"].append(f"No news data available for {ticker_symbol}")
        except Exception as e:
            error_msg = f"Error fetching news for {ticker_symbol}: {str(e)}"
            entity_data["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Store the collected data
        self._data[f"entity_{entity_value}"] = entity_data
        
        # Update context with entity-specific information
        await self.update_context({
            f"market_data_{entity_value}": {
                "ticker_symbol": ticker_symbol,
                "data_collected": len([k for k, v in entity_data.items() 
                                    if k not in ["entity_value", "entity_type", "errors"] and v is not None]),
                "has_errors": len(entity_data["errors"]) > 0
            }
        })
    
    async def evaluate_goal(self) -> bool:
        """
        Evaluate whether sufficient market data has been collected.
        
        The goal is considered achieved if:
        1. We have processed all entities
        2. We have collected at least 2 types of data for most entities
        3. We have some successful data collection (not all failures)
        
        Returns:
            True if sufficient market data has been collected
        """
        if not self.entities:
            logger.warning(f"MarketDataAgent {self.agent_id} has no entities to process")
            return True
        
        # Check if we have data for all entities
        entities_with_data = len([key for key in self._data.keys() if key.startswith("entity_")])
        if entities_with_data < len(self.entities):
            logger.debug(f"MarketDataAgent {self.agent_id} still processing entities: {entities_with_data}/{len(self.entities)}")
            return False
        
        # Check data quality - we should have collected at least 2 types of data
        if len(self._collected_data_types) < 2:
            logger.debug(f"MarketDataAgent {self.agent_id} needs more data types: {len(self._collected_data_types)}/2 minimum")
            return False
        
        # Check that we have successful data collection for most entities
        successful_entities = 0
        for key, entity_data in self._data.items():
            if key.startswith("entity_"):
                data_count = len([k for k, v in entity_data.items() 
                                if k not in ["entity_value", "entity_type", "errors"] and v is not None])
                if data_count >= 1:  # At least one successful data collection
                    successful_entities += 1
        
        success_rate = successful_entities / len(self.entities) if self.entities else 0
        goal_achieved = success_rate >= 0.5  # At least 50% success rate
        
        if goal_achieved:
            logger.info(f"MarketDataAgent {self.agent_id} achieved goal: {successful_entities}/{len(self.entities)} entities with data, {len(self._collected_data_types)} data types collected")
        else:
            logger.debug(f"MarketDataAgent {self.agent_id} goal not achieved: {successful_entities}/{len(self.entities)} entities successful, {success_rate:.2%} success rate")
        
        return goal_achieved
    
    async def handle_failure(self, error: Exception) -> bool:
        """
        Handle failures specific to market data collection with enhanced error handling.
        
        Args:
            error: The exception that caused the failure
            
        Returns:
            True if the agent should retry
        """
        from app.backend.agent.error_handling import ErrorClassifier, ErrorCategory
        
        # Classify the error
        error_category, error_severity = ErrorClassifier.classify_error(
            error,
            context={
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "query_id": self.query_id,
                "retry_count": self._retry_count,
                "entities_count": len(self.entities)
            }
        )
        
        # Market data specific handling
        if error_category == ErrorCategory.API_RATE_LIMIT:
            logger.warning(f"MarketDataAgent {self.agent_id} hit rate limit, will retry with longer delay")
            # The error handler will automatically apply longer delays for rate limits
            return True
        
        if error_category == ErrorCategory.API_ERROR:
            logger.warning(f"MarketDataAgent {self.agent_id} encountered API error: {str(error)}")
            # Allow retries for most API errors
            return self._retry_count < self.config.max_retries
        
        if error_category == ErrorCategory.NETWORK:
            logger.warning(f"MarketDataAgent {self.agent_id} encountered network error: {str(error)}")
            # Always retry network errors (within limits)
            return self._retry_count < self.config.max_retries
        
        # Use enhanced base class handling for other errors
        return await super().handle_failure(error)


class CompanyResearchAgent(BaseSubAgent):
    """
    Specialized sub-agent for company research and analysis.
    
    This agent focuses on gathering comprehensive company information including
    news, financial data, analyst recommendations, and trends.
    """
    
    def __init__(
        self,
        query_id: str,
        entities: List[Dict[str, Any]],
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the company research agent.
        
        Args:
            query_id: ID of the query this agent is processing
            entities: List of company entities to research
            config: Agent configuration
            agent_id: Unique agent ID
        """
        super().__init__(AgentType.COMPANY_RESEARCH, query_id, config, agent_id)
        self.entities = entities or []
        self._required_data_types = {"company_news", "stock_trends", "financial_data"}
        self._collected_data_types = set()
        
        # Import market data service for company research tools
        from app.backend.services.market_data import get_market_data
        self.market_service = get_market_data()
        
        logger.info(f"CompanyResearchAgent {self.agent_id} initialized with {len(self.entities)} entities")
    
    async def execute(self) -> None:
        """
        Execute company research for all entities.
        
        Collects company news, analyst recommendations, trends, and financial
        data for each company entity provided.
        """
        logger.info(f"CompanyResearchAgent {self.agent_id} starting company research")
        
        for entity in self.entities:
            entity_value = entity.get("value", "")
            entity_type = entity.get("entity_type", "")
            
            # Only process company and ticker entities
            if entity_type not in ["company", "ticker"]:
                logger.debug(f"Skipping non-company entity: {entity_type} - {entity_value}")
                continue
            
            try:
                await self._research_company(entity_value, entity_type)
            except Exception as e:
                logger.error(f"Error researching company {entity_value}: {str(e)}")
                # Continue with other entities even if one fails
                continue
        
        # Update shared context with research results
        await self.update_shared_context({
            "company_research_completed": True,
            "companies_researched": len([e for e in self.entities if e.get("entity_type") in ["company", "ticker"]]),
            "research_data_types": list(self._collected_data_types)
        })
        
        logger.info(f"CompanyResearchAgent {self.agent_id} completed company research")
    
    async def _research_company(self, entity_value: str, entity_type: str) -> None:
        """
        Conduct comprehensive research for a specific company.
        
        Args:
            entity_value: The company name or ticker symbol
            entity_type: The type of entity (company or ticker)
        """
        logger.debug(f"Researching company {entity_type}: {entity_value}")
        
        # Initialize company research data structure
        company_data = {
            "entity_value": entity_value,
            "entity_type": entity_type,
            "ticker_symbol": None,
            "company_news": None,
            "stock_trends": None,
            "financial_data": None,
            "research_summary": {},
            "errors": []
        }
        
        # Step 1: Get ticker symbol if we have a company name
        ticker_symbol = entity_value
        if entity_type == "company":
            try:
                ticker_info = self.market_service.search_ticker(entity_value)
                if ticker_info:
                    ticker_symbol = ticker_info["symbol"]
                    company_data["ticker_symbol"] = ticker_symbol
                    company_data["company_info"] = ticker_info
                    logger.debug(f"Found ticker {ticker_symbol} for company {entity_value}")
                else:
                    company_data["errors"].append(f"No ticker found for company: {entity_value}")
                    logger.warning(f"No ticker found for company: {entity_value}")
                    # Store the company data even if we couldn't find a ticker
                    self._data[f"company_{entity_value}"] = company_data
                    return
            except Exception as e:
                error_msg = f"Error searching ticker for {entity_value}: {str(e)}"
                company_data["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Step 2: Get company news
        try:
            news_data = self.market_service.fetch_company_news(ticker_symbol)
            if news_data and "error" not in news_data:
                company_data["company_news"] = news_data
                self._collected_data_types.add("company_news")
                
                # Extract key insights from news
                news_summary = self._analyze_news_data(news_data)
                company_data["research_summary"]["news_insights"] = news_summary
                
                logger.debug(f"Collected company news for {ticker_symbol}")
            else:
                company_data["errors"].append(f"No company news available for {ticker_symbol}")
        except Exception as e:
            error_msg = f"Error fetching company news for {ticker_symbol}: {str(e)}"
            company_data["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Step 3: Get stock trends and analyst recommendations
        try:
            trends_data = self.market_service.fetch_stock_trends(ticker_symbol)
            if trends_data and "error" not in trends_data:
                company_data["stock_trends"] = trends_data
                self._collected_data_types.add("stock_trends")
                
                # Extract key insights from trends
                trends_summary = self._analyze_trends_data(trends_data)
                company_data["research_summary"]["trends_insights"] = trends_summary
                
                logger.debug(f"Collected stock trends for {ticker_symbol}")
            else:
                company_data["errors"].append(f"No stock trends available for {ticker_symbol}")
        except Exception as e:
            error_msg = f"Error fetching stock trends for {ticker_symbol}: {str(e)}"
            company_data["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Step 4: Get additional financial data (earnings, time series for context)
        try:
            earnings_data = self.market_service.fetch_earnings(ticker_symbol)
            time_series_data = self.market_service.fetch_time_series_market_data(ticker_symbol, period="1mo")
            
            financial_data = {}
            if earnings_data and "error" not in earnings_data:
                financial_data["earnings"] = earnings_data
            if time_series_data and "error" not in time_series_data:
                financial_data["recent_performance"] = time_series_data
            
            if financial_data:
                company_data["financial_data"] = financial_data
                self._collected_data_types.add("financial_data")
                
                # Extract key financial insights
                financial_summary = self._analyze_financial_data(financial_data)
                company_data["research_summary"]["financial_insights"] = financial_summary
                
                logger.debug(f"Collected financial data for {ticker_symbol}")
            else:
                company_data["errors"].append(f"No financial data available for {ticker_symbol}")
        except Exception as e:
            error_msg = f"Error fetching financial data for {ticker_symbol}: {str(e)}"
            company_data["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Step 5: Create comprehensive company profile
        company_data["research_summary"]["profile_completeness"] = self._assess_profile_completeness(company_data)
        
        # Store the collected research data
        self._data[f"company_{entity_value}"] = company_data
        
        # Update context with company-specific insights
        await self.update_context({
            f"company_research_{entity_value}": {
                "ticker_symbol": ticker_symbol,
                "data_types_collected": len([k for k, v in company_data.items() 
                                           if k not in ["entity_value", "entity_type", "errors", "research_summary"] and v is not None]),
                "profile_completeness": company_data["research_summary"]["profile_completeness"],
                "has_errors": len(company_data["errors"]) > 0,
                "key_insights": company_data["research_summary"]
            }
        })
    
    def _analyze_news_data(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news data to extract key insights.
        
        Args:
            news_data: Raw news data from the API
            
        Returns:
            Dictionary containing news insights
        """
        insights = {
            "total_articles": 0,
            "sentiment_summary": "neutral",
            "key_topics": [],
            "recent_developments": []
        }
        
        try:
            if "feed" in news_data:
                insights["total_articles"] = len(news_data["feed"])
                
                # Extract recent developments (first few articles)
                for article in news_data["feed"][:3]:
                    insights["recent_developments"].append({
                        "title": article.get("title", ""),
                        "summary": article.get("summary", "")[:200] + "..." if len(article.get("summary", "")) > 200 else article.get("summary", ""),
                        "source": article.get("source", "")
                    })
            
            # Analyze overall sentiment if available
            if "overall_sentiment_score" in news_data:
                score = float(news_data["overall_sentiment_score"])
                if score > 0.1:
                    insights["sentiment_summary"] = "positive"
                elif score < -0.1:
                    insights["sentiment_summary"] = "negative"
                else:
                    insights["sentiment_summary"] = "neutral"
        
        except Exception as e:
            logger.warning(f"Error analyzing news data: {str(e)}")
        
        return insights
    
    def _analyze_trends_data(self, trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trends data to extract key insights.
        
        Args:
            trends_data: Raw trends data from the API
            
        Returns:
            Dictionary containing trends insights
        """
        insights = {
            "recommendation_summary": "hold",
            "analyst_count": 0,
            "consensus_strength": "weak"
        }
        
        try:
            if isinstance(trends_data, list) and trends_data:
                latest_recommendation = trends_data[0]
                
                # Calculate recommendation summary
                buy = latest_recommendation.get("buy", 0)
                hold = latest_recommendation.get("hold", 0)
                sell = latest_recommendation.get("sell", 0)
                strong_buy = latest_recommendation.get("strongBuy", 0)
                strong_sell = latest_recommendation.get("strongSell", 0)
                
                total_analysts = buy + hold + sell + strong_buy + strong_sell
                insights["analyst_count"] = total_analysts
                
                if total_analysts > 0:
                    buy_ratio = (buy + strong_buy) / total_analysts
                    sell_ratio = (sell + strong_sell) / total_analysts
                    
                    if buy_ratio > 0.6:
                        insights["recommendation_summary"] = "buy"
                        insights["consensus_strength"] = "strong" if buy_ratio > 0.8 else "moderate"
                    elif sell_ratio > 0.6:
                        insights["recommendation_summary"] = "sell"
                        insights["consensus_strength"] = "strong" if sell_ratio > 0.8 else "moderate"
                    else:
                        insights["recommendation_summary"] = "hold"
                        insights["consensus_strength"] = "moderate"
        
        except Exception as e:
            logger.warning(f"Error analyzing trends data: {str(e)}")
        
        return insights
    
    def _analyze_financial_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial data to extract key insights.
        
        Args:
            financial_data: Raw financial data
            
        Returns:
            Dictionary containing financial insights
        """
        insights = {
            "has_earnings": False,
            "has_recent_performance": False,
            "performance_trend": "stable"
        }
        
        try:
            if "earnings" in financial_data and financial_data["earnings"]:
                insights["has_earnings"] = True
            
            if "recent_performance" in financial_data and financial_data["recent_performance"]:
                insights["has_recent_performance"] = True
                # Could add more sophisticated performance analysis here
        
        except Exception as e:
            logger.warning(f"Error analyzing financial data: {str(e)}")
        
        return insights
    
    def _assess_profile_completeness(self, company_data: Dict[str, Any]) -> float:
        """
        Assess how complete the company research profile is.
        
        Args:
            company_data: The collected company data
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        data_points = ["company_news", "stock_trends", "financial_data"]
        collected_points = sum(1 for point in data_points if company_data.get(point) is not None)
        return collected_points / len(data_points)
    
    async def evaluate_goal(self) -> bool:
        """
        Evaluate whether comprehensive company research has been completed.
        
        The goal is considered achieved if:
        1. We have processed all company entities
        2. We have collected at least 2 types of research data for most companies
        3. We have achieved good profile completeness for most companies
        
        Returns:
            True if comprehensive company research has been completed
        """
        company_entities = [e for e in self.entities if e.get("entity_type") in ["company", "ticker"]]
        
        if not company_entities:
            logger.warning(f"CompanyResearchAgent {self.agent_id} has no company entities to research")
            return True
        
        # Check if we have research data for all company entities
        companies_with_data = len([key for key in self._data.keys() if key.startswith("company_")])
        if companies_with_data < len(company_entities):
            logger.debug(f"CompanyResearchAgent {self.agent_id} still researching companies: {companies_with_data}/{len(company_entities)}")
            return False
        
        # Check data quality - we should have collected at least 2 types of research data
        if len(self._collected_data_types) < 2:
            logger.debug(f"CompanyResearchAgent {self.agent_id} needs more research data types: {len(self._collected_data_types)}/2 minimum")
            return False
        
        # Check profile completeness for companies
        total_completeness = 0
        successful_companies = 0
        
        for key, company_data in self._data.items():
            if key.startswith("company_"):
                completeness = company_data.get("research_summary", {}).get("profile_completeness", 0)
                total_completeness += completeness
                if completeness >= 0.5:  # At least 50% complete profile
                    successful_companies += 1
        
        avg_completeness = total_completeness / len(company_entities) if company_entities else 0
        success_rate = successful_companies / len(company_entities) if company_entities else 0
        
        goal_achieved = avg_completeness >= 0.6 and success_rate >= 0.5
        
        if goal_achieved:
            logger.info(f"CompanyResearchAgent {self.agent_id} achieved goal: {successful_companies}/{len(company_entities)} companies with good profiles, {avg_completeness:.2%} avg completeness")
        else:
            logger.debug(f"CompanyResearchAgent {self.agent_id} goal not achieved: {avg_completeness:.2%} avg completeness, {success_rate:.2%} success rate")
        
        return goal_achieved
    
    async def handle_failure(self, error: Exception) -> bool:
        """
        Handle failures specific to company research.
        
        Args:
            error: The exception that caused the failure
            
        Returns:
            True if the agent should retry
        """
        # Handle API-specific errors
        if "API" in str(error) or "rate limit" in str(error).lower():
            logger.warning(f"CompanyResearchAgent {self.agent_id} encountered API error, will retry: {str(error)}")
            # Wait longer for API errors
            await asyncio.sleep(5)
            return True
        
        # Handle network errors
        if "network" in str(error).lower() or "connection" in str(error).lower():
            logger.warning(f"CompanyResearchAgent {self.agent_id} encountered network error, will retry: {str(error)}")
            await asyncio.sleep(2)
            return True
        
        # Use default handling for other errors
        return await super().handle_failure(error)


class TopicAnalysisAgent(BaseSubAgent):
    """
    Specialized sub-agent for topic and sector analysis.
    
    This agent focuses on gathering thematic financial information including
    topic news, sector trends, and thematic analysis for financial topics.
    """
    
    def __init__(
        self,
        query_id: str,
        entities: List[Dict[str, Any]],
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the topic analysis agent.
        
        Args:
            query_id: ID of the query this agent is processing
            entities: List of topic/sector entities to analyze
            config: Agent configuration
            agent_id: Unique agent ID
        """
        super().__init__(AgentType.TOPIC_ANALYSIS, query_id, config, agent_id)
        self.entities = entities or []
        self._required_data_types = {"topic_news", "sector_analysis", "thematic_trends"}
        self._collected_data_types = set()
        
        # Import market data service for topic analysis tools
        from app.backend.services.market_data import get_market_data
        self.market_service = get_market_data()
        
        # Mapping of topic names to API ticker names
        self._topic_mapping = {
            "blockchain": "blockchain",
            "cryptocurrency": "blockchain",
            "crypto": "blockchain",
            "earnings": "earnings",
            "ipo": "ipo",
            "initial public offering": "ipo",
            "mergers": "mergers_and_acquisitions",
            "acquisitions": "mergers_and_acquisitions",
            "m&a": "mergers_and_acquisitions",
            "financial markets": "financial_markets",
            "markets": "financial_markets",
            "fiscal policy": "economy_fiscal",
            "tax": "economy_fiscal",
            "government spending": "economy_fiscal",
            "monetary policy": "economy_monetary",
            "interest rates": "economy_monetary",
            "inflation": "economy_monetary",
            "fed": "economy_monetary",
            "economy": "economy_macro",
            "economic": "economy_macro",
            "energy": "energy_transportation",
            "transportation": "energy_transportation",
            "oil": "energy_transportation",
            "finance": "finance",
            "banking": "finance",
            "risk": "finance",
            "risk exposure": "finance",
            "risk management": "finance",
            "life sciences": "life_sciences",
            "biotech": "life_sciences",
            "pharmaceutical": "life_sciences",
            "manufacturing": "manufacturing",
            "industrial": "manufacturing",
            "real estate": "real_estate",
            "construction": "real_estate",
            "retail": "retail_wholesale",
            "wholesale": "retail_wholesale",
            "consumer": "retail_wholesale",
            "technology": "technology",
            "tech": "technology",
            "software": "technology",
            "ai": "technology",
            "artificial intelligence": "technology"
        }
        
        logger.info(f"TopicAnalysisAgent {self.agent_id} initialized with {len(self.entities)} entities")
    
    async def execute(self) -> None:
        """
        Execute topic analysis for all entities.
        
        Collects topic news, sector analysis, and thematic trends
        for each topic/sector entity provided.
        """
        logger.info(f"TopicAnalysisAgent {self.agent_id} starting topic analysis")
        
        # Group entities by type for efficient processing
        topic_entities = []
        sector_entities = []
        
        for entity in self.entities:
            entity_type = entity.get("entity_type", "")
            if entity_type == "topic":
                topic_entities.append(entity)
            elif entity_type == "sector":
                sector_entities.append(entity)
        
        # Process topic entities
        if topic_entities:
            await self._analyze_topics(topic_entities)
        
        # Process sector entities
        if sector_entities:
            await self._analyze_sectors(sector_entities)
        
        # Update shared context with analysis results
        await self.update_shared_context({
            "topic_analysis_completed": True,
            "topics_analyzed": len(topic_entities),
            "sectors_analyzed": len(sector_entities),
            "analysis_data_types": list(self._collected_data_types)
        })
        
        logger.info(f"TopicAnalysisAgent {self.agent_id} completed topic analysis")
    
    async def _analyze_topics(self, topic_entities: List[Dict[str, Any]]) -> None:
        """
        Analyze multiple topics concurrently.
        
        Args:
            topic_entities: List of topic entities to analyze
        """
        logger.debug(f"Analyzing {len(topic_entities)} topics")
        
        # Map topics to API ticker names
        api_topics = []
        topic_mapping = {}
        
        for entity in topic_entities:
            topic_value = entity.get("value", "").lower()
            api_topic = self._map_topic_to_api(topic_value)
            if api_topic:
                api_topics.append(api_topic)
                topic_mapping[api_topic] = entity
        
        if not api_topics:
            logger.warning("No valid API topics found for analysis")
            return
        
        try:
            # Fetch topic news for all topics at once
            topic_news_data = self.market_service.fetch_topic_news(api_topics)
            
            if topic_news_data and "error" not in topic_news_data:
                await self._process_topic_news(topic_news_data, topic_mapping)
                self._collected_data_types.add("topic_news")
                logger.debug(f"Collected topic news for {len(api_topics)} topics")
            else:
                logger.warning(f"No topic news data available for topics: {api_topics}")
                
        except Exception as e:
            logger.error(f"Error fetching topic news: {str(e)}")
    
    async def _analyze_sectors(self, sector_entities: List[Dict[str, Any]]) -> None:
        """
        Analyze sector entities.
        
        Args:
            sector_entities: List of sector entities to analyze
        """
        logger.debug(f"Analyzing {len(sector_entities)} sectors")
        
        for entity in sector_entities:
            sector_value = entity.get("value", "")
            
            try:
                await self._analyze_single_sector(sector_value, entity)
            except Exception as e:
                logger.error(f"Error analyzing sector {sector_value}: {str(e)}")
                continue
    
    async def _analyze_single_sector(self, sector_value: str, entity: Dict[str, Any]) -> None:
        """
        Analyze a single sector.
        
        Args:
            sector_value: The sector name
            entity: The sector entity data
        """
        logger.debug(f"Analyzing sector: {sector_value}")
        
        # Initialize sector analysis data
        sector_data = {
            "entity_value": sector_value,
            "entity_type": "sector",
            "sector_news": None,
            "sector_trends": None,
            "analysis_summary": {},
            "errors": []
        }
        
        # Map sector to topic for news analysis
        api_topic = self._map_topic_to_api(sector_value.lower())
        
        if api_topic:
            try:
                # Fetch sector-related news
                sector_news = self.market_service.fetch_topic_news([api_topic])
                if sector_news and "error" not in sector_news:
                    sector_data["sector_news"] = sector_news
                    self._collected_data_types.add("sector_analysis")
                    
                    # Analyze sector insights
                    sector_insights = self._analyze_sector_news(sector_news, sector_value)
                    sector_data["analysis_summary"]["sector_insights"] = sector_insights
                    
                    logger.debug(f"Collected sector news for {sector_value}")
                else:
                    sector_data["errors"].append(f"No sector news available for {sector_value}")
                    
            except Exception as e:
                error_msg = f"Error fetching sector news for {sector_value}: {str(e)}"
                sector_data["errors"].append(error_msg)
                logger.error(error_msg)
        else:
            sector_data["errors"].append(f"Could not map sector {sector_value} to API topic")
        
        # Store sector analysis data
        self._data[f"sector_{sector_value}"] = sector_data
        
        # Update context with sector-specific insights
        await self.update_context({
            f"sector_analysis_{sector_value}": {
                "api_topic": api_topic,
                "has_news": sector_data["sector_news"] is not None,
                "has_errors": len(sector_data["errors"]) > 0,
                "insights": sector_data["analysis_summary"]
            }
        })
    
    async def _process_topic_news(self, news_data: Dict[str, Any], topic_mapping: Dict[str, Dict[str, Any]]) -> None:
        """
        Process topic news data and create analysis for each topic.
        
        Args:
            news_data: Raw news data from the API
            topic_mapping: Mapping from API topics to original entities
        """
        logger.debug("Processing topic news data")
        
        # Create analysis for each topic
        for api_topic, entity in topic_mapping.items():
            topic_value = entity.get("value", "")
            
            # Initialize topic analysis data
            topic_data = {
                "entity_value": topic_value,
                "entity_type": "topic",
                "api_topic": api_topic,
                "topic_news": news_data,
                "thematic_trends": None,
                "analysis_summary": {},
                "errors": []
            }
            
            # Analyze topic-specific insights
            topic_insights = self._analyze_topic_news(news_data, topic_value, api_topic)
            topic_data["analysis_summary"]["topic_insights"] = topic_insights
            
            # Generate thematic trends analysis
            thematic_analysis = self._generate_thematic_analysis(news_data, topic_value)
            topic_data["thematic_trends"] = thematic_analysis
            topic_data["analysis_summary"]["thematic_analysis"] = thematic_analysis.get("summary", {})
            
            if thematic_analysis:
                self._collected_data_types.add("thematic_trends")
            
            # Store topic analysis data
            self._data[f"topic_{topic_value}"] = topic_data
            
            # Update context with topic-specific insights
            await self.update_context({
                f"topic_analysis_{topic_value}": {
                    "api_topic": api_topic,
                    "has_news": True,
                    "has_thematic_analysis": bool(thematic_analysis),
                    "insights": topic_data["analysis_summary"]
                }
            })
    
    def _map_topic_to_api(self, topic_value: str) -> Optional[str]:
        """
        Map a topic value to an API ticker name.
        
        Args:
            topic_value: The topic value to map
            
        Returns:
            API ticker name or None if not found
        """
        topic_lower = topic_value.lower().strip()
        
        # Direct mapping
        if topic_lower in self._topic_mapping:
            return self._topic_mapping[topic_lower]
        
        # Partial matching
        for key, api_topic in self._topic_mapping.items():
            if key in topic_lower or topic_lower in key:
                return api_topic
        
        # Default to technology for tech-related terms
        tech_keywords = ["tech", "digital", "innovation", "startup", "app", "platform"]
        if any(keyword in topic_lower for keyword in tech_keywords):
            return "technology"
        
        # Default to finance for finance-related terms
        finance_keywords = ["financial", "investment", "trading", "market", "stock", "bond"]
        if any(keyword in topic_lower for keyword in finance_keywords):
            return "finance"
        
        logger.warning(f"Could not map topic '{topic_value}' to API ticker")
        return None
    
    def _analyze_topic_news(self, news_data: Dict[str, Any], topic_value: str, api_topic: str) -> Dict[str, Any]:
        """
        Analyze news data for topic-specific insights.
        
        Args:
            news_data: Raw news data
            topic_value: Original topic value
            api_topic: API topic name
            
        Returns:
            Dictionary containing topic insights
        """
        insights = {
            "topic": topic_value,
            "api_topic": api_topic,
            "total_articles": 0,
            "sentiment_summary": "neutral",
            "key_developments": [],
            "relevance_score": 0.0
        }
        
        try:
            if "feed" in news_data and news_data["feed"]:
                articles = news_data["feed"]
                insights["total_articles"] = len(articles)
                
                # Filter articles relevant to the specific topic
                relevant_articles = []
                topic_keywords = topic_value.lower().split()
                
                for article in articles:
                    title = article.get("title", "").lower()
                    summary = article.get("summary", "").lower()
                    
                    # Check relevance based on keyword matching
                    relevance = sum(1 for keyword in topic_keywords if keyword in title or keyword in summary)
                    if relevance > 0:
                        relevant_articles.append((article, relevance))
                
                # Sort by relevance and take top articles
                relevant_articles.sort(key=lambda x: x[1], reverse=True)
                
                # Extract key developments from most relevant articles
                for article, relevance in relevant_articles[:3]:
                    insights["key_developments"].append({
                        "title": article.get("title", ""),
                        "summary": article.get("summary", "")[:200] + "..." if len(article.get("summary", "")) > 200 else article.get("summary", ""),
                        "source": article.get("source", ""),
                        "relevance": relevance
                    })
                
                # Calculate overall relevance score
                if articles:
                    insights["relevance_score"] = len(relevant_articles) / len(articles)
            
            # Analyze overall sentiment if available
            if "overall_sentiment_score" in news_data:
                score = float(news_data["overall_sentiment_score"])
                if score > 0.1:
                    insights["sentiment_summary"] = "positive"
                elif score < -0.1:
                    insights["sentiment_summary"] = "negative"
                else:
                    insights["sentiment_summary"] = "neutral"
        
        except Exception as e:
            logger.warning(f"Error analyzing topic news for {topic_value}: {str(e)}")
        
        return insights
    
    def _analyze_sector_news(self, news_data: Dict[str, Any], sector_value: str) -> Dict[str, Any]:
        """
        Analyze news data for sector-specific insights.
        
        Args:
            news_data: Raw news data
            sector_value: Sector name
            
        Returns:
            Dictionary containing sector insights
        """
        insights = {
            "sector": sector_value,
            "total_articles": 0,
            "sentiment_summary": "neutral",
            "sector_trends": [],
            "market_impact": "low"
        }
        
        try:
            if "feed" in news_data and news_data["feed"]:
                insights["total_articles"] = len(news_data["feed"])
                
                # Extract sector trends from recent articles
                for article in news_data["feed"][:5]:
                    insights["sector_trends"].append({
                        "title": article.get("title", ""),
                        "summary": article.get("summary", "")[:150] + "..." if len(article.get("summary", "")) > 150 else article.get("summary", ""),
                        "source": article.get("source", "")
                    })
                
                # Assess market impact based on article count and sentiment
                if insights["total_articles"] > 10:
                    insights["market_impact"] = "high"
                elif insights["total_articles"] > 5:
                    insights["market_impact"] = "medium"
            
            # Analyze overall sentiment if available
            if "overall_sentiment_score" in news_data:
                score = float(news_data["overall_sentiment_score"])
                if score > 0.1:
                    insights["sentiment_summary"] = "positive"
                elif score < -0.1:
                    insights["sentiment_summary"] = "negative"
                else:
                    insights["sentiment_summary"] = "neutral"
        
        except Exception as e:
            logger.warning(f"Error analyzing sector news for {sector_value}: {str(e)}")
        
        return insights
    
    def _generate_thematic_analysis(self, news_data: Dict[str, Any], topic_value: str) -> Dict[str, Any]:
        """
        Generate thematic analysis based on news data.
        
        Args:
            news_data: Raw news data
            topic_value: Topic name
            
        Returns:
            Dictionary containing thematic analysis
        """
        analysis = {
            "topic": topic_value,
            "trend_direction": "stable",
            "market_attention": "low",
            "key_themes": [],
            "summary": {}
        }
        
        try:
            if "feed" in news_data and news_data["feed"]:
                articles = news_data["feed"]
                article_count = len(articles)
                
                # Assess market attention based on article volume
                if article_count > 20:
                    analysis["market_attention"] = "high"
                elif article_count > 10:
                    analysis["market_attention"] = "medium"
                else:
                    analysis["market_attention"] = "low"
                
                # Extract key themes from article titles and summaries
                theme_keywords = {}
                for article in articles:
                    title = article.get("title", "").lower()
                    summary = article.get("summary", "").lower()
                    
                    # Simple keyword extraction (could be enhanced with NLP)
                    words = (title + " " + summary).split()
                    for word in words:
                        if len(word) > 4 and word.isalpha():  # Filter meaningful words
                            theme_keywords[word] = theme_keywords.get(word, 0) + 1
                
                # Get top themes
                sorted_themes = sorted(theme_keywords.items(), key=lambda x: x[1], reverse=True)
                analysis["key_themes"] = [theme for theme, count in sorted_themes[:5]]
                
                # Determine trend direction based on sentiment and volume
                if "overall_sentiment_score" in news_data:
                    sentiment_score = float(news_data["overall_sentiment_score"])
                    if sentiment_score > 0.2 and article_count > 10:
                        analysis["trend_direction"] = "bullish"
                    elif sentiment_score < -0.2 and article_count > 10:
                        analysis["trend_direction"] = "bearish"
                    else:
                        analysis["trend_direction"] = "stable"
                
                # Create summary
                analysis["summary"] = {
                    "trend_direction": analysis["trend_direction"],
                    "market_attention": analysis["market_attention"],
                    "article_count": article_count,
                    "top_themes": analysis["key_themes"][:3]
                }
        
        except Exception as e:
            logger.warning(f"Error generating thematic analysis for {topic_value}: {str(e)}")
        
        return analysis
    
    async def evaluate_goal(self) -> bool:
        """
        Evaluate whether relevant topic insights have been gathered.
        
        The goal is considered achieved if:
        1. We have processed all topic/sector entities
        2. We have collected topic news or sector analysis data
        3. We have generated thematic analysis for most topics
        
        Returns:
            True if relevant topic insights have been gathered
        """
        topic_sector_entities = [e for e in self.entities if e.get("entity_type") in ["topic", "sector"]]
        
        if not topic_sector_entities:
            logger.warning(f"TopicAnalysisAgent {self.agent_id} has no topic/sector entities to analyze")
            return True
        
        # Check if we have analysis data for all topic/sector entities
        entities_with_data = len([key for key in self._data.keys() if key.startswith(("topic_", "sector_"))])
        if entities_with_data < len(topic_sector_entities):
            logger.debug(f"TopicAnalysisAgent {self.agent_id} still analyzing entities: {entities_with_data}/{len(topic_sector_entities)}")
            return False
        
        # Check data quality - we should have collected at least 1 type of analysis data
        if len(self._collected_data_types) < 1:
            logger.debug(f"TopicAnalysisAgent {self.agent_id} needs more analysis data types: {len(self._collected_data_types)}/1 minimum")
            return False
        
        # Check that we have meaningful analysis for most entities
        entities_with_insights = 0
        logger.debug(f"TopicAnalysisAgent {self.agent_id} evaluating {len(self._data)} data entries")
        for key, entity_data in self._data.items():
            if key.startswith(("topic_", "sector_")):
                has_news = entity_data.get("topic_news") is not None or entity_data.get("sector_news") is not None
                has_analysis = bool(entity_data.get("analysis_summary", {}))
                logger.debug(f"Entity {key}: has_news={has_news}, has_analysis={has_analysis}")
                if has_news and has_analysis:
                    entities_with_insights += 1
        
        success_rate = entities_with_insights / len(topic_sector_entities) if topic_sector_entities else 0
        goal_achieved = success_rate >= 0.6  # At least 60% success rate
        
        if goal_achieved:
            logger.info(f"TopicAnalysisAgent {self.agent_id} achieved goal: {entities_with_insights}/{len(topic_sector_entities)} entities with insights, {len(self._collected_data_types)} data types collected")
        else:
            logger.debug(f"TopicAnalysisAgent {self.agent_id} goal not achieved: {entities_with_insights}/{len(topic_sector_entities)} entities with insights, {success_rate:.2%} success rate")
        
        return goal_achieved
    
    async def handle_failure(self, error: Exception) -> bool:
        """
        Handle failures specific to topic analysis.
        
        Args:
            error: The exception that caused the failure
            
        Returns:
            True if the agent should retry
        """
        # Handle API-specific errors
        if "API" in str(error) or "rate limit" in str(error).lower():
            logger.warning(f"TopicAnalysisAgent {self.agent_id} encountered API error, will retry: {str(error)}")
            # Wait longer for API errors
            await asyncio.sleep(5)
            return True
        
        # Handle network errors
        if "network" in str(error).lower() or "connection" in str(error).lower():
            logger.warning(f"TopicAnalysisAgent {self.agent_id} encountered network error, will retry: {str(error)}")
            await asyncio.sleep(2)
            return True
        
        # Use default handling for other errors
        return await super().handle_failure(error)


class RiskAnalysisAgent(BaseSubAgent):
    """
    Specialized sub-agent for risk analysis and assessment.
    
    This agent focuses on calculating risk metrics, volatility analysis,
    and correlation analysis for financial entities and portfolios.
    """
    
    def __init__(
        self,
        query_id: str,
        entities: List[Dict[str, Any]],
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the risk analysis agent.
        
        Args:
            query_id: ID of the query this agent is processing
            entities: List of financial entities to analyze for risk
            config: Agent configuration
            agent_id: Unique agent ID
        """
        super().__init__(AgentType.RISK_ANALYSIS, query_id, config, agent_id)
        self.entities = entities or []
        self._required_data_types = {"volatility_analysis", "risk_metrics", "correlation_analysis"}
        self._collected_data_types = set()
        
        # Import market data service for risk analysis
        from app.backend.services.market_data import get_market_data
        self.market_service = get_market_data()
        
        logger.info(f"RiskAnalysisAgent {self.agent_id} initialized with {len(self.entities)} entities")
    
    async def execute(self) -> None:
        """
        Execute risk analysis for all entities.
        
        Calculates risk metrics, volatility analysis, and correlation
        analysis for each financial entity provided.
        """
        logger.info(f"RiskAnalysisAgent {self.agent_id} starting risk analysis")
        
        # First, collect market data for all entities
        entity_market_data = {}
        for entity in self.entities:
            entity_value = entity.get("value", "")
            entity_type = entity.get("entity_type", "")
            
            # Only analyze entities that can have market data
            if entity_type in ["company", "ticker"]:
                try:
                    market_data = await self._collect_market_data_for_risk(entity_value, entity_type)
                    if market_data:
                        entity_market_data[entity_value] = market_data
                except Exception as e:
                    logger.error(f"Error collecting market data for risk analysis of {entity_value}: {str(e)}")
                    continue
        
        # Perform risk analysis on collected data
        if entity_market_data:
            await self._perform_risk_analysis(entity_market_data)
        
        # Update shared context with risk analysis results
        await self.update_shared_context({
            "risk_analysis_completed": True,
            "entities_analyzed": len(entity_market_data),
            "risk_data_types": list(self._collected_data_types)
        })
        
        logger.info(f"RiskAnalysisAgent {self.agent_id} completed risk analysis")
    
    async def _collect_market_data_for_risk(self, entity_value: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Collect market data needed for risk analysis.
        
        Args:
            entity_value: The entity value (company name or ticker)
            entity_type: The type of entity
            
        Returns:
            Market data suitable for risk analysis or None if unavailable
        """
        logger.debug(f"Collecting market data for risk analysis: {entity_type} - {entity_value}")
        
        # Get ticker symbol if we have a company name
        ticker_symbol = entity_value
        if entity_type == "company":
            try:
                ticker_info = self.market_service.search_ticker(entity_value)
                if ticker_info:
                    ticker_symbol = ticker_info["symbol"]
                else:
                    logger.warning(f"No ticker found for company: {entity_value}")
                    return None
            except Exception as e:
                logger.error(f"Error searching ticker for {entity_value}: {str(e)}")
                return None
        
        # Collect time series data for different periods for risk analysis
        market_data = {
            "ticker_symbol": ticker_symbol,
            "entity_value": entity_value,
            "time_series_data": {},
            "errors": []
        }
        
        # Collect data for different time periods
        periods = ["1mo", "3mo", "6mo", "1y"]
        for period in periods:
            try:
                time_series = self.market_service.fetch_time_series_market_data(ticker_symbol, period=period)
                if time_series and "error" not in time_series:
                    market_data["time_series_data"][period] = time_series
                    logger.debug(f"Collected {period} data for {ticker_symbol}")
                else:
                    market_data["errors"].append(f"No {period} data available for {ticker_symbol}")
            except Exception as e:
                error_msg = f"Error fetching {period} data for {ticker_symbol}: {str(e)}"
                market_data["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Return data only if we have at least some time series data
        if market_data["time_series_data"]:
            return market_data
        else:
            logger.warning(f"No usable market data collected for {entity_value}")
            return None
    
    async def _perform_risk_analysis(self, entity_market_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Perform comprehensive risk analysis on collected market data.
        
        Args:
            entity_market_data: Dictionary of market data for each entity
        """
        logger.debug(f"Performing risk analysis on {len(entity_market_data)} entities")
        
        # Analyze each entity individually
        for entity_value, market_data in entity_market_data.items():
            try:
                risk_analysis = await self._analyze_entity_risk(entity_value, market_data)
                self._data[f"risk_{entity_value}"] = risk_analysis
            except Exception as e:
                logger.error(f"Error analyzing risk for {entity_value}: {str(e)}")
                continue
        
        # Perform correlation analysis if we have multiple entities
        if len(entity_market_data) > 1:
            try:
                correlation_analysis = await self._perform_correlation_analysis(entity_market_data)
                self._data["portfolio_correlation"] = correlation_analysis
                self._collected_data_types.add("correlation_analysis")
            except Exception as e:
                logger.error(f"Error performing correlation analysis: {str(e)}")
    
    async def _analyze_entity_risk(self, entity_value: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze risk metrics for a single entity.
        
        Args:
            entity_value: The entity value
            market_data: Market data for the entity
            
        Returns:
            Dictionary containing risk analysis results
        """
        logger.debug(f"Analyzing risk for entity: {entity_value}")
        
        risk_analysis = {
            "entity_value": entity_value,
            "ticker_symbol": market_data.get("ticker_symbol"),
            "volatility_metrics": {},
            "risk_metrics": {},
            "risk_profile": {},
            "errors": market_data.get("errors", [])
        }
        
        # Analyze volatility for each time period
        for period, time_series_data in market_data.get("time_series_data", {}).items():
            try:
                volatility_metrics = self._calculate_volatility_metrics(time_series_data, period)
                risk_analysis["volatility_metrics"][period] = volatility_metrics
                self._collected_data_types.add("volatility_analysis")
            except Exception as e:
                error_msg = f"Error calculating volatility for {period}: {str(e)}"
                risk_analysis["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Calculate overall risk metrics
        try:
            overall_risk_metrics = self._calculate_risk_metrics(risk_analysis["volatility_metrics"])
            risk_analysis["risk_metrics"] = overall_risk_metrics
            self._collected_data_types.add("risk_metrics")
        except Exception as e:
            error_msg = f"Error calculating risk metrics: {str(e)}"
            risk_analysis["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Generate risk profile
        try:
            risk_profile = self._generate_risk_profile(risk_analysis)
            risk_analysis["risk_profile"] = risk_profile
        except Exception as e:
            error_msg = f"Error generating risk profile: {str(e)}"
            risk_analysis["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Update context with entity risk information
        await self.update_context({
            f"risk_analysis_{entity_value}": {
                "ticker_symbol": risk_analysis["ticker_symbol"],
                "risk_level": risk_analysis["risk_profile"].get("risk_level", "unknown"),
                "volatility_periods": list(risk_analysis["volatility_metrics"].keys()),
                "has_errors": len(risk_analysis["errors"]) > 0
            }
        })
        
        return risk_analysis
    
    def _calculate_volatility_metrics(self, time_series_data: Dict[str, Any], period: str) -> Dict[str, Any]:
        """
        Calculate volatility metrics from time series data.
        
        Args:
            time_series_data: Time series market data
            period: Time period for the data
            
        Returns:
            Dictionary containing volatility metrics
        """
        volatility_metrics = {
            "period": period,
            "data_points": 0,
            "price_volatility": 0.0,
            "returns_volatility": 0.0,
            "max_drawdown": 0.0,
            "price_range": {"min": 0.0, "max": 0.0, "range": 0.0}
        }
        
        try:
            # Handle different data formats (AlphaVantage vs yfinance)
            prices = []
            
            if isinstance(time_series_data, dict):
                # AlphaVantage format
                if "Time Series (Daily)" in time_series_data:
                    daily_data = time_series_data["Time Series (Daily)"]
                    for date, data in daily_data.items():
                        close_price = float(data.get("4. close", 0))
                        if close_price > 0:
                            prices.append(close_price)
                
                # yfinance format (converted to dict)
                elif "Close" in time_series_data:
                    close_data = time_series_data["Close"]
                    for date, price in close_data.items():
                        if price > 0:
                            prices.append(float(price))
            
            if len(prices) < 2:
                logger.warning(f"Insufficient price data for volatility calculation: {len(prices)} points")
                return volatility_metrics
            
            # Reverse to get chronological order
            prices = list(reversed(prices))
            volatility_metrics["data_points"] = len(prices)
            
            # Calculate price range
            min_price = min(prices)
            max_price = max(prices)
            volatility_metrics["price_range"] = {
                "min": min_price,
                "max": max_price,
                "range": max_price - min_price
            }
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(daily_return)
            
            if returns:
                # Calculate volatility metrics
                import statistics
                import math
                
                # Price volatility (coefficient of variation)
                mean_price = statistics.mean(prices)
                price_std = statistics.stdev(prices) if len(prices) > 1 else 0
                volatility_metrics["price_volatility"] = (price_std / mean_price) if mean_price > 0 else 0
                
                # Returns volatility (standard deviation of returns)
                returns_std = statistics.stdev(returns) if len(returns) > 1 else 0
                # Annualize volatility (assuming daily data)
                volatility_metrics["returns_volatility"] = returns_std * math.sqrt(252)  # 252 trading days per year
                
                # Calculate maximum drawdown
                peak = prices[0]
                max_drawdown = 0
                for price in prices:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                volatility_metrics["max_drawdown"] = max_drawdown
        
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
        
        return volatility_metrics
    
    def _calculate_risk_metrics(self, volatility_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall risk metrics from volatility data.
        
        Args:
            volatility_metrics: Volatility metrics for different periods
            
        Returns:
            Dictionary containing overall risk metrics
        """
        risk_metrics = {
            "overall_volatility": 0.0,
            "risk_consistency": 0.0,
            "max_drawdown_worst": 0.0,
            "risk_trend": "stable",
            "data_quality": "low"
        }
        
        try:
            if not volatility_metrics:
                return risk_metrics
            
            # Calculate average volatility across periods
            volatilities = []
            drawdowns = []
            
            for period, metrics in volatility_metrics.items():
                if metrics.get("returns_volatility", 0) > 0:
                    volatilities.append(metrics["returns_volatility"])
                if metrics.get("max_drawdown", 0) > 0:
                    drawdowns.append(metrics["max_drawdown"])
            
            if volatilities:
                import statistics
                
                risk_metrics["overall_volatility"] = statistics.mean(volatilities)
                
                # Risk consistency (lower standard deviation means more consistent risk)
                if len(volatilities) > 1:
                    vol_std = statistics.stdev(volatilities)
                    risk_metrics["risk_consistency"] = 1.0 - min(vol_std / risk_metrics["overall_volatility"], 1.0)
                else:
                    risk_metrics["risk_consistency"] = 1.0
                
                # Determine risk trend (comparing short-term vs long-term volatility)
                if len(volatilities) >= 2:
                    short_term_vol = volatilities[0]  # Assuming first is shortest period
                    long_term_vol = volatilities[-1]  # Assuming last is longest period
                    
                    if short_term_vol > long_term_vol * 1.2:
                        risk_metrics["risk_trend"] = "increasing"
                    elif short_term_vol < long_term_vol * 0.8:
                        risk_metrics["risk_trend"] = "decreasing"
                    else:
                        risk_metrics["risk_trend"] = "stable"
            
            if drawdowns:
                risk_metrics["max_drawdown_worst"] = max(drawdowns)
            
            # Assess data quality
            total_data_points = sum(metrics.get("data_points", 0) for metrics in volatility_metrics.values())
            if total_data_points > 200:
                risk_metrics["data_quality"] = "high"
            elif total_data_points > 50:
                risk_metrics["data_quality"] = "medium"
            else:
                risk_metrics["data_quality"] = "low"
        
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
        
        return risk_metrics
    
    def _generate_risk_profile(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive risk profile for the entity.
        
        Args:
            risk_analysis: Complete risk analysis data
            
        Returns:
            Dictionary containing risk profile
        """
        risk_profile = {
            "risk_level": "medium",
            "risk_category": "moderate",
            "volatility_assessment": "average",
            "investment_suitability": "moderate_risk_tolerance",
            "key_risk_factors": [],
            "risk_score": 0.5  # 0.0 = very low risk, 1.0 = very high risk
        }
        
        try:
            risk_metrics = risk_analysis.get("risk_metrics", {})
            overall_volatility = risk_metrics.get("overall_volatility", 0)
            max_drawdown = risk_metrics.get("max_drawdown_worst", 0)
            
            # Calculate risk score based on volatility and drawdown
            volatility_score = min(overall_volatility / 0.5, 1.0)  # Normalize to 0.5 as high volatility
            drawdown_score = min(max_drawdown / 0.3, 1.0)  # Normalize to 0.3 as high drawdown
            risk_profile["risk_score"] = (volatility_score + drawdown_score) / 2
            
            # Determine risk level
            if risk_profile["risk_score"] < 0.2:
                risk_profile["risk_level"] = "very_low"
                risk_profile["risk_category"] = "conservative"
                risk_profile["volatility_assessment"] = "low"
                risk_profile["investment_suitability"] = "conservative_investors"
            elif risk_profile["risk_score"] < 0.4:
                risk_profile["risk_level"] = "low"
                risk_profile["risk_category"] = "moderate_conservative"
                risk_profile["volatility_assessment"] = "below_average"
                risk_profile["investment_suitability"] = "low_risk_tolerance"
            elif risk_profile["risk_score"] < 0.6:
                risk_profile["risk_level"] = "medium"
                risk_profile["risk_category"] = "moderate"
                risk_profile["volatility_assessment"] = "average"
                risk_profile["investment_suitability"] = "moderate_risk_tolerance"
            elif risk_profile["risk_score"] < 0.8:
                risk_profile["risk_level"] = "high"
                risk_profile["risk_category"] = "aggressive"
                risk_profile["volatility_assessment"] = "above_average"
                risk_profile["investment_suitability"] = "high_risk_tolerance"
            else:
                risk_profile["risk_level"] = "very_high"
                risk_profile["risk_category"] = "very_aggressive"
                risk_profile["volatility_assessment"] = "high"
                risk_profile["investment_suitability"] = "very_high_risk_tolerance"
            
            # Identify key risk factors
            if overall_volatility > 0.3:
                risk_profile["key_risk_factors"].append("High price volatility")
            if max_drawdown > 0.2:
                risk_profile["key_risk_factors"].append("Significant drawdown potential")
            if risk_metrics.get("risk_trend") == "increasing":
                risk_profile["key_risk_factors"].append("Increasing risk trend")
            if risk_metrics.get("data_quality") == "low":
                risk_profile["key_risk_factors"].append("Limited historical data")
        
        except Exception as e:
            logger.error(f"Error generating risk profile: {str(e)}")
        
        return risk_profile
    
    async def _perform_correlation_analysis(self, entity_market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform correlation analysis between multiple entities.
        
        Args:
            entity_market_data: Market data for multiple entities
            
        Returns:
            Dictionary containing correlation analysis results
        """
        logger.debug("Performing correlation analysis")
        
        correlation_analysis = {
            "entities": list(entity_market_data.keys()),
            "correlation_matrix": {},
            "diversification_score": 0.0,
            "portfolio_risk": "medium",
            "correlation_insights": []
        }
        
        try:
            # Extract price data for correlation calculation
            entity_prices = {}
            
            for entity_value, market_data in entity_market_data.items():
                # Use the longest available time series for correlation
                time_series_data = market_data.get("time_series_data", {})
                
                # Try to get data from longest period available
                for period in ["1y", "6mo", "3mo", "1mo"]:
                    if period in time_series_data:
                        prices = self._extract_prices_for_correlation(time_series_data[period])
                        if len(prices) > 10:  # Need sufficient data points
                            entity_prices[entity_value] = prices
                            break
            
            if len(entity_prices) < 2:
                logger.warning("Insufficient data for correlation analysis")
                return correlation_analysis
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(entity_prices)
            correlation_analysis["correlation_matrix"] = correlation_matrix
            
            # Calculate diversification score
            diversification_score = self._calculate_diversification_score(correlation_matrix)
            correlation_analysis["diversification_score"] = diversification_score
            
            # Determine portfolio risk level
            avg_correlation = sum(sum(row.values()) for row in correlation_matrix.values()) / (len(correlation_matrix) ** 2)
            if avg_correlation > 0.7:
                correlation_analysis["portfolio_risk"] = "high"
            elif avg_correlation < 0.3:
                correlation_analysis["portfolio_risk"] = "low"
            else:
                correlation_analysis["portfolio_risk"] = "medium"
            
            # Generate correlation insights
            correlation_analysis["correlation_insights"] = self._generate_correlation_insights(correlation_matrix)
        
        except Exception as e:
            logger.error(f"Error performing correlation analysis: {str(e)}")
        
        return correlation_analysis
    
    def _extract_prices_for_correlation(self, time_series_data: Dict[str, Any]) -> List[float]:
        """
        Extract price data suitable for correlation analysis.
        
        Args:
            time_series_data: Time series market data
            
        Returns:
            List of prices
        """
        prices = []
        
        try:
            # Handle different data formats
            if isinstance(time_series_data, dict):
                # AlphaVantage format
                if "Time Series (Daily)" in time_series_data:
                    daily_data = time_series_data["Time Series (Daily)"]
                    for date, data in sorted(daily_data.items()):
                        close_price = float(data.get("4. close", 0))
                        if close_price > 0:
                            prices.append(close_price)
                
                # yfinance format
                elif "Close" in time_series_data:
                    close_data = time_series_data["Close"]
                    for date, price in sorted(close_data.items()):
                        if price > 0:
                            prices.append(float(price))
        
        except Exception as e:
            logger.error(f"Error extracting prices for correlation: {str(e)}")
        
        return prices
    
    def _calculate_correlation_matrix(self, entity_prices: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between entities.
        
        Args:
            entity_prices: Dictionary of price lists for each entity
            
        Returns:
            Correlation matrix
        """
        correlation_matrix = {}
        entities = list(entity_prices.keys())
        
        try:
            # Align price data to same length (use minimum length)
            min_length = min(len(prices) for prices in entity_prices.values())
            aligned_prices = {entity: prices[-min_length:] for entity, prices in entity_prices.items()}
            
            # Calculate correlations
            for entity1 in entities:
                correlation_matrix[entity1] = {}
                for entity2 in entities:
                    if entity1 == entity2:
                        correlation_matrix[entity1][entity2] = 1.0
                    else:
                        correlation = self._calculate_correlation(
                            aligned_prices[entity1],
                            aligned_prices[entity2]
                        )
                        correlation_matrix[entity1][entity2] = correlation
        
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
        
        return correlation_matrix
    
    def _calculate_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """
        Calculate correlation coefficient between two price series.
        
        Args:
            prices1: First price series
            prices2: Second price series
            
        Returns:
            Correlation coefficient
        """
        try:
            if len(prices1) != len(prices2) or len(prices1) < 2:
                return 0.0
            
            # Calculate returns
            returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1] for i in range(1, len(prices1))]
            returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1] for i in range(1, len(prices2))]
            
            if len(returns1) < 2:
                return 0.0
            
            # Calculate correlation coefficient
            import statistics
            
            mean1 = statistics.mean(returns1)
            mean2 = statistics.mean(returns2)
            
            numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(returns1, returns2))
            
            sum_sq1 = sum((r1 - mean1) ** 2 for r1 in returns1)
            sum_sq2 = sum((r2 - mean2) ** 2 for r2 in returns2)
            
            denominator = (sum_sq1 * sum_sq2) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
        
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
    
    def _calculate_diversification_score(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate diversification score based on correlation matrix.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Diversification score (0.0 = no diversification, 1.0 = perfect diversification)
        """
        try:
            if not correlation_matrix:
                return 0.0
            
            # Calculate average correlation (excluding diagonal)
            total_correlation = 0
            count = 0
            
            for entity1, correlations in correlation_matrix.items():
                for entity2, correlation in correlations.items():
                    if entity1 != entity2:
                        total_correlation += abs(correlation)
                        count += 1
            
            if count == 0:
                return 1.0
            
            avg_correlation = total_correlation / count
            # Diversification score is inverse of average correlation
            diversification_score = 1.0 - avg_correlation
            
            return max(0.0, min(1.0, diversification_score))
        
        except Exception as e:
            logger.error(f"Error calculating diversification score: {str(e)}")
            return 0.0
    
    def _generate_correlation_insights(self, correlation_matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Generate insights from correlation analysis.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            List of correlation insights
        """
        insights = []
        
        try:
            entities = list(correlation_matrix.keys())
            
            # Find highly correlated pairs
            high_correlations = []
            low_correlations = []
            
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    correlation = correlation_matrix[entity1][entity2]
                    
                    if correlation > 0.7:
                        high_correlations.append((entity1, entity2, correlation))
                    elif correlation < 0.3:
                        low_correlations.append((entity1, entity2, correlation))
            
            # Generate insights
            if high_correlations:
                insights.append(f"High correlation detected between {len(high_correlations)} pairs - limited diversification benefit")
                for entity1, entity2, corr in high_correlations[:3]:  # Top 3
                    insights.append(f"{entity1} and {entity2} are highly correlated ({corr:.2f})")
            
            if low_correlations:
                insights.append(f"Good diversification potential with {len(low_correlations)} low-correlation pairs")
                for entity1, entity2, corr in low_correlations[:2]:  # Top 2
                    insights.append(f"{entity1} and {entity2} provide diversification ({corr:.2f})")
            
            if not high_correlations and not low_correlations:
                insights.append("Moderate correlation levels across portfolio - balanced risk profile")
        
        except Exception as e:
            logger.error(f"Error generating correlation insights: {str(e)}")
        
        return insights
    
    async def evaluate_goal(self) -> bool:
        """
        Evaluate whether complete risk profile has been assembled.
        
        The goal is considered achieved if:
        1. We have processed all relevant entities
        2. We have calculated risk metrics for most entities
        3. We have performed correlation analysis if multiple entities exist
        
        Returns:
            True if complete risk profile has been assembled
        """
        relevant_entities = [e for e in self.entities if e.get("entity_type") in ["company", "ticker"]]
        
        if not relevant_entities:
            logger.warning(f"RiskAnalysisAgent {self.agent_id} has no relevant entities for risk analysis")
            return True
        
        # Check if we have risk analysis for entities
        entities_with_risk_data = len([key for key in self._data.keys() if key.startswith("risk_")])
        if entities_with_risk_data < len(relevant_entities):
            logger.debug(f"RiskAnalysisAgent {self.agent_id} still analyzing entities: {entities_with_risk_data}/{len(relevant_entities)}")
            return False
        
        # Check data quality - we should have collected at least 2 types of risk data
        if len(self._collected_data_types) < 2:
            logger.debug(f"RiskAnalysisAgent {self.agent_id} needs more risk data types: {len(self._collected_data_types)}/2 minimum")
            return False
        
        # Check that we have meaningful risk analysis for most entities
        entities_with_complete_analysis = 0
        for key, risk_data in self._data.items():
            if key.startswith("risk_"):
                has_volatility = bool(risk_data.get("volatility_metrics"))
                has_risk_metrics = bool(risk_data.get("risk_metrics"))
                has_risk_profile = bool(risk_data.get("risk_profile"))
                
                if has_volatility and has_risk_metrics and has_risk_profile:
                    entities_with_complete_analysis += 1
        
        success_rate = entities_with_complete_analysis / len(relevant_entities) if relevant_entities else 0
        
        # If we have multiple entities, check for correlation analysis
        correlation_requirement_met = True
        if len(relevant_entities) > 1:
            correlation_requirement_met = "portfolio_correlation" in self._data
        
        goal_achieved = success_rate >= 0.6 and correlation_requirement_met
        
        if goal_achieved:
            logger.info(f"RiskAnalysisAgent {self.agent_id} achieved goal: {entities_with_complete_analysis}/{len(relevant_entities)} entities with complete analysis, correlation: {correlation_requirement_met}")
        else:
            logger.debug(f"RiskAnalysisAgent {self.agent_id} goal not achieved: {success_rate:.2%} success rate, correlation: {correlation_requirement_met}")
        
        return goal_achieved
    
    async def handle_failure(self, error: Exception) -> bool:
        """
        Handle failures specific to risk analysis.
        
        Args:
            error: The exception that caused the failure
            
        Returns:
            True if the agent should retry
        """
        # Handle API-specific errors
        if "API" in str(error) or "rate limit" in str(error).lower():
            logger.warning(f"RiskAnalysisAgent {self.agent_id} encountered API error, will retry: {str(error)}")
            # Wait longer for API errors
            await asyncio.sleep(5)
            return True
        
        # Handle network errors
        if "network" in str(error).lower() or "connection" in str(error).lower():
            logger.warning(f"RiskAnalysisAgent {self.agent_id} encountered network error, will retry: {str(error)}")
            await asyncio.sleep(2)
            return True
        
        # Handle calculation errors (might be due to insufficient data)
        if "calculation" in str(error).lower() or "insufficient" in str(error).lower():
            logger.warning(f"RiskAnalysisAgent {self.agent_id} encountered calculation error, will retry with different approach: {str(error)}")
            return True
        
        # Use default handling for other errors
        return await super().handle_failure(error)


class MockSubAgent(BaseSubAgent):
    """
    Mock sub-agent implementation for testing purposes.
    
    This agent can be configured to succeed, fail, or timeout
    for testing the base agent functionality.
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        query_id: str,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        should_succeed: bool = True,
        execution_delay: float = 0.1,
        goal_achievement_cycles: int = 1
    ):
        """
        Initialize the mock sub-agent.
        
        Args:
            agent_type: Type of the agent
            query_id: Query ID
            config: Agent configuration
            agent_id: Agent ID
            should_succeed: Whether the agent should succeed or fail
            execution_delay: Delay in seconds for each execution cycle
            goal_achievement_cycles: Number of cycles before goal is achieved
        """
        super().__init__(agent_type, query_id, config, agent_id)
        self.should_succeed = should_succeed
        self.execution_delay = execution_delay
        self.goal_achievement_cycles = goal_achievement_cycles
        self._execution_count = 0
    
    async def execute(self) -> None:
        """Mock execution that can be configured to succeed or fail."""
        await asyncio.sleep(self.execution_delay)
        self._execution_count += 1
        
        if not self.should_succeed:
            raise RuntimeError("Mock agent configured to fail")
        
        # Add some mock data
        self._data[f"execution_{self._execution_count}"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle": self._execution_count
        }
        
        # Add some mock context
        await self.update_context({
            f"context_cycle_{self._execution_count}": f"Context from cycle {self._execution_count}"
        })
    
    async def evaluate_goal(self) -> bool:
        """Mock goal evaluation."""
        return self._execution_count >= self.goal_achievement_cycles