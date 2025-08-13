"""
Orchestrator agent for coordinating multi-agent financial data queries.

This module implements the main orchestrator agent that manages the entire
multi-agent workflow, from query analysis to result synthesis.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.backend.models.models import (
    QueryAnalysis, FinancialResponse, ExecutionResults, AgentResult,
    ExecutionMetadata, OrchestratorConfig, AgentConfig
)
from app.backend.models.enums import AgentType, AgentStatus
from app.backend.agent.query_analyzer import QueryAnalyzer
from app.backend.agent.subagents import (
    MarketDataAgent, CompanyResearchAgent, TopicAnalysisAgent, RiskAnalysisAgent
)
from app.backend.agent.factory import get_sub_agent_factory, ResourceLimits
from app.backend.services.context_store import get_context_store
from app.backend.services.synthesis import get_llm_service
from app.backend.agent.error_handling import get_error_handler, RetryConfig, TimeoutConfig

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates multi-agent financial data queries.
    
    The orchestrator manages the complete workflow:
    1. Analyze incoming queries to identify entities and required agents
    2. Spawn appropriate sub-agents based on analysis
    3. Coordinate sub-agent execution and lifecycle management
    4. Collect and aggregate results from all sub-agents
    5. Synthesize final comprehensive response
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator agent.
        
        Args:
            config: Orchestrator configuration, uses defaults if None
        """
        self.config = config or OrchestratorConfig()
        self.orchestrator_id = f"orchestrator_{uuid.uuid4().hex[:8]}"
        
        # Initialize services
        self.query_analyzer = QueryAnalyzer()
        self.context_store = get_context_store()
        self.llm_service = get_llm_service()
        
        # Initialize enhanced error handling
        self.error_handler = get_error_handler()
        
        # Configure timeout settings based on orchestrator config
        self.timeout_config = TimeoutConfig(
            agent_timeout=300.0,  # 5 minutes per agent
            operation_timeout=60.0,  # 1 minute per operation
            cycle_timeout=self.config.cycle_timeout_seconds,
            total_timeout=self.config.cycle_timeout_seconds * self.config.max_cycles
        )
        
        # Initialize sub-agent factory with resource limits based on config
        factory_resource_limits = ResourceLimits(
            max_concurrent_agents=self.config.max_concurrent_agents,
            max_total_resource_weight=self.config.max_concurrent_agents * 2.0,  # Allow some resource flexibility
            max_agents_per_type=3,
            spawn_timeout_seconds=30
        )
        self.sub_agent_factory = get_sub_agent_factory(factory_resource_limits)
        
        logger.info(f"Initialized OrchestratorAgent {self.orchestrator_id}")
    
    async def process_query(self, query: str) -> FinancialResponse:
        """
        Process a financial query through the complete multi-agent workflow.
        
        This is the main entry point that orchestrates the entire process:
        1. Analyze the query to extract entities and determine required agents
        2. Execute multiple cycles of sub-agent coordination until completion
        3. Synthesize final results into a comprehensive response
        
        Args:
            query: The user's financial query
            
        Returns:
            FinancialResponse with synthesized answer and supporting data
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Orchestrator {self.orchestrator_id} processing query: {query}")
        
        try:
            # Step 1: Analyze the query with enhanced error handling
            analysis = await self.error_handler.execute_with_retry(
                operation=lambda: self.analyze_query(query),
                agent_id=self.orchestrator_id,
                agent_type=AgentType.ORCHESTRATOR,
                operation_name="query_analysis",
                context={"query": query}
            )
            logger.info(f"Query analysis completed: {len(analysis.entities)} entities, {len(analysis.required_agents)} agent types")
            
            # Initialize execution metadata
            execution_metadata = ExecutionMetadata(
                query_id=analysis.query_id,
                orchestrator_id=self.orchestrator_id,
                started_at=start_time,
                synthesis_model=self.config.synthesis_model
            )
            
            # Step 2: Execute multi-agent coordination cycles with timeout
            final_results = await asyncio.wait_for(
                self._execute_coordination_cycles(analysis),
                timeout=self.timeout_config.total_timeout
            )
            
            # Step 3: Synthesize final response with enhanced error handling
            response = await self.error_handler.execute_with_retry(
                operation=lambda: self.synthesize_results(query, final_results, execution_metadata),
                agent_id=self.orchestrator_id,
                agent_type=AgentType.ORCHESTRATOR,
                operation_name="result_synthesis",
                context={"query": query, "results_count": len(final_results.results)}
            )
            
            # Update execution metadata with final information
            execution_metadata.completed_at = datetime.utcnow()
            execution_metadata.total_cycles = final_results.shared_context.get("total_cycles", 0)
            response.execution_metadata = execution_metadata
            
            # Add error summary to response metadata
            error_summary = self.error_handler.get_error_summary()
            if error_summary["total_errors"] > 0:
                response.supporting_data["error_summary"] = error_summary
            
            logger.info(f"Orchestrator {self.orchestrator_id} completed query processing in {(datetime.utcnow() - start_time).total_seconds():.2f}s")
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Orchestrator {self.orchestrator_id} timed out after {self.timeout_config.total_timeout}s")
            return FinancialResponse(
                query=query,
                answer="I'm sorry, but your query took too long to process and timed out. Please try a simpler query or try again later.",
                confidence=0.0,
                sources=["timeout"],
                execution_metadata=ExecutionMetadata(
                    query_id=str(uuid.uuid4()),
                    orchestrator_id=self.orchestrator_id,
                    started_at=start_time,
                    completed_at=datetime.utcnow()
                )
            )
            
        except Exception as e:
            logger.error(f"Error in orchestrator {self.orchestrator_id} processing query: {str(e)}", exc_info=True)
            
            # Get error summary for debugging
            error_summary = self.error_handler.get_error_summary()
            
            # Return error response with error details
            return FinancialResponse(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question.",
                confidence=0.0,
                sources=["error"],
                supporting_data={"error_summary": error_summary} if error_summary["total_errors"] > 0 else {},
                execution_metadata=ExecutionMetadata(
                    query_id=str(uuid.uuid4()),
                    orchestrator_id=self.orchestrator_id,
                    started_at=start_time,
                    completed_at=datetime.utcnow()
                )
            )
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze the query to extract entities and determine required agents.
        
        Args:
            query: The user's financial query
            
        Returns:
            QueryAnalysis with extracted entities and required agent types
        """
        logger.debug(f"Analyzing query: {query}")
        analysis = await self.query_analyzer.analyze(query)
        
        # Store the analysis in context for reference
        await self.context_store.store_context(
            f"analysis_{analysis.query_id}",
            analysis.to_dict()
        )
        
        return analysis
    
    async def _execute_coordination_cycles(self, analysis: QueryAnalysis) -> ExecutionResults:
        """
        Execute multiple cycles of sub-agent coordination until completion.
        
        This implements the core orchestration logic with sophisticated decision-making
        to prevent infinite loops and ensure comprehensive information gathering.
        
        Args:
            analysis: Query analysis with entities and required agents
            
        Returns:
            ExecutionResults with all agent results and shared context
        """
        cycle_count = 0
        all_results = []
        consecutive_low_value_cycles = 0
        
        # Track cycle performance for infinite loop prevention
        cycle_performance_history = []
        
        while cycle_count < self.config.max_cycles:
            cycle_count += 1
            cycle_start_time = datetime.utcnow()
            
            logger.info(f"Starting coordination cycle {cycle_count}/{self.config.max_cycles}")
            
            try:
                # Execute cycle with timeout management
                cycle_results = await asyncio.wait_for(
                    self._execute_agent_cycle(analysis, cycle_count),
                    timeout=self.timeout_config.cycle_timeout
                )
                cycle_successful_count = len(cycle_results.get_successful_results())
                
                # Track cycle performance
                cycle_performance = {
                    "cycle": cycle_count,
                    "successful_agents": cycle_successful_count,
                    "total_agents": len(cycle_results.results),
                    "execution_time": (datetime.utcnow() - cycle_start_time).total_seconds()
                }
                cycle_performance_history.append(cycle_performance)
                
                # Add results to overall collection
                all_results.extend(cycle_results.results)
                
                # Enhanced completion decision with infinite loop prevention
                should_complete = await self.error_handler.execute_with_retry(
                    operation=lambda: self._should_complete_execution(analysis, cycle_results, all_results),
                    agent_id=self.orchestrator_id,
                    agent_type=AgentType.ORCHESTRATOR,
                    operation_name="completion_decision",
                    context={"cycle": cycle_count, "successful_agents": cycle_successful_count}
                )
                
                # Check for infinite loop patterns
                if self._detect_infinite_loop_pattern(cycle_performance_history):
                    logger.warning("Infinite loop pattern detected, completing execution")
                    break
                
                # Check for consecutive low-value cycles
                if cycle_successful_count == 0:
                    consecutive_low_value_cycles += 1
                    if consecutive_low_value_cycles >= 2:
                        logger.warning("Multiple consecutive cycles with no successful agents, completing execution")
                        break
                else:
                    consecutive_low_value_cycles = 0
                
                if should_complete:
                    logger.info(f"Execution completed after {cycle_count} cycles based on completion criteria")
                    break
                
                # If not the last cycle, prepare for next iteration
                if cycle_count < self.config.max_cycles:
                    logger.info(f"Continuing to cycle {cycle_count + 1} - need more information")
                    
                    # Adaptive delay based on cycle performance
                    delay = self._calculate_adaptive_delay(cycle_performance_history)
                    await asyncio.sleep(delay)
                
            except asyncio.TimeoutError:
                logger.error(f"Coordination cycle {cycle_count} timed out after {self.timeout_config.cycle_timeout}s")
                # Continue with available results rather than failing completely
                break
                
            except Exception as e:
                logger.error(f"Error in coordination cycle {cycle_count}: {str(e)}")
                # Continue with available results rather than failing completely
                break
        
        # Final loop prevention check
        if cycle_count >= self.config.max_cycles:
            logger.warning(f"Reached maximum cycles ({self.config.max_cycles}), completing execution")
        
        # Get final shared context
        shared_context = await self.context_store.get_shared_context(analysis.query_id)
        
        # Create final execution results with enhanced metadata
        final_results = ExecutionResults(
            query_id=analysis.query_id,
            results=all_results,
            shared_context=shared_context
        )
        
        # Add cycle performance to shared context for analysis
        shared_context["cycle_performance"] = cycle_performance_history
        shared_context["total_cycles"] = cycle_count
        shared_context["completion_reason"] = self._determine_completion_reason(
            cycle_count, cycle_performance_history, all_results
        )
        
        logger.info(f"Coordination completed: {len(all_results)} total agent executions across {cycle_count} cycles")
        return final_results
    
    def _detect_infinite_loop_pattern(self, cycle_history: List[Dict[str, Any]]) -> bool:
        """
        Detect patterns that indicate infinite loops or diminishing returns.
        
        Args:
            cycle_history: History of cycle performance metrics
            
        Returns:
            True if infinite loop pattern detected
        """
        if len(cycle_history) < 3:
            return False
        
        # Check for repeated patterns of zero successful agents
        recent_cycles = cycle_history[-3:]
        zero_success_cycles = sum(1 for cycle in recent_cycles if cycle["successful_agents"] == 0)
        
        if zero_success_cycles >= 2:
            logger.debug("Pattern detected: Multiple cycles with zero successful agents")
            return True
        
        # Check for stagnant performance (no improvement in success rate)
        if len(cycle_history) >= 4:
            recent_success_rates = [
                cycle["successful_agents"] / max(1, cycle["total_agents"]) 
                for cycle in cycle_history[-4:]
            ]
            
            # If success rates are consistently low and not improving
            if all(rate <= 0.3 for rate in recent_success_rates):
                logger.debug("Pattern detected: Consistently low success rates")
                return True
        
        return False
    
    def _calculate_adaptive_delay(self, cycle_history: List[Dict[str, Any]]) -> float:
        """
        Calculate adaptive delay between cycles based on performance.
        
        Args:
            cycle_history: History of cycle performance metrics
            
        Returns:
            Delay in seconds
        """
        if not cycle_history:
            return 1.0
        
        last_cycle = cycle_history[-1]
        
        # Shorter delay for successful cycles, longer for failed ones
        if last_cycle["successful_agents"] > 0:
            return 0.5  # Quick turnaround for successful cycles
        else:
            return 2.0  # Longer delay for failed cycles to avoid rapid failures
    
    def _determine_completion_reason(
        self, 
        cycle_count: int, 
        cycle_history: List[Dict[str, Any]], 
        all_results: List[AgentResult]
    ) -> str:
        """
        Determine the reason for completion for analysis and debugging.
        
        Args:
            cycle_count: Total number of cycles executed
            cycle_history: History of cycle performance
            all_results: All agent results
            
        Returns:
            String describing completion reason
        """
        successful_results = [r for r in all_results if r.status == AgentStatus.COMPLETED]
        
        if cycle_count >= self.config.max_cycles:
            return "max_cycles_reached"
        elif not successful_results:
            return "no_successful_results"
        elif self._detect_infinite_loop_pattern(cycle_history):
            return "infinite_loop_prevention"
        elif len(successful_results) >= len(all_results) * 0.8:
            return "high_success_rate"
        else:
            return "sufficient_information_gathered"
    
    async def _execute_agent_cycle(self, analysis: QueryAnalysis, cycle_number: int) -> ExecutionResults:
        """
        Execute a single cycle of sub-agent coordination.
        
        Args:
            analysis: Query analysis with entities and required agents
            cycle_number: Current cycle number
            
        Returns:
            ExecutionResults for this cycle
        """
        logger.debug(f"Executing agent cycle {cycle_number}")
        
        # Spawn sub-agents based on analysis
        agents = await self.spawn_subagents(analysis, cycle_number)
        
        if not agents:
            logger.warning(f"No agents spawned for cycle {cycle_number}")
            return ExecutionResults(query_id=analysis.query_id, results=[])
        
        # Execute agents concurrently with coordination
        results = await self.coordinate_execution(agents)
        
        return results
    
    async def spawn_subagents(self, analysis: QueryAnalysis, cycle_number: int) -> List:
        """
        Spawn appropriate sub-agents based on query analysis using the factory.
        
        Args:
            analysis: Query analysis with entities and required agents
            cycle_number: Current cycle number for agent naming
            
        Returns:
            List of spawned sub-agent instances
        """
        try:
            # Create agent instances using the factory
            agent_instances = await self.sub_agent_factory.create_agents_for_query(
                analysis=analysis,
                cycle_number=cycle_number
            )
            
            if not agent_instances:
                logger.warning(f"No agent instances created for cycle {cycle_number}")
                return []
            
            # Spawn agents concurrently with resource management
            spawned_instances = await self.sub_agent_factory.spawn_agents_concurrently(agent_instances)
            
            # Extract the actual agent objects for backward compatibility
            agents = [instance.agent for instance in spawned_instances]
            
            logger.info(f"Successfully spawned {len(agents)} agents for cycle {cycle_number}")
            return agents
            
        except Exception as e:
            logger.error(f"Error spawning agents for cycle {cycle_number}: {str(e)}")
            return []
    

    
    async def coordinate_execution(self, agents: List) -> ExecutionResults:
        """
        Coordinate the execution of multiple sub-agents concurrently using factory management.
        
        Args:
            agents: List of sub-agent instances to execute
            
        Returns:
            ExecutionResults with results from all agents
        """
        if not agents:
            return ExecutionResults(query_id="", results=[])
        
        query_id = agents[0].query_id
        logger.info(f"Coordinating execution of {len(agents)} agents")
        
        # Get agent instances from the factory for the current query
        agent_instances = self.sub_agent_factory.get_agents_by_query(query_id)
        
        # If no instances found, create them from the agents (fallback)
        if not agent_instances:
            logger.warning("No agent instances found in factory, using direct execution")
            return await self._execute_agents_directly(agents)
        
        # Use factory's managed execution context
        async with self.sub_agent_factory.managed_agent_execution(agent_instances) as managed_instances:
            # Execute agents concurrently with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
            
            async def execute_agent_with_semaphore(agent_instance):
                async with semaphore:
                    try:
                        # Update instance status
                        agent_instance.status = AgentStatus.IN_PROGRESS
                        
                        # Execute the agent
                        result = await agent_instance.agent.run()
                        
                        # Update instance status based on result
                        agent_instance.status = result.status
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error executing agent {agent_instance.agent_id}: {str(e)}")
                        
                        # Update instance status
                        agent_instance.status = AgentStatus.FAILED
                        
                        # Return a failed result instead of raising
                        result = AgentResult(
                            agent_id=agent_instance.agent_id,
                            agent_type=agent_instance.agent_type,
                            status=AgentStatus.FAILED,
                            error_message=str(e)
                        )
                        result.mark_failed(str(e))
                        return result
            
            # Execute all agents concurrently
            tasks = [execute_agent_with_semaphore(instance) for instance in managed_instances]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Filter out any None results and ensure we have AgentResult objects
            valid_results = []
            for result in results:
                if isinstance(result, AgentResult):
                    valid_results.append(result)
                else:
                    logger.warning(f"Invalid result type: {type(result)}")
            
            # Get shared context after all agents complete
            shared_context = await self.context_store.get_shared_context(query_id)
            
            # Add factory resource usage stats to shared context
            resource_stats = self.sub_agent_factory.get_resource_usage_stats()
            shared_context["factory_resource_stats"] = resource_stats
            
            execution_results = ExecutionResults(
                query_id=query_id,
                results=valid_results,
                shared_context=shared_context
            )
            
            successful_count = len(execution_results.get_successful_results())
            failed_count = len(execution_results.get_failed_results())
            
            logger.info(f"Agent coordination completed: {successful_count} successful, {failed_count} failed")
            return execution_results
    
    async def _execute_agents_directly(self, agents: List) -> ExecutionResults:
        """
        Fallback method to execute agents directly without factory management.
        
        Args:
            agents: List of agent instances to execute
            
        Returns:
            ExecutionResults with results from all agents
        """
        query_id = agents[0].query_id
        
        # Execute agents concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        
        async def execute_agent_with_semaphore(agent):
            async with semaphore:
                try:
                    return await agent.run()
                except Exception as e:
                    logger.error(f"Error executing agent {agent.agent_id}: {str(e)}")
                    # Return a failed result instead of raising
                    result = AgentResult(
                        agent_id=agent.agent_id,
                        agent_type=agent.agent_type,
                        status=AgentStatus.FAILED,
                        error_message=str(e)
                    )
                    result.mark_failed(str(e))
                    return result
        
        # Execute all agents concurrently
        tasks = [execute_agent_with_semaphore(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Filter out any None results and ensure we have AgentResult objects
        valid_results = []
        for result in results:
            if isinstance(result, AgentResult):
                valid_results.append(result)
            else:
                logger.warning(f"Invalid result type: {type(result)}")
        
        # Get shared context after all agents complete
        shared_context = await self.context_store.get_shared_context(query_id)
        
        execution_results = ExecutionResults(
            query_id=query_id,
            results=valid_results,
            shared_context=shared_context
        )
        
        return execution_results
    
    async def _should_complete_execution(self, analysis: QueryAnalysis, cycle_results: ExecutionResults, all_results: List[AgentResult]) -> bool:
        """
        Determine if execution should be completed based on current results.
        
        This implements sophisticated decision-making logic to determine when
        sufficient information has been gathered to provide a comprehensive answer.
        
        Args:
            analysis: Original query analysis
            cycle_results: Results from the current cycle
            all_results: All results from previous cycles
            
        Returns:
            True if execution should be completed, False to continue
        """
        # Check if we have at least one successful result overall
        all_successful = [r for r in all_results if r.status == AgentStatus.COMPLETED]
        if not all_successful:
            logger.debug("No successful results across all cycles, continuing execution")
            return False
        
        # Evaluate information completeness using LLM
        completion_decision = await self._evaluate_information_completeness(
            analysis.original_query, all_successful
        )
        
        if completion_decision == "COMPLETE":
            logger.info("LLM evaluation indicates sufficient information for completion")
            return True
        else:
            logger.info(f"LLM evaluation indicates more information needed: {completion_decision}")
        
        # Check agent type coverage - need at least 60% of required agent types
        required_agent_types = set(analysis.required_agents)
        completed_agent_types = set(result.agent_type for result in all_successful)
        coverage_ratio = len(completed_agent_types) / len(required_agent_types)
        
        if coverage_ratio >= 0.6:
            logger.info(f"High agent coverage achieved: {coverage_ratio:.2%}")
            return True
        
        # Check data quality - if we have high-quality data from fewer agents, we can complete
        data_quality_score = self._calculate_data_quality_score(all_successful)
        if data_quality_score >= 0.7:
            logger.info(f"High data quality achieved: {data_quality_score:.2f}")
            return True
        
        # Check for diminishing returns - if recent cycles aren't adding much value
        if len(all_results) > 3:  # Only check after multiple cycles
            recent_value_added = self._assess_recent_value_added(all_results)
            if recent_value_added < 0.2:
                logger.info("Diminishing returns detected, completing execution")
                return True
        
        logger.debug(f"Continuing execution - Coverage: {coverage_ratio:.2%}, Quality: {data_quality_score:.2f}")
        return False
    
    async def _evaluate_information_completeness(self, query: str, successful_results: List[AgentResult]) -> str:
        """
        Use LLM to evaluate if we have sufficient information to answer the query.
        
        Args:
            query: Original user query
            successful_results: List of successful agent results
            
        Returns:
            "COMPLETE" if sufficient information, "CONTINUE" if more needed
        """
        try:
            # Prepare context summary for evaluation
            context_summary = []
            for result in successful_results:
                agent_summary = f"{result.agent_type.value} Agent Results:\n"
                
                # Summarize key data points
                data_summary = self._summarize_agent_data(result.data)
                agent_summary += f"- Entities processed: {data_summary['entities_processed']}\n"
                agent_summary += f"- Data points collected: {data_summary['data_points']}\n"
                
                # Add key insights if available
                if result.context.get("key_insights"):
                    agent_summary += f"- Key insights: {', '.join(result.context['key_insights'])}\n"
                
                # Add specific data types
                if result.context.get("data_types_collected"):
                    agent_summary += f"- Data types: {', '.join(result.context['data_types_collected'])}\n"
                
                context_summary.append(agent_summary)
            
            # Use existing LLM service method for evaluation
            evaluation = await self.llm_service.evaluate_context(query, context_summary)
            logger.debug(f"LLM evaluation raw result: '{evaluation}'")
            
            # Map the evaluation result to our decision format
            if "CONTINUE" in evaluation.upper():
                logger.debug("Mapped to CONTINUE")
                return "CONTINUE"
            elif "REPLAN" in evaluation.upper():
                logger.debug("Mapped to CONTINUE (from REPLAN)")
                return "CONTINUE"  # REPLAN means we need more information
            else:
                logger.debug("Mapped to COMPLETE (default)")
                return "COMPLETE"
                
        except Exception as e:
            logger.error(f"Error evaluating information completeness: {str(e)}")
            # Default to continue if evaluation fails
            return "CONTINUE"
    
    def _calculate_data_quality_score(self, successful_results: List[AgentResult]) -> float:
        """
        Calculate a data quality score based on the richness of collected information.
        
        Args:
            successful_results: List of successful agent results
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not successful_results:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for result in successful_results:
            # Base score for having a successful result
            result_score = 0.2
            max_result_score = 1.0
            
            # Score based on data richness
            data_summary = self._summarize_agent_data(result.data)
            
            # Points for entities processed
            if data_summary['entities_processed'] > 0:
                result_score += min(0.3, data_summary['entities_processed'] * 0.1)
            
            # Points for data points collected
            if data_summary['data_points'] > 0:
                result_score += min(0.3, data_summary['data_points'] * 0.05)
            
            # Points for having insights
            if result.context.get("key_insights"):
                result_score += min(0.2, len(result.context["key_insights"]) * 0.05)
            
            # Penalty for errors
            if data_summary.get('has_errors'):
                result_score *= 0.8
            
            total_score += result_score
            max_possible_score += max_result_score
        
        return min(1.0, total_score / max_possible_score) if max_possible_score > 0 else 0.0
    
    def _assess_recent_value_added(self, all_results: List[AgentResult]) -> float:
        """
        Assess how much value recent cycles have added compared to earlier ones.
        
        Args:
            all_results: All agent results from all cycles
            
        Returns:
            Value added score between 0.0 and 1.0
        """
        if len(all_results) < 4:
            return 1.0  # Not enough data to assess
        
        # Sort results by completion time
        sorted_results = sorted(
            [r for r in all_results if r.completed_at],
            key=lambda x: x.completed_at
        )
        
        if len(sorted_results) < 4:
            return 1.0
        
        # Compare data richness of recent vs earlier results
        mid_point = len(sorted_results) // 2
        earlier_results = sorted_results[:mid_point]
        recent_results = sorted_results[mid_point:]
        
        earlier_quality = self._calculate_data_quality_score(earlier_results)
        recent_quality = self._calculate_data_quality_score(recent_results)
        
        # Calculate improvement ratio
        if earlier_quality == 0:
            return 1.0 if recent_quality > 0 else 0.0
        
        improvement_ratio = (recent_quality - earlier_quality) / earlier_quality
        
        # Normalize to 0-1 scale (positive improvement = higher value)
        return max(0.0, min(1.0, (improvement_ratio + 1.0) / 2.0))
    
    async def synthesize_results(
        self, 
        query: str, 
        results: ExecutionResults, 
        execution_metadata: ExecutionMetadata
    ) -> FinancialResponse:
        """
        Synthesize final response from all agent results with comprehensive supporting data.
        
        This method integrates with the existing LLMService to generate high-quality
        synthesized responses with rich supporting data and metadata.
        
        Args:
            query: Original user query
            results: Execution results from all agents
            execution_metadata: Metadata about the execution
            
        Returns:
            FinancialResponse with synthesized answer and comprehensive supporting data
        """
        logger.info(f"Synthesizing results from {len(results.results)} agent executions")
        
        try:
            # Prepare comprehensive context for synthesis
            synthesis_context = self._prepare_enhanced_synthesis_context(results)
            
            # Generate synthesized response using enhanced LLM integration
            synthesized_answer = await self._generate_enhanced_synthesized_response(
                query, synthesis_context
            )
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_enhanced_response_confidence(results, synthesis_context)
            
            # Extract comprehensive sources from agent results
            sources = self._extract_comprehensive_sources(results)
            
            # Prepare rich supporting data
            supporting_data = self._prepare_comprehensive_supporting_data(results, synthesis_context)
            
            # Update execution metadata with synthesis information
            execution_metadata.total_cycles = results.shared_context.get("total_cycles", 0)
            
            response = FinancialResponse(
                query=query,
                answer=synthesized_answer,
                supporting_data=supporting_data,
                confidence=confidence,
                sources=sources,
                execution_metadata=execution_metadata
            )
            
            logger.info(f"Enhanced response synthesis completed with confidence {confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {str(e)}")
            
            # Enhanced fallback response with partial data
            return self._create_fallback_response(query, results, execution_metadata, str(e))
    
    def _prepare_enhanced_synthesis_context(self, results: ExecutionResults) -> Dict[str, Any]:
        """
        Prepare comprehensive context for response synthesis.
        
        Args:
            results: Execution results from all agents
            
        Returns:
            Enhanced dictionary with organized context for synthesis
        """
        successful_results = results.get_successful_results()
        failed_results = results.get_failed_results()
        
        context = {
            "successful_results": [],
            "failed_results_summary": [],
            "entities_processed": set(),
            "data_types_collected": set(),
            "key_insights": [],
            "data_quality_metrics": {},
            "execution_metrics": {},
            "shared_context": results.shared_context,
            "agent_coverage": {}
        }
        
        # Process successful results
        for result in successful_results:
            result_data = {
                "agent_type": result.agent_type.value,
                "agent_id": result.agent_id,
                "execution_time": result.execution_time,
                "retry_count": result.retry_count,
                "data_summary": self._create_detailed_data_summary(result.data),
                "key_insights": result.context.get("key_insights", []),
                "entities_processed": result.context.get("entities_processed", []),
                "data_types_collected": result.context.get("data_types_collected", [])
            }
            
            context["successful_results"].append(result_data)
            
            # Aggregate entities and data types
            if result_data["entities_processed"]:
                context["entities_processed"].update(result_data["entities_processed"])
            if result_data["data_types_collected"]:
                context["data_types_collected"].update(result_data["data_types_collected"])
            if result_data["key_insights"]:
                context["key_insights"].extend(result_data["key_insights"])
        
        # Process failed results for context
        for result in failed_results:
            failure_summary = {
                "agent_type": result.agent_type.value,
                "error_message": result.error_message,
                "retry_count": result.retry_count
            }
            context["failed_results_summary"].append(failure_summary)
        
        # Calculate agent coverage
        all_agent_types = set(result.agent_type for result in results.results)
        successful_agent_types = set(result.agent_type for result in successful_results)
        
        for agent_type in all_agent_types:
            context["agent_coverage"][agent_type.value] = {
                "attempted": sum(1 for r in results.results if r.agent_type == agent_type),
                "successful": sum(1 for r in successful_results if r.agent_type == agent_type),
                "success_rate": sum(1 for r in successful_results if r.agent_type == agent_type) / 
                              max(1, sum(1 for r in results.results if r.agent_type == agent_type))
            }
        
        # Convert sets to lists for JSON serialization
        context["entities_processed"] = list(context["entities_processed"])
        context["data_types_collected"] = list(context["data_types_collected"])
        
        # Calculate data quality metrics
        context["data_quality_metrics"] = {
            "total_data_points": sum(
                result["data_summary"]["total_data_points"] 
                for result in context["successful_results"]
            ),
            "entities_with_data": sum(
                result["data_summary"]["entities_with_data"] 
                for result in context["successful_results"]
            ),
            "average_data_richness": sum(
                result["data_summary"]["data_richness"] 
                for result in context["successful_results"]
            ) / max(1, len(context["successful_results"]))
        }
        
        # Calculate execution metrics
        if results.execution_summary:
            context["execution_metrics"] = {
                "total_execution_time": results.execution_summary.total_execution_time,
                "average_execution_time": results.execution_summary.average_execution_time,
                "success_rate": results.execution_summary.successful_agents / 
                               max(1, results.execution_summary.total_agents),
                "total_retries": results.execution_summary.total_retries
            }
        
        return context
    
    def _create_detailed_data_summary(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed summary of agent data for synthesis.
        
        Args:
            agent_data: Raw data from agent execution
            
        Returns:
            Detailed data summary with metrics
        """
        summary = {
            "total_data_points": 0,
            "entities_with_data": 0,
            "entities_without_data": 0,
            "data_richness": 0.0,
            "error_count": 0,
            "data_categories": set()
        }
        
        for key, value in agent_data.items():
            if key.startswith("entity_"):
                if isinstance(value, dict):
                    # Count data fields (excluding metadata fields)
                    data_fields = [k for k, v in value.items() 
                                 if k not in ["entity_value", "entity_type", "errors", "metadata"] 
                                 and v is not None and v != ""]
                    
                    if data_fields:
                        summary["entities_with_data"] += 1
                        summary["total_data_points"] += len(data_fields)
                        
                        # Categorize data types
                        for field in data_fields:
                            if "price" in field.lower() or "value" in field.lower():
                                summary["data_categories"].add("pricing")
                            elif "volume" in field.lower():
                                summary["data_categories"].add("volume")
                            elif "news" in field.lower() or "article" in field.lower():
                                summary["data_categories"].add("news")
                            elif "earnings" in field.lower() or "revenue" in field.lower():
                                summary["data_categories"].add("financials")
                            elif "risk" in field.lower() or "volatility" in field.lower():
                                summary["data_categories"].add("risk")
                            else:
                                summary["data_categories"].add("other")
                    else:
                        summary["entities_without_data"] += 1
                    
                    # Count errors
                    if value.get("errors"):
                        summary["error_count"] += len(value["errors"])
        
        # Calculate data richness score
        total_entities = summary["entities_with_data"] + summary["entities_without_data"]
        if total_entities > 0:
            summary["data_richness"] = summary["entities_with_data"] / total_entities
        
        # Convert set to list for JSON serialization
        summary["data_categories"] = list(summary["data_categories"])
        
        return summary
    
    async def _generate_enhanced_synthesized_response(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate enhanced synthesized response using LLM with rich context.
        
        Args:
            query: Original user query
            context: Enhanced synthesis context
            
        Returns:
            Synthesized response text
        """
        # Prepare context for the existing LLM service
        context_strings = []
        
        # Add successful agent results
        for result in context["successful_results"]:
            agent_context = f"{result['agent_type'].title()} Agent Analysis:\n"
            
            if result["data_summary"]["total_data_points"] > 0:
                agent_context += f"- Processed {result['data_summary']['entities_with_data']} entities with data\n"
                agent_context += f"- Collected {result['data_summary']['total_data_points']} data points\n"
                agent_context += f"- Data categories: {', '.join(result['data_summary']['data_categories'])}\n"
            
            if result["key_insights"]:
                agent_context += f"- Key insights: {'; '.join(result['key_insights'])}\n"
            
            agent_context += f"- Execution time: {result['execution_time']:.2f}s\n"
            
            context_strings.append(agent_context)
        
        # Add execution summary
        if context["execution_metrics"]:
            metrics = context["execution_metrics"]
            summary_context = f"Execution Summary:\n"
            summary_context += f"- Success rate: {metrics['success_rate']:.1%}\n"
            summary_context += f"- Total execution time: {metrics['total_execution_time']:.2f}s\n"
            summary_context += f"- Data quality score: {context['data_quality_metrics']['average_data_richness']:.2f}\n"
            context_strings.append(summary_context)
        
        # Add coverage information
        coverage_context = "Agent Coverage:\n"
        for agent_type, coverage in context["agent_coverage"].items():
            coverage_context += f"- {agent_type}: {coverage['successful']}/{coverage['attempted']} successful ({coverage['success_rate']:.1%})\n"
        context_strings.append(coverage_context)
        
        try:
            # Use the existing synthesize_with_context method
            synthesized_response = await self.llm_service.synthesize_with_context(
                query, context_strings
            )
            
            return synthesized_response
            
        except Exception as e:
            logger.error(f"Error in enhanced synthesis: {str(e)}")
            
            # Fallback to basic synthesis
            basic_context = [f"Analysis from {len(context['successful_results'])} specialized agents"]
            if context["key_insights"]:
                basic_context.append(f"Key insights: {'; '.join(context['key_insights'][:5])}")
            
            return await self.llm_service.synthesize_with_context(query, basic_context)
    
    def _calculate_enhanced_response_confidence(
        self, 
        results: ExecutionResults, 
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate enhanced confidence score based on multiple quality factors.
        
        Args:
            results: Execution results from all agents
            context: Enhanced synthesis context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results.results:
            return 0.0
        
        successful_results = results.get_successful_results()
        if not successful_results:
            return 0.1
        
        # Factor 1: Success rate (30% weight)
        success_rate = len(successful_results) / len(results.results)
        success_factor = success_rate * 0.3
        
        # Factor 2: Data quality (25% weight)
        data_quality = context["data_quality_metrics"]["average_data_richness"]
        data_factor = data_quality * 0.25
        
        # Factor 3: Agent coverage (20% weight)
        coverage_scores = [
            coverage["success_rate"] 
            for coverage in context["agent_coverage"].values()
        ]
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
        coverage_factor = avg_coverage * 0.2
        
        # Factor 4: Data richness (15% weight)
        total_data_points = context["data_quality_metrics"]["total_data_points"]
        data_richness = min(1.0, total_data_points / 20.0)  # Normalize to 20 data points = 1.0
        richness_factor = data_richness * 0.15
        
        # Factor 5: Insight quality (10% weight)
        insight_count = len(context["key_insights"])
        insight_factor = min(1.0, insight_count / 10.0) * 0.1  # Normalize to 10 insights = 1.0
        
        # Combine all factors
        total_confidence = (
            success_factor + data_factor + coverage_factor + 
            richness_factor + insight_factor
        )
        
        # Apply penalties for failures
        if context["failed_results_summary"]:
            failure_penalty = len(context["failed_results_summary"]) / len(results.results) * 0.1
            total_confidence = max(0.1, total_confidence - failure_penalty)
        
        return min(1.0, max(0.1, total_confidence))
    
    def _extract_comprehensive_sources(self, results: ExecutionResults) -> List[str]:
        """
        Extract comprehensive sources from agent results.
        
        Args:
            results: Execution results from all agents
            
        Returns:
            List of detailed source identifiers
        """
        sources = set()
        
        for result in results.get_successful_results():
            # Add agent type as source
            sources.add(f"{result.agent_type.value}_agent")
            
            # Add specific data sources based on agent data
            for key, value in result.data.items():
                if key.startswith("entity_") and isinstance(value, dict):
                    # Check for specific data types
                    if any(field in value for field in ["market_data", "price", "volume"]):
                        sources.add("market_data_service")
                    if any(field in value for field in ["news", "articles"]):
                        sources.add("financial_news_service")
                    if any(field in value for field in ["earnings", "revenue", "financials"]):
                        sources.add("earnings_data_service")
                    if any(field in value for field in ["risk", "volatility"]):
                        sources.add("risk_analysis_service")
        
        # Add execution metadata as source
        sources.add("multi_agent_orchestrator")
        
        return sorted(list(sources))
    
    def _prepare_comprehensive_supporting_data(
        self, 
        results: ExecutionResults, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive supporting data for the response.
        
        Args:
            results: Execution results from all agents
            context: Enhanced synthesis context
            
        Returns:
            Dictionary with comprehensive supporting data
        """
        supporting_data = {
            # Execution summary
            "execution_summary": {
                "total_agents": len(results.results),
                "successful_agents": len(results.get_successful_results()),
                "failed_agents": len(results.get_failed_results()),
                "success_rate": len(results.get_successful_results()) / max(1, len(results.results)),
                "total_cycles": context["shared_context"].get("total_cycles", 1),
                "completion_reason": context["shared_context"].get("completion_reason", "unknown")
            },
            
            # Data quality metrics
            "data_quality": context["data_quality_metrics"],
            
            # Agent coverage
            "agent_coverage": context["agent_coverage"],
            
            # Execution metrics
            "execution_metrics": context["execution_metrics"],
            
            # Content summary
            "content_summary": {
                "entities_processed": len(context["entities_processed"]),
                "data_types_collected": context["data_types_collected"],
                "key_insights_count": len(context["key_insights"]),
                "total_data_points": context["data_quality_metrics"]["total_data_points"]
            },
            
            # Error summary
            "error_summary": {
                "failed_agents": len(context["failed_results_summary"]),
                "error_types": [
                    failure["error_message"][:50] + "..." if len(failure["error_message"]) > 50 
                    else failure["error_message"]
                    for failure in context["failed_results_summary"]
                ]
            }
        }
        
        # Add execution timeline if available
        if results.execution_summary:
            supporting_data["execution_timeline"] = {
                "total_execution_time": results.execution_summary.total_execution_time,
                "average_agent_time": results.execution_summary.average_execution_time,
                "total_retries": results.execution_summary.total_retries
            }
        
        return supporting_data
    
    def _create_fallback_response(
        self, 
        query: str, 
        results: ExecutionResults, 
        execution_metadata: ExecutionMetadata, 
        error_message: str
    ) -> FinancialResponse:
        """
        Create a fallback response when synthesis fails.
        
        Args:
            query: Original user query
            results: Execution results (may be partial)
            execution_metadata: Execution metadata
            error_message: Error that caused synthesis failure
            
        Returns:
            Fallback FinancialResponse with available information
        """
        successful_results = results.get_successful_results()
        
        # Create basic answer based on available data
        if successful_results:
            answer = (
                f"I was able to gather information from {len(successful_results)} specialized agents "
                f"about your query, but encountered issues during final synthesis. "
                f"The analysis covered {len(set(r.agent_type for r in successful_results))} "
                f"different financial domains. Please try rephrasing your question or ask for "
                f"specific aspects of the analysis."
            )
            confidence = 0.4
        else:
            answer = (
                "I encountered difficulties gathering information for your query. "
                "This might be due to data availability issues or network problems. "
                "Please try again or rephrase your question."
            )
            confidence = 0.1
        
        # Basic supporting data
        supporting_data = {
            "execution_summary": {
                "total_agents": len(results.results),
                "successful_agents": len(successful_results),
                "error_occurred": True,
                "error_message": error_message[:100] + "..." if len(error_message) > 100 else error_message
            },
            "partial_data_available": len(successful_results) > 0
        }
        
        return FinancialResponse(
            query=query,
            answer=answer,
            supporting_data=supporting_data,
            confidence=confidence,
            sources=["partial_results", "error_recovery"],
            execution_metadata=execution_metadata
        )
    
    def _prepare_synthesis_context(self, results: ExecutionResults) -> Dict[str, Any]:
        """
        Prepare context for response synthesis from agent results.
        
        Args:
            results: Execution results from all agents
            
        Returns:
            Dictionary with organized context for synthesis
        """
        context = {
            "successful_results": [],
            "entities": [],
            "data_types": [],
            "shared_context": results.shared_context
        }
        
        for result in results.get_successful_results():
            # Add result data to context
            result_summary = {
                "agent_type": result.agent_type.value,
                "data_summary": self._summarize_agent_data(result.data),
                "key_insights": result.context.get("key_insights", []),
                "execution_time": result.execution_time
            }
            context["successful_results"].append(result_summary)
            
            # Extract data types collected
            if "data_types_collected" in result.context:
                context["data_types"].extend(result.context["data_types_collected"])
        
        # Remove duplicates from data types
        context["data_types"] = list(set(context["data_types"]))
        
        return context
    
    def _summarize_agent_data(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of agent data for synthesis.
        
        Args:
            agent_data: Raw data from agent execution
            
        Returns:
            Summarized data for synthesis
        """
        summary = {
            "entities_processed": 0,
            "data_points": 0,
            "has_errors": False
        }
        
        for key, value in agent_data.items():
            if key.startswith("entity_"):
                summary["entities_processed"] += 1
                if isinstance(value, dict):
                    # Count non-null data fields
                    data_fields = [k for k, v in value.items() 
                                 if k not in ["entity_value", "entity_type", "errors"] and v is not None]
                    summary["data_points"] += len(data_fields)
                    
                    # Check for errors
                    if value.get("errors"):
                        summary["has_errors"] = True
        
        return summary
    
    async def _generate_synthesized_response(self, query: str, context: Dict[str, Any]) -> str:
        """
        Generate synthesized response using LLM.
        
        Args:
            query: Original user query
            context: Prepared synthesis context
            
        Returns:
            Synthesized response text
        """
        system_prompt = """You are a financial analysis expert synthesizing information from multiple specialized agents.

Your task is to provide a comprehensive, accurate answer to the user's financial query based on the collected data.

Guidelines:
1. Synthesize information from all successful agent results
2. Provide specific data points and insights where available
3. Acknowledge any limitations or missing information
4. Structure your response clearly and professionally
5. Focus on answering the user's specific question
6. Include relevant context and supporting details

The context includes results from specialized agents for market data, company research, topic analysis, and risk analysis."""
        
        context_text = f"""
Query: {query}

Agent Results Summary:
- Successful agents: {len(context['successful_results'])}
- Data types collected: {', '.join(context['data_types']) if context['data_types'] else 'None'}

Detailed Results:
"""
        
        for result in context['successful_results']:
            context_text += f"\n{result['agent_type'].title()} Agent:\n"
            context_text += f"- Entities processed: {result['data_summary']['entities_processed']}\n"
            context_text += f"- Data points collected: {result['data_summary']['data_points']}\n"
            context_text += f"- Execution time: {result['execution_time']:.2f}s\n"
            
            if result['key_insights']:
                context_text += f"- Key insights: {', '.join(result['key_insights'])}\n"
        
        if context['shared_context']:
            context_text += f"\nShared Context: {context['shared_context']}\n"
        
        try:
            response = await self.llm_service.generate(
                messages=[{
                    "role": "user",
                    "content": context_text
                }],
                model=self.config.synthesis_model,
                system=system_prompt,
                temperature=0.3
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating synthesized response: {str(e)}")
            return f"Based on the available data from {len(context['successful_results'])} specialized agents, I can provide some insights about your query, though I encountered issues with the final synthesis. Please let me know if you'd like me to focus on any specific aspect."
    
    def _calculate_response_confidence(self, results: ExecutionResults) -> float:
        """
        Calculate confidence score for the response based on result quality.
        
        Args:
            results: Execution results from all agents
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results.results:
            return 0.0
        
        successful_results = results.get_successful_results()
        if not successful_results:
            return 0.1
        
        # Base confidence on success rate
        success_rate = len(successful_results) / len(results.results)
        
        # Adjust based on data quality
        total_data_points = 0
        total_entities = 0
        
        for result in successful_results:
            summary = self._summarize_agent_data(result.data)
            total_data_points += summary["data_points"]
            total_entities += summary["entities_processed"]
        
        # Higher confidence with more data points and entities
        data_quality_factor = min(1.0, (total_data_points + total_entities) / 10.0)
        
        # Combine factors
        confidence = (success_rate * 0.6) + (data_quality_factor * 0.4)
        
        return min(1.0, max(0.1, confidence))
    
    def _extract_sources(self, results: ExecutionResults) -> List[str]:
        """
        Extract sources from agent results.
        
        Args:
            results: Execution results from all agents
            
        Returns:
            List of source identifiers
        """
        sources = set()
        
        for result in results.get_successful_results():
            sources.add(f"{result.agent_type.value}_agent")
            
            # Add specific data sources if available
            if "market_data" in result.data:
                sources.add("market_data_api")
            if "company_news" in result.data:
                sources.add("financial_news")
            if "earnings" in result.data:
                sources.add("earnings_data")
        
        return list(sources)