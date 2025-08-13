"""
Core data models for the multi-agent finance orchestrator system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from .enums import AgentType, EntityType, AgentStatus, Priority


@dataclass
class FinancialEntity:
    """Represents a financial entity extracted from a query."""
    entity_type: EntityType
    value: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the entity after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.value.strip():
            raise ValueError("Entity value cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_type": self.entity_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialEntity":
        """Create instance from dictionary."""
        return cls(
            entity_type=EntityType(data["entity_type"]),
            value=data["value"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {})
        )


@dataclass
class QueryAnalysis:
    """Analysis results of a financial query."""
    query_id: str
    original_query: str
    entities: List[FinancialEntity]
    required_agents: List[AgentType]
    priority: Priority
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate the query analysis after initialization."""
        if not self.query_id.strip():
            raise ValueError("Query ID cannot be empty")
        if not self.original_query.strip():
            raise ValueError("Original query cannot be empty")
        if not self.entities:
            raise ValueError("At least one entity must be identified")
        if not self.required_agents:
            raise ValueError("At least one agent type must be required")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "entities": [entity.to_dict() for entity in self.entities],
            "required_agents": [agent.value for agent in self.required_agents],
            "priority": self.priority.value,
            "context_requirements": self.context_requirements,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryAnalysis":
        """Create instance from dictionary."""
        return cls(
            query_id=data["query_id"],
            original_query=data["original_query"],
            entities=[FinancialEntity.from_dict(e) for e in data["entities"]],
            required_agents=[AgentType(a) for a in data["required_agents"]],
            priority=Priority(data["priority"]),
            context_requirements=data.get("context_requirements", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class AgentResult:
    """Result from a sub-agent execution."""
    agent_id: str
    agent_type: AgentType
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_details: Dict[str, Any] = field(default_factory=dict)  # Enhanced error information
    performance_metrics: Dict[str, Any] = field(default_factory=dict)  # Performance tracking
    
    def __post_init__(self):
        """Validate the agent result after initialization."""
        if not self.agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        if self.retry_count < 0:
            raise ValueError("Retry count cannot be negative")
    
    def mark_completed(self):
        """Mark the agent result as completed."""
        self.status = AgentStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str, error_details: Dict[str, Any] = None):
        """Mark the agent result as failed with error message and details."""
        self.status = AgentStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        if error_details:
            self.error_details = error_details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "data": self.data,
            "context": self.context,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_details": self.error_details,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        """Create instance from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=AgentType(data["agent_type"]),
            status=AgentStatus(data["status"]),
            data=data.get("data", {}),
            context=data.get("context", {}),
            execution_time=data.get("execution_time", 0.0),
            retry_count=data.get("retry_count", 0),
            error_message=data.get("error_message"),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_details=data.get("error_details", {}),
            performance_metrics=data.get("performance_metrics", {})
        )


@dataclass
class ExecutionSummary:
    """Summary of execution metrics and statistics."""
    total_agents: int
    successful_agents: int
    failed_agents: int
    total_execution_time: float
    average_execution_time: float
    total_retries: int
    
    def __post_init__(self):
        """Validate execution summary after initialization."""
        if self.total_agents < 0:
            raise ValueError("Total agents cannot be negative")
        if self.successful_agents < 0:
            raise ValueError("Successful agents cannot be negative")
        if self.failed_agents < 0:
            raise ValueError("Failed agents cannot be negative")
        if self.successful_agents + self.failed_agents > self.total_agents:
            raise ValueError("Sum of successful and failed agents cannot exceed total")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_agents": self.total_agents,
            "successful_agents": self.successful_agents,
            "failed_agents": self.failed_agents,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.average_execution_time,
            "total_retries": self.total_retries
        }


@dataclass
class ExecutionMetadata:
    """Metadata about the execution process."""
    query_id: str
    orchestrator_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_cycles: int = 0
    synthesis_model: str = "gemini-2.5-pro-latest"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "orchestrator_id": self.orchestrator_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_cycles": self.total_cycles,
            "synthesis_model": self.synthesis_model
        }


@dataclass
class ExecutionResults:
    """Results from all sub-agent executions."""
    query_id: str
    results: List[AgentResult]
    shared_context: Dict[str, Any] = field(default_factory=dict)
    execution_summary: Optional[ExecutionSummary] = None
    
    def __post_init__(self):
        """Validate execution results and compute summary if not provided."""
        if not self.query_id.strip():
            raise ValueError("Query ID cannot be empty")
        
        if self.execution_summary is None:
            self._compute_summary()
    
    def _compute_summary(self):
        """Compute execution summary from results."""
        if not self.results:
            self.execution_summary = ExecutionSummary(0, 0, 0, 0.0, 0.0, 0)
            return
        
        total_agents = len(self.results)
        successful_agents = sum(1 for r in self.results if r.status == AgentStatus.COMPLETED)
        failed_agents = sum(1 for r in self.results if r.status == AgentStatus.FAILED)
        total_execution_time = sum(r.execution_time for r in self.results)
        average_execution_time = total_execution_time / total_agents if total_agents > 0 else 0.0
        total_retries = sum(r.retry_count for r in self.results)
        
        self.execution_summary = ExecutionSummary(
            total_agents=total_agents,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            total_execution_time=total_execution_time,
            average_execution_time=average_execution_time,
            total_retries=total_retries
        )
    
    def add_result(self, result: AgentResult):
        """Add a new agent result and recompute summary."""
        self.results.append(result)
        self._compute_summary()
    
    def get_successful_results(self) -> List[AgentResult]:
        """Get only the successful agent results."""
        return [r for r in self.results if r.status == AgentStatus.COMPLETED]
    
    def get_failed_results(self) -> List[AgentResult]:
        """Get only the failed agent results."""
        return [r for r in self.results if r.status == AgentStatus.FAILED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "results": [result.to_dict() for result in self.results],
            "shared_context": self.shared_context,
            "execution_summary": self.execution_summary.to_dict() if self.execution_summary else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResults":
        """Create instance from dictionary."""
        return cls(
            query_id=data["query_id"],
            results=[AgentResult.from_dict(r) for r in data["results"]],
            shared_context=data.get("shared_context", {}),
            execution_summary=ExecutionSummary(**data["execution_summary"]) if data.get("execution_summary") else None
        )


@dataclass
class FinancialResponse:
    """Final response from the orchestrator system."""
    query: str
    answer: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    execution_metadata: Optional[ExecutionMetadata] = None
    
    def __post_init__(self):
        """Validate the financial response after initialization."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "supporting_data": self.supporting_data,
            "confidence": self.confidence,
            "sources": self.sources,
            "execution_metadata": self.execution_metadata.to_dict() if self.execution_metadata else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for API responses."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialResponse":
        """Create instance from dictionary."""
        return cls(
            query=data["query"],
            answer=data["answer"],
            supporting_data=data.get("supporting_data", {}),
            confidence=data.get("confidence", 0.0),
            sources=data.get("sources", []),
            execution_metadata=ExecutionMetadata(**data["execution_metadata"]) if data.get("execution_metadata") else None
        )


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    max_retries: int = 3
    timeout_seconds: int = 300
    goal_evaluation_interval: int = 10
    tools: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate agent configuration."""
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.goal_evaluation_interval <= 0:
            raise ValueError("Goal evaluation interval must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "goal_evaluation_interval": self.goal_evaluation_interval,
            "tools": self.tools
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator agent."""
    max_concurrent_agents: int = 5
    synthesis_model: str = "gemini-2.5-pro-latest"
    context_sharing_enabled: bool = True
    max_cycles: int = 3
    cycle_timeout_seconds: int = 600
    
    def __post_init__(self):
        """Validate orchestrator configuration."""
        if self.max_concurrent_agents <= 0:
            raise ValueError("Max concurrent agents must be positive")
        if not self.synthesis_model.strip():
            raise ValueError("Synthesis model cannot be empty")
        if self.max_cycles <= 0:
            raise ValueError("Max cycles must be positive")
        if self.cycle_timeout_seconds <= 0:
            raise ValueError("Cycle timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_concurrent_agents": self.max_concurrent_agents,
            "synthesis_model": self.synthesis_model,
            "context_sharing_enabled": self.context_sharing_enabled,
            "max_cycles": self.max_cycles,
            "cycle_timeout_seconds": self.cycle_timeout_seconds
        }