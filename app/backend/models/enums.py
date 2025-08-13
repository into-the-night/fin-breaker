"""
Enums for the multi-agent finance orchestrator system.
"""

from enum import Enum, auto


class AgentType(Enum):
    """Types of specialized sub-agents in the system."""
    MARKET_DATA = "market_data"
    COMPANY_RESEARCH = "company_research"
    TOPIC_ANALYSIS = "topic_analysis"
    RISK_ANALYSIS = "risk_analysis"
    ORCHESTRATOR = "orchestrator"


class EntityType(Enum):
    """Types of financial entities that can be extracted from queries."""
    COMPANY = "company"
    TICKER = "ticker"
    TOPIC = "topic"
    SECTOR = "sector"
    CURRENCY = "currency"
    COMMODITY = "commodity"


class AgentStatus(Enum):
    """Status of agent execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Priority(Enum):
    """Priority levels for query processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"