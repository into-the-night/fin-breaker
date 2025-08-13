"""
Comprehensive error handling for the multi-agent finance orchestrator system.

This module provides retry logic with exponential backoff, graceful degradation,
timeout management, and comprehensive error logging and audit trail.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.backend.models.enums import AgentStatus, AgentType


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better handling."""
    NETWORK = "network"
    API_RATE_LIMIT = "api_rate_limit"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorDetails:
    """Detailed information about an error."""
    error_id: str
    timestamp: datetime
    agent_id: str
    agent_type: AgentType
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    error_message: str
    exception_type: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "error_category": self.error_category.value,
            "error_severity": self.error_severity.value,
            "error_message": self.error_message,
            "exception_type": self.exception_type,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "retry_count": self.retry_count,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful
        }


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 1.0
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt with exponential backoff."""
        if attempt <= 0:
            return 0.0
        
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 1))
        delay = self.base_delay * (self.exponential_base ** (attempt - 1)) * self.backoff_multiplier
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.0, delay)


@dataclass
class TimeoutConfig:
    """Configuration for timeout management."""
    agent_timeout: float = 300.0  # 5 minutes per agent
    operation_timeout: float = 60.0  # 1 minute per operation
    cycle_timeout: float = 600.0  # 10 minutes per cycle
    total_timeout: float = 1800.0  # 30 minutes total
    
    def get_timeout_for_operation(self, operation_type: str) -> float:
        """Get timeout for specific operation type."""
        timeout_map = {
            "agent_execution": self.agent_timeout,
            "api_call": self.operation_timeout,
            "data_processing": self.operation_timeout,
            "cycle_execution": self.cycle_timeout,
            "total_execution": self.total_timeout
        }
        return timeout_map.get(operation_type, self.operation_timeout)


class ErrorClassifier:
    """Classifies errors into categories and determines appropriate handling."""
    
    @staticmethod
    def classify_error(error: Exception, context: Dict[str, Any] = None) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error into category and severity.
        
        Args:
            error: The exception to classify
            context: Additional context about the error
            
        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_str for keyword in ["connection", "network", "dns", "socket", "timeout"]):
            if "timeout" in error_str:
                return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        # API rate limiting
        if any(keyword in error_str for keyword in ["rate limit", "too many requests", "429"]):
            return ErrorCategory.API_RATE_LIMIT, ErrorSeverity.LOW
        
        # API errors
        if any(keyword in error_str for keyword in ["api", "http", "400", "401", "403", "404", "500", "502", "503"]):
            if any(code in error_str for code in ["401", "403"]):
                return ErrorCategory.API_ERROR, ErrorSeverity.HIGH
            elif any(code in error_str for code in ["500", "502", "503"]):
                return ErrorCategory.API_ERROR, ErrorSeverity.MEDIUM
            return ErrorCategory.API_ERROR, ErrorSeverity.LOW
        
        # Validation errors
        if error_type in ["ValueError", "TypeError", "ValidationError"] or "validation" in error_str:
            return ErrorCategory.VALIDATION, ErrorSeverity.HIGH
        
        # Resource exhaustion
        if any(keyword in error_str for keyword in ["memory", "disk", "resource", "limit exceeded"]):
            return ErrorCategory.RESOURCE_EXHAUSTION, ErrorSeverity.HIGH
        
        # Configuration errors
        if any(keyword in error_str for keyword in ["config", "setting", "parameter", "missing"]):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH
        
        # Timeout errors
        if error_type in ["TimeoutError", "asyncio.TimeoutError"] or "timeout" in error_str:
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        
        # Default to unknown
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    @staticmethod
    def should_retry(error_category: ErrorCategory, error_severity: ErrorSeverity, retry_count: int, max_retries: int) -> bool:
        """
        Determine if an error should be retried based on its characteristics.
        
        Args:
            error_category: Category of the error
            error_severity: Severity of the error
            retry_count: Current retry count
            max_retries: Maximum allowed retries
            
        Returns:
            True if the error should be retried
        """
        if retry_count >= max_retries:
            return False
        
        # Never retry validation or configuration errors
        if error_category in [ErrorCategory.VALIDATION, ErrorCategory.CONFIGURATION]:
            return False
        
        # Don't retry critical errors
        if error_severity == ErrorSeverity.CRITICAL:
            return False
        
        # Always retry network and timeout errors (with limits)
        if error_category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            return True
        
        # Retry API rate limits with longer delays
        if error_category == ErrorCategory.API_RATE_LIMIT:
            return True
        
        # Retry some API errors
        if error_category == ErrorCategory.API_ERROR and error_severity != ErrorSeverity.HIGH:
            return True
        
        # Retry resource exhaustion with caution
        if error_category == ErrorCategory.RESOURCE_EXHAUSTION and retry_count < 2:
            return True
        
        # Default: retry unknown errors once
        if error_category == ErrorCategory.UNKNOWN and retry_count < 1:
            return True
        
        return False


class ErrorAuditTrail:
    """Maintains a comprehensive audit trail of all errors and recovery attempts."""
    
    def __init__(self):
        self.errors: List[ErrorDetails] = []
        self.error_stats: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.ErrorAuditTrail")
    
    def record_error(self, error_details: ErrorDetails) -> None:
        """Record an error in the audit trail."""
        self.errors.append(error_details)
        self._update_stats(error_details)
        
        # Log the error with appropriate level
        log_level = self._get_log_level(error_details.error_severity)
        self.logger.log(
            log_level,
            f"Error recorded: {error_details.error_id} - {error_details.error_message}",
            extra={"error_details": error_details.to_dict()}
        )
    
    def _update_stats(self, error_details: ErrorDetails) -> None:
        """Update error statistics."""
        if not self.error_stats:
            self.error_stats = {
                "total_errors": 0,
                "by_category": {},
                "by_severity": {},
                "by_agent_type": {},
                "recovery_success_rate": 0.0
            }
        
        self.error_stats["total_errors"] += 1
        
        # Update category stats
        category = error_details.error_category.value
        self.error_stats["by_category"][category] = self.error_stats["by_category"].get(category, 0) + 1
        
        # Update severity stats
        severity = error_details.error_severity.value
        self.error_stats["by_severity"][severity] = self.error_stats["by_severity"].get(severity, 0) + 1
        
        # Update agent type stats
        agent_type = error_details.agent_type.value
        self.error_stats["by_agent_type"][agent_type] = self.error_stats["by_agent_type"].get(agent_type, 0) + 1
        
        # Update recovery success rate
        recovery_attempts = sum(1 for e in self.errors if e.recovery_attempted)
        recovery_successes = sum(1 for e in self.errors if e.recovery_successful)
        self.error_stats["recovery_success_rate"] = recovery_successes / recovery_attempts if recovery_attempts > 0 else 0.0
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get appropriate logging level for error severity."""
        level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_map.get(severity, logging.WARNING)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all recorded errors."""
        return {
            "total_errors": len(self.errors),
            "error_stats": self.error_stats,
            "recent_errors": [e.to_dict() for e in self.errors[-10:]],  # Last 10 errors
            "error_timeline": self._get_error_timeline()
        }
    
    def _get_error_timeline(self) -> List[Dict[str, Any]]:
        """Get a timeline of errors for analysis."""
        timeline = []
        for error in self.errors:
            timeline.append({
                "timestamp": error.timestamp.isoformat(),
                "agent_id": error.agent_id,
                "category": error.error_category.value,
                "severity": error.error_severity.value,
                "retry_count": error.retry_count
            })
        return timeline


class EnhancedErrorHandler:
    """Enhanced error handler with retry logic, graceful degradation, and comprehensive logging."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None, timeout_config: Optional[TimeoutConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.timeout_config = timeout_config or TimeoutConfig()
        self.audit_trail = ErrorAuditTrail()
        self.logger = logging.getLogger(f"{__name__}.EnhancedErrorHandler")
        
        # Circuit breaker state for graceful degradation
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    async def execute_with_retry(
        self,
        operation: Callable,
        agent_id: str,
        agent_type: AgentType,
        operation_name: str = "operation",
        context: Dict[str, Any] = None,
        custom_retry_config: Optional[RetryConfig] = None
    ) -> Any:
        """
        Execute an operation with retry logic and comprehensive error handling.
        
        Args:
            operation: The async operation to execute
            agent_id: ID of the agent performing the operation
            agent_type: Type of the agent
            operation_name: Name of the operation for logging
            context: Additional context for error handling
            custom_retry_config: Custom retry configuration for this operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries are exhausted
        """
        retry_config = custom_retry_config or self.retry_config
        context = context or {}
        retry_count = 0
        last_error = None
        
        while retry_count <= retry_config.max_retries:
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open(agent_type, operation_name):
                    raise Exception(f"Circuit breaker open for {agent_type.value}:{operation_name}")
                
                # Execute with timeout
                timeout = self.timeout_config.get_timeout_for_operation(operation_name)
                result = await asyncio.wait_for(operation(), timeout=timeout)
                
                # Success - reset circuit breaker and return
                self._reset_circuit_breaker(agent_type, operation_name)
                
                if retry_count > 0:
                    self.logger.info(f"Operation {operation_name} succeeded after {retry_count} retries for agent {agent_id}")
                
                return result
                
            except Exception as error:
                last_error = error
                
                # Classify the error
                error_category, error_severity = ErrorClassifier.classify_error(error, context)
                
                # Create error details
                error_details = ErrorDetails(
                    error_id=f"{agent_id}_{operation_name}_{int(time.time())}_{retry_count}",
                    timestamp=datetime.utcnow(),
                    agent_id=agent_id,
                    agent_type=agent_type,
                    error_category=error_category,
                    error_severity=error_severity,
                    error_message=str(error),
                    exception_type=type(error).__name__,
                    stack_trace=traceback.format_exc(),
                    context=context,
                    retry_count=retry_count
                )
                
                # Record the error
                self.audit_trail.record_error(error_details)
                
                # Update circuit breaker
                self._update_circuit_breaker(agent_type, operation_name, error_category)
                
                # Determine if we should retry
                should_retry = ErrorClassifier.should_retry(
                    error_category, error_severity, retry_count, retry_config.max_retries
                )
                
                if not should_retry or retry_count >= retry_config.max_retries:
                    self.logger.error(
                        f"Operation {operation_name} failed permanently for agent {agent_id} after {retry_count} retries: {str(error)}"
                    )
                    break
                
                # Calculate delay and wait
                delay = retry_config.calculate_delay(retry_count + 1)
                
                # Special handling for rate limits - longer delay
                if error_category == ErrorCategory.API_RATE_LIMIT:
                    delay = max(delay, 30.0)  # At least 30 seconds for rate limits
                
                self.logger.warning(
                    f"Operation {operation_name} failed for agent {agent_id} (attempt {retry_count + 1}), retrying in {delay:.2f}s: {str(error)}"
                )
                
                if delay > 0:
                    await asyncio.sleep(delay)
                
                retry_count += 1
        
        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise Exception(f"Operation {operation_name} failed for unknown reasons")
    
    def _is_circuit_breaker_open(self, agent_type: AgentType, operation_name: str) -> bool:
        """Check if circuit breaker is open for a specific agent type and operation."""
        key = f"{agent_type.value}:{operation_name}"
        breaker = self.circuit_breakers.get(key)
        
        if not breaker:
            return False
        
        # Check if circuit breaker should be reset (half-open state)
        if datetime.utcnow() > breaker["reset_time"]:
            breaker["state"] = "half_open"
            return False
        
        return breaker["state"] == "open"
    
    def _update_circuit_breaker(self, agent_type: AgentType, operation_name: str, error_category: ErrorCategory) -> None:
        """Update circuit breaker state based on error."""
        key = f"{agent_type.value}:{operation_name}"
        
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "failure_count": 0,
                "state": "closed",
                "reset_time": datetime.utcnow()
            }
        
        breaker = self.circuit_breakers[key]
        
        # Increment failure count for certain error types
        if error_category in [ErrorCategory.API_ERROR, ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            breaker["failure_count"] += 1
            
            # Open circuit breaker after 5 consecutive failures
            if breaker["failure_count"] >= 5:
                breaker["state"] = "open"
                breaker["reset_time"] = datetime.utcnow() + timedelta(minutes=5)  # 5-minute cooldown
                self.logger.warning(f"Circuit breaker opened for {key} due to repeated failures")
    
    def _reset_circuit_breaker(self, agent_type: AgentType, operation_name: str) -> None:
        """Reset circuit breaker after successful operation."""
        key = f"{agent_type.value}:{operation_name}"
        
        if key in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "failure_count": 0,
                "state": "closed",
                "reset_time": datetime.utcnow()
            }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary including circuit breaker states."""
        summary = self.audit_trail.get_error_summary()
        summary["circuit_breakers"] = {
            key: {
                "state": breaker["state"],
                "failure_count": breaker["failure_count"],
                "reset_time": breaker["reset_time"].isoformat()
            }
            for key, breaker in self.circuit_breakers.items()
        }
        return summary
    
    async def graceful_degradation(
        self,
        primary_operation: Callable,
        fallback_operation: Optional[Callable],
        agent_id: str,
        agent_type: AgentType,
        operation_name: str = "operation",
        context: Dict[str, Any] = None
    ) -> tuple[Any, bool]:
        """
        Execute operation with graceful degradation to fallback if primary fails.
        
        Args:
            primary_operation: Primary operation to attempt
            fallback_operation: Fallback operation if primary fails
            agent_id: ID of the agent
            agent_type: Type of the agent
            operation_name: Name of the operation
            context: Additional context
            
        Returns:
            Tuple of (result, used_fallback)
        """
        try:
            result = await self.execute_with_retry(
                primary_operation, agent_id, agent_type, operation_name, context
            )
            return result, False
            
        except Exception as primary_error:
            self.logger.warning(f"Primary operation {operation_name} failed for agent {agent_id}, attempting fallback")
            
            if fallback_operation:
                try:
                    result = await self.execute_with_retry(
                        fallback_operation, agent_id, agent_type, f"{operation_name}_fallback", context
                    )
                    self.logger.info(f"Fallback operation succeeded for agent {agent_id}")
                    return result, True
                    
                except Exception as fallback_error:
                    self.logger.error(f"Both primary and fallback operations failed for agent {agent_id}")
                    raise primary_error  # Raise the original error
            else:
                raise primary_error


# Global error handler instance
_global_error_handler: Optional[EnhancedErrorHandler] = None


def get_error_handler() -> EnhancedErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = EnhancedErrorHandler()
    return _global_error_handler


def configure_error_handler(retry_config: Optional[RetryConfig] = None, timeout_config: Optional[TimeoutConfig] = None) -> None:
    """Configure the global error handler."""
    global _global_error_handler
    _global_error_handler = EnhancedErrorHandler(retry_config, timeout_config)