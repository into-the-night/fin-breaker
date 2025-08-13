"""
Enhanced logging configuration for comprehensive error tracking and audit trail.

This module provides structured logging with error categorization, performance metrics,
and audit trail capabilities for the multi-agent finance orchestrator system.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs for better analysis."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add thread and process info
        if hasattr(record, 'thread'):
            log_data["thread_id"] = record.thread
        if hasattr(record, 'process'):
            log_data["process_id"] = record.process
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'exc_info', 'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str, separators=(',', ':'))


class ErrorAuditHandler(logging.Handler):
    """Custom handler that maintains an in-memory audit trail of errors."""
    
    def __init__(self, max_records: int = 1000):
        super().__init__()
        self.max_records = max_records
        self.error_records = []
        self.error_stats = {
            "total_errors": 0,
            "by_level": {},
            "by_module": {},
            "by_agent": {},
            "recent_errors": []
        }
    
    def emit(self, record: logging.LogRecord) -> None:
        """Process and store error records."""
        if record.levelno >= logging.WARNING:  # Only store warnings and above
            error_record = {
                "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception details if present
            if record.exc_info:
                error_record["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None
                }
            
            # Extract agent information if available
            if hasattr(record, 'agent_id'):
                error_record["agent_id"] = record.agent_id
            if hasattr(record, 'agent_type'):
                error_record["agent_type"] = record.agent_type
            if hasattr(record, 'query_id'):
                error_record["query_id"] = record.query_id
            
            # Add to records list
            self.error_records.append(error_record)
            
            # Maintain max records limit
            if len(self.error_records) > self.max_records:
                self.error_records.pop(0)
            
            # Update statistics
            self._update_stats(error_record)
    
    def _update_stats(self, error_record: Dict[str, Any]) -> None:
        """Update error statistics."""
        self.error_stats["total_errors"] += 1
        
        # Update by level
        level = error_record["level"]
        self.error_stats["by_level"][level] = self.error_stats["by_level"].get(level, 0) + 1
        
        # Update by module
        module = error_record["module"]
        self.error_stats["by_module"][module] = self.error_stats["by_module"].get(module, 0) + 1
        
        # Update by agent if available
        if "agent_id" in error_record:
            agent_id = error_record["agent_id"]
            self.error_stats["by_agent"][agent_id] = self.error_stats["by_agent"].get(agent_id, 0) + 1
        
        # Keep recent errors (last 20)
        self.error_stats["recent_errors"] = self.error_records[-20:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        return {
            "total_records": len(self.error_records),
            "stats": self.error_stats,
            "all_errors": self.error_records
        }
    
    def clear_records(self) -> None:
        """Clear all stored error records."""
        self.error_records.clear()
        self.error_stats = {
            "total_errors": 0,
            "by_level": {},
            "by_module": {},
            "by_agent": {},
            "recent_errors": []
        }


class PerformanceMetricsHandler(logging.Handler):
    """Handler that tracks performance metrics from log messages."""
    
    def __init__(self):
        super().__init__()
        self.metrics = {
            "agent_execution_times": [],
            "operation_times": [],
            "retry_counts": [],
            "timeout_events": 0,
            "circuit_breaker_events": 0
        }
    
    def emit(self, record: logging.LogRecord) -> None:
        """Extract and store performance metrics from log records."""
        message = record.getMessage().lower()
        
        # Track execution times
        if hasattr(record, 'execution_time'):
            if 'agent' in message:
                self.metrics["agent_execution_times"].append(record.execution_time)
            else:
                self.metrics["operation_times"].append(record.execution_time)
        
        # Track retry counts
        if hasattr(record, 'retry_count'):
            self.metrics["retry_counts"].append(record.retry_count)
        
        # Track timeout events
        if 'timeout' in message or 'timed out' in message:
            self.metrics["timeout_events"] += 1
        
        # Track circuit breaker events
        if 'circuit breaker' in message:
            self.metrics["circuit_breaker_events"] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {
            "timeout_events": self.metrics["timeout_events"],
            "circuit_breaker_events": self.metrics["circuit_breaker_events"]
        }
        
        # Calculate execution time statistics
        if self.metrics["agent_execution_times"]:
            times = self.metrics["agent_execution_times"]
            summary["agent_execution"] = {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }
        
        if self.metrics["operation_times"]:
            times = self.metrics["operation_times"]
            summary["operation_execution"] = {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }
        
        # Calculate retry statistics
        if self.metrics["retry_counts"]:
            retries = self.metrics["retry_counts"]
            summary["retries"] = {
                "total_operations": len(retries),
                "operations_with_retries": len([r for r in retries if r > 0]),
                "avg_retries": sum(retries) / len(retries),
                "max_retries": max(retries)
            }
        
        return summary


# Global handlers
_error_audit_handler: Optional[ErrorAuditHandler] = None
_performance_handler: Optional[PerformanceMetricsHandler] = None


def setup_enhanced_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = True,
    enable_audit: bool = True,
    enable_performance: bool = True
) -> None:
    """
    Set up enhanced logging configuration for the multi-agent system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Enable console logging
        enable_structured: Enable structured JSON logging
        enable_audit: Enable error audit trail
        enable_performance: Enable performance metrics tracking
    """
    global _error_audit_handler, _performance_handler
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if enable_structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        if enable_structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            ))
        root_logger.addHandler(file_handler)
    
    # Error audit handler
    if enable_audit:
        _error_audit_handler = ErrorAuditHandler()
        root_logger.addHandler(_error_audit_handler)
    
    # Performance metrics handler
    if enable_performance:
        _performance_handler = PerformanceMetricsHandler()
        root_logger.addHandler(_performance_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info("Enhanced logging configuration completed")


def get_error_audit_summary() -> Dict[str, Any]:
    """Get error audit summary from the global handler."""
    if _error_audit_handler:
        return _error_audit_handler.get_error_summary()
    return {"message": "Error audit handler not initialized"}


def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics from the global handler."""
    if _performance_handler:
        return _performance_handler.get_metrics_summary()
    return {"message": "Performance metrics handler not initialized"}


def clear_audit_records() -> None:
    """Clear all stored audit records."""
    if _error_audit_handler:
        _error_audit_handler.clear_records()


def log_agent_error(
    logger: logging.Logger,
    agent_id: str,
    agent_type: str,
    error: Exception,
    context: Dict[str, Any] = None,
    query_id: Optional[str] = None
) -> None:
    """
    Log an agent error with structured information.
    
    Args:
        logger: Logger instance to use
        agent_id: ID of the agent that encountered the error
        agent_type: Type of the agent
        error: The exception that occurred
        context: Additional context information
        query_id: Query ID if available
    """
    extra_data = {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    
    if query_id:
        extra_data["query_id"] = query_id
    
    if context:
        extra_data["context"] = context
    
    logger.error(
        f"Agent {agent_id} ({agent_type}) encountered error: {str(error)}",
        exc_info=True,
        extra=extra_data
    )


def log_agent_performance(
    logger: logging.Logger,
    agent_id: str,
    agent_type: str,
    operation: str,
    execution_time: float,
    retry_count: int = 0,
    success: bool = True,
    context: Dict[str, Any] = None
) -> None:
    """
    Log agent performance metrics.
    
    Args:
        logger: Logger instance to use
        agent_id: ID of the agent
        agent_type: Type of the agent
        operation: Name of the operation
        execution_time: Time taken to execute
        retry_count: Number of retries performed
        success: Whether the operation was successful
        context: Additional context information
    """
    extra_data = {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "operation": operation,
        "execution_time": execution_time,
        "retry_count": retry_count,
        "success": success
    }
    
    if context:
        extra_data["context"] = context
    
    level = logging.INFO if success else logging.WARNING
    status = "completed" if success else "failed"
    
    logger.log(
        level,
        f"Agent {agent_id} ({agent_type}) {status} {operation} in {execution_time:.2f}s (retries: {retry_count})",
        extra=extra_data
    )