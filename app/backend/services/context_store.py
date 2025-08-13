"""
Shared context store for multi-agent coordination.

This module provides a thread-safe, in-memory context store that allows
agents to share context and coordinate their activities.
"""

import asyncio
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
import threading
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Represents a single context entry with metadata."""
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def mark_accessed(self):
        """Mark this entry as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def update_data(self, new_data: Dict[str, Any]):
        """Update the data and timestamp."""
        self.data.update(new_data)
        self.updated_at = datetime.utcnow()


class SharedContextStore:
    """
    Thread-safe, in-memory context store for agent coordination.
    
    This store manages context data for individual agents and shared context
    across multiple agents working on the same query.
    """
    
    def __init__(self, cleanup_interval_minutes: int = 60, max_age_hours: int = 24):
        """
        Initialize the context store.
        
        Args:
            cleanup_interval_minutes: How often to run cleanup (default: 60 minutes)
            max_age_hours: Maximum age of context entries before cleanup (default: 24 hours)
        """
        self._agent_contexts: Dict[str, ContextEntry] = {}
        self._shared_contexts: Dict[str, ContextEntry] = {}
        self._query_agents: Dict[str, Set[str]] = {}  # query_id -> set of agent_ids
        self._lock = threading.RLock()
        self._cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self._max_age = timedelta(hours=max_age_hours)
        self._last_cleanup = datetime.utcnow()
        
        logger.info(f"SharedContextStore initialized with cleanup_interval={cleanup_interval_minutes}min, max_age={max_age_hours}h")
    
    async def store_context(self, agent_id: str, context: Dict[str, Any]) -> None:
        """
        Store context for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            context: Context data to store
            
        Raises:
            ValueError: If agent_id is empty or context is None
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        if context is None:
            raise ValueError("Context cannot be None")
        
        with self._lock:
            if agent_id in self._agent_contexts:
                # Update existing context
                self._agent_contexts[agent_id].update_data(context)
                logger.debug(f"Updated context for agent {agent_id}")
            else:
                # Create new context entry
                self._agent_contexts[agent_id] = ContextEntry(data=context.copy())
                logger.debug(f"Stored new context for agent {agent_id}")
        
        # Run cleanup if needed
        await self._cleanup_if_needed()
    
    async def get_context(self, agent_id: str) -> Dict[str, Any]:
        """
        Retrieve context for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dict containing the agent's context, empty dict if not found
            
        Raises:
            ValueError: If agent_id is empty
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        with self._lock:
            if agent_id in self._agent_contexts:
                entry = self._agent_contexts[agent_id]
                entry.mark_accessed()
                logger.debug(f"Retrieved context for agent {agent_id} (access count: {entry.access_count})")
                return entry.data.copy()
            else:
                logger.debug(f"No context found for agent {agent_id}")
                return {}
    
    async def get_shared_context(self, query_id: str) -> Dict[str, Any]:
        """
        Retrieve shared context for a query.
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            Dict containing the shared context, empty dict if not found
            
        Raises:
            ValueError: If query_id is empty
        """
        if not query_id or not query_id.strip():
            raise ValueError("Query ID cannot be empty")
        
        with self._lock:
            if query_id in self._shared_contexts:
                entry = self._shared_contexts[query_id]
                entry.mark_accessed()
                logger.debug(f"Retrieved shared context for query {query_id} (access count: {entry.access_count})")
                return entry.data.copy()
            else:
                logger.debug(f"No shared context found for query {query_id}")
                return {}
    
    async def update_shared_context(self, query_id: str, updates: Dict[str, Any]) -> None:
        """
        Update shared context for a query.
        
        Args:
            query_id: Unique identifier for the query
            updates: Updates to apply to the shared context
            
        Raises:
            ValueError: If query_id is empty or updates is None
        """
        if not query_id or not query_id.strip():
            raise ValueError("Query ID cannot be empty")
        if updates is None:
            raise ValueError("Updates cannot be None")
        
        with self._lock:
            if query_id in self._shared_contexts:
                # Update existing shared context
                self._shared_contexts[query_id].update_data(updates)
                logger.debug(f"Updated shared context for query {query_id}")
            else:
                # Create new shared context entry
                self._shared_contexts[query_id] = ContextEntry(data=updates.copy())
                logger.debug(f"Created new shared context for query {query_id}")
        
        # Run cleanup if needed
        await self._cleanup_if_needed()
    
    async def register_agent_for_query(self, query_id: str, agent_id: str) -> None:
        """
        Register an agent as working on a specific query.
        
        This helps track which agents are associated with which queries
        for context sharing and cleanup purposes.
        
        Args:
            query_id: Unique identifier for the query
            agent_id: Unique identifier for the agent
            
        Raises:
            ValueError: If query_id or agent_id is empty
        """
        if not query_id or not query_id.strip():
            raise ValueError("Query ID cannot be empty")
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        with self._lock:
            if query_id not in self._query_agents:
                self._query_agents[query_id] = set()
            self._query_agents[query_id].add(agent_id)
            logger.debug(f"Registered agent {agent_id} for query {query_id}")
    
    async def get_agents_for_query(self, query_id: str) -> Set[str]:
        """
        Get all agents registered for a specific query.
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            Set of agent IDs working on the query
            
        Raises:
            ValueError: If query_id is empty
        """
        if not query_id or not query_id.strip():
            raise ValueError("Query ID cannot be empty")
        
        with self._lock:
            return self._query_agents.get(query_id, set()).copy()
    
    async def remove_context(self, agent_id: str) -> bool:
        """
        Remove context for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if context was removed, False if not found
            
        Raises:
            ValueError: If agent_id is empty
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        
        with self._lock:
            if agent_id in self._agent_contexts:
                del self._agent_contexts[agent_id]
                logger.debug(f"Removed context for agent {agent_id}")
                return True
            else:
                logger.debug(f"No context to remove for agent {agent_id}")
                return False
    
    async def remove_shared_context(self, query_id: str) -> bool:
        """
        Remove shared context for a specific query.
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            True if context was removed, False if not found
            
        Raises:
            ValueError: If query_id is empty
        """
        if not query_id or not query_id.strip():
            raise ValueError("Query ID cannot be empty")
        
        with self._lock:
            removed_shared = query_id in self._shared_contexts
            removed_agents = query_id in self._query_agents
            
            if removed_shared:
                del self._shared_contexts[query_id]
            if removed_agents:
                del self._query_agents[query_id]
            
            if removed_shared or removed_agents:
                logger.debug(f"Removed shared context and agent mappings for query {query_id}")
                return True
            else:
                logger.debug(f"No shared context to remove for query {query_id}")
                return False
    
    async def get_context_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the context store.
        
        Returns:
            Dict containing store statistics
        """
        with self._lock:
            agent_contexts_count = len(self._agent_contexts)
            shared_contexts_count = len(self._shared_contexts)
            query_mappings_count = len(self._query_agents)
            
            # Calculate total access counts
            total_agent_accesses = sum(entry.access_count for entry in self._agent_contexts.values())
            total_shared_accesses = sum(entry.access_count for entry in self._shared_contexts.values())
            
            return {
                "agent_contexts_count": agent_contexts_count,
                "shared_contexts_count": shared_contexts_count,
                "query_mappings_count": query_mappings_count,
                "total_agent_accesses": total_agent_accesses,
                "total_shared_accesses": total_shared_accesses,
                "last_cleanup": self._last_cleanup.isoformat(),
                "cleanup_interval_minutes": self._cleanup_interval.total_seconds() / 60,
                "max_age_hours": self._max_age.total_seconds() / 3600
            }
    
    async def clear_all(self) -> None:
        """
        Clear all contexts from the store.
        
        This is primarily useful for testing and cleanup.
        """
        with self._lock:
            agent_count = len(self._agent_contexts)
            shared_count = len(self._shared_contexts)
            query_count = len(self._query_agents)
            
            self._agent_contexts.clear()
            self._shared_contexts.clear()
            self._query_agents.clear()
            
            logger.info(f"Cleared all contexts: {agent_count} agent contexts, {shared_count} shared contexts, {query_count} query mappings")
    
    async def _cleanup_if_needed(self) -> None:
        """
        Run cleanup if enough time has passed since the last cleanup.
        """
        now = datetime.utcnow()
        if now - self._last_cleanup >= self._cleanup_interval:
            await self._cleanup_old_contexts()
    
    async def _cleanup_old_contexts(self) -> None:
        """
        Remove old context entries that exceed the maximum age.
        """
        now = datetime.utcnow()
        cutoff_time = now - self._max_age
        
        with self._lock:
            # Cleanup agent contexts
            old_agent_contexts = [
                agent_id for agent_id, entry in self._agent_contexts.items()
                if entry.updated_at < cutoff_time
            ]
            for agent_id in old_agent_contexts:
                del self._agent_contexts[agent_id]
            
            # Cleanup shared contexts
            old_shared_contexts = [
                query_id for query_id, entry in self._shared_contexts.items()
                if entry.updated_at < cutoff_time
            ]
            for query_id in old_shared_contexts:
                del self._shared_contexts[query_id]
                # Also remove query agent mappings
                if query_id in self._query_agents:
                    del self._query_agents[query_id]
            
            self._last_cleanup = now
            
            if old_agent_contexts or old_shared_contexts:
                logger.info(f"Cleanup completed: removed {len(old_agent_contexts)} agent contexts, {len(old_shared_contexts)} shared contexts")


# Global instance for the application
_context_store: Optional[SharedContextStore] = None


def get_context_store() -> SharedContextStore:
    """
    Get the global context store instance.
    
    Returns:
        SharedContextStore instance
    """
    global _context_store
    if _context_store is None:
        _context_store = SharedContextStore()
    return _context_store


def initialize_context_store(cleanup_interval_minutes: int = 60, max_age_hours: int = 24) -> SharedContextStore:
    """
    Initialize the global context store with custom settings.
    
    Args:
        cleanup_interval_minutes: How often to run cleanup
        max_age_hours: Maximum age of context entries before cleanup
        
    Returns:
        SharedContextStore instance
    """
    global _context_store
    _context_store = SharedContextStore(cleanup_interval_minutes, max_age_hours)
    return _context_store