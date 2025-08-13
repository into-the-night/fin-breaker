"""
Configuration management system for the multi-agent orchestrator.

This module provides environment-based configuration loading, default configurations
for different deployment scenarios, and comprehensive validation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from app.backend.models.models import AgentConfig, OrchestratorConfig
from app.backend.models.enums import AgentType


logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration settings."""
    name: str
    orchestrator_config: OrchestratorConfig
    default_agent_config: AgentConfig
    agent_specific_configs: Dict[AgentType, AgentConfig] = field(default_factory=dict)
    
    def get_agent_config(self, agent_type: AgentType) -> AgentConfig:
        """Get configuration for a specific agent type."""
        return self.agent_specific_configs.get(agent_type, self.default_agent_config)


class ConfigurationManager:
    """
    Manages configuration loading and validation for the multi-agent system.
    
    Supports:
    - Environment-based configuration loading
    - Default configurations for different deployment scenarios
    - Configuration validation and error handling
    - Runtime configuration updates
    """
    
    def __init__(self, environment: Optional[Union[str, DeploymentEnvironment]] = None):
        """
        Initialize configuration manager.
        
        Args:
            environment: Deployment environment (defaults to DEVELOPMENT)
        """
        if isinstance(environment, str):
            try:
                self.environment = DeploymentEnvironment(environment.lower())
            except ValueError:
                logger.warning(f"Unknown environment '{environment}', defaulting to DEVELOPMENT")
                self.environment = DeploymentEnvironment.DEVELOPMENT
        else:
            self.environment = environment or DeploymentEnvironment.DEVELOPMENT
        
        self._config_cache: Dict[str, EnvironmentConfig] = {}
        self._load_environment_configs()
    
    def _load_environment_configs(self) -> None:
        """Load all environment configurations."""
        try:
            # Load from environment variables first
            env_config = self._load_from_environment()
            if env_config:
                self._config_cache["environment"] = env_config
            
            # Load from config files
            config_file = self._get_config_file_path()
            if config_file and config_file.exists():
                file_config = self._load_from_file(config_file)
                if file_config:
                    self._config_cache["file"] = file_config
            
            # Load default configurations
            self._load_default_configurations()
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _get_config_file_path(self) -> Optional[Path]:
        """Get configuration file path based on environment."""
        config_dir = Path("config")
        if not config_dir.exists():
            return None
        
        config_file = config_dir / f"{self.environment.value}.json"
        if config_file.exists():
            return config_file
        
        # Fallback to default config
        default_config = config_dir / "default.json"
        return default_config if default_config.exists() else None
    
    def _load_from_environment(self) -> Optional[EnvironmentConfig]:
        """Load configuration from environment variables."""
        try:
            # Only load from environment if at least one environment variable is set
            env_vars = [
                "ORCHESTRATOR_MAX_CONCURRENT_AGENTS", "ORCHESTRATOR_SYNTHESIS_MODEL",
                "ORCHESTRATOR_CONTEXT_SHARING", "ORCHESTRATOR_MAX_CYCLES", "ORCHESTRATOR_CYCLE_TIMEOUT",
                "AGENT_MAX_RETRIES", "AGENT_TIMEOUT_SECONDS", "AGENT_GOAL_EVALUATION_INTERVAL"
            ]
            
            # Check for agent-specific environment variables
            for agent_type in AgentType:
                prefix = f"AGENT_{agent_type.value.upper()}_"
                env_vars.extend([f"{prefix}MAX_RETRIES", f"{prefix}TIMEOUT_SECONDS", f"{prefix}GOAL_EVALUATION_INTERVAL"])
            
            # If no environment variables are set, return None
            if not any(os.getenv(var) for var in env_vars):
                return None
            
            # Orchestrator configuration from environment
            orchestrator_config = OrchestratorConfig(
                max_concurrent_agents=int(os.getenv("ORCHESTRATOR_MAX_CONCURRENT_AGENTS", "5")),
                synthesis_model=os.getenv("ORCHESTRATOR_SYNTHESIS_MODEL", "gemini-2.5-pro-latest"),
                context_sharing_enabled=os.getenv("ORCHESTRATOR_CONTEXT_SHARING", "true").lower() == "true",
                max_cycles=int(os.getenv("ORCHESTRATOR_MAX_CYCLES", "3")),
                cycle_timeout_seconds=int(os.getenv("ORCHESTRATOR_CYCLE_TIMEOUT", "600"))
            )
            
            # Default agent configuration from environment
            default_agent_config = AgentConfig(
                max_retries=int(os.getenv("AGENT_MAX_RETRIES", "3")),
                timeout_seconds=int(os.getenv("AGENT_TIMEOUT_SECONDS", "300")),
                goal_evaluation_interval=int(os.getenv("AGENT_GOAL_EVALUATION_INTERVAL", "10")),
                tools=os.getenv("AGENT_DEFAULT_TOOLS", "").split(",") if os.getenv("AGENT_DEFAULT_TOOLS") else []
            )
            
            # Agent-specific configurations from environment
            agent_configs = {}
            for agent_type in AgentType:
                prefix = f"AGENT_{agent_type.value.upper()}_"
                if any(key.startswith(prefix) for key in os.environ):
                    agent_configs[agent_type] = AgentConfig(
                        max_retries=int(os.getenv(f"{prefix}MAX_RETRIES", str(default_agent_config.max_retries))),
                        timeout_seconds=int(os.getenv(f"{prefix}TIMEOUT_SECONDS", str(default_agent_config.timeout_seconds))),
                        goal_evaluation_interval=int(os.getenv(f"{prefix}GOAL_EVALUATION_INTERVAL", str(default_agent_config.goal_evaluation_interval))),
                        tools=os.getenv(f"{prefix}TOOLS", "").split(",") if os.getenv(f"{prefix}TOOLS") else default_agent_config.tools
                    )
            
            return EnvironmentConfig(
                name="environment",
                orchestrator_config=orchestrator_config,
                default_agent_config=default_agent_config,
                agent_specific_configs=agent_configs
            )
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to load configuration from environment: {e}")
            return None
    
    def _load_from_file(self, config_file: Path) -> Optional[EnvironmentConfig]:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Parse orchestrator configuration
            orchestrator_data = config_data.get("orchestrator", {})
            orchestrator_config = OrchestratorConfig(
                max_concurrent_agents=orchestrator_data.get("max_concurrent_agents", 5),
                synthesis_model=orchestrator_data.get("synthesis_model", "gemini-2.5-pro-latest"),
                context_sharing_enabled=orchestrator_data.get("context_sharing_enabled", True),
                max_cycles=orchestrator_data.get("max_cycles", 3),
                cycle_timeout_seconds=orchestrator_data.get("cycle_timeout_seconds", 600)
            )
            
            # Parse default agent configuration
            agent_data = config_data.get("default_agent", {})
            default_agent_config = AgentConfig(
                max_retries=agent_data.get("max_retries", 3),
                timeout_seconds=agent_data.get("timeout_seconds", 300),
                goal_evaluation_interval=agent_data.get("goal_evaluation_interval", 10),
                tools=agent_data.get("tools", [])
            )
            
            # Parse agent-specific configurations
            agent_configs = {}
            agents_data = config_data.get("agents", {})
            for agent_type_str, agent_config_data in agents_data.items():
                try:
                    agent_type = AgentType(agent_type_str.lower())
                    agent_configs[agent_type] = AgentConfig(
                        max_retries=agent_config_data.get("max_retries", default_agent_config.max_retries),
                        timeout_seconds=agent_config_data.get("timeout_seconds", default_agent_config.timeout_seconds),
                        goal_evaluation_interval=agent_config_data.get("goal_evaluation_interval", default_agent_config.goal_evaluation_interval),
                        tools=agent_config_data.get("tools", default_agent_config.tools)
                    )
                except ValueError:
                    logger.warning(f"Unknown agent type in config: {agent_type_str}")
            
            return EnvironmentConfig(
                name=f"file_{config_file.stem}",
                orchestrator_config=orchestrator_config,
                default_agent_config=default_agent_config,
                agent_specific_configs=agent_configs
            )
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to load configuration from file {config_file}: {e}")
            return None
    
    def _load_default_configurations(self) -> None:
        """Load default configurations for different deployment scenarios."""
        
        # Development configuration
        dev_config = EnvironmentConfig(
            name="development",
            orchestrator_config=OrchestratorConfig(
                max_concurrent_agents=3,
                synthesis_model="gemini-2.5-pro-latest",
                context_sharing_enabled=True,
                max_cycles=2,
                cycle_timeout_seconds=300
            ),
            default_agent_config=AgentConfig(
                max_retries=2,
                timeout_seconds=180,
                goal_evaluation_interval=5,
                tools=[]
            ),
            agent_specific_configs={
                AgentType.MARKET_DATA: AgentConfig(max_retries=3, timeout_seconds=120),
                AgentType.COMPANY_RESEARCH: AgentConfig(max_retries=2, timeout_seconds=200),
                AgentType.TOPIC_ANALYSIS: AgentConfig(max_retries=2, timeout_seconds=150),
                AgentType.RISK_ANALYSIS: AgentConfig(max_retries=2, timeout_seconds=180)
            }
        )
        
        # Testing configuration
        test_config = EnvironmentConfig(
            name="testing",
            orchestrator_config=OrchestratorConfig(
                max_concurrent_agents=2,
                synthesis_model="gemini-2.5-pro-latest",
                context_sharing_enabled=True,
                max_cycles=1,
                cycle_timeout_seconds=60
            ),
            default_agent_config=AgentConfig(
                max_retries=1,
                timeout_seconds=30,
                goal_evaluation_interval=2,
                tools=[]
            ),
            agent_specific_configs={
                AgentType.MARKET_DATA: AgentConfig(max_retries=1, timeout_seconds=20),
                AgentType.COMPANY_RESEARCH: AgentConfig(max_retries=1, timeout_seconds=25),
                AgentType.TOPIC_ANALYSIS: AgentConfig(max_retries=1, timeout_seconds=20),
                AgentType.RISK_ANALYSIS: AgentConfig(max_retries=1, timeout_seconds=25)
            }
        )
        
        # Staging configuration
        staging_config = EnvironmentConfig(
            name="staging",
            orchestrator_config=OrchestratorConfig(
                max_concurrent_agents=4,
                synthesis_model="gemini-2.5-pro-latest",
                context_sharing_enabled=True,
                max_cycles=3,
                cycle_timeout_seconds=450
            ),
            default_agent_config=AgentConfig(
                max_retries=3,
                timeout_seconds=240,
                goal_evaluation_interval=8,
                tools=[]
            ),
            agent_specific_configs={
                AgentType.MARKET_DATA: AgentConfig(max_retries=3, timeout_seconds=180),
                AgentType.COMPANY_RESEARCH: AgentConfig(max_retries=2, timeout_seconds=300),
                AgentType.TOPIC_ANALYSIS: AgentConfig(max_retries=2, timeout_seconds=240),
                AgentType.RISK_ANALYSIS: AgentConfig(max_retries=3, timeout_seconds=200)
            }
        )
        
        # Production configuration
        prod_config = EnvironmentConfig(
            name="production",
            orchestrator_config=OrchestratorConfig(
                max_concurrent_agents=5,
                synthesis_model="gemini-2.5-pro-latest",
                context_sharing_enabled=True,
                max_cycles=3,
                cycle_timeout_seconds=600
            ),
            default_agent_config=AgentConfig(
                max_retries=3,
                timeout_seconds=300,
                goal_evaluation_interval=10,
                tools=[]
            ),
            agent_specific_configs={
                AgentType.MARKET_DATA: AgentConfig(max_retries=3, timeout_seconds=300),
                AgentType.COMPANY_RESEARCH: AgentConfig(max_retries=2, timeout_seconds=400),
                AgentType.TOPIC_ANALYSIS: AgentConfig(max_retries=2, timeout_seconds=350),
                AgentType.RISK_ANALYSIS: AgentConfig(max_retries=2, timeout_seconds=300)
            }
        )
        
        # Store default configurations
        self._config_cache["development"] = dev_config
        self._config_cache["testing"] = test_config
        self._config_cache["staging"] = staging_config
        self._config_cache["production"] = prod_config
    
    def get_config(self, config_name: Optional[str] = None) -> EnvironmentConfig:
        """
        Get configuration by name or current environment.
        
        Args:
            config_name: Specific configuration name, defaults to current environment
            
        Returns:
            EnvironmentConfig for the specified configuration
            
        Raises:
            ConfigurationError: If configuration is not found or invalid
        """
        if config_name is None:
            config_name = self.environment.value
        
        # Priority order for current environment: environment variables > file > defaults
        if config_name == self.environment.value:
            # Try environment variables first
            if "environment" in self._config_cache:
                config = self._config_cache["environment"]
                try:
                    self._validate_config(config)
                    logger.debug(f"Using environment configuration")
                    return config
                except Exception as e:
                    logger.warning(f"Environment configuration validation failed: {e}")
            
            # Try file configuration next
            if "file" in self._config_cache:
                config = self._config_cache["file"]
                try:
                    self._validate_config(config)
                    logger.debug(f"Using file configuration")
                    return config
                except Exception as e:
                    logger.warning(f"File configuration validation failed: {e}")
            
            # Fall back to default configuration
            if config_name in self._config_cache:
                config = self._config_cache[config_name]
                try:
                    self._validate_config(config)
                    logger.debug(f"Using default configuration: {config_name}")
                    return config
                except Exception as e:
                    logger.warning(f"Default configuration '{config_name}' validation failed: {e}")
        
        # If specific config requested, try it directly
        if config_name in self._config_cache:
            config = self._config_cache[config_name]
            try:
                self._validate_config(config)
                return config
            except Exception as e:
                logger.warning(f"Configuration '{config_name}' validation failed: {e}")
        
        raise ConfigurationError(f"No valid configuration found for '{config_name}'")
    
    def get_orchestrator_config(self, config_name: Optional[str] = None) -> OrchestratorConfig:
        """Get orchestrator configuration."""
        return self.get_config(config_name).orchestrator_config
    
    def get_agent_config(self, agent_type: AgentType, config_name: Optional[str] = None) -> AgentConfig:
        """Get agent configuration for specific agent type."""
        return self.get_config(config_name).get_agent_config(agent_type)
    
    def get_default_agent_config(self, config_name: Optional[str] = None) -> AgentConfig:
        """Get default agent configuration."""
        return self.get_config(config_name).default_agent_config
    
    def _validate_config(self, config: EnvironmentConfig) -> None:
        """
        Validate configuration for consistency and correctness.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Validate orchestrator config (validation happens in __post_init__)
            config.orchestrator_config.__post_init__()
            
            # Validate default agent config
            config.default_agent_config.__post_init__()
            
            # Validate agent-specific configs
            for agent_type, agent_config in config.agent_specific_configs.items():
                agent_config.__post_init__()
            
            # Additional business logic validation
            if config.orchestrator_config.max_concurrent_agents > 10:
                logger.warning("High concurrent agent count may impact performance")
            
            if config.default_agent_config.timeout_seconds > config.orchestrator_config.cycle_timeout_seconds:
                raise ConfigurationError(
                    "Agent timeout cannot exceed orchestrator cycle timeout"
                )
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def update_config(self, config_name: str, config: EnvironmentConfig) -> None:
        """
        Update configuration at runtime.
        
        Args:
            config_name: Name of the configuration to update
            config: New configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self._validate_config(config)
            self._config_cache[config_name] = config
            logger.info(f"Configuration '{config_name}' updated successfully")
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration '{config_name}': {e}")
    
    def list_available_configs(self) -> list[str]:
        """List all available configuration names."""
        return list(self._config_cache.keys())
    
    def export_config(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Export configuration as dictionary for serialization.
        
        Args:
            config_name: Configuration name to export, defaults to current environment
            
        Returns:
            Dictionary representation of the configuration
        """
        config = self.get_config(config_name)
        
        return {
            "name": config.name,
            "orchestrator": config.orchestrator_config.to_dict(),
            "default_agent": config.default_agent_config.to_dict(),
            "agents": {
                agent_type.value: agent_config.to_dict()
                for agent_type, agent_config in config.agent_specific_configs.items()
            }
        }


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(environment: Optional[Union[str, DeploymentEnvironment]] = None) -> ConfigurationManager:
    """
    Get or create global configuration manager instance.
    
    Args:
        environment: Deployment environment (only used on first call)
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        env = environment or os.getenv("DEPLOYMENT_ENVIRONMENT", "development")
        _config_manager = ConfigurationManager(env)
    return _config_manager


def reset_config_manager() -> None:
    """Reset global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None