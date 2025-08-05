"""
MLflow Configuration for SutazAI System
High-volume experiment tracking for 69+ AI agents
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class MLflowConfig:
    """Central configuration for MLflow tracking system"""
    
    # Tracking Server Configuration
    tracking_uri: str = "postgresql://mlflow:mlflow_secure_pwd@localhost:5432/mlflow_db"
    tracking_server_host: str = "0.0.0.0"
    tracking_server_port: int = 5000
    
    # Backend Store Configuration
    backend_store_uri: str = "postgresql://mlflow:mlflow_secure_pwd@localhost:5432/mlflow_db"
    
    # Artifact Store Configuration
    artifact_root: str = "/opt/sutazaiapp/backend/mlflow_artifacts"
    s3_artifact_root: Optional[str] = None
    
    # Performance Configuration
    max_concurrent_experiments: int = 50
    batch_logging_size: int = 100
    batch_logging_timeout: int = 30
    
    # High-Volume Settings
    enable_async_logging: bool = True
    enable_compression: bool = True
    artifact_compression_level: int = 6
    
    # Registry Configuration
    model_registry_uri: Optional[str] = None
    
    # Security Configuration
    enable_auth: bool = True
    auth_config_path: str = "/opt/sutazaiapp/backend/mlflow_system/auth.yaml"
    
    # Agent Integration
    agent_tracking_enabled: bool = True
    auto_log_models: bool = True
    auto_log_params: bool = True
    auto_log_metrics: bool = True
    
    # Database Settings
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    
    # Cleanup Settings
    artifact_retention_days: int = 90
    experiment_retention_days: int = 365
    enable_auto_cleanup: bool = True
    
    # Monitoring
    enable_prometheus_metrics: bool = True
    metrics_port: int = 8080
    
    def __post_init__(self):
        """Initialize paths and validate configuration"""
        # Ensure artifact directory exists
        Path(self.artifact_root).mkdir(parents=True, exist_ok=True)
        
        # Set model registry URI if not provided
        if not self.model_registry_uri:
            self.model_registry_uri = self.backend_store_uri


@dataclass  
class ExperimentConfig:
    """Configuration for individual experiments"""
    
    name: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Agent Configuration
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_version: Optional[str] = None
    
    # Model Configuration
    model_type: str = "neural_network"
    framework: str = "pytorch"
    
    # Training Configuration
    auto_log_frequency: int = 10  # Log every N iterations
    log_model_signature: bool = True
    log_input_example: bool = True
    
    # Resource Tracking
    track_system_metrics: bool = True
    track_gpu_metrics: bool = False  # CPU-only by default
    track_memory_usage: bool = True
    
    # Artifacts
    log_artifacts: bool = True
    artifact_compression: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow"""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags
        }


class MLflowConfigManager:
    """Manages MLflow configuration across the system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "/opt/sutazaiapp/backend/mlflow_system/mlflow_config.yaml"
        self._config: Optional[MLflowConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._config = MLflowConfig(**config_data)
        else:
            self._config = MLflowConfig()
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        config_dict = {
            'tracking_uri': self._config.tracking_uri,
            'tracking_server_host': self._config.tracking_server_host,
            'tracking_server_port': self._config.tracking_server_port,
            'backend_store_uri': self._config.backend_store_uri,
            'artifact_root': self._config.artifact_root,
            's3_artifact_root': self._config.s3_artifact_root,
            'max_concurrent_experiments': self._config.max_concurrent_experiments,
            'batch_logging_size': self._config.batch_logging_size,
            'batch_logging_timeout': self._config.batch_logging_timeout,
            'enable_async_logging': self._config.enable_async_logging,
            'enable_compression': self._config.enable_compression,
            'artifact_compression_level': self._config.artifact_compression_level,
            'model_registry_uri': self._config.model_registry_uri,
            'enable_auth': self._config.enable_auth,
            'auth_config_path': self._config.auth_config_path,
            'agent_tracking_enabled': self._config.agent_tracking_enabled,
            'auto_log_models': self._config.auto_log_models,
            'auto_log_params': self._config.auto_log_params,
            'auto_log_metrics': self._config.auto_log_metrics,
            'db_pool_size': self._config.db_pool_size,
            'db_max_overflow': self._config.db_max_overflow,
            'db_pool_timeout': self._config.db_pool_timeout,
            'db_pool_recycle': self._config.db_pool_recycle,
            'artifact_retention_days': self._config.artifact_retention_days,
            'experiment_retention_days': self._config.experiment_retention_days,
            'enable_auto_cleanup': self._config.enable_auto_cleanup,
            'enable_prometheus_metrics': self._config.enable_prometheus_metrics,
            'metrics_port': self._config.metrics_port
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @property
    def config(self) -> MLflowConfig:
        """Get current configuration"""
        return self._config
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        self.save_config()
    
    def get_agent_experiment_config(self, agent_id: str, agent_type: str) -> ExperimentConfig:
        """Create experiment configuration for specific agent"""
        return ExperimentConfig(
            name=f"agent_{agent_id}_{agent_type}",
            description=f"Experiment tracking for {agent_type} agent {agent_id}",
            tags={
                "agent_id": agent_id,
                "agent_type": agent_type,
                "system": "sutazai",
                "environment": "production"
            },
            agent_id=agent_id,
            agent_type=agent_type
        )


# Global configuration instance
config_manager = MLflowConfigManager()
mlflow_config = config_manager.config