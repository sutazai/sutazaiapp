"""
Automated Experiment Tracking for SutazAI Agents
Comprehensive tracking system for all 69+ AI agents
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import inspect
import sys
import os

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import psutil
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException

from .config import mlflow_config, ExperimentConfig
from .metrics import mlflow_metrics
from .database import mlflow_database


logger = logging.getLogger(__name__)


class AgentExperimentTracker:
    """Automated experiment tracking for individual agents"""
    
    def __init__(self, agent_id: str, agent_type: str, agent_config: Dict = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.agent_config = agent_config or {}
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        self.client = MlflowClient()
        
        # Experiment configuration
        self.experiment_config = ExperimentConfig(
            name=f"agent_{agent_id}_{agent_type}",
            description=f"Automated tracking for {agent_type} agent {agent_id}",
            tags={
                "agent_id": agent_id,
                "agent_type": agent_type,
                "system": "sutazai",
                "auto_tracking": "true",
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            agent_id=agent_id,
            agent_type=agent_type
        )
        
        # Tracking state
        self.experiment_id: Optional[str] = None
        self.current_run_id: Optional[str] = None
        self.tracking_enabled = True
        self.batch_metrics = []
        self.batch_params = {}
        self.batch_tags = {}
        
        # Performance monitoring
        self.start_time = time.time()
        self.last_metric_time = time.time()
        self.metrics_logged_count = 0
        self.params_logged_count = 0
        
        # System resource tracking
        self.track_system_resources = mlflow_config.agent_tracking_enabled
        self.resource_logging_interval = 60  # seconds
        self.last_resource_log = time.time()
    
    async def initialize(self):
        """Initialize experiment tracking for the agent"""
        try:
            # Create or get experiment
            self.experiment_id = await self._ensure_experiment()
            
            # Set up auto-logging based on agent type
            await self._setup_auto_logging()
            
            logger.info(f"Initialized tracking for agent {self.agent_id} (experiment: {self.experiment_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracking for agent {self.agent_id}: {e}")
            self.tracking_enabled = False
    
    async def _ensure_experiment(self) -> str:
        """Create or retrieve experiment for this agent"""
        try:
            # Try to get existing experiment
            experiment = self.client.get_experiment_by_name(self.experiment_config.name)
            
            if experiment is None:
                # Create new experiment
                experiment_id = self.client.create_experiment(
                    name=self.experiment_config.name,
                    artifact_location=os.path.join(
                        mlflow_config.artifact_root, 
                        f"agent_{self.agent_id}"
                    ),
                    tags=self.experiment_config.tags
                )
                
                # Update metrics
                mlflow_metrics.experiments_created.labels(
                    agent_type=self.agent_type,
                    experiment_type="agent_tracking"
                ).inc()
                
                logger.info(f"Created experiment {experiment_id} for agent {self.agent_id}")
                return experiment_id
            else:
                return experiment.experiment_id
                
        except Exception as e:
            logger.error(f"Failed to create experiment for agent {self.agent_id}: {e}")
            raise
    
    async def _setup_auto_logging(self):
        """Set up automatic logging based on agent type and framework"""
        try:
            # Enable MLflow autologging based on detected frameworks
            if "pytorch" in self.agent_config.get("framework", "").lower():
                mlflow.pytorch.autolog(
                    log_models=mlflow_config.auto_log_models,
                    log_every_n_epoch=10,
                    log_every_n_step=None,
                    disable=False
                )
            
            if "tensorflow" in self.agent_config.get("framework", "").lower():
                mlflow.tensorflow.autolog(
                    log_models=mlflow_config.auto_log_models,
                    log_every_n_epoch=10,
                    disable=False
                )
            
            if "sklearn" in self.agent_config.get("framework", "").lower():
                mlflow.sklearn.autolog(
                    log_models=mlflow_config.auto_log_models,
                    log_input_examples=True,
                    log_model_signatures=True,
                    disable=False
                )
            
            # Set up system metrics tracking
            if self.track_system_resources:
                asyncio.create_task(self._track_system_resources())
            
        except Exception as e:
            logger.warning(f"Auto-logging setup failed for agent {self.agent_id}: {e}")
    
    async def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run for this agent"""
        if not self.tracking_enabled:
            return "tracking_disabled"
        
        try:
            # Generate run name if not provided
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{self.agent_type}_{timestamp}"
            
            # Combine tags
            run_tags = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "run_name": run_name,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "auto_tracked": "true"
            }
            
            if tags:
                run_tags.update(tags)
            
            # Start MLflow run
            run = self.client.create_run(
                experiment_id=self.experiment_id,
                tags=run_tags,
                run_name=run_name
            )
            
            self.current_run_id = run.info.run_id
            
            # Log initial parameters
            await self._log_initial_params()
            
            # Update metrics
            mlflow_metrics.runs_started.labels(
                experiment_name=self.experiment_config.name,
                agent_id=self.agent_id
            ).inc()
            
            logger.info(f"Started run {self.current_run_id} for agent {self.agent_id}")
            return self.current_run_id
            
        except Exception as e:
            logger.error(f"Failed to start run for agent {self.agent_id}: {e}")
            return "error"
    
    async def _log_initial_params(self):
        """Log initial parameters for the run"""
        try:
            initial_params = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "tracking_system": "sutazai_mlflow",
                "python_version": sys.version.split()[0],
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            # Add agent-specific config
            for key, value in self.agent_config.items():
                if isinstance(value, (str, int, float, bool)):
                    initial_params[f"config_{key}"] = str(value)
            
            await self.log_params(initial_params)
            
        except Exception as e:
            logger.warning(f"Failed to log initial params for agent {self.agent_id}: {e}")
    
    @mlflow_metrics.track_logging_operation("log_metrics")
    async def log_metric(self, key: str, value: float, step: int = None, timestamp: int = None):
        """Log a single metric"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Use current timestamp if not provided
            if timestamp is None:
                timestamp = int(time.time() * 1000)
            
            # Log to MLflow
            self.client.log_metric(
                run_id=self.current_run_id,
                key=key,
                value=float(value),
                timestamp=timestamp,
                step=step
            )
            
            self.metrics_logged_count += 1
            self.last_metric_time = time.time()
            
            # Update Prometheus metrics
            mlflow_metrics.metrics_logged.labels(
                experiment_name=self.experiment_config.name,
                metric_name=key
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to log metric {key} for agent {self.agent_id}: {e}")
    
    @mlflow_metrics.track_logging_operation("log_metrics_batch")
    async def log_metrics(self, metrics: Dict[str, float], step: int = None, timestamp: int = None):
        """Log multiple metrics efficiently"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Use current timestamp if not provided
            if timestamp is None:
                timestamp = int(time.time() * 1000)
            
            # Batch log metrics
            for key, value in metrics.items():
                self.client.log_metric(
                    run_id=self.current_run_id,
                    key=key,
                    value=float(value),
                    timestamp=timestamp,
                    step=step
                )
            
            self.metrics_logged_count += len(metrics)
            self.last_metric_time = time.time()
            
            # Update Prometheus metrics
            for key in metrics.keys():
                mlflow_metrics.metrics_logged.labels(
                    experiment_name=self.experiment_config.name,
                    metric_name=key
                ).inc()
            
        except Exception as e:
            logger.error(f"Failed to log metrics batch for agent {self.agent_id}: {e}")
    
    @mlflow_metrics.track_logging_operation("log_params")
    async def log_param(self, key: str, value: Any):
        """Log a single parameter"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Convert value to string if needed
            param_value = str(value) if not isinstance(value, str) else value
            
            # Truncate if too long
            if len(param_value) > 500:
                param_value = param_value[:497] + "..."
            
            self.client.log_param(
                run_id=self.current_run_id,
                key=key,
                value=param_value
            )
            
            self.params_logged_count += 1
            
            # Update Prometheus metrics
            mlflow_metrics.params_logged.labels(
                experiment_name=self.experiment_config.name
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to log param {key} for agent {self.agent_id}: {e}")
    
    @mlflow_metrics.track_logging_operation("log_params_batch")
    async def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters efficiently"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Convert and truncate parameters
            processed_params = {}
            for key, value in params.items():
                param_value = str(value) if not isinstance(value, str) else value
                if len(param_value) > 500:
                    param_value = param_value[:497] + "..."
                processed_params[key] = param_value
            
            # Batch log parameters
            for key, value in processed_params.items():
                self.client.log_param(
                    run_id=self.current_run_id,
                    key=key,
                    value=value
                )
            
            self.params_logged_count += len(processed_params)
            
            # Update Prometheus metrics
            mlflow_metrics.params_logged.labels(
                experiment_name=self.experiment_config.name
            ).inc(len(processed_params))
            
        except Exception as e:
            logger.error(f"Failed to log params batch for agent {self.agent_id}: {e}")
    
    async def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact file"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Ensure file exists
            if not os.path.exists(local_path):
                logger.warning(f"Artifact file not found: {local_path}")
                return
            
            # Get file size for metrics
            file_size = os.path.getsize(local_path)
            
            # Log artifact
            self.client.log_artifact(
                run_id=self.current_run_id,
                local_path=local_path,
                artifact_path=artifact_path
            )
            
            # Determine artifact type
            artifact_type = Path(local_path).suffix.lower() or "unknown"
            
            # Update Prometheus metrics
            mlflow_metrics.artifacts_logged.labels(
                experiment_name=self.experiment_config.name,
                artifact_type=artifact_type
            ).inc()
            
            mlflow_metrics.artifacts_size_bytes.labels(
                experiment_name=self.experiment_config.name,
                artifact_type=artifact_type
            ).observe(file_size)
            
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path} for agent {self.agent_id}: {e}")
    
    async def log_model(self, model: Any, artifact_path: str = "model", 
                       model_type: str = None, signature=None, input_example=None):
        """Log a trained model"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Determine model type if not provided
            if model_type is None:
                model_type = type(model).__name__.lower()
            
            # Log model based on type
            if "sklearn" in model_type or hasattr(model, 'fit'):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            elif "pytorch" in model_type or hasattr(model, 'state_dict'):
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            elif "tensorflow" in model_type or hasattr(model, 'save'):
                mlflow.tensorflow.log_model(
                    tf_saved_model_dir=model,
                    tf_meta_graph_tags=None,
                    tf_signature_def_key=None,
                    artifact_path=artifact_path
                )
            else:
                # Generic model logging
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model
                )
            
            # Update metrics
            mlflow_metrics.models_registered.labels(
                model_name=f"{self.agent_type}_model",
                agent_type=self.agent_type
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to log model for agent {self.agent_id}: {e}")
    
    async def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run"""
        if not self.tracking_enabled or not self.current_run_id:
            return
        
        try:
            # Log final metrics
            run_duration = time.time() - self.start_time
            await self.log_metric("run_duration_seconds", run_duration)
            await self.log_metric("total_metrics_logged", self.metrics_logged_count)
            await self.log_metric("total_params_logged", self.params_logged_count)
            
            # End the run
            run_status = getattr(RunStatus, status, RunStatus.FINISHED)
            self.client.set_terminated(
                run_id=self.current_run_id,
                status=run_status
            )
            
            # Update metrics
            mlflow_metrics.runs_completed.labels(
                experiment_name=self.experiment_config.name,
                agent_id=self.agent_id,
                status=status
            ).inc()
            
            mlflow_metrics.run_duration.labels(
                experiment_name=self.experiment_config.name,
                agent_id=self.agent_id
            ).observe(run_duration)
            
            logger.info(f"Ended run {self.current_run_id} for agent {self.agent_id} with status {status}")
            
            # Reset tracking state
            self.current_run_id = None
            self.start_time = time.time()
            self.metrics_logged_count = 0
            self.params_logged_count = 0
            
        except Exception as e:
            logger.error(f"Failed to end run for agent {self.agent_id}: {e}")
    
    async def _track_system_resources(self):
        """Track system resource usage periodically"""
        while self.tracking_enabled and self.current_run_id:
            try:
                current_time = time.time()
                
                # Only log if enough time has passed
                if current_time - self.last_resource_log >= self.resource_logging_interval:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Log system metrics
                    await self.log_metrics({
                        "system_cpu_percent": cpu_percent,
                        "system_memory_percent": memory.percent,
                        "system_memory_used_gb": memory.used / (1024**3),
                        "system_memory_available_gb": memory.available / (1024**3)
                    })
                    
                    self.last_resource_log = current_time
                
                # Sleep before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"System resource tracking error for agent {self.agent_id}: {e}")
                await asyncio.sleep(60)
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get current tracking statistics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "experiment_id": self.experiment_id,
            "current_run_id": self.current_run_id,
            "tracking_enabled": self.tracking_enabled,
            "metrics_logged": self.metrics_logged_count,
            "params_logged": self.params_logged_count,
            "uptime_seconds": time.time() - self.start_time,
            "last_metric_time": self.last_metric_time
        }


class SutazAIAgentTrackingManager:
    """Central manager for tracking all SutazAI agents"""
    
    def __init__(self):
        self.agent_trackers: Dict[str, AgentExperimentTracker] = {}
        self.tracking_enabled = True
        self.client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
        
        # Load agent configurations
        self.agent_configs = self._load_agent_configurations()
    
    def _load_agent_configurations(self) -> Dict[str, Dict]:
        """Load configurations for all agents"""
        configs = {}
        
        try:
            # Try to load from agent registry
            registry_path = "/opt/sutazaiapp/backend/agents/agent_registry.json"
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                    
                for agent_id, agent_info in registry.items():
                    configs[agent_id] = {
                        "agent_type": agent_info.get("type", "unknown"),
                        "framework": agent_info.get("framework", "pytorch"),
                        "description": agent_info.get("description", ""),
                        "version": agent_info.get("version", "1.0.0")
                    }
            else:
                logger.warning("Agent registry not found, using default configurations")
                
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
        
        return configs
    
    async def initialize_agent_tracking(self, agent_id: str, agent_type: str = None, 
                                      agent_config: Dict = None) -> AgentExperimentTracker:
        """Initialize tracking for a specific agent"""
        try:
            # Get agent configuration
            if agent_config is None:
                agent_config = self.agent_configs.get(agent_id, {})
            
            if agent_type is None:
                agent_type = agent_config.get("agent_type", "unknown")
            
            # Create tracker
            tracker = AgentExperimentTracker(
                agent_id=agent_id,
                agent_type=agent_type,
                agent_config=agent_config
            )
            
            # Initialize tracker
            await tracker.initialize()
            
            # Store tracker
            self.agent_trackers[agent_id] = tracker
            
            # Update metrics
            mlflow_metrics.agent_experiments.labels(
                agent_id=agent_id,
                agent_type=agent_type
            ).set(1)
            
            logger.info(f"Initialized tracking for agent {agent_id}")
            return tracker
            
        except Exception as e:
            logger.error(f"Failed to initialize tracking for agent {agent_id}: {e}")
            raise
    
    async def initialize_all_agents(self):
        """Initialize tracking for all configured agents"""
        logger.info("Initializing tracking for all agents...")
        
        try:
            # Initialize from registry
            for agent_id, config in self.agent_configs.items():
                try:
                    await self.initialize_agent_tracking(agent_id, config.get("agent_type"), config)
                except Exception as e:
                    logger.error(f"Failed to initialize agent {agent_id}: {e}")
            
            # Also check for running agents
            await self._discover_running_agents()
            
            logger.info(f"Initialized tracking for {len(self.agent_trackers)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize all agents: {e}")
    
    async def _discover_running_agents(self):
        """Discover and initialize tracking for currently running agents"""
        try:
            # This would integrate with the SutazAI agent discovery system
            # For now, we'll implement a basic discovery mechanism
            
            agents_dir = Path("/opt/sutazaiapp/backend/agents")
            if agents_dir.exists():
                for agent_dir in agents_dir.iterdir():
                    if agent_dir.is_dir() and agent_dir.name not in ["__pycache__", "configs"]:
                        agent_id = agent_dir.name
                        
                        if agent_id not in self.agent_trackers:
                            # Try to determine agent type from directory name
                            agent_type = agent_id.replace("-", "_").replace("_", " ").title()
                            
                            config = {
                                "agent_type": agent_type,
                                "framework": "pytorch",  # default
                                "discovered": True
                            }
                            
                            await self.initialize_agent_tracking(agent_id, agent_type, config)
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
    
    def get_agent_tracker(self, agent_id: str) -> Optional[AgentExperimentTracker]:
        """Get tracker for specific agent"""
        return self.agent_trackers.get(agent_id)
    
    def get_all_tracking_stats(self) -> Dict[str, Any]:
        """Get tracking statistics for all agents"""
        stats = {
            "total_agents": len(self.agent_trackers),
            "tracking_enabled": self.tracking_enabled,
            "agents": {}
        }
        
        for agent_id, tracker in self.agent_trackers.items():
            stats["agents"][agent_id] = tracker.get_tracking_stats()
        
        return stats
    
    async def shutdown_all_tracking(self):
        """Shutdown tracking for all agents"""
        logger.info("Shutting down all agent tracking...")
        
        for agent_id, tracker in self.agent_trackers.items():
            try:
                if tracker.current_run_id:
                    await tracker.end_run(status="KILLED")
            except Exception as e:
                logger.error(f"Failed to shutdown tracking for agent {agent_id}: {e}")
        
        self.agent_trackers.clear()
        logger.info("All agent tracking shutdown complete")


# Global tracking manager
agent_tracking_manager = SutazAIAgentTrackingManager()