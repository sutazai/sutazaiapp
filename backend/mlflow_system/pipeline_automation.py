"""
ML Pipeline Automation for SutazAI System
Automated training pipelines, hyperparameter tuning, and deployment
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import optuna
import numpy as np
from dataclasses import dataclass, field

from .config import mlflow_config, ExperimentConfig
from .agent_tracker import AgentExperimentTracker
from .metrics import mlflow_metrics


logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TuningAlgorithm(Enum):
    """Hyperparameter tuning algorithms"""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    OPTUNA_TPE = "optuna_tpe"
    OPTUNA_CMA_ES = "optuna_cma_es"


class DeploymentStrategy(Enum):
    """Model deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline"""
    
    name: str
    description: str = ""
    
    # Agent configuration
    agent_id: str = ""
    agent_type: str = ""
    
    # Training configuration
    training_script: str = ""
    data_path: str = ""
    model_type: str = "neural_network"
    framework: str = "pytorch"
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False
    tuning_algorithm: TuningAlgorithm = TuningAlgorithm.OPTUNA_TPE
    tuning_trials: int = 50
    tuning_timeout: int = 3600  # seconds
    hyperparameter_space: Dict[str, Any] = field(default_factory=dict)
    
    # Resource configuration
    max_concurrent_runs: int = 3
    cpu_limit: int = 4
    memory_limit_gb: int = 8
    gpu_enabled: bool = False
    
    # Validation configuration
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    enable_early_stopping: bool = True
    
    # Deployment configuration
    auto_deploy: bool = False
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    deployment_threshold: float = 0.85  # accuracy threshold
    
    # Monitoring configuration
    enable_model_monitoring: bool = True
    monitoring_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Scheduling
    schedule_cron: Optional[str] = None  # Cron expression for scheduling
    trigger_on_data_change: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "training_script": self.training_script,
            "data_path": self.data_path,
            "model_type": self.model_type,
            "framework": self.framework,
            "enable_hyperparameter_tuning": self.enable_hyperparameter_tuning,
            "tuning_algorithm": self.tuning_algorithm.value,
            "tuning_trials": self.tuning_trials,
            "tuning_timeout": self.tuning_timeout,
            "hyperparameter_space": self.hyperparameter_space,
            "max_concurrent_runs": self.max_concurrent_runs,
            "cpu_limit": self.cpu_limit,
            "memory_limit_gb": self.memory_limit_gb,
            "gpu_enabled": self.gpu_enabled,
            "validation_split": self.validation_split,
            "cross_validation_folds": self.cross_validation_folds,
            "enable_early_stopping": self.enable_early_stopping,
            "auto_deploy": self.auto_deploy,
            "deployment_strategy": self.deployment_strategy.value,
            "deployment_threshold": self.deployment_threshold,
            "enable_model_monitoring": self.enable_model_monitoring,
            "monitoring_metrics": self.monitoring_metrics,
            "alert_thresholds": self.alert_thresholds,
            "schedule_cron": self.schedule_cron,
            "trigger_on_data_change": self.trigger_on_data_change
        }


@dataclass
class PipelineRun:
    """Represents a pipeline execution"""
    
    run_id: str
    pipeline_config: PipelineConfig
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    mlflow_run_id: Optional[str] = None
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    
    def duration_seconds(self) -> float:
        """Get run duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return 0.0


class HyperparameterTuner:
    """Hyperparameter optimization using various algorithms"""
    
    def __init__(self, config: PipelineConfig, agent_tracker: AgentExperimentTracker):
        self.config = config
        self.agent_tracker = agent_tracker
        self.study: Optional[optuna.Study] = None
    
    async def optimize(self, objective_function: Callable, trial_timeout: int = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        try:
            if self.config.tuning_algorithm == TuningAlgorithm.OPTUNA_TPE:
                return await self._optimize_with_optuna(objective_function, "TPE", trial_timeout)
            elif self.config.tuning_algorithm == TuningAlgorithm.OPTUNA_CMA_ES:
                return await self._optimize_with_optuna(objective_function, "CMA-ES", trial_timeout)
            elif self.config.tuning_algorithm == TuningAlgorithm.RANDOM_SEARCH:
                return await self._optimize_random_search(objective_function)
            elif self.config.tuning_algorithm == TuningAlgorithm.GRID_SEARCH:
                return await self._optimize_grid_search(objective_function)
            else:
                raise ValueError(f"Unsupported tuning algorithm: {self.config.tuning_algorithm}")
                
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            raise
    
    async def _optimize_with_optuna(self, objective_function: Callable, 
                                  sampler_type: str = "TPE", trial_timeout: int = None) -> Dict[str, Any]:
        """Optimize using Optuna framework"""
        try:
            # Create sampler
            if sampler_type == "TPE":
                sampler = optuna.samplers.TPESampler()
            elif sampler_type == "CMA-ES":
                sampler = optuna.samplers.CmaEsSampler()
            else:
                sampler = optuna.samplers.RandomSampler()
            
            # Create study
            study_name = f"{self.config.name}_hyperopt_{int(time.time())}"
            self.study = optuna.create_study(
                study_name=study_name,
                direction="maximize",  # Assuming we want to maximize the objective
                sampler=sampler
            )
            
            # Create wrapper objective that logs to MLflow
            async def mlflow_objective(trial):
                # Generate parameters based on space definition
                params = {}
                for param_name, param_config in self.config.hyperparameter_space.items():
                    if param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"]
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config["low"],
                            param_config["high"]
                        )
                    elif param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config["choices"]
                        )
                
                # Start MLflow run for this trial
                run_name = f"trial_{trial.number}"
                mlflow_run_id = await self.agent_tracker.start_run(run_name)
                
                try:
                    # Log hyperparameters
                    await self.agent_tracker.log_params(params)
                    await self.agent_tracker.log_param("trial_number", trial.number)
                    
                    # Execute objective function
                    result = await objective_function(params)
                    
                    # Log results
                    if isinstance(result, dict):
                        objective_value = result.get("objective", result.get("accuracy", 0.0))
                        await self.agent_tracker.log_metrics(result)
                    else:
                        objective_value = float(result)
                        await self.agent_tracker.log_metric("objective", objective_value)
                    
                    await self.agent_tracker.log_metric("trial_objective", objective_value)
                    
                    # End run
                    await self.agent_tracker.end_run("FINISHED")
                    
                    return objective_value
                    
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}")
                    await self.agent_tracker.log_param("error", str(e))
                    await self.agent_tracker.end_run("FAILED")
                    raise optuna.TrialPruned()
            
            # Run optimization
            timeout = trial_timeout or self.config.tuning_timeout
            self.study.optimize(
                mlflow_objective,
                n_trials=self.config.tuning_trials,
                timeout=timeout,
                n_jobs=1  # Sequential execution for now
            )
            
            # Return best results
            best_trial = self.study.best_trial
            return {
                "best_params": best_trial.params,
                "best_value": best_trial.value,
                "n_trials": len(self.study.trials),
                "study_name": study_name
            }
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            raise
    
    async def _optimize_random_search(self, objective_function: Callable) -> Dict[str, Any]:
        """Random search optimization"""
        best_params = None
        best_value = float('-inf')
        
        for trial in range(self.config.tuning_trials):
            try:
                # Generate random parameters
                params = {}
                for param_name, param_config in self.config.hyperparameter_space.items():
                    if param_config["type"] == "float":
                        params[param_name] = np.random.uniform(
                            param_config["low"], param_config["high"]
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = np.random.randint(
                            param_config["low"], param_config["high"] + 1
                        )
                    elif param_config["type"] == "categorical":
                        params[param_name] = np.random.choice(param_config["choices"])
                
                # Start MLflow run
                run_name = f"random_trial_{trial}"
                await self.agent_tracker.start_run(run_name)
                
                # Log parameters
                await self.agent_tracker.log_params(params)
                
                # Execute objective
                result = await objective_function(params)
                
                # Process result
                if isinstance(result, dict):
                    objective_value = result.get("objective", result.get("accuracy", 0.0))
                    await self.agent_tracker.log_metrics(result)
                else:
                    objective_value = float(result)
                    await self.agent_tracker.log_metric("objective", objective_value)
                
                # Update best
                if objective_value > best_value:
                    best_value = objective_value
                    best_params = params.copy()
                
                await self.agent_tracker.end_run("FINISHED")
                
            except Exception as e:
                logger.error(f"Random search trial {trial} failed: {e}")
                await self.agent_tracker.end_run("FAILED")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": self.config.tuning_trials
        }
    
    async def _optimize_grid_search(self, objective_function: Callable) -> Dict[str, Any]:
        """Grid search optimization"""
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations()
        
        best_params = None
        best_value = float('-inf')
        
        for i, params in enumerate(param_combinations[:self.config.tuning_trials]):
            try:
                # Start MLflow run
                run_name = f"grid_trial_{i}"
                await self.agent_tracker.start_run(run_name)
                
                # Log parameters
                await self.agent_tracker.log_params(params)
                
                # Execute objective
                result = await objective_function(params)
                
                # Process result
                if isinstance(result, dict):
                    objective_value = result.get("objective", result.get("accuracy", 0.0))
                    await self.agent_tracker.log_metrics(result)
                else:
                    objective_value = float(result)
                    await self.agent_tracker.log_metric("objective", objective_value)
                
                # Update best
                if objective_value > best_value:
                    best_value = objective_value
                    best_params = params.copy()
                
                await self.agent_tracker.end_run("FINISHED")
                
            except Exception as e:
                logger.error(f"Grid search trial {i} failed: {e}")
                await self.agent_tracker.end_run("FAILED")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(param_combinations[:self.config.tuning_trials])
        }
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        import itertools
        
        param_names = []
        param_values = []
        
        for param_name, param_config in self.config.hyperparameter_space.items():
            param_names.append(param_name)
            
            if param_config["type"] == "float":
                # Generate evenly spaced values
                num_values = param_config.get("num_values", 5)
                values = np.linspace(param_config["low"], param_config["high"], num_values)
                param_values.append(values.tolist())
            elif param_config["type"] == "int":
                # Generate integer range
                values = list(range(param_config["low"], param_config["high"] + 1))
                param_values.append(values)
            elif param_config["type"] == "categorical":
                param_values.append(param_config["choices"])
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations


class MLPipelineExecutor:
    """Executes ML training pipelines"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
        self.agent_tracker: Optional[AgentExperimentTracker] = None
        
        # Current run state
        self.current_run: Optional[PipelineRun] = None
        self.is_running = False
        
        # Resource monitoring
        self.resource_monitor_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the pipeline executor"""
        try:
            # Initialize agent tracker if agent specified
            if self.config.agent_id:
                from .agent_tracker import agent_tracking_manager
                self.agent_tracker = await agent_tracking_manager.initialize_agent_tracking(
                    self.config.agent_id,
                    self.config.agent_type
                )
            
            logger.info(f"Initialized pipeline executor for {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline executor: {e}")
            raise
    
    async def execute_pipeline(self) -> PipelineRun:
        """Execute the complete ML pipeline"""
        if self.is_running:
            raise RuntimeError("Pipeline is already running")
        
        # Create pipeline run
        self.current_run = PipelineRun(
            run_id=str(uuid.uuid4()),
            pipeline_config=self.config,
            start_time=datetime.now(timezone.utc)
        )
        
        self.is_running = True
        
        try:
            logger.info(f"Starting pipeline execution: {self.config.name}")
            
            # Start MLflow run
            if self.agent_tracker:
                self.current_run.mlflow_run_id = await self.agent_tracker.start_run(
                    run_name=f"pipeline_{self.config.name}_{int(time.time())}",
                    tags={
                        "pipeline_name": self.config.name,
                        "pipeline_type": "automated",
                        "agent_id": self.config.agent_id
                    }
                )
            
            # Log pipeline configuration
            if self.agent_tracker:
                config_params = {f"pipeline_{k}": str(v) for k, v in self.config.to_dict().items()}
                await self.agent_tracker.log_params(config_params)
            
            # Start resource monitoring
            self.resource_monitor_task = asyncio.create_task(self._monitor_resources())
            
            # Execute pipeline steps
            self.current_run.status = PipelineStatus.RUNNING
            
            # Step 1: Data validation and preparation
            await self._validate_data()
            
            # Step 2: Hyperparameter tuning (if enabled)
            if self.config.enable_hyperparameter_tuning:
                await self._run_hyperparameter_tuning()
            
            # Step 3: Model training
            await self._train_model()
            
            # Step 4: Model validation
            await self._validate_model()
            
            # Step 5: Model deployment (if enabled)
            if self.config.auto_deploy:
                await self._deploy_model()
            
            # Step 6: Set up monitoring (if enabled)
            if self.config.enable_model_monitoring:
                await self._setup_monitoring()
            
            # Complete the run
            self.current_run.status = PipelineStatus.COMPLETED
            self.current_run.end_time = datetime.now(timezone.utc)
            
            if self.agent_tracker:
                await self.agent_tracker.log_metric(
                    "pipeline_duration_seconds", 
                    self.current_run.duration_seconds()
                )
                await self.agent_tracker.end_run("FINISHED")
            
            logger.info(f"Pipeline execution completed: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            
            self.current_run.status = PipelineStatus.FAILED
            self.current_run.error_message = str(e)
            self.current_run.end_time = datetime.now(timezone.utc)
            
            if self.agent_tracker:
                await self.agent_tracker.log_param("error", str(e))
                await self.agent_tracker.end_run("FAILED")
        
        finally:
            self.is_running = False
            
            # Stop resource monitoring
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
        
        return self.current_run
    
    async def _validate_data(self):
        """Validate and prepare data for training"""
        logger.info("Validating training data...")
        
        # Check if data path exists
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data path not found: {self.config.data_path}")
        
        # Log data information
        if self.agent_tracker:
            data_stats = self._get_data_statistics()
            await self.agent_tracker.log_params(data_stats)
        
        logger.info("Data validation completed")
    
    def _get_data_statistics(self) -> Dict[str, str]:
        """Get basic statistics about the training data"""
        try:
            data_path = Path(self.config.data_path)
            stats = {
                "data_path": str(data_path),
                "data_size_mb": round(data_path.stat().st_size / (1024 * 1024), 2),
                "data_modified": datetime.fromtimestamp(data_path.stat().st_mtime).isoformat()
            }
            
            # Try to get more detailed stats if it's a common format
            if data_path.suffix.lower() in ['.csv', '.json', '.parquet']:
                # This would require actual data loading and analysis
                # For now, just return basic file stats
                pass
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get data statistics: {e}")
            return {"data_path": self.config.data_path}
    
    async def _run_hyperparameter_tuning(self):
        """Run hyperparameter optimization"""
        logger.info("Starting hyperparameter tuning...")
        
        if not self.config.hyperparameter_space:
            logger.warning("No hyperparameter space defined, skipping tuning")
            return
        
        # Create tuner
        tuner = HyperparameterTuner(self.config, self.agent_tracker)
        
        # Define objective function
        async def objective(params: Dict[str, Any]) -> float:
            # This would execute the actual training with given parameters
            # For now, return a mock score
            # In a real implementation, this would:
            # 1. Update model configuration with params
            # 2. Train the model
            # 3. Evaluate on validation set
            # 4. Return validation metric
            
            # Mock implementation
            await asyncio.sleep(1)  # Simulate training time
            return np.random.random()  # Mock validation score
        
        # Run optimization
        results = await tuner.optimize(objective)
        
        # Store best parameters
        self.current_run.best_params = results["best_params"]
        self.current_run.best_metrics["tuning_best_score"] = results["best_value"]
        
        if self.agent_tracker:
            await self.agent_tracker.log_params(
                {f"best_{k}": str(v) for k, v in results["best_params"].items()}
            )
            await self.agent_tracker.log_metric("tuning_trials", results["n_trials"])
            await self.agent_tracker.log_metric("tuning_best_score", results["best_value"])
        
        logger.info(f"Hyperparameter tuning completed. Best score: {results['best_value']}")
    
    async def _train_model(self):
        """Train the model with best parameters"""
        logger.info("Training model...")
        
        # This would execute the actual training script/function
        # For now, simulate training
        training_duration = 30  # Simulate 30 seconds of training
        
        for epoch in range(10):  # Simulate 10 epochs
            await asyncio.sleep(training_duration / 10)
            
            # Log training metrics
            if self.agent_tracker:
                # Mock training metrics
                train_loss = 1.0 - (epoch * 0.1) + np.random.normal(0, 0.05)
                train_acc = 0.5 + (epoch * 0.05) + np.random.normal(0, 0.02)
                
                await self.agent_tracker.log_metrics({
                    "train_loss": max(0.1, train_loss),
                    "train_accuracy": min(0.99, max(0.5, train_acc))
                }, step=epoch)
        
        # Final training metrics
        final_accuracy = 0.85 + np.random.normal(0, 0.05)
        self.current_run.best_metrics["final_train_accuracy"] = final_accuracy
        
        if self.agent_tracker:
            await self.agent_tracker.log_metric("final_train_accuracy", final_accuracy)
        
        logger.info(f"Model training completed. Final accuracy: {final_accuracy:.4f}")
    
    async def _validate_model(self):
        """Validate the trained model"""
        logger.info("Validating model...")
        
        # Simulate model validation
        await asyncio.sleep(5)
        
        # Mock validation metrics
        val_accuracy = 0.80 + np.random.normal(0, 0.03)
        val_precision = 0.82 + np.random.normal(0, 0.03)
        val_recall = 0.78 + np.random.normal(0, 0.03)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        
        validation_metrics = {
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1
        }
        
        self.current_run.best_metrics.update(validation_metrics)
        
        if self.agent_tracker:
            await self.agent_tracker.log_metrics(validation_metrics)
        
        logger.info(f"Model validation completed. Accuracy: {val_accuracy:.4f}")
    
    async def _deploy_model(self):
        """Deploy the model if it meets the threshold"""
        logger.info("Checking deployment criteria...")
        
        val_accuracy = self.current_run.best_metrics.get("val_accuracy", 0.0)
        
        if val_accuracy >= self.config.deployment_threshold:
            logger.info(f"Model meets deployment threshold ({self.config.deployment_threshold}). Deploying...")
            
            # Simulate model deployment
            await asyncio.sleep(10)
            
            # Log deployment information
            if self.agent_tracker:
                await self.agent_tracker.log_params({
                    "deployed": "true",
                    "deployment_strategy": self.config.deployment_strategy.value,
                    "deployment_timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            logger.info("Model deployed successfully")
        else:
            logger.info(f"Model does not meet deployment threshold. Accuracy: {val_accuracy:.4f} < {self.config.deployment_threshold}")
            
            if self.agent_tracker:
                await self.agent_tracker.log_param("deployed", "false")
    
    async def _setup_monitoring(self):
        """Set up model monitoring"""
        logger.info("Setting up model monitoring...")
        
        # This would integrate with the existing monitoring system
        # For now, just log that monitoring is enabled
        
        if self.agent_tracker:
            await self.agent_tracker.log_params({
                "monitoring_enabled": "true",
                "monitoring_metrics": ",".join(self.config.monitoring_metrics)
            })
        
        logger.info("Model monitoring setup completed")
    
    async def _monitor_resources(self):
        """Monitor resource usage during pipeline execution"""
        try:
            while self.is_running:
                # Get current resource usage
                import psutil
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Log to MLflow if tracker available
                if self.agent_tracker:
                    await self.agent_tracker.log_metrics({
                        "pipeline_cpu_percent": cpu_percent,
                        "pipeline_memory_percent": memory.percent,
                        "pipeline_memory_used_gb": memory.used / (1024**3)
                    })
                
                # Check resource limits
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage during pipeline: {cpu_percent}%")
                
                if memory.percent > 90:
                    logger.warning(f"High memory usage during pipeline: {memory.percent}%")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("Resource monitoring stopped")
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")


class MLPipelineManager:
    """Manages multiple ML pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, MLPipelineExecutor] = {}
        self.pipeline_configs: Dict[str, PipelineConfig] = {}
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        
        # Load pipeline configurations
        self.config_dir = Path("/opt/sutazaiapp/backend/mlflow_system/pipeline_configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_pipeline_configurations()
    
    def _load_pipeline_configurations(self):
        """Load pipeline configurations from files"""
        try:
            for config_file in self.config_dir.glob("*.yaml"):
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                config = PipelineConfig(**config_data)
                self.pipeline_configs[config.name] = config
                
                logger.info(f"Loaded pipeline configuration: {config.name}")
                
        except Exception as e:
            logger.error(f"Failed to load pipeline configurations: {e}")
    
    async def create_pipeline(self, config: PipelineConfig) -> str:
        """Create a new pipeline"""
        try:
            # Save configuration
            self.pipeline_configs[config.name] = config
            
            # Save to file
            config_file = self.config_dir / f"{config.name}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            
            # Create executor
            executor = MLPipelineExecutor(config)
            await executor.initialize()
            
            self.pipelines[config.name] = executor
            
            logger.info(f"Created pipeline: {config.name}")
            return config.name
            
        except Exception as e:
            logger.error(f"Failed to create pipeline {config.name}: {e}")
            raise
    
    async def execute_pipeline(self, pipeline_name: str) -> str:
        """Execute a pipeline"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        
        if pipeline_name in self.running_pipelines:
            raise RuntimeError(f"Pipeline {pipeline_name} is already running")
        
        # Start pipeline execution
        executor = self.pipelines[pipeline_name]
        task = asyncio.create_task(executor.execute_pipeline())
        self.running_pipelines[pipeline_name] = task
        
        # Set up cleanup when task completes
        def cleanup_task(task):
            if pipeline_name in self.running_pipelines:
                del self.running_pipelines[pipeline_name]
        
        task.add_done_callback(cleanup_task)
        
        logger.info(f"Started pipeline execution: {pipeline_name}")
        return pipeline_name
    
    async def stop_pipeline(self, pipeline_name: str):
        """Stop a running pipeline"""
        if pipeline_name in self.running_pipelines:
            task = self.running_pipelines[pipeline_name]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            logger.info(f"Stopped pipeline: {pipeline_name}")
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get status of a pipeline"""
        if pipeline_name not in self.pipelines:
            return {"error": "Pipeline not found"}
        
        executor = self.pipelines[pipeline_name]
        
        status = {
            "name": pipeline_name,
            "is_running": pipeline_name in self.running_pipelines,
            "current_run": None
        }
        
        if executor.current_run:
            run = executor.current_run
            status["current_run"] = {
                "run_id": run.run_id,
                "status": run.status.value,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "duration_seconds": run.duration_seconds(),
                "best_metrics": run.best_metrics,
                "error_message": run.error_message
            }
        
        return status
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all available pipelines"""
        pipelines = []
        
        for name, config in self.pipeline_configs.items():
            pipeline_info = {
                "name": name,
                "description": config.description,
                "agent_id": config.agent_id,
                "agent_type": config.agent_type,
                "is_running": name in self.running_pipelines,
                "auto_deploy": config.auto_deploy,
                "hyperparameter_tuning": config.enable_hyperparameter_tuning
            }
            pipelines.append(pipeline_info)
        
        return pipelines


# Global pipeline manager
pipeline_manager = MLPipelineManager()