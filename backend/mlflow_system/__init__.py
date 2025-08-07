"""
SutazAI MLflow System
Comprehensive experiment tracking and ML pipeline automation for SutazAI's 69+ AI agents
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from .config import mlflow_config, MLflowConfig, config_manager
from .tracking_server import MLflowTrackingServer
from .agent_tracker import agent_tracking_manager, AgentExperimentTracker
from .pipeline_automation import pipeline_manager, MLPipelineManager
from .analysis_tools import experiment_analyzer
from .database import mlflow_database
from .metrics import mlflow_metrics
from .integration import sutazai_mlflow_integration


__version__ = "1.0.0"
__author__ = "SutazAI Team"
__description__ = "MLflow-based experiment tracking and ML pipeline automation for SutazAI"


logger = logging.getLogger(__name__)


class SutazAIMLflowSystem:
    """Main system class for SutazAI MLflow integration"""
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or mlflow_config
        self.is_running = False
        self.initialization_complete = False
        
        # System components
        self.integration = sutazai_mlflow_integration
        self.agent_tracking = agent_tracking_manager
        self.pipeline_manager = pipeline_manager
        self.experiment_analyzer = experiment_analyzer
        
        # Performance tracking
        self.startup_time = None
        self.total_experiments_created = 0
        self.total_runs_executed = 0
        self.total_models_registered = 0
    
    async def initialize(self) -> bool:
        """Initialize the complete MLflow system"""
        logger.info("🚀 Initializing SutazAI MLflow System...")
        
        try:
            # Initialize the integration layer
            await self.integration.initialize()
            
            # System is now ready
            self.initialization_complete = True
            self.is_running = True
            
            logger.info("✅ SutazAI MLflow System initialized successfully")
            logger.info(f"📊 Tracking URI: {self.config.tracking_uri}")
            logger.info(f"📁 Artifact Root: {self.config.artifact_root}")
            logger.info(f"🤖 Agent Tracking: {self.config.agent_tracking_enabled}")
            logger.info(f"⚙️ Pipeline Automation: Enabled")
            logger.info(f"📈 Monitoring: {self.config.enable_prometheus_metrics}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SutazAI MLflow System: {e}")
            self.is_running = False
            return False
    
    async def start_experiment_tracking(self, agent_id: str, agent_type: str = None, 
                                      agent_config: Dict[str, Any] = None) -> Optional[AgentExperimentTracker]:
        """Start experiment tracking for a specific agent"""
        if not self.is_running:
            logger.error("MLflow system is not running")
            return None
        
        try:
            tracker = await self.agent_tracking.initialize_agent_tracking(
                agent_id, agent_type, agent_config
            )
            
            self.total_experiments_created += 1
            logger.info(f"✅ Started experiment tracking for agent {agent_id}")
            return tracker
            
        except Exception as e:
            logger.error(f"❌ Failed to start tracking for agent {agent_id}: {e}")
            return None
    
    async def create_pipeline(self, pipeline_config: Dict[str, Any]) -> bool:
        """Create a new ML pipeline"""
        if not self.is_running:
            logger.error("MLflow system is not running")
            return False
        
        try:
            from .pipeline_automation import PipelineConfig
            
            # Convert dict to PipelineConfig
            config = PipelineConfig(**pipeline_config)
            
            # Create pipeline
            pipeline_name = await self.pipeline_manager.create_pipeline(config)
            
            logger.info(f"✅ Created pipeline: {pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create pipeline: {e}")
            return False
    
    async def execute_pipeline(self, pipeline_name: str) -> bool:
        """Execute an ML pipeline"""
        if not self.is_running:
            logger.error("MLflow system is not running")
            return False
        
        try:
            await self.pipeline_manager.execute_pipeline(pipeline_name)
            
            self.total_runs_executed += 1
            logger.info(f"✅ Started pipeline execution: {pipeline_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute pipeline {pipeline_name}: {e}")
            return False
    
    async def compare_experiments(self, experiment_ids: list, 
                                 metrics_to_compare: list = None) -> Optional[Dict[str, Any]]:
        """Compare multiple experiments"""
        if not self.is_running:
            logger.error("MLflow system is not running")
            return None
        
        try:
            comparison = await self.experiment_analyzer.compare_experiments(
                experiment_ids, metrics_to_compare
            )
            
            logger.info(f"✅ Completed experiment comparison for {len(experiment_ids)} experiments")
            return comparison.to_dict()
            
        except Exception as e:
            logger.error(f"❌ Failed to compare experiments: {e}")
            return None
    
    async def analyze_model_performance(self, model_name: str, 
                                      time_window_days: int = 30) -> Optional[Dict[str, Any]]:
        """Analyze model performance over time"""
        if not self.is_running:
            logger.error("MLflow system is not running")
            return None
        
        try:
            analysis = await self.experiment_analyzer.analyze_model_performance(
                model_name, time_window_days
            )
            
            logger.info(f"✅ Completed performance analysis for model {model_name}")
            return analysis.to_dict()
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze model performance: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                "system": {
                    "running": self.is_running,
                    "initialized": self.initialization_complete,
                    "version": __version__,
                    "config": {
                        "tracking_uri": self.config.tracking_uri,
                        "artifact_root": self.config.artifact_root,
                        "agent_tracking_enabled": self.config.agent_tracking_enabled,
                        "auto_log_models": self.config.auto_log_models,
                        "max_concurrent_experiments": self.config.max_concurrent_experiments
                    }
                },
                "statistics": {
                    "total_experiments_created": self.total_experiments_created,
                    "total_runs_executed": self.total_runs_executed,
                    "total_models_registered": self.total_models_registered
                },
                "components": self.integration.get_system_status() if self.integration else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the MLflow system gracefully"""
        logger.info("🛑 Shutting down SutazAI MLflow System...")
        
        try:
            self.is_running = False
            
            # Shutdown integration layer
            await self.integration.shutdown()
            
            logger.info("✅ SutazAI MLflow System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# Global system instance
mlflow_system = SutazAIMLflowSystem()


# Convenience functions for easy access
async def initialize_system(config: Optional[MLflowConfig] = None) -> bool:
    """Initialize the MLflow system"""
    if config:
        global mlflow_system
        mlflow_system = SutazAIMLflowSystem(config)
    
    return await mlflow_system.initialize()


async def start_agent_tracking(agent_id: str, agent_type: str = None, 
                              agent_config: Dict[str, Any] = None) -> Optional[AgentExperimentTracker]:
    """Start tracking for an agent"""
    return await mlflow_system.start_experiment_tracking(agent_id, agent_type, agent_config)


async def create_ml_pipeline(pipeline_config: Dict[str, Any]) -> bool:
    """Create an ML pipeline"""
    return await mlflow_system.create_pipeline(pipeline_config)


async def run_pipeline(pipeline_name: str) -> bool:
    """Execute an ML pipeline"""
    return await mlflow_system.execute_pipeline(pipeline_name)


async def compare_experiments_simple(experiment_ids: list, metrics: list = None) -> Optional[Dict[str, Any]]:
    """Compare experiments (simplified interface)"""
    return await mlflow_system.compare_experiments(experiment_ids, metrics)


async def analyze_model(model_name: str, days: int = 30) -> Optional[Dict[str, Any]]:
    """Analyze model performance (simplified interface)"""
    return await mlflow_system.analyze_model_performance(model_name, days)


def get_status() -> Dict[str, Any]:
    """Get system status"""
    return mlflow_system.get_system_status()


async def shutdown_system():
    """Shutdown the system"""
    await mlflow_system.shutdown()


# Export key classes and functions
__all__ = [
    # Main system
    'SutazAIMLflowSystem',
    'mlflow_system',
    
    # Configuration
    'MLflowConfig',
    'mlflow_config',
    'config_manager',
    
    # Core components
    'MLflowTrackingServer',
    'AgentExperimentTracker',
    'MLPipelineManager',
    'agent_tracking_manager',
    'pipeline_manager',
    'experiment_analyzer',
    
    # Convenience functions
    'initialize_system',
    'start_agent_tracking',
    'create_ml_pipeline',
    'run_pipeline',
    'compare_experiments_simple',
    'analyze_model',
    'get_status',
    'shutdown_system',
    
    # Integration
    'sutazai_mlflow_integration',
    
    # Database and metrics
    'mlflow_database',
    'mlflow_metrics'
]