"""
MLflow Tracking Server for SutazAI System
High-performance tracking server optimized for 69+ agents
"""

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.tracking
from mlflow.server import get_app
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME
import psutil
import uvicorn
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from .config import mlflow_config, MLflowConfig
from .database import MLflowDatabase
from .metrics import MLflowMetrics


logger = logging.getLogger(__name__)


class MLflowTrackingServer:
    """High-performance MLflow tracking server for SutazAI"""
    
    def __init__(self, config: MLflowConfig = None):
        self.config = config or mlflow_config
        self.database = MLflowDatabase(self.config)
        self.metrics = MLflowMetrics()
        self.server_process = None
        self.monitoring_thread = None
        self.cleanup_thread = None
        self._shutdown_event = threading.Event()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_experiments,
            thread_name_prefix="mlflow_worker"
        )
        
        # MLflow client for server operations
        self.client: Optional[MlflowClient] = None
        
        # Performance monitoring
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
    async def initialize(self):
        """Initialize the tracking server"""
        logger.info("Initializing MLflow tracking server...")
        
        try:
            # Initialize database
            await self.database.initialize()
            
            # Set MLflow environment variables
            self._setup_environment()
            
            # Initialize MLflow client
            self.client = MlflowClient(
                tracking_uri=self.config.tracking_uri,
                registry_uri=self.config.model_registry_uri
            )
            
            # Create default experiment if needed
            await self._ensure_default_experiment()
            
            logger.info("MLflow tracking server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracking server: {e}")
            raise
    
    def _setup_environment(self):
        """Set up MLflow environment variables"""
        os.environ["MLFLOW_TRACKING_URI"] = self.config.tracking_uri
        os.environ["MLFLOW_BACKEND_STORE_URI"] = self.config.backend_store_uri
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = self.config.artifact_root
        
        if self.config.s3_artifact_root:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.s3_artifact_root
        
        # Performance optimizations
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"  # For development
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        
        # Database connection optimization
        os.environ["MLFLOW_SQLALCHEMY_DATABASE_URI"] = self.config.backend_store_uri
        os.environ["MLFLOW_SQLALCHEMY_POOL_SIZE"] = str(self.config.db_pool_size)
        os.environ["MLFLOW_SQLALCHEMY_MAX_OVERFLOW"] = str(self.config.db_max_overflow)
        os.environ["MLFLOW_SQLALCHEMY_POOL_TIMEOUT"] = str(self.config.db_pool_timeout)
        os.environ["MLFLOW_SQLALCHEMY_POOL_RECYCLE"] = str(self.config.db_pool_recycle)
    
    async def _ensure_default_experiment(self):
        """Ensure default SutazAI experiment exists"""
        try:
            experiment_name = "SutazAI_Default"
            
            # Check if experiment exists
            experiment = self.client.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                # Create default experiment
                experiment_id = self.client.create_experiment(
                    name=experiment_name,
                    artifact_location=os.path.join(self.config.artifact_root, "default"),
                    tags={
                        "system": "sutazai",
                        "version": "1.0.0",
                        "purpose": "default",
                        "created_by": "mlflow_system"
                    }
                )
                logger.info(f"Created default experiment: {experiment_id}")
            else:
                logger.info(f"Default experiment exists: {experiment.experiment_id}")
                
        except Exception as e:
            logger.error(f"Failed to create default experiment: {e}")
    
    async def start_server(self):
        """Start the MLflow tracking server"""
        logger.info("Starting MLflow tracking server...")
        
        try:
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitor_server,
                daemon=True
            )
            self.monitoring_thread.start()
            
            # Start cleanup thread
            if self.config.enable_auto_cleanup:
                self.cleanup_thread = threading.Thread(
                    target=self._cleanup_old_data,
                    daemon=True
                )
                self.cleanup_thread.start()
            
            # Configure uvicorn settings
            config = uvicorn.Config(
                app=get_app(
                    backend_store_uri=self.config.backend_store_uri,
                    default_artifact_root=self.config.artifact_root,
                    serve_artifacts=True
                ),
                host=self.config.tracking_server_host,
                port=self.config.tracking_server_port,
                workers=1,
                loop="asyncio",
                log_level="info"
            )
            
            # Start server
            server = uvicorn.Server(config)
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            logger.info(f"MLflow server starting on {self.config.tracking_server_host}:{self.config.tracking_server_port}")
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start tracking server: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_event.set()
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _monitor_server(self):
        """Monitor server performance and health"""
        logger.info("Starting server monitoring...")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage(self.config.artifact_root)
                
                # Update Prometheus metrics
                self.metrics.system_cpu_usage.set(cpu_percent)
                self.metrics.system_memory_usage.set(memory.percent)
                self.metrics.disk_usage.set(disk.percent)
                self.metrics.active_experiments.set(self._count_active_experiments())
                
                # Log health status
                if cpu_percent > 80 or memory.percent > 85:
                    logger.warning(f"High resource usage: CPU {cpu_percent}%, Memory {memory.percent}%")
                
                # Check database connectivity
                if not self._check_database_health():
                    logger.error("Database connectivity issues detected")
                    self.metrics.database_errors.inc()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _cleanup_old_data(self):
        """Clean up old experiments and artifacts"""
        logger.info("Starting data cleanup service...")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up old artifacts
                self._cleanup_artifacts()
                
                # Clean up old experiments (if configured)
                if self.config.experiment_retention_days > 0:
                    self._cleanup_experiments()
                
                # Sleep for 1 hour between cleanup cycles
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(3600)
    
    def _cleanup_artifacts(self):
        """Clean up old artifact files"""
        try:
            artifact_path = Path(self.config.artifact_root)
            current_time = time.time()
            retention_seconds = self.config.artifact_retention_days * 24 * 3600
            
            cleaned_count = 0
            for file_path in artifact_path.rglob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > retention_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old artifact files")
                self.metrics.artifacts_cleaned.inc(cleaned_count)
                
        except Exception as e:
            logger.error(f"Artifact cleanup failed: {e}")
    
    def _cleanup_experiments(self):
        """Archive old experiments (placeholder for future implementation)"""
        # This would require careful implementation to avoid data loss
        # For now, just log the intent
        logger.debug("Experiment cleanup check completed")
    
    def _count_active_experiments(self) -> int:
        """Count currently active experiments"""
        try:
            if self.client:
                experiments = self.client.search_experiments(
                    view_type=mlflow.tracking.ViewType.ACTIVE_ONLY,
                    max_results=SEARCH_MAX_RESULTS_DEFAULT
                )
                return len(experiments)
            return 0
        except Exception:
            return 0
    
    def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            engine = create_engine(
                self.config.backend_store_uri,
                poolclass=QueuePool,
                pool_size=1,
                max_overflow=0,
                pool_timeout=5
            )
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the tracking server"""
        logger.info("Shutting down MLflow tracking server...")
        
        try:
            # Set shutdown event
            self._shutdown_event.set()
            
            # Wait for threads to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=10)
            
            # Shutdown executor
            self.executor.shutdown(wait=True, timeout=30)
            
            # Close database connections
            await self.database.close()
            
            logger.info("MLflow tracking server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_server_stats(self) -> Dict:
        """Get current server statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "active_experiments": self._count_active_experiments(),
            "database_healthy": self._check_database_health(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage(self.config.artifact_root).percent
        }


async def main():
    """Main function to run the tracking server"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = MLflowTrackingServer()
    
    try:
        await server.initialize()
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())