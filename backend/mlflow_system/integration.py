"""
MLflow System Integration with SutazAI Infrastructure
Seamless integration with existing databases, APIs, and services
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import redis
import aioredis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import httpx

from .config import mlflow_config, MLflowConfig
from .tracking_server import MLflowTrackingServer
from .agent_tracker import agent_tracking_manager, AgentExperimentTracker
from .pipeline_automation import pipeline_manager
from .database import mlflow_database
from .metrics import mlflow_metrics


logger = logging.getLogger(__name__)


class SutazAIMLflowIntegration:
    """Main integration class for MLflow with SutazAI systems"""
    
    def __init__(self):
        self.config = mlflow_config
        self.tracking_server: Optional[MLflowTrackingServer] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_engine = None
        
        # Integration components
        self.api_integration = APIIntegration()
        self.database_integration = DatabaseIntegration()
        self.messaging_integration = MessagingIntegration()
        self.monitoring_integration = MonitoringIntegration()
        
        # System state
        self.is_initialized = False
        self.startup_time = None
        
    async def initialize(self):
        """Initialize the complete MLflow integration system"""
        logger.info("Initializing SutazAI MLflow integration...")
        
        try:
            self.startup_time = datetime.now(timezone.utc)
            
            # Step 1: Initialize database connections
            await self._initialize_databases()
            
            # Step 2: Initialize Redis for messaging
            await self._initialize_redis()
            
            # Step 3: Initialize MLflow tracking server
            await self._initialize_tracking_server()
            
            # Step 4: Initialize agent tracking
            await self._initialize_agent_tracking()
            
            # Step 5: Initialize pipeline automation
            await self._initialize_pipeline_automation()
            
            # Step 6: Initialize API integration
            await self._initialize_api_integration()
            
            # Step 7: Initialize monitoring
            await self._initialize_monitoring()
            
            # Step 8: Start background services
            await self._start_background_services()
            
            self.is_initialized = True
            logger.info("SutazAI MLflow integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow integration: {e}")
            raise
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        logger.info("Initializing database connections...")
        
        try:
            # Initialize MLflow database
            await mlflow_database.initialize()
            
            # Initialize SutazAI database connection
            await self.database_integration.initialize()
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection for messaging"""
        logger.info("Initializing Redis connection...")
        
        try:
            # Connect to Redis
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize messaging integration
            await self.messaging_integration.initialize(self.redis_client)
            
            logger.info("Redis connection initialized")
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            raise
    
    async def _initialize_tracking_server(self):
        """Initialize MLflow tracking server"""
        logger.info("Initializing MLflow tracking server...")
        
        try:
            # Create and initialize tracking server
            self.tracking_server = MLflowTrackingServer(self.config)
            await self.tracking_server.initialize()
            
            # Start server in background
            asyncio.create_task(self.tracking_server.start_server())
            
            # Wait for server to be ready
            await asyncio.sleep(5)
            
            logger.info("MLflow tracking server initialized")
            
        except Exception as e:
            logger.error(f"Tracking server initialization failed: {e}")
            raise
    
    async def _initialize_agent_tracking(self):
        """Initialize agent tracking system"""
        logger.info("Initializing agent tracking...")
        
        try:
            # Initialize all agents
            await agent_tracking_manager.initialize_all_agents()
            
            logger.info("Agent tracking initialized")
            
        except Exception as e:
            logger.error(f"Agent tracking initialization failed: {e}")
            raise
    
    async def _initialize_pipeline_automation(self):
        """Initialize pipeline automation"""
        logger.info("Initializing pipeline automation...")
        
        try:
            # Load and initialize pipelines
            # Pipeline manager is already initialized through import
            
            logger.info("Pipeline automation initialized")
            
        except Exception as e:
            logger.error(f"Pipeline automation initialization failed: {e}")
            raise
    
    async def _initialize_api_integration(self):
        """Initialize API integration"""
        logger.info("Initializing API integration...")
        
        try:
            await self.api_integration.initialize(self.config)
            
            logger.info("API integration initialized")
            
        except Exception as e:
            logger.error(f"API integration initialization failed: {e}")
            raise
    
    async def _initialize_monitoring(self):
        """Initialize monitoring integration"""
        logger.info("Initializing monitoring integration...")
        
        try:
            await self.monitoring_integration.initialize()
            
            logger.info("Monitoring integration initialized")
            
        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
            raise
    
    async def _start_background_services(self):
        """Start background services"""
        logger.info("Starting background services...")
        
        try:
            # Start health check service
            asyncio.create_task(self._health_check_service())
            
            # Start metrics collection service
            asyncio.create_task(self._metrics_collection_service())
            
            # Start cleanup service
            asyncio.create_task(self._cleanup_service())
            
            logger.info("Background services started")
            
        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
            raise
    
    async def _health_check_service(self):
        """Background health check service"""
        while True:
            try:
                # Check MLflow server health
                if self.tracking_server:
                    stats = self.tracking_server.get_server_stats()
                    
                    # Update metrics
                    mlflow_metrics.system_cpu_usage.set(stats.get('cpu_usage', 0))
                    mlflow_metrics.system_memory_usage.set(stats.get('memory_usage', 0))
                    mlflow_metrics.disk_usage.set(stats.get('disk_usage', 0))
                
                # Check database health
                if await mlflow_database._check_database_health():
                    mlflow_metrics.database_connections_active.set(1)
                else:
                    mlflow_metrics.database_errors.labels(error_type="connection_failed").inc()
                
                # Check Redis health
                if self.redis_client:
                    try:
                        await self.redis_client.ping()
                    except Exception:
                        logger.warning("Redis health check failed")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check service error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_service(self):
        """Background metrics collection service"""
        while True:
            try:
                # Collect system metrics
                uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
                mlflow_metrics.tracking_server_uptime.set(uptime)
                
                # Collect agent metrics
                agent_stats = agent_tracking_manager.get_all_tracking_stats()
                total_agents = agent_stats.get('total_agents', 0)
                mlflow_metrics.active_experiments.set(total_agents)
                
                # Collect database stats
                db_stats = await mlflow_database.get_database_stats()
                if db_stats:
                    pool_size = db_stats.get('pool_size', 0)
                    mlflow_metrics.database_connections_active.set(pool_size)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Metrics collection service error: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_service(self):
        """Background cleanup service"""
        while True:
            try:
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
                # Clean up old experiment data
                if self.config.enable_auto_cleanup:
                    cleaned_runs = await mlflow_database.cleanup_old_runs(
                        self.config.experiment_retention_days
                    )
                    
                    if cleaned_runs > 0:
                        logger.info(f"Cleaned up {cleaned_runs} old runs")
                        mlflow_metrics.artifacts_cleaned.inc(cleaned_runs)
                
            except Exception as e:
                logger.error(f"Cleanup service error: {e}")
                await asyncio.sleep(3600)
    
    async def shutdown(self):
        """Gracefully shutdown the integration system"""
        logger.info("Shutting down SutazAI MLflow integration...")
        
        try:
            # Shutdown agent tracking
            await agent_tracking_manager.shutdown_all_tracking()
            
            # Shutdown tracking server
            if self.tracking_server:
                await self.tracking_server.shutdown()
            
            # Close database connections
            await mlflow_database.close()
            await self.database_integration.close()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Shutdown other components
            await self.api_integration.shutdown()
            await self.monitoring_integration.shutdown()
            
            logger.info("SutazAI MLflow integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "initialized": self.is_initialized,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "uptime_seconds": (datetime.now(timezone.utc) - self.startup_time).total_seconds() if self.startup_time else 0,
            "components": {}
        }
        
        # Component status
        if self.tracking_server:
            status["components"]["tracking_server"] = self.tracking_server.get_server_stats()
        
        status["components"]["agent_tracking"] = agent_tracking_manager.get_all_tracking_stats()
        status["components"]["pipelines"] = pipeline_manager.list_pipelines()
        
        return status


class APIIntegration:
    """Integration with SutazAI REST APIs"""
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = "http://localhost:8000"  # SutazAI API base URL
        
    async def initialize(self, config: MLflowConfig):
        """Initialize API integration"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"User-Agent": "SutazAI-MLflow-Integration/1.0"}
        )
        
        # Test API connectivity
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                logger.info("API integration initialized successfully")
            else:
                logger.warning(f"API health check returned {response.status_code}")
        except Exception as e:
            logger.warning(f"API connectivity test failed: {e}")
    
    async def notify_experiment_started(self, experiment_id: str, agent_id: str):
        """Notify SutazAI API that an experiment has started"""
        try:
            payload = {
                "event": "experiment_started",
                "experiment_id": experiment_id,
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = await self.client.post("/api/v1/mlflow/events", json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to notify experiment started: {e}")
    
    async def notify_model_deployed(self, model_name: str, version: str, agent_id: str):
        """Notify SutazAI API that a model has been deployed"""
        try:
            payload = {
                "event": "model_deployed",
                "model_name": model_name,
                "version": version,
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = await self.client.post("/api/v1/mlflow/events", json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to notify model deployed: {e}")
    
    async def get_agent_configuration(self, agent_id: str) -> Dict[str, Any]:
        """Get agent configuration from SutazAI API"""
        try:
            response = await self.client.get(f"/api/v1/agents/{agent_id}/config")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get agent configuration: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown API integration"""
        if self.client:
            await self.client.aclose()


class DatabaseIntegration:
    """Integration with SutazAI PostgreSQL database"""
    
    def __init__(self):
        self.engine = None
        self.connection_string = "postgresql+asyncpg://sutazai:sutazai_secure@localhost:5432/sutazai"
    
    async def initialize(self):
        """Initialize database integration"""
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            
            self.engine = create_async_engine(
                self.connection_string,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("Database integration initialized")
            
        except Exception as e:
            logger.error(f"Database integration failed: {e}")
            raise
    
    async def get_agent_metadata(self, agent_id: str) -> Dict[str, Any]:
        """Get agent metadata from SutazAI database"""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT * FROM agents WHERE agent_id = :agent_id"),
                    {"agent_id": agent_id}
                )
                
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get agent metadata: {e}")
            return {}
    
    async def store_experiment_results(self, experiment_data: Dict[str, Any]):
        """Store experiment results in SutazAI database"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO experiment_results 
                        (experiment_id, agent_id, metrics, parameters, timestamp)
                        VALUES (:experiment_id, :agent_id, :metrics, :parameters, :timestamp)
                    """),
                    {
                        "experiment_id": experiment_data["experiment_id"],
                        "agent_id": experiment_data["agent_id"],
                        "metrics": json.dumps(experiment_data["metrics"]),
                        "parameters": json.dumps(experiment_data["parameters"]),
                        "timestamp": datetime.now(timezone.utc)
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to store experiment results: {e}")
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()


class MessagingIntegration:
    """Integration with Redis messaging system"""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.subscribers = {}
    
    async def initialize(self, redis_client: aioredis.Redis):
        """Initialize messaging integration"""
        self.redis = redis_client
        
        # Subscribe to relevant channels
        await self._setup_subscriptions()
        
        logger.info("Messaging integration initialized")
    
    async def _setup_subscriptions(self):
        """Set up Redis channel subscriptions"""
        try:
            # Subscribe to agent events
            asyncio.create_task(self._subscribe_to_channel("agent_events", self._handle_agent_event))
            
            # Subscribe to system events
            asyncio.create_task(self._subscribe_to_channel("system_events", self._handle_system_event))
            
        except Exception as e:
            logger.error(f"Failed to setup subscriptions: {e}")
    
    async def _subscribe_to_channel(self, channel: str, handler):
        """Subscribe to a Redis channel"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await handler(message["data"])
                    
        except Exception as e:
            logger.error(f"Subscription error for channel {channel}: {e}")
    
    async def _handle_agent_event(self, message: str):
        """Handle agent events from Redis"""
        try:
            event_data = json.loads(message)
            event_type = event_data.get("type")
            agent_id = event_data.get("agent_id")
            
            if event_type == "agent_started":
                # Initialize tracking for new agent
                await agent_tracking_manager.initialize_agent_tracking(
                    agent_id, 
                    event_data.get("agent_type"),
                    event_data.get("config", {})
                )
            elif event_type == "agent_stopped":
                # End tracking for stopped agent
                tracker = agent_tracking_manager.get_agent_tracker(agent_id)
                if tracker and tracker.current_run_id:
                    await tracker.end_run("KILLED")
            
        except Exception as e:
            logger.error(f"Failed to handle agent event: {e}")
    
    async def _handle_system_event(self, message: str):
        """Handle system events from Redis"""
        try:
            event_data = json.loads(message)
            event_type = event_data.get("type")
            
            if event_type == "system_shutdown":
                # Graceful shutdown requested
                logger.info("System shutdown event received")
                # Could trigger graceful shutdown here
            
        except Exception as e:
            logger.error(f"Failed to handle system event: {e}")
    
    async def publish_experiment_event(self, event_type: str, experiment_id: str, data: Dict[str, Any]):
        """Publish experiment event to Redis"""
        try:
            event_data = {
                "type": event_type,
                "experiment_id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data
            }
            
            await self.redis.publish("experiment_events", json.dumps(event_data))
            
        except Exception as e:
            logger.error(f"Failed to publish experiment event: {e}")


class MonitoringIntegration:
    """Integration with SutazAI monitoring systems"""
    
    def __init__(self):
        self.prometheus_client = None
        self.grafana_client = None
    
    async def initialize(self):
        """Initialize monitoring integration"""
        try:
            # Set up Prometheus metrics export
            if mlflow_config.enable_prometheus_metrics:
                await self._setup_prometheus_export()
            
            logger.info("Monitoring integration initialized")
            
        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
            raise
    
    async def _setup_prometheus_export(self):
        """Set up Prometheus metrics export"""
        try:
            from prometheus_client import start_http_server
            
            # Start metrics HTTP server
            start_http_server(mlflow_config.metrics_port)
            
            logger.info(f"Prometheus metrics server started on port {mlflow_config.metrics_port}")
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    async def send_alert(self, level: str, message: str, details: Dict[str, Any] = None):
        """Send alert to monitoring system"""
        try:
            alert_data = {
                "level": level,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "mlflow_system"
            }
            
            # In a real implementation, this would send to alerting system
            logger.warning(f"ALERT [{level}]: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def shutdown(self):
        """Shutdown monitoring integration"""
        pass


# Global integration instance
sutazai_mlflow_integration = SutazAIMLflowIntegration()