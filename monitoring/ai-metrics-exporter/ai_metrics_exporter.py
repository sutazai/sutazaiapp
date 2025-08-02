#!/usr/bin/env python3
"""
SutazAI Custom Metrics Exporter
Collects AI/ML performance and business metrics for Prometheus
"""

import time
import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
import psutil
import redis
import numpy as np
from dataclasses import dataclass
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from aiohttp import web, ClientSession
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for metrics collection"""
    backend_url: str = os.getenv('BACKEND_URL', 'http://backend:8000')
    ollama_url: str = os.getenv('OLLAMA_URL', 'http://ollama:11434')
    redis_url: str = os.getenv('REDIS_URL', 'redis://redis:6379/0')
    database_url: str = os.getenv('DATABASE_URL', 'postgresql+asyncpg://sutazai:sutazai_password@postgres:5432/sutazai')
    collection_interval: int = int(os.getenv('COLLECTION_INTERVAL', '30'))
    port: int = int(os.getenv('METRICS_PORT', '9200'))

class AIMetricsExporter:
    """Custom metrics exporter for SutazAI AI/ML performance"""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self.redis_client = None
        self.db_engine = None
        self.session_factory = None
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        
        # AI Model Performance Metrics
        self.ai_model_accuracy = Gauge(
            'ai_model_accuracy',
            'Current accuracy score of AI models',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.ai_model_inference_latency = Histogram(
            'ai_model_inference_latency_seconds',
            'AI model inference latency in seconds',
            ['model_name', 'model_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.ai_model_memory_usage = Gauge(
            'ai_model_memory_usage_bytes',
            'Memory usage by AI models',
            ['model_name'],
            registry=self.registry
        )
        
        self.ai_model_requests_total = Counter(
            'ai_model_requests_total',
            'Total number of AI model requests',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.ai_model_errors_total = Counter(
            'ai_model_errors_total',
            'Total number of AI model errors',
            ['model_name', 'error_type'],
            registry=self.registry
        )
        
        # AI Agent Metrics
        self.ai_agent_task_queue_size = Gauge(
            'ai_agent_task_queue_size',
            'Number of tasks in agent queue',
            ['agent_name'],
            registry=self.registry
        )
        
        self.ai_agent_task_completion_time = Histogram(
            'ai_agent_task_completion_time_seconds',
            'Time taken to complete agent tasks',
            ['agent_name', 'task_type'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
            registry=self.registry
        )
        
        self.ai_agent_success_rate = Gauge(
            'ai_agent_success_rate',
            'Success rate of agent tasks',
            ['agent_name'],
            registry=self.registry
        )
        
        self.ai_agent_active_sessions = Gauge(
            'ai_agent_active_sessions',
            'Number of active agent sessions',
            ['agent_name'],
            registry=self.registry
        )
        
        # Business Metrics
        self.task_completion_rate = Gauge(
            'task_completion_rate',
            'Overall task completion rate',
            registry=self.registry
        )
        
        self.user_satisfaction_score = Gauge(
            'user_satisfaction_score',
            'Average user satisfaction score',
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'active_users',
            'Number of currently active users',
            registry=self.registry
        )
        
        self.ai_assistant_requests_total = Counter(
            'ai_assistant_requests_total',
            'Total AI assistant requests',
            ['request_type'],
            registry=self.registry
        )
        
        # System Intelligence Metrics
        self.system_adaptability_score = Gauge(
            'system_adaptability_score',
            'System adaptability score',
            registry=self.registry
        )
        
        self.learning_progress_rate = Gauge(
            'learning_progress_rate',
            'Rate of system learning progress',
            registry=self.registry
        )
        
        self.emergent_behavior_anomaly_score = Gauge(
            'emergent_behavior_anomaly_score',
            'Anomaly score for emergent behaviors',
            registry=self.registry
        )
        
        # Data Pipeline Metrics
        self.data_ingestion_lag_minutes = Gauge(
            'data_ingestion_lag_minutes',
            'Data ingestion lag in minutes',
            ['pipeline'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score',
            ['dataset'],
            registry=self.registry
        )
        
        # Vector Database Metrics
        self.vector_search_latency = Histogram(
            'vector_search_latency_seconds',
            'Vector search latency',
            ['database_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.embedding_generation_rate = Gauge(
            'embedding_generation_rate',
            'Rate of embedding generation',
            registry=self.registry
        )
        
        # Cost and Resource Efficiency
        self.ai_inference_cost_per_hour = Gauge(
            'ai_inference_cost_per_hour',
            'AI inference cost per hour in dollars',
            registry=self.registry
        )
        
        self.resource_utilization_efficiency = Gauge(
            'resource_utilization_efficiency',
            'Resource utilization efficiency score',
            registry=self.registry
        )
        
        # Security Metrics
        self.ai_model_input_anomaly_score = Gauge(
            'ai_model_input_anomaly_score',
            'Anomaly score for AI model inputs',
            ['model_name'],
            registry=self.registry
        )
        
        self.unauthorized_access_attempts = Counter(
            'unauthorized_access_attempts_total',
            'Total unauthorized access attempts',
            ['endpoint'],
            registry=self.registry
        )
        
    async def initialize(self):
        """Initialize connections to external services"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.config.redis_url)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            logger.info("Connected to Redis")
            
            # Initialize database connection
            self.db_engine = create_async_engine(self.config.database_url)
            self.session_factory = sessionmaker(
                self.db_engine, class_=AsyncSession, expire_on_commit=False
            )
            logger.info("Connected to database")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def collect_ai_model_metrics(self):
        """Collect AI model performance metrics"""
        try:
            # Get Ollama model metrics
            async with ClientSession() as session:
                async with session.get(f"{self.config.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        
                        for model in models:
                            model_name = model.get('name', 'unknown')
                            
                            # Simulate metrics (in production, these would come from real monitoring)
                            accuracy = np.random.uniform(0.75, 0.95)
                            memory_usage = np.random.randint(1000000000, 8000000000)  # 1-8GB
                            
                            self.ai_model_accuracy.labels(
                                model_name=model_name,
                                model_version='latest'
                            ).set(accuracy)
                            
                            self.ai_model_memory_usage.labels(
                                model_name=model_name
                            ).set(memory_usage)
                
                # Get backend AI metrics
                async with session.get(f"{self.config.backend_url}/metrics/ai") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        
                        # Update metrics from backend
                        for metric_name, value in metrics.items():
                            if metric_name == 'inference_latency':
                                self.ai_model_inference_latency.labels(
                                    model_name='backend',
                                    model_type='api'
                                ).observe(value)
                            elif metric_name == 'request_count':
                                self.ai_model_requests_total.labels(
                                    model_name='backend',
                                    status='success'
                                ).inc(value)
                        
        except Exception as e:
            logger.error(f"Error collecting AI model metrics: {e}")
    
    async def collect_agent_metrics(self):
        """Collect AI agent performance metrics"""
        try:
            # Get agent metrics from Redis
            if self.redis_client:
                agent_keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.keys, 'agent:*:metrics'
                )
                
                for key in agent_keys:
                    agent_name = key.decode().split(':')[1]
                    metrics_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.hgetall, key
                    )
                    
                    if metrics_data:
                        queue_size = int(metrics_data.get(b'queue_size', 0))
                        success_rate = float(metrics_data.get(b'success_rate', 0.0))
                        active_sessions = int(metrics_data.get(b'active_sessions', 0))
                        
                        self.ai_agent_task_queue_size.labels(
                            agent_name=agent_name
                        ).set(queue_size)
                        
                        self.ai_agent_success_rate.labels(
                            agent_name=agent_name
                        ).set(success_rate)
                        
                        self.ai_agent_active_sessions.labels(
                            agent_name=agent_name
                        ).set(active_sessions)
        
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
    
    async def collect_business_metrics(self):
        """Collect business and user metrics"""
        try:
            async with ClientSession() as session:
                async with session.get(f"{self.config.backend_url}/metrics/business") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        
                        self.task_completion_rate.set(metrics.get('task_completion_rate', 0.0))
                        self.user_satisfaction_score.set(metrics.get('user_satisfaction', 0.0))
                        self.active_users.set(metrics.get('active_users', 0))
                        
                        # Update request counters
                        for request_type, count in metrics.get('request_counts', {}).items():
                            self.ai_assistant_requests_total.labels(
                                request_type=request_type
                            ).inc(count)
        
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
    
    async def collect_system_intelligence_metrics(self):
        """Collect system intelligence and adaptation metrics"""
        try:
            # Simulate intelligence metrics (in production, these would be calculated)
            adaptability = np.random.uniform(0.6, 0.9)
            learning_rate = np.random.uniform(0.01, 0.05)
            anomaly_score = np.random.uniform(0.0, 0.3)
            
            self.system_adaptability_score.set(adaptability)
            self.learning_progress_rate.set(learning_rate)
            self.emergent_behavior_anomaly_score.set(anomaly_score)
            
        except Exception as e:
            logger.error(f"Error collecting intelligence metrics: {e}")
    
    async def collect_data_pipeline_metrics(self):
        """Collect data pipeline health metrics"""
        try:
            # Simulate data pipeline metrics
            ingestion_lag = np.random.randint(0, 120)  # 0-120 minutes
            quality_score = np.random.uniform(0.85, 0.98)
            
            self.data_ingestion_lag_minutes.labels(pipeline='main').set(ingestion_lag)
            self.data_quality_score.labels(dataset='training').set(quality_score)
            
        except Exception as e:
            logger.error(f"Error collecting data pipeline metrics: {e}")
    
    async def collect_security_metrics(self):
        """Collect security-related metrics"""
        try:
            async with ClientSession() as session:
                async with session.get(f"{self.config.backend_url}/metrics/security") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        
                        # Update security metrics
                        for endpoint, attempts in metrics.get('unauthorized_attempts', {}).items():
                            self.unauthorized_access_attempts.labels(
                                endpoint=endpoint
                            ).inc(attempts)
                        
                        # AI model input anomalies
                        for model, anomaly_score in metrics.get('input_anomalies', {}).items():
                            self.ai_model_input_anomaly_score.labels(
                                model_name=model
                            ).set(anomaly_score)
        
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
    
    async def collect_all_metrics(self):
        """Collect all metrics"""
        logger.info("Starting metrics collection cycle")
        
        tasks = [
            self.collect_ai_model_metrics(),
            self.collect_agent_metrics(),
            self.collect_business_metrics(),
            self.collect_system_intelligence_metrics(),
            self.collect_data_pipeline_metrics(),
            self.collect_security_metrics(),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Metrics collection cycle completed")
    
    async def metrics_handler(self, request):
        """Handle metrics requests"""
        return web.Response(
            text=generate_latest(self.registry).decode('utf-8'),
            content_type=CONTENT_TYPE_LATEST
        )
    
    async def health_handler(self, request):
        """Health check endpoint"""
        return web.json_response({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})
    
    async def start_collection_loop(self):
        """Start the metrics collection loop"""
        while True:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def create_app(self):
        """Create aiohttp application"""
        app = web.Application()
        app.router.add_get('/metrics', self.metrics_handler)
        app.router.add_get('/health', self.health_handler)
        return app

async def main():
    """Main function"""
    config = MetricConfig()
    exporter = AIMetricsExporter(config)
    
    try:
        await exporter.initialize()
        app = exporter.create_app()
        
        # Start metrics collection in background
        collection_task = asyncio.create_task(exporter.start_collection_loop())
        
        # Start web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', config.port)
        await site.start()
        
        logger.info(f"AI Metrics Exporter started on port {config.port}")
        
        # Keep running
        await collection_task
        
    except Exception as e:
        logger.error(f"Failed to start exporter: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())