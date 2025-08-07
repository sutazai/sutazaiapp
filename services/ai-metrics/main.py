#!/usr/bin/env python3
"""
SutazAI AI Metrics Exporter
Collects and exports metrics from AI services for Prometheus monitoring
"""

import asyncio
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
import os
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
AI_SERVICE_UP = Gauge('ai_service_up', 'AI service availability', ['service'])
AI_MODEL_LOADED = Gauge('ai_model_loaded', 'Number of loaded models', ['service'])
AI_REQUESTS_TOTAL = Counter('ai_requests_total', 'Total AI requests', ['service', 'model'])
AI_REQUEST_DURATION = Histogram('ai_request_duration_seconds', 'AI request duration', ['service'])
AI_MEMORY_USAGE = Gauge('ai_memory_usage_bytes', 'AI service memory usage', ['service'])
AI_GPU_USAGE = Gauge('ai_gpu_usage_percent', 'GPU usage percentage', ['device'])

# Vector database metrics
VECTOR_DB_SIZE = Gauge('vector_db_size_total', 'Vector database size', ['database'])
VECTOR_DB_QUERIES = Counter('vector_db_queries_total', 'Vector database queries', ['database', 'operation'])
VECTOR_DB_RESPONSE_TIME = Histogram('vector_db_response_time_seconds', 'Vector DB response time', ['database'])

# Graph database metrics
GRAPH_DB_NODES = Gauge('graph_db_nodes_total', 'Total nodes in graph database')
GRAPH_DB_RELATIONSHIPS = Gauge('graph_db_relationships_total', 'Total relationships in graph database')
GRAPH_DB_QUERIES = Counter('graph_db_queries_total', 'Graph database queries', ['query_type'])

class AIMetricsCollector:
    def __init__(self):
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')
        self.chromadb_url = os.getenv('CHROMADB_URL', 'http://chromadb:8000')
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://qdrant:6333')
        self.neo4j_url = os.getenv('NEO4J_URL', 'bolt://neo4j:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'sutazai_neo4j')
        
        self.neo4j_driver = None
        self.last_metrics_collection = time.time()
        
    async def initialize(self):
        """Initialize connections"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info("AI Metrics Collector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection: {e}")
    
    async def collect_ollama_metrics(self):
        """Collect metrics from Ollama service"""
        try:
            # Check service health
            response = requests.get(f"{self.ollama_url}/", timeout=5)
            AI_SERVICE_UP.labels(service='ollama').set(1 if response.status_code == 200 else 0)
            
            # Get loaded models
            models_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                loaded_models = len(models_data.get('models', []))
                AI_MODEL_LOADED.labels(service='ollama').set(loaded_models)
                
                # Log model information
                for model in models_data.get('models', []):
                    logger.info(f"Ollama model loaded: {model.get('name', 'unknown')}")
            
            # Get performance stats (if available)
            try:
                stats_response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    for model_stat in stats_data.get('models', []):
                        model_name = model_stat.get('name', 'unknown')
                        size_vram = model_stat.get('size_vram', 0)
                        AI_MEMORY_USAGE.labels(service=f'ollama-{model_name}').set(size_vram)
            except Exception as e:
                logger.debug(f"Ollama stats not available: {e}")
                
        except Exception as e:
            AI_SERVICE_UP.labels(service='ollama').set(0)
            logger.error(f"Failed to collect Ollama metrics: {e}")
    
    async def collect_chromadb_metrics(self):
        """Collect metrics from ChromaDB"""
        try:
            # Check service health
            response = requests.get(f"{self.chromadb_url}/api/v1/heartbeat", timeout=5)
            AI_SERVICE_UP.labels(service='chromadb').set(1 if response.status_code == 200 else 0)
            
            # Get collections info
            collections_response = requests.get(f"{self.chromadb_url}/api/v1/collections", timeout=5)
            if collections_response.status_code == 200:
                collections = collections_response.json()
                total_vectors = 0
                for collection in collections:
                    collection_name = collection.get('name', 'unknown')
                    # Get collection count
                    count_response = requests.get(
                        f"{self.chromadb_url}/api/v1/collections/{collection_name}/count",
                        timeout=5
                    )
                    if count_response.status_code == 200:
                        count = count_response.json()
                        total_vectors += count
                        VECTOR_DB_SIZE.labels(database=f'chromadb-{collection_name}').set(count)
                
                VECTOR_DB_SIZE.labels(database='chromadb').set(total_vectors)
                
        except Exception as e:
            AI_SERVICE_UP.labels(service='chromadb').set(0)
            logger.error(f"Failed to collect ChromaDB metrics: {e}")
    
    async def collect_qdrant_metrics(self):
        """Collect metrics from Qdrant"""
        try:
            # Check service health
            response = requests.get(f"{self.qdrant_url}/", timeout=5)
            AI_SERVICE_UP.labels(service='qdrant').set(1 if response.status_code == 200 else 0)
            
            # Get collections info
            collections_response = requests.get(f"{self.qdrant_url}/collections", timeout=5)
            if collections_response.status_code == 200:
                collections_data = collections_response.json()
                collections = collections_data.get('result', {}).get('collections', [])
                
                total_vectors = 0
                for collection in collections:
                    collection_name = collection.get('name', 'unknown')
                    # Get collection info
                    info_response = requests.get(
                        f"{self.qdrant_url}/collections/{collection_name}",
                        timeout=5
                    )
                    if info_response.status_code == 200:
                        info_data = info_response.json()
                        vectors_count = info_data.get('result', {}).get('vectors_count', 0)
                        total_vectors += vectors_count
                        VECTOR_DB_SIZE.labels(database=f'qdrant-{collection_name}').set(vectors_count)
                
                VECTOR_DB_SIZE.labels(database='qdrant').set(total_vectors)
                
        except Exception as e:
            AI_SERVICE_UP.labels(service='qdrant').set(0)
            logger.error(f"Failed to collect Qdrant metrics: {e}")
    
    async def collect_neo4j_metrics(self):
        """Collect metrics from Neo4j"""
        try:
            if not self.neo4j_driver:
                AI_SERVICE_UP.labels(service='neo4j').set(0)
                return
                
            with self.neo4j_driver.session() as session:
                # Check connection
                result = session.run("RETURN 1")
                result.consume()
                AI_SERVICE_UP.labels(service='neo4j').set(1)
                
                # Get node count
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()['node_count']
                GRAPH_DB_NODES.set(node_count)
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()['rel_count']
                GRAPH_DB_RELATIONSHIPS.set(rel_count)
                
        except Exception as e:
            AI_SERVICE_UP.labels(service='neo4j').set(0)
            logger.error(f"Failed to collect Neo4j metrics: {e}")
    
    async def collect_faiss_metrics(self):
        """Collect metrics from FAISS service"""
        try:
            # Check if FAISS service is running
            response = requests.get("http://faiss-vector:8000/health", timeout=5)
            AI_SERVICE_UP.labels(service='faiss').set(1 if response.status_code == 200 else 0)
            
            # Get index statistics
            stats_response = requests.get("http://faiss-vector:8000/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                total_vectors = stats.get('total_vectors', 0)
                VECTOR_DB_SIZE.labels(database='faiss').set(total_vectors)
                
        except Exception as e:
            AI_SERVICE_UP.labels(service='faiss').set(0)
            logger.error(f"Failed to collect FAISS metrics: {e}")
    
    async def collect_all_metrics(self):
        """Collect all AI service metrics"""
        logger.info("Starting metrics collection cycle")
        
        # Collect metrics from all services concurrently
        await asyncio.gather(
            self.collect_ollama_metrics(),
            self.collect_chromadb_metrics(),
            self.collect_qdrant_metrics(),
            self.collect_neo4j_metrics(),
            self.collect_faiss_metrics(),
            return_exceptions=True
        )
        
        self.last_metrics_collection = time.time()
        logger.info("Metrics collection cycle completed")
    
    async def run_collector(self):
        """Main collector loop"""
        await self.initialize()
        
        while True:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

def main():
    """Main function"""
    logger.info("Starting AI Metrics Exporter")
    
    # Start Prometheus metrics server
    metrics_port = int(os.getenv('METRICS_PORT', '9200'))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # Create and run collector
    collector = AIMetricsCollector()
    
    # Run the collector
    try:
        asyncio.run(collector.run_collector())
    except KeyboardInterrupt:
        logger.info("Shutting down AI Metrics Exporter")
    finally:
        if collector.neo4j_driver:
            collector.neo4j_driver.close()

if __name__ == "__main__":
    main()