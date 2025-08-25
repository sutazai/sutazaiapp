#!/usr/bin/env python3
"""
AI Services Validation Tests
===========================

Comprehensive validation tests for AI and vector database services:
- Ollama LLM service with model management and inference
- ChromaDB vector database with embedding operations  
- Qdrant vector search with performance testing
- FAISS similarity search with index management
- Edge inference capabilities and model optimization

Focus on actual AI workloads and performance validation.
"""

import asyncio
import aiohttp
import json
import logging
import time
import numpy as np
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import httpx
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class AIServiceTestResult:
    """AI service test execution result"""
    service: str
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class AIServicesValidator:
    """Comprehensive AI services validation"""
    
    def __init__(self):
        self.results: List[AIServiceTestResult] = []
        
        # AI service configurations from port registry  
        self.config = {
            "ollama": {
                "host": "localhost",
                "port": 10104,
                "models": ["tinyllama", "llama3.2:1b"],  # Expected available models
                "performance_threshold_ms": 5000,
                "max_tokens": 500
            },
            "chromadb": {
                "host": "localhost", 
                "port": 10100,
                "api_version": "v1",
                "performance_threshold_ms": 100,
                "test_collection": "infrastructure_test"
            },
            "qdrant": {
                "host": "localhost",
                "port": 10101,
                "grpc_port": 10102,
                "performance_threshold_ms": 50,
                "test_collection": "infrastructure_test",
                "vector_size": 384
            },
            "faiss": {
                "host": "localhost",
                "port": 10103,
                "performance_threshold_ms": 10,
                "vector_size": 384,
                "index_type": "IVF"
            }
        }
        
        # Test data for AI services
        self.test_documents = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming modern technology.",
            "Vector databases enable efficient similarity search.",
            "Machine learning models require careful validation.",
            "Natural language processing has many applications."
        ]
        
        # Initialize sentence transformer for embeddings (if available)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_available = True
        except Exception as e:
            logger.warning(f"Sentence transformer not available: {e}")
            self.embedding_model = None
            self.embedding_available = False
    
    async def run_all_ai_services_tests(self) -> List[AIServiceTestResult]:
        """Execute all AI services validation tests"""
        logger.info("Starting comprehensive AI services validation")
        
        # Run AI service tests with proper dependency handling
        test_methods = [
            # Core AI inference service
            ("ollama", self.test_ollama_comprehensive),
            
            # Vector databases (can run in parallel after Ollama)
            ("chromadb", self.test_chromadb_comprehensive),
            ("qdrant", self.test_qdrant_comprehensive), 
            ("faiss", self.test_faiss_comprehensive)
        ]
        
        # Execute Ollama test first (other services may depend on it)
        await test_methods[0][1]()  # ollama
        
        # Execute vector database tests in parallel
        vector_db_tasks = []
        for service, method in test_methods[1:]:
            task = asyncio.create_task(method())
            vector_db_tasks.append((service, task))
        
        # Wait for vector database tests
        for service, task in vector_db_tasks:
            try:
                await task
            except Exception as e:
                logger.error(f"AI service test {service} failed: {e}")
        
        return self.results
    
    async def test_ollama_comprehensive(self) -> None:
        """Comprehensive Ollama LLM service validation"""
        config = self.config["ollama"]
        start_time = time.time()
        
        try:
            base_url = f"http://{config['host']}:{config['port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test Ollama API availability
                async with session.get(f"{base_url}/api/tags",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    api_available = response.status == 200
                    if api_available:
                        models_data = await response.json()
                        available_models = [model["name"] for model in models_data.get("models", [])]
                    else:
                        models_data = {}
                        available_models = []
                
                # Test model inference if models are available
                inference_results = {}
                inference_success = False
                
                if available_models:
                    # Use the first available model for testing
                    test_model = available_models[0]
                    
                    # Test simple inference
                    inference_payload = {
                        "model": test_model,
                        "prompt": "What is artificial intelligence?",
                        "stream": False,
                        "options": {
                            "num_ctx": 512,
                            "temperature": 0.1
                        }
                    }
                    
                    inference_start = time.time()
                    try:
                        async with session.post(f"{base_url}/api/generate",
                                              json=inference_payload,
                                              timeout=aiohttp.ClientTimeout(total=30)) as response:
                            inference_success = response.status == 200
                            if inference_success:
                                inference_data = await response.json()
                                inference_duration = (time.time() - inference_start) * 1000
                                
                                inference_results = {
                                    "model": test_model,
                                    "response_length": len(inference_data.get("response", "")),
                                    "duration_ms": inference_duration,
                                    "tokens_generated": inference_data.get("eval_count", 0),
                                    "tokens_per_second": inference_data.get("eval_count", 0) / (inference_duration / 1000) if inference_duration > 0 else 0,
                                    "context_length": inference_data.get("context", []),
                                    "load_duration_ms": inference_data.get("load_duration", 0) / 1000000,  # Convert nanoseconds
                                    "prompt_eval_duration_ms": inference_data.get("prompt_eval_duration", 0) / 1000000
                                }
                            else:
                                error_data = await response.text()
                                logger.warning(f"Ollama inference failed: {error_data}")
                    
                    except asyncio.TimeoutError:
                        logger.warning("Ollama inference timed out")
                        inference_results = {"timeout": True}
                
                # Test embeddings endpoint if available
                embeddings_success = False
                embeddings_results = {}
                
                try:
                    embeddings_payload = {
                        "model": available_models[0] if available_models else "nomic-embed-text",
                        "prompt": "Test embedding generation"
                    }
                    
                    async with session.post(f"{base_url}/api/embeddings",
                                          json=embeddings_payload,
                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                        embeddings_success = response.status == 200
                        if embeddings_success:
                            embeddings_data = await response.json()
                            embeddings_results = {
                                "embedding_dimensions": len(embeddings_data.get("embedding", [])),
                                "embedding_available": True
                            }
                
                except Exception as e:
                    logger.info(f"Ollama embeddings not available: {e}")
                    embeddings_results = {"embedding_available": False}
            
            duration = time.time() - start_time
            
            # Calculate performance grade
            if inference_results.get("duration_ms", float('inf')) < 2000:
                performance_grade = "excellent"
            elif inference_results.get("duration_ms", float('inf')) < 5000:
                performance_grade = "good" 
            else:
                performance_grade = "poor"
            
            self.results.append(AIServiceTestResult(
                service="ollama",
                test_name="comprehensive_validation",
                success=api_available and (inference_success or len(available_models) == 0),
                duration=duration,
                metrics={
                    "api_available": api_available,
                    "models_available": len(available_models),
                    "model_list": available_models,
                    "inference_working": inference_success,
                    "inference_metrics": inference_results,
                    "embeddings_available": embeddings_success,
                    "embeddings_metrics": embeddings_results,
                    "performance_grade": performance_grade,
                    "tokens_per_second": inference_results.get("tokens_per_second", 0)
                }
            ))
            
            logger.info(f"Ollama validation - API: {api_available}, Models: {len(available_models)}, Inference: {inference_success}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(AIServiceTestResult(
                service="ollama",
                test_name="comprehensive_validation",
                success=False,
                duration=duration, 
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Ollama validation failed: {e}")
    
    async def test_chromadb_comprehensive(self) -> None:
        """Comprehensive ChromaDB vector database validation"""
        config = self.config["chromadb"]
        start_time = time.time()
        
        try:
            base_url = f"http://{config['host']}:{config['port']}"
            collection_name = config["test_collection"]
            
            async with aiohttp.ClientSession() as session:
                # Test ChromaDB API availability
                async with session.get(f"{base_url}/api/{config['api_version']}/heartbeat",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    api_available = response.status == 200
                    if api_available:
                        heartbeat_data = await response.json()
                    else:
                        heartbeat_data = {}
                
                # Test version endpoint
                async with session.get(f"{base_url}/api/{config['api_version']}/version",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    version_available = response.status == 200
                    if version_available:
                        version_data = await response.json()
                    else:
                        version_data = {}
                
                collection_operations = {}
                embedding_operations = {}
                
                if api_available:
                    # Test collection operations
                    try:
                        # Create test collection
                        create_payload = {
                            "name": collection_name,
                            "metadata": {"description": "Infrastructure test collection"},
                            "get_or_create": True
                        }
                        
                        async with session.post(f"{base_url}/api/{config['api_version']}/collections",
                                              json=create_payload,
                                              timeout=aiohttp.ClientTimeout(total=30)) as response:
                            collection_created = response.status in [200, 201]
                            if collection_created:
                                collection_data = await response.json()
                            
                        collection_operations["create"] = collection_created
                        
                        # List collections
                        async with session.get(f"{base_url}/api/{config['api_version']}/collections",
                                             timeout=aiohttp.ClientTimeout(total=30)) as response:
                            collections_listed = response.status == 200
                            if collections_listed:
                                collections_data = await response.json()
                            
                        collection_operations["list"] = collections_listed
                        
                        # Test embedding operations if we can generate embeddings
                        if self.embedding_available and collection_created:
                            # Generate embeddings for test documents
                            embeddings = self.embedding_model.encode(self.test_documents).tolist()
                            
                            # Add documents with embeddings
                            add_payload = {
                                "embeddings": embeddings,
                                "documents": self.test_documents,
                                "ids": [f"doc_{i}" for i in range(len(self.test_documents))],
                                "metadatas": [{"source": f"test_doc_{i}"} for i in range(len(self.test_documents))]
                            }
                            
                            embed_start = time.time()
                            async with session.post(f"{base_url}/api/{config['api_version']}/collections/{collection_name}/add",
                                                  json=add_payload,
                                                  timeout=aiohttp.ClientTimeout(total=30)) as response:
                                add_success = response.status in [200, 201]
                                add_duration = (time.time() - embed_start) * 1000
                            
                            embedding_operations["add_documents"] = add_success
                            embedding_operations["add_duration_ms"] = add_duration
                            
                            # Test similarity search
                            if add_success:
                                query_embedding = self.embedding_model.encode(["What is machine learning?"]).tolist()[0]
                                
                                query_payload = {
                                    "query_embeddings": [query_embedding],
                                    "n_results": 3
                                }
                                
                                search_start = time.time()
                                async with session.post(f"{base_url}/api/{config['api_version']}/collections/{collection_name}/query",
                                                      json=query_payload,
                                                      timeout=aiohttp.ClientTimeout(total=30)) as response:
                                    search_success = response.status == 200
                                    search_duration = (time.time() - search_start) * 1000
                                    
                                    if search_success:
                                        search_results = await response.json()
                                        embedding_operations["search_results_count"] = len(search_results.get("documents", [[]])[0])
                                
                                embedding_operations["search"] = search_success
                                embedding_operations["search_duration_ms"] = search_duration
                        
                        # Cleanup test collection
                        async with session.delete(f"{base_url}/api/{config['api_version']}/collections/{collection_name}",
                                                timeout=aiohttp.ClientTimeout(total=30)) as response:
                            cleanup_success = response.status == 200
                        
                        collection_operations["cleanup"] = cleanup_success
                        
                    except Exception as coll_error:
                        logger.warning(f"ChromaDB collection operations failed: {coll_error}")
                        collection_operations["error"] = str(coll_error)
            
            duration = time.time() - start_time
            
            # Calculate performance grade
            avg_operation_time = embedding_operations.get("add_duration_ms", 0) + embedding_operations.get("search_duration_ms", 0)
            if avg_operation_time < 100:
                performance_grade = "excellent"
            elif avg_operation_time < 300:
                performance_grade = "good"
            else:
                performance_grade = "poor"
            
            self.results.append(AIServiceTestResult(
                service="chromadb",
                test_name="comprehensive_validation",
                success=api_available,
                duration=duration,
                metrics={
                    "api_available": api_available,
                    "version": version_data.get("version", "unknown"),
                    "heartbeat": heartbeat_data,
                    "collection_operations": collection_operations,
                    "embedding_operations": embedding_operations,
                    "embedding_model_available": self.embedding_available,
                    "performance_grade": performance_grade
                }
            ))
            
            logger.info(f"ChromaDB validation - API: {api_available}, Collections: {collection_operations.get('create', False)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(AIServiceTestResult(
                service="chromadb",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"ChromaDB validation failed: {e}")
    
    async def test_qdrant_comprehensive(self) -> None:
        """Comprehensive Qdrant vector search validation"""
        config = self.config["qdrant"]
        start_time = time.time()
        
        try:
            base_url = f"http://{config['host']}:{config['port']}"
            collection_name = config["test_collection"]
            
            async with aiohttp.ClientSession() as session:
                # Test Qdrant health
                async with session.get(f"{base_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    health_available = response.status == 200
                
                # Test Qdrant info
                async with session.get(f"{base_url}",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    info_available = response.status == 200
                    if info_available:
                        info_data = await response.json()
                    else:
                        info_data = {}
                
                collection_operations = {}
                vector_operations = {}
                
                if health_available:
                    try:
                        # Create test collection
                        create_payload = {
                            "vectors": {
                                "size": config["vector_size"],
                                "distance": "Cosine"
                            }
                        }
                        
                        async with session.put(f"{base_url}/collections/{collection_name}",
                                             json=create_payload,
                                             timeout=aiohttp.ClientTimeout(total=30)) as response:
                            collection_created = response.status in [200, 201]
                        
                        collection_operations["create"] = collection_created
                        
                        # List collections
                        async with session.get(f"{base_url}/collections",
                                             timeout=aiohttp.ClientTimeout(total=30)) as response:
                            collections_success = response.status == 200
                            if collections_success:
                                collections_data = await response.json()
                        
                        collection_operations["list"] = collections_success
                        
                        # Test vector operations if we can generate embeddings
                        if self.embedding_available and collection_created:
                            # Generate embeddings for test documents
                            embeddings = self.embedding_model.encode(self.test_documents)
                            
                            # Upsert vectors
                            points = [
                                {
                                    "id": i,
                                    "vector": embedding.tolist(),
                                    "payload": {
                                        "text": doc,
                                        "source": f"test_doc_{i}"
                                    }
                                }
                                for i, (doc, embedding) in enumerate(zip(self.test_documents, embeddings))
                            ]
                            
                            upsert_payload = {"points": points}
                            
                            upsert_start = time.time()
                            async with session.put(f"{base_url}/collections/{collection_name}/points",
                                                 json=upsert_payload,
                                                 timeout=aiohttp.ClientTimeout(total=30)) as response:
                                upsert_success = response.status == 200
                                upsert_duration = (time.time() - upsert_start) * 1000
                            
                            vector_operations["upsert"] = upsert_success
                            vector_operations["upsert_duration_ms"] = upsert_duration
                            
                            # Test vector search
                            if upsert_success:
                                query_vector = self.embedding_model.encode(["What is machine learning?"]).tolist()
                                
                                search_payload = {
                                    "vector": query_vector,
                                    "limit": 3,
                                    "with_payload": True,
                                    "with_vector": False
                                }
                                
                                search_start = time.time()
                                async with session.post(f"{base_url}/collections/{collection_name}/points/search",
                                                      json=search_payload,
                                                      timeout=aiohttp.ClientTimeout(total=30)) as response:
                                    search_success = response.status == 200
                                    search_duration = (time.time() - search_start) * 1000
                                    
                                    if search_success:
                                        search_results = await response.json()
                                        vector_operations["search_results_count"] = len(search_results.get("result", []))
                                        vector_operations["top_score"] = search_results.get("result", [{}])[0].get("score", 0) if search_results.get("result") else 0
                                
                                vector_operations["search"] = search_success
                                vector_operations["search_duration_ms"] = search_duration
                        
                        # Get collection info
                        async with session.get(f"{base_url}/collections/{collection_name}",
                                             timeout=aiohttp.ClientTimeout(total=30)) as response:
                            collection_info_success = response.status == 200
                            if collection_info_success:
                                collection_info = await response.json()
                        
                        collection_operations["info"] = collection_info_success
                        
                        # Cleanup test collection
                        async with session.delete(f"{base_url}/collections/{collection_name}",
                                                timeout=aiohttp.ClientTimeout(total=30)) as response:
                            cleanup_success = response.status == 200
                        
                        collection_operations["cleanup"] = cleanup_success
                        
                    except Exception as coll_error:
                        logger.warning(f"Qdrant collection operations failed: {coll_error}")
                        collection_operations["error"] = str(coll_error)
            
            duration = time.time() - start_time
            
            # Calculate performance grade
            avg_operation_time = vector_operations.get("upsert_duration_ms", 0) + vector_operations.get("search_duration_ms", 0)
            if avg_operation_time < 50:
                performance_grade = "excellent"
            elif avg_operation_time < 150:
                performance_grade = "good"
            else:
                performance_grade = "poor"
            
            self.results.append(AIServiceTestResult(
                service="qdrant",
                test_name="comprehensive_validation",
                success=health_available,
                duration=duration,
                metrics={
                    "health_available": health_available,
                    "info_available": info_available,
                    "version": info_data.get("version", "unknown"),
                    "collection_operations": collection_operations,
                    "vector_operations": vector_operations,
                    "embedding_model_available": self.embedding_available,
                    "performance_grade": performance_grade
                }
            ))
            
            logger.info(f"Qdrant validation - Health: {health_available}, Collections: {collection_operations.get('create', False)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(AIServiceTestResult(
                service="qdrant",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Qdrant validation failed: {e}")
    
    async def test_faiss_comprehensive(self) -> None:
        """Comprehensive FAISS similarity search validation"""
        config = self.config["faiss"]
        start_time = time.time()
        
        try:
            base_url = f"http://{config['host']}:{config['port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test FAISS service health
                async with session.get(f"{base_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    health_available = response.status == 200
                    if health_available:
                        health_data = await response.json()
                    else:
                        health_data = {}
                
                # Test FAISS info endpoint
                async with session.get(f"{base_url}/info",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    info_available = response.status == 200
                    if info_available:
                        info_data = await response.json()
                    else:
                        info_data = {}
                
                index_operations = {}
                search_operations = {}
                
                if health_available:
                    try:
                        # Test index creation if embeddings are available
                        if self.embedding_available:
                            # Generate embeddings for test documents
                            embeddings = self.embedding_model.encode(self.test_documents)
                            
                            # Create index
                            create_payload = {
                                "index_name": "test_index",
                                "dimension": config["vector_size"],
                                "index_type": config["index_type"],
                                "metric": "cosine"
                            }
                            
                            async with session.post(f"{base_url}/create_index",
                                                  json=create_payload,
                                                  timeout=aiohttp.ClientTimeout(total=30)) as response:
                                index_created = response.status in [200, 201]
                            
                            index_operations["create"] = index_created
                            
                            # Add vectors to index
                            if index_created:
                                add_payload = {
                                    "index_name": "test_index",
                                    "vectors": embeddings.tolist(),
                                    "ids": list(range(len(embeddings)))
                                }
                                
                                add_start = time.time()
                                async with session.post(f"{base_url}/add_vectors",
                                                      json=add_payload,
                                                      timeout=aiohttp.ClientTimeout(total=30)) as response:
                                    add_success = response.status == 200
                                    add_duration = (time.time() - add_start) * 1000
                                
                                index_operations["add_vectors"] = add_success
                                index_operations["add_duration_ms"] = add_duration
                                
                                # Test vector search
                                if add_success:
                                    query_vector = self.embedding_model.encode(["What is machine learning?"])
                                    
                                    search_payload = {
                                        "index_name": "test_index",
                                        "query_vector": query_vector.tolist()[0],
                                        "k": 3
                                    }
                                    
                                    search_start = time.time()
                                    async with session.post(f"{base_url}/search",
                                                          json=search_payload,
                                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                                        search_success = response.status == 200
                                        search_duration = (time.time() - search_start) * 1000
                                        
                                        if search_success:
                                            search_results = await response.json()
                                            search_operations["results_count"] = len(search_results.get("indices", []))
                                            search_operations["top_distance"] = search_results.get("distances", [float('inf')])[0] if search_results.get("distances") else float('inf')
                                    
                                    search_operations["search"] = search_success
                                    search_operations["search_duration_ms"] = search_duration
                                
                                # List indices
                                async with session.get(f"{base_url}/list_indices",
                                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                                    list_success = response.status == 200
                                    if list_success:
                                        indices_data = await response.json()
                                
                                index_operations["list"] = list_success
                                
                                # Cleanup test index
                                async with session.delete(f"{base_url}/delete_index/test_index",
                                                         timeout=aiohttp.ClientTimeout(total=30)) as response:
                                    cleanup_success = response.status == 200
                                
                                index_operations["cleanup"] = cleanup_success
                    
                    except Exception as idx_error:
                        logger.warning(f"FAISS index operations failed: {idx_error}")
                        index_operations["error"] = str(idx_error)
            
            duration = time.time() - start_time
            
            # Calculate performance grade
            search_time = search_operations.get("search_duration_ms", float('inf'))
            if search_time < 10:
                performance_grade = "excellent"
            elif search_time < 50:
                performance_grade = "good"
            else:
                performance_grade = "poor"
            
            self.results.append(AIServiceTestResult(
                service="faiss",
                test_name="comprehensive_validation",
                success=health_available,
                duration=duration,
                metrics={
                    "health_available": health_available,
                    "info_available": info_available,
                    "service_info": info_data,
                    "index_operations": index_operations,
                    "search_operations": search_operations,
                    "embedding_model_available": self.embedding_available,
                    "performance_grade": performance_grade
                }
            ))
            
            logger.info(f"FAISS validation - Health: {health_available}, Index ops: {index_operations.get('create', False)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(AIServiceTestResult(
                service="faiss",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"FAISS validation failed: {e}")
    
    def generate_ai_services_report(self) -> Dict[str, Any]:
        """Generate comprehensive AI services validation report"""
        total_services = len(self.results)
        successful_services = len([r for r in self.results if r.success])
        
        # Group results by service
        service_results = {}
        for result in self.results:
            service_results[result.service] = result
        
        # Calculate AI services health
        ai_services = ["ollama", "chromadb", "qdrant", "faiss"]
        ai_success = sum(1 for svc in ai_services 
                        if svc in service_results and service_results[svc].success)
        
        ai_grade = "EXCELLENT" if ai_success == len(ai_services) else \
                  "GOOD" if ai_success >= len(ai_services) - 1 else \
                  "POOR"
        
        # Performance analysis
        performance_summary = {}
        inference_metrics = {}
        
        for result in self.results:
            if result.success:
                performance_summary[result.service] = result.metrics.get("performance_grade", "unknown")
                
                # Collect inference-specific metrics
                if result.service == "ollama":
                    inference_metrics = result.metrics.get("inference_metrics", {})
        
        # Vector database analysis
        vector_db_analysis = {}
        for service in ["chromadb", "qdrant", "faiss"]:
            if service in service_results and service_results[service].success:
                metrics = service_results[service].metrics
                vector_db_analysis[service] = {
                    "embedding_support": metrics.get("embedding_model_available", False),
                    "operations_success": len([k for k, v in metrics.items() if isinstance(v, dict) and v.get("create", False)]) > 0
                }
        
        return {
            "summary": {
                "total_ai_services": total_services,
                "successful_services": successful_services,
                "success_rate": round(successful_services / max(total_services, 1) * 100, 2),
                "ai_services_grade": ai_grade,
                "core_ai_health": f"{ai_success}/{len(ai_services)}"
            },
            "service_details": {
                service: {
                    "status": "success" if result.success else "failed",
                    "duration_seconds": round(result.duration, 3),
                    "key_metrics": result.metrics,
                    "error": result.error_message
                }
                for service, result in service_results.items()
            },
            "performance_analysis": performance_summary,
            "inference_capabilities": inference_metrics,
            "vector_database_analysis": vector_db_analysis,
            "embedding_model_available": self.embedding_available,
            "recommendations": self._generate_ai_recommendations(service_results)
        }
    
    def _generate_ai_recommendations(self, service_results: Dict) -> List[str]:
        """Generate AI services improvement recommendations"""
        recommendations = []
        
        # Check critical AI services
        for service, result in service_results.items():
            if not result.success:
                if service == "ollama":
                    recommendations.append(f"🔴 CRITICAL: Ollama LLM service is not available - AI inference capabilities disabled")
                else:
                    recommendations.append(f"🟡 WARNING: {service} vector database is not accessible - similarity search limited")
        
        # Performance recommendations
        if "ollama" in service_results and service_results["ollama"].success:
            ollama_metrics = service_results["ollama"].metrics
            inference_metrics = ollama_metrics.get("inference_metrics", {})
            
            tokens_per_second = inference_metrics.get("tokens_per_second", 0)
            if tokens_per_second < 10:
                recommendations.append(f"🟡 PERFORMANCE: Ollama inference is slow ({tokens_per_second:.1f} tokens/sec) - consider model optimization")
            
            if len(ollama_metrics.get("model_list", [])) == 0:
                recommendations.append("🟡 SETUP: No models available in Ollama - download models for AI inference")
        
        # Vector database recommendations
        vector_services = ["chromadb", "qdrant", "faiss"]
        working_vector_dbs = sum(1 for svc in vector_services 
                               if svc in service_results and service_results[svc].success)
        
        if working_vector_dbs == 0:
            recommendations.append("🔴 CRITICAL: No vector databases are working - similarity search and RAG capabilities disabled")
        elif working_vector_dbs < len(vector_services):
            recommendations.append(f"🟡 REDUNDANCY: Only {working_vector_dbs}/{len(vector_services)} vector databases working - consider fixing for redundancy")
        
        # Embedding model recommendation
        if not self.embedding_available:
            recommendations.append("🟡 FEATURE: Sentence transformer not available - install for full vector database testing")
        
        return recommendations if recommendations else ["✅ AI services are operating optimally"]

async def main():
    """Main execution for AI services validation"""
    validator = AIServicesValidator()
    
    print("🤖 Starting AI Services Validation Tests")
    print("=" * 60)
    
    results = await validator.run_all_ai_services_tests()
    report = validator.generate_ai_services_report()
    
    print("\n" + "=" * 60)
    print("📊 AI SERVICES VALIDATION COMPLETE")
    print("=" * 60)
    
    # Print summary
    summary = report["summary"]
    print(f"AI Services Tested: {summary['total_ai_services']}")
    print(f"Successful: {summary['successful_services']} ({summary['success_rate']}%)")
    print(f"AI Services Grade: {summary['ai_services_grade']}")
    print(f"Core AI Health: {summary['core_ai_health']}")
    
    # Print service details
    print("\n🔍 AI Service Status:")
    for service, details in report["service_details"].items():
        status_icon = "✅" if details["status"] == "success" else "❌"
        duration = details["duration_seconds"]
        print(f"  {status_icon} {service}: {details['status']} ({duration:.2f}s)")
        
        # Show key metrics for successful services
        if details["status"] == "success":
            metrics = details["key_metrics"]
            if service == "ollama":
                models = len(metrics.get("model_list", []))
                inference = "✅" if metrics.get("inference_working") else "❌"
                print(f"    📚 Models: {models}, Inference: {inference}")
            elif "performance_grade" in metrics:
                grade = metrics["performance_grade"]
                print(f"    ⚡ Performance: {grade}")
        
        if details["error"]:
            print(f"    ⚠️  {details['error']}")
    
    # Print inference capabilities
    inference_caps = report["inference_capabilities"]
    if inference_caps:
        print(f"\n🧠 Inference Capabilities:")
        print(f"  Tokens/sec: {inference_caps.get('tokens_per_second', 0):.1f}")
        print(f"  Model: {inference_caps.get('model', 'none')}")
    
    # Print vector database analysis
    vector_analysis = report["vector_database_analysis"]
    if vector_analysis:
        print(f"\n🔍 Vector Database Status:")
        for db, analysis in vector_analysis.items():
            embed_support = "✅" if analysis["embedding_support"] else "❌"
            ops_success = "✅" if analysis["operations_success"] else "❌"
            print(f"  {db}: Embeddings {embed_support}, Operations {ops_success}")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  {rec}")
    
    # Save detailed report
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"ai_services_validation_report_{timestamp}.json"
    
    import json
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return summary["ai_services_grade"] in ["EXCELLENT", "GOOD"]

if __name__ == "__main__":
    success = asyncio.run(main())
    import sys
    sys.exit(0 if success else 1)