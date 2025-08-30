"""
Service connections for all external services
Manages connections to Redis, RabbitMQ, Vector DBs, Neo4j, etc.
"""

import redis.asyncio as redis
import aio_pika
from aio_pika import Connection, Channel
import httpx
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional, Dict, Any
import logging
import asyncio
from app.core.config import settings

logger = logging.getLogger(__name__)


class ServiceConnections:
    """Singleton class managing all service connections"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.redis_client: Optional[redis.Redis] = None
            self.rabbitmq_connection: Optional[Connection] = None
            self.rabbitmq_channel: Optional[Channel] = None
            self.neo4j_driver = None
            self.qdrant_client: Optional[AsyncQdrantClient] = None
            self.chroma_client = None
            self.faiss_client: Optional[httpx.AsyncClient] = None
            self.consul_client: Optional[httpx.AsyncClient] = None
            self.kong_client: Optional[httpx.AsyncClient] = None
            self.ollama_client: Optional[httpx.AsyncClient] = None
            self._initialized = True
    
    async def connect_all(self):
        """Initialize all service connections"""
        await self.connect_redis()
        await self.connect_rabbitmq()
        await self.connect_neo4j()
        await self.connect_vector_dbs()
        await self.connect_service_mesh()
        logger.info("All service connections established")
    
    async def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        try:
            self.rabbitmq_connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL
            )
            self.rabbitmq_channel = await self.rabbitmq_connection.channel()
            logger.info("RabbitMQ connection established")
        except Exception as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            raise
    
    async def connect_neo4j(self):
        """Connect to Neo4j"""
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URL,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Test connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    async def connect_vector_dbs(self):
        """Connect to vector databases"""
        # ChromaDB - Using v2 API configuration
        try:
            # Use Client factory with proper v2 API settings
            self.chroma_client = chromadb.Client(
                ChromaSettings(
                    chroma_api_impl="chromadb.api.segment.SegmentAPI",
                    chroma_server_host=settings.CHROMADB_HOST,
                    chroma_server_http_port=settings.CHROMADB_PORT,
                    chroma_server_ssl_enabled=False,
                    chroma_client_auth_provider=None,
                    chroma_client_auth_credentials=None
                )
            )
            # Test the connection with v2 API
            heartbeat_result = self.chroma_client.heartbeat()
            logger.info(f"ChromaDB v2 API connection established (heartbeat: {heartbeat_result})")
        except Exception as e:
            logger.error(f"ChromaDB v2 API connection failed: {e}")
            # Try fallback to HttpClient if Client factory fails
            try:
                self.chroma_client = chromadb.HttpClient(
                    host=settings.CHROMADB_HOST,
                    port=settings.CHROMADB_PORT
                )
                logger.info("ChromaDB connection established using HttpClient fallback")
            except Exception as fallback_error:
                logger.error(f"ChromaDB HttpClient fallback also failed: {fallback_error}")
                logger.warning("Continuing without ChromaDB connection")
                self.chroma_client = None
        
        # Qdrant
        try:
            self.qdrant_client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_HTTP_PORT,
                https=False
            )
            logger.info("Qdrant connection established")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            logger.warning("Continuing without Qdrant connection")
            self.qdrant_client = None
        
        # FAISS (HTTP client)
        try:
            self.faiss_client = httpx.AsyncClient(
                base_url=f"http://{settings.FAISS_HOST}:{settings.FAISS_PORT}"
            )
            # Test connection
            response = await self.faiss_client.get("/health")
            response.raise_for_status()
            logger.info("FAISS connection established")
        except Exception as e:
            logger.error(f"FAISS connection failed: {e}")
            logger.warning("Continuing without FAISS connection")
            self.faiss_client = None
    
    async def connect_service_mesh(self):
        """Connect to service mesh components"""
        # Consul
        try:
            self.consul_client = httpx.AsyncClient(
                base_url=f"http://{settings.CONSUL_HOST}:{settings.CONSUL_PORT}"
            )
            # Test connection
            response = await self.consul_client.get("/v1/status/leader")
            response.raise_for_status()
            logger.info("Consul connection established")
        except Exception as e:
            logger.warning(f"Consul connection failed (non-critical): {e}")
            self.consul_client = None
        
        # Kong Admin API - with retry logic and timeout
        max_retries = 5
        retry_delay = 2
        for retry in range(max_retries):
            try:
                self.kong_client = httpx.AsyncClient(
                    base_url=f"http://{settings.KONG_HOST}:{settings.KONG_ADMIN_PORT}",
                    timeout=httpx.Timeout(5.0, connect=10.0)
                )
                # Test connection with retry
                response = await self.kong_client.get("/status")
                response.raise_for_status()
                logger.info("Kong Admin API connection established")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Kong connection attempt {retry + 1} failed, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Kong connection failed after {max_retries} attempts: {e}")
                    self.kong_client = None
        
        # Ollama
        try:
            self.ollama_client = httpx.AsyncClient(
                base_url=f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}"
            )
            # Test connection
            response = await self.ollama_client.get("/api/tags")
            response.raise_for_status()
            logger.info("Ollama connection established")
        except Exception as e:
            logger.warning(f"Ollama connection failed (non-critical): {e}")
            self.ollama_client = None
    
    async def disconnect_all(self):
        """Close all service connections"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()
        
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        
        if self.faiss_client:
            await self.faiss_client.aclose()
        
        if self.consul_client:
            await self.consul_client.aclose()
        
        if self.kong_client:
            await self.kong_client.aclose()
        
        if self.ollama_client:
            await self.ollama_client.aclose()
        
        logger.info("All service connections closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        health = {
            "redis": False,
            "rabbitmq": False,
            "neo4j": False,
            "chromadb": False,
            "qdrant": False,
            "faiss": False,
            "consul": False,
            "kong": False,
            "ollama": False
        }
        
        # Redis
        try:
            await self.redis_client.ping()
            health["redis"] = True
        except:
            pass
        
        # RabbitMQ
        try:
            if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
                health["rabbitmq"] = True
        except:
            pass
        
        # Neo4j
        try:
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            health["neo4j"] = True
        except:
            pass
        
        # ChromaDB
        try:
            if self.chroma_client:
                # ChromaDB doesn't have async heartbeat, use sync
                heartbeat = self.chroma_client.heartbeat()
                health["chromadb"] = heartbeat is not None
            else:
                health["chromadb"] = False
        except Exception as e:
            logger.debug(f"ChromaDB health check failed: {e}")
            health["chromadb"] = False
        
        # Qdrant
        try:
            if self.qdrant_client:
                await self.qdrant_client.get_collections()
                health["qdrant"] = True
            else:
                health["qdrant"] = False
        except:
            health["qdrant"] = False
        
        # FAISS
        try:
            if self.faiss_client:
                response = await self.faiss_client.get("/health")
                health["faiss"] = response.status_code == 200
            else:
                health["faiss"] = False
        except:
            health["faiss"] = False
        
        # Consul
        try:
            if self.consul_client:
                response = await self.consul_client.get("/v1/status/leader")
                health["consul"] = response.status_code == 200
            else:
                health["consul"] = False
        except:
            health["consul"] = False
        
        # Kong
        try:
            if self.kong_client:
                response = await self.kong_client.get("/status")
                health["kong"] = response.status_code == 200
            else:
                health["kong"] = False
        except:
            health["kong"] = False
        
        # Ollama
        try:
            if self.ollama_client:
                response = await self.ollama_client.get("/api/tags")
                health["ollama"] = response.status_code == 200
            else:
                health["ollama"] = False
        except:
            health["ollama"] = False
        
        return health


# Global instance
service_connections = ServiceConnections()