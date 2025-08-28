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
from typing import Optional
import logging
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
        # ChromaDB
        try:
            # Try simple connection without auth first
            self.chroma_client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT
            )
            logger.info("ChromaDB connection established")
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {e}")
            # Continue without ChromaDB for now
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
        
        # Kong Admin API
        try:
            self.kong_client = httpx.AsyncClient(
                base_url=f"http://{settings.KONG_HOST}:{settings.KONG_ADMIN_PORT}"
            )
            # Test connection
            response = await self.kong_client.get("/status")
            response.raise_for_status()
            logger.info("Kong Admin API connection established")
        except Exception as e:
            logger.warning(f"Kong connection failed (non-critical): {e}")
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
    
    async def health_check(self) -> dict:
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
            # ChromaDB doesn't have async heartbeat, use sync
            self.chroma_client.heartbeat()
            health["chromadb"] = True
        except:
            pass
        
        # Qdrant
        try:
            await self.qdrant_client.get_collections()
            health["qdrant"] = True
        except:
            pass
        
        # FAISS
        try:
            response = await self.faiss_client.get("/health")
            health["faiss"] = response.status_code == 200
        except:
            pass
        
        # Consul
        try:
            response = await self.consul_client.get("/v1/status/leader")
            health["consul"] = response.status_code == 200
        except:
            pass
        
        # Kong
        try:
            response = await self.kong_client.get("/status")
            health["kong"] = response.status_code == 200
        except:
            pass
        
        # Ollama
        try:
            response = await self.ollama_client.get("/api/tags")
            health["ollama"] = response.status_code == 200
        except:
            pass
        
        return health


# Global instance
service_connections = ServiceConnections()