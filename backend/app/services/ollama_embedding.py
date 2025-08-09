"""
Ollama Embedding Service
Provides embedding generation using TinyLlama model
"""

import httpx
import logging
from typing import List, Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class OllamaEmbeddingService:
    """Service for generating embeddings using Ollama"""
    
    def __init__(self, base_url: str = "http://sutazai-ollama:11434"):
        self.base_url = base_url
        self.model = "tinyllama"
        self.client = None
    
    async def initialize(self):
        """Initialize the embedding service"""
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Test connection and model availability
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                if self.model in model_names:
                    logger.info(f"Ollama embedding service initialized with {self.model}")
                    return True
                else:
                    logger.error(f"Model {self.model} not found in Ollama. Available: {model_names}")
                    return False
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Ollama embedding service initialization failed: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using TinyLlama
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding, or None if failed
        """
        if not self.client:
            await self.initialize()
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Use Ollama embeddings endpoint
            request_data = {
                "model": self.model,
                "prompt": text.strip()
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding")
                
                if embedding and isinstance(embedding, list):
                    logger.debug(f"Generated embedding of size {len(embedding)}")
                    return embedding
                else:
                    logger.error("Invalid embedding format from Ollama")
                    return None
            else:
                logger.error(f"Ollama embedding failed: {response.status_code} - {response.text}")
                return None
                
        except httpx.TimeoutException:
            logger.error("Ollama embedding request timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (or None for failed ones)
        """
        if not texts:
            return []
        
        # Process in parallel with semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def embed_with_semaphore(text):
            async with semaphore:
                return await self.generate_embedding(text)
        
        tasks = [embed_with_semaphore(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, result in enumerate(embeddings):
            if isinstance(result, Exception):
                logger.error(f"Error embedding text {i}: {result}")
                results.append(None)
            else:
                results.append(result)
        
        return results
    
    async def similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1, or None if failed
        """
        embeddings = await self.generate_embeddings_batch([text1, text2])
        
        if len(embeddings) != 2 or None in embeddings:
            logger.error("Failed to generate embeddings for similarity calculation")
            return None
        
        try:
            import numpy as np
            
            # Convert to numpy arrays
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of the embedding service
        
        Returns:
            Health status dictionary
        """
        status = {
            "service": "ollama_embedding",
            "status": "unhealthy",
            "model": self.model,
            "base_url": self.base_url,
            "details": {}
        }
        
        try:
            if not self.client:
                await self.initialize()
            
            # Test basic connectivity
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                status["details"]["available_models"] = model_names
                status["details"]["connectivity"] = "ok"
                
                if self.model in model_names:
                    # Test embedding generation
                    test_embedding = await self.generate_embedding("test")
                    
                    if test_embedding:
                        status["status"] = "healthy"
                        status["details"]["embedding_size"] = len(test_embedding)
                        status["details"]["test_embedding"] = "success"
                    else:
                        status["details"]["test_embedding"] = "failed"
                else:
                    status["details"]["model_available"] = False
            else:
                status["details"]["connectivity"] = f"error_{response.status_code}"
                
        except Exception as e:
            status["details"]["error"] = str(e)
        
        return status
    
    async def shutdown(self):
        """Shutdown the embedding service"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Ollama embedding service shutdown")

# Global service instance
_embedding_service = None

async def get_ollama_embedding_service() -> OllamaEmbeddingService:
    """Get or create the global Ollama embedding service"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = OllamaEmbeddingService()
        await _embedding_service.initialize()
    
    return _embedding_service