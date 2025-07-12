"""
Vector Memory System for SutazAI
Provides persistent memory storage with embeddings and semantic search capabilities
"""

import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import hashlib
from datetime import datetime
import uuid
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using basic similarity search")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available, using simple embeddings")

from .database import get_db_manager

logger = logging.getLogger(__name__)

class VectorMemory:
    """Advanced Vector Memory System with semantic search capabilities"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 vector_dimension: int = 384, index_path: str = "/opt/sutazaiapp/data/vector_index"):
        self.embedding_model_name = embedding_model
        self.vector_dimension = vector_dimension
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_manager = get_db_manager()
        self.embedding_model = None
        self.faiss_index = None
        self.id_mapping = {}  # Maps FAISS index positions to embedding IDs
        self.initialized = False
        
    async def initialize(self):
        """Initialize the vector memory system"""
        if self.initialized:
            return
            
        try:
            # Initialize embedding model
            await self._load_embedding_model()
            
            # Initialize FAISS index
            await self._load_faiss_index()
            
            # Load existing embeddings from database
            await self._load_existing_embeddings()
            
            self.initialized = True
            logger.info("Vector memory system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector memory: {e}")
            raise
    
    async def _load_embedding_model(self):
        """Load the sentence embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            else:
                logger.warning("Using simple embedding fallback")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def _load_faiss_index(self):
        """Load or create FAISS index"""
        try:
            if not FAISS_AVAILABLE:
                logger.warning("FAISS not available, using simple similarity search")
                return
                
            index_file = self.index_path / "vector_index.faiss"
            mapping_file = self.index_path / "id_mapping.json"
            
            if index_file.exists() and mapping_file.exists():
                # Load existing index
                self.faiss_index = faiss.read_index(str(index_file))
                
                with open(mapping_file, 'r') as f:
                    self.id_mapping = json.load(f)
                    
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Create new index
                self.faiss_index = faiss.IndexFlatIP(self.vector_dimension)  # Inner Product for cosine similarity
                self.id_mapping = {}
                logger.info("Created new FAISS index")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
    
    async def _load_existing_embeddings(self):
        """Load existing embeddings from database into FAISS index"""
        try:
            # This would be implemented when we have existing embeddings
            # For now, we start with an empty index
            pass
            
        except Exception as e:
            logger.error(f"Failed to load existing embeddings: {e}")
    
    async def _save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            if not FAISS_AVAILABLE or not self.faiss_index:
                return
                
            index_file = self.index_path / "vector_index.faiss"
            mapping_file = self.index_path / "id_mapping.json"
            
            faiss.write_index(self.faiss_index, str(index_file))
            
            with open(mapping_file, 'w') as f:
                json.dump(self.id_mapping, f)
                
            logger.debug("Saved FAISS index to disk")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            if self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                return embedding.astype(np.float32)
            else:
                # Simple fallback: hash-based embedding
                return self._simple_embedding(text)
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding using text hash"""
        # Create a simple embedding based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to vector of appropriate dimension
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8)
        
        # Pad or truncate to match vector dimension
        if len(embedding) > self.vector_dimension:
            embedding = embedding[:self.vector_dimension]
        else:
            padding = np.zeros(self.vector_dimension - len(embedding), dtype=np.uint8)
            embedding = np.concatenate([embedding, padding])
        
        # Normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def store_memory(self, memory_id: str, content: str, source_type: str = "general",
                          source_id: str = None, metadata: Dict = None) -> bool:
        """Store content in vector memory"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Generate unique embedding ID if not provided
            if not memory_id:
                memory_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Store in database
            success = await self.db_manager.save_embedding(
                embedding_id=memory_id,
                source_type=source_type,
                source_id=source_id or memory_id,
                content=content,
                embedding_vector=embedding.tolist(),
                metadata=metadata
            )
            
            if not success:
                return False
            
            # Add to FAISS index if available
            if self.faiss_index and FAISS_AVAILABLE:
                embedding_reshaped = embedding.reshape(1, -1)
                self.faiss_index.add(embedding_reshaped)
                
                # Update ID mapping
                index_position = self.faiss_index.ntotal - 1
                self.id_mapping[str(index_position)] = memory_id
                
                # Save index periodically
                if self.faiss_index.ntotal % 100 == 0:
                    await self._save_faiss_index()
            
            logger.debug(f"Stored memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory_id}: {e}")
            return False
    
    async def search_memory(self, query: str, top_k: int = 5, 
                           source_type: str = None, similarity_threshold: float = 0.5) -> List[Dict]:
        """Search for similar memories"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            if self.faiss_index and FAISS_AVAILABLE and self.faiss_index.ntotal > 0:
                # Use FAISS for efficient search
                return await self._faiss_search(query_embedding, top_k, source_type, similarity_threshold)
            else:
                # Use database-based search
                return await self._database_search(query, query_embedding, top_k, source_type, similarity_threshold)
                
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    async def _faiss_search(self, query_embedding: np.ndarray, top_k: int,
                           source_type: str, similarity_threshold: float) -> List[Dict]:
        """Search using FAISS index"""
        try:
            query_reshaped = query_embedding.reshape(1, -1)
            
            # Search in FAISS
            similarities, indices = self.faiss_index.search(query_reshaped, min(top_k * 2, self.faiss_index.ntotal))
            
            results = []
            for similarity, index in zip(similarities[0], indices[0]):
                if similarity < similarity_threshold:
                    continue
                    
                # Get embedding ID from mapping
                embedding_id = self.id_mapping.get(str(index))
                if not embedding_id:
                    continue
                
                # Get full data from database
                async with self.db_manager.get_connection() as db:
                    cursor = await db.execute("""
                        SELECT * FROM vector_embeddings WHERE embedding_id = ?
                    """, (embedding_id,))
                    row = await cursor.fetchone()
                    
                    if row:
                        row_dict = dict(row)
                        
                        # Filter by source type if specified
                        if source_type and row_dict['source_type'] != source_type:
                            continue
                            
                        row_dict['similarity'] = float(similarity)
                        results.append(row_dict)
                        
                        if len(results) >= top_k:
                            break
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def _database_search(self, query: str, query_embedding: np.ndarray, 
                              top_k: int, source_type: str, similarity_threshold: float) -> List[Dict]:
        """Search using database with similarity calculation"""
        try:
            async with self.db_manager.get_connection() as db:
                # Get all embeddings (filtered by source type if specified)
                if source_type:
                    cursor = await db.execute("""
                        SELECT * FROM vector_embeddings WHERE source_type = ?
                    """, (source_type,))
                else:
                    cursor = await db.execute("SELECT * FROM vector_embeddings")
                
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    row_dict = dict(row)
                    
                    # Calculate similarity
                    if row_dict['embedding_vector']:
                        stored_embedding = np.array(json.loads(row_dict['embedding_vector']))
                        similarity = np.dot(query_embedding, stored_embedding)
                        
                        if similarity >= similarity_threshold:
                            row_dict['similarity'] = float(similarity)
                            results.append(row_dict)
                
                # Sort by similarity and return top_k
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get specific memory by ID"""
        try:
            async with self.db_manager.get_connection() as db:
                cursor = await db.execute("""
                    SELECT * FROM vector_embeddings WHERE embedding_id = ?
                """, (memory_id,))
                row = await cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def update_memory(self, memory_id: str, content: str = None, 
                           metadata: Dict = None) -> bool:
        """Update existing memory"""
        try:
            # Get existing memory
            existing = await self.get_memory(memory_id)
            if not existing:
                return False
            
            # Update content and regenerate embedding if content changed
            if content and content != existing['content']:
                embedding = self._generate_embedding(content)
                
                async with self.db_manager.get_connection() as db:
                    await db.execute("""
                        UPDATE vector_embeddings 
                        SET content = ?, embedding_vector = ?, metadata = ?
                        WHERE embedding_id = ?
                    """, (content, json.dumps(embedding.tolist()), 
                          json.dumps(metadata or {}), memory_id))
                    await db.commit()
                
                # Update FAISS index (rebuild required for updates)
                await self._rebuild_faiss_index()
                
            elif metadata:
                # Update only metadata
                async with self.db_manager.get_connection() as db:
                    await db.execute("""
                        UPDATE vector_embeddings 
                        SET metadata = ?
                        WHERE embedding_id = ?
                    """, (json.dumps(metadata), memory_id))
                    await db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID"""
        try:
            async with self.db_manager.get_connection() as db:
                await db.execute("""
                    DELETE FROM vector_embeddings WHERE embedding_id = ?
                """, (memory_id,))
                await db.commit()
            
            # Rebuild FAISS index to remove deleted embedding
            await self._rebuild_faiss_index()
            
            logger.debug(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def _rebuild_faiss_index(self):
        """Rebuild FAISS index from database"""
        try:
            if not FAISS_AVAILABLE:
                return
                
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.vector_dimension)
            self.id_mapping = {}
            
            # Load all embeddings from database
            async with self.db_manager.get_connection() as db:
                cursor = await db.execute("SELECT embedding_id, embedding_vector FROM vector_embeddings")
                rows = await cursor.fetchall()
                
                embeddings = []
                embedding_ids = []
                
                for row in rows:
                    embedding_id, embedding_vector_str = row
                    if embedding_vector_str:
                        embedding = np.array(json.loads(embedding_vector_str), dtype=np.float32)
                        embeddings.append(embedding)
                        embedding_ids.append(embedding_id)
                
                if embeddings:
                    # Add all embeddings to index
                    embeddings_matrix = np.vstack(embeddings)
                    self.faiss_index.add(embeddings_matrix)
                    
                    # Update ID mapping
                    for i, embedding_id in enumerate(embedding_ids):
                        self.id_mapping[str(i)] = embedding_id
                    
                    # Save updated index
                    await self._save_faiss_index()
                    
                    logger.info(f"Rebuilt FAISS index with {len(embeddings)} embeddings")
                
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    async def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        try:
            async with self.db_manager.get_connection() as db:
                # Total memories
                cursor = await db.execute("SELECT COUNT(*) FROM vector_embeddings")
                total_memories = (await cursor.fetchone())[0]
                
                # Memories by source type
                cursor = await db.execute("""
                    SELECT source_type, COUNT(*) as count
                    FROM vector_embeddings
                    GROUP BY source_type
                """)
                source_stats = [dict(row) for row in await cursor.fetchall()]
                
                # Memory size estimation
                cursor = await db.execute("""
                    SELECT AVG(LENGTH(content)) as avg_content_size,
                           AVG(LENGTH(embedding_vector)) as avg_embedding_size
                    FROM vector_embeddings
                """)
                size_stats = dict(await cursor.fetchone())
                
                return {
                    'total_memories': total_memories,
                    'source_type_distribution': source_stats,
                    'average_content_size': size_stats.get('avg_content_size', 0),
                    'average_embedding_size': size_stats.get('avg_embedding_size', 0),
                    'faiss_available': FAISS_AVAILABLE,
                    'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
                    'embedding_model': self.embedding_model_name,
                    'vector_dimension': self.vector_dimension
                }
                
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    async def clear_memories(self, source_type: str = None) -> bool:
        """Clear memories (all or by source type)"""
        try:
            async with self.db_manager.get_connection() as db:
                if source_type:
                    await db.execute("""
                        DELETE FROM vector_embeddings WHERE source_type = ?
                    """, (source_type,))
                else:
                    await db.execute("DELETE FROM vector_embeddings")
                
                await db.commit()
            
            # Rebuild FAISS index
            await self._rebuild_faiss_index()
            
            logger.info(f"Cleared memories" + (f" for source type: {source_type}" if source_type else ""))
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False
    
    async def get_memories_by_source(self, source_type: str, source_id: str = None) -> List[Dict]:
        """Get all memories for a specific source"""
        try:
            async with self.db_manager.get_connection() as db:
                if source_id:
                    cursor = await db.execute("""
                        SELECT * FROM vector_embeddings 
                        WHERE source_type = ? AND source_id = ?
                        ORDER BY created_at DESC
                    """, (source_type, source_id))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM vector_embeddings 
                        WHERE source_type = ?
                        ORDER BY created_at DESC
                    """, (source_type,))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get memories by source: {e}")
            return []
    
    def get_memory_count(self) -> int:
        """Get total memory count (synchronous for UI)"""
        try:
            import sqlite3
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM vector_embeddings")
                return cursor.fetchone()[0]
        except:
            return 0

# Global vector memory instance
vector_memory = VectorMemory()

async def init_vector_memory():
    """Initialize the global vector memory instance"""
    await vector_memory.initialize()

def get_vector_memory() -> VectorMemory:
    """Get the global vector memory instance"""
    return vector_memory