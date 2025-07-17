#!/usr/bin/env python3
"""
Model Registry - Central repository for managing local AI models
Provides version control, metadata management, and model discovery
"""

import json
import os
import hashlib
import shutil
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model type enumeration"""
    LANGUAGE_MODEL = "language_model"
    EMBEDDING_MODEL = "embedding_model"
    VISION_MODEL = "vision_model"
    AUDIO_MODEL = "audio_model"
    MULTIMODAL_MODEL = "multimodal_model"
    CUSTOM_MODEL = "custom_model"

class ModelFormat(Enum):
    """Model format enumeration"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    LLAMACPP = "llamacpp"
    GGUF = "gguf"
    SAFETENSORS = "safetensors"

class ModelStatus(Enum):
    """Model status enumeration"""
    AVAILABLE = "available"
    LOADING = "loading"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    DEPRECATED = "deprecated"

@dataclass
class ModelMetadata:
    """Model metadata container"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    model_format: ModelFormat
    description: str
    file_path: str
    file_size: int
    file_hash: str
    parameters: int
    quantization: Optional[str] = None
    architecture: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    tags: List[str] = None
    capabilities: List[str] = None
    requirements: Dict[str, str] = None
    performance_metrics: Dict[str, Any] = None
    status: ModelStatus = ModelStatus.AVAILABLE
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
        if self.tags is None:
            self.tags = []
        if self.capabilities is None:
            self.capabilities = []
        if self.requirements is None:
            self.requirements = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    model_id: str
    file_path: str
    file_hash: str
    created_at: datetime
    changelog: str = ""
    is_active: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

class ModelRegistry:
    """
    Central model registry for managing local AI models
    Provides version control, metadata management, and model discovery
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".sutazai" / "models"
        self.registry_db = self.storage_path / "registry.db"
        self.models_dir = self.storage_path / "models"
        self.cache_dir = self.storage_path / "cache"
        self.temp_dir = self.storage_path / "temp"
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Model registry initialized at {self.storage_path}")
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.registry_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_format TEXT NOT NULL,
                    description TEXT,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT,
                    parameters INTEGER,
                    quantization TEXT,
                    architecture TEXT,
                    license TEXT,
                    author TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    tags TEXT,
                    capabilities TEXT,
                    requirements TEXT,
                    performance_metrics TEXT,
                    status TEXT DEFAULT 'available'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    changelog TEXT,
                    is_active BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_type ON models(model_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_status ON models(status)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_versions ON model_versions(model_id, version)
            ''')
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _serialize_list(self, data: List[str]) -> str:
        """Serialize list to JSON string"""
        return json.dumps(data) if data else "[]"
    
    def _deserialize_list(self, data: str) -> List[str]:
        """Deserialize JSON string to list"""
        try:
            return json.loads(data) if data else []
        except json.JSONDecodeError:
            return []
    
    def _serialize_dict(self, data: Dict[str, Any]) -> str:
        """Serialize dictionary to JSON string"""
        return json.dumps(data) if data else "{}"
    
    def _deserialize_dict(self, data: str) -> Dict[str, Any]:
        """Deserialize JSON string to dictionary"""
        try:
            return json.loads(data) if data else {}
        except json.JSONDecodeError:
            return {}
    
    async def register_model(self, model_metadata: ModelMetadata, 
                           model_file: Union[str, Path], 
                           copy_file: bool = True) -> bool:
        """
        Register a new model in the registry
        
        Args:
            model_metadata: Model metadata
            model_file: Path to model file
            copy_file: Whether to copy the file to registry storage
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            with self._lock:
                model_file = Path(model_file)
                
                if not model_file.exists():
                    logger.error(f"Model file not found: {model_file}")
                    return False
                
                # Calculate file hash
                file_hash = self._calculate_file_hash(model_file)
                model_metadata.file_hash = file_hash
                model_metadata.file_size = model_file.stat().st_size
                
                # Determine storage path
                if copy_file:
                    # Copy file to registry storage
                    storage_path = self.models_dir / model_metadata.model_id
                    storage_path.mkdir(parents=True, exist_ok=True)
                    
                    target_file = storage_path / f"{model_metadata.version}_{model_file.name}"
                    shutil.copy2(model_file, target_file)
                    model_metadata.file_path = str(target_file)
                else:
                    # Store reference to original file
                    model_metadata.file_path = str(model_file.absolute())
                
                # Store in database
                with sqlite3.connect(self.registry_db) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO models (
                            model_id, name, version, model_type, model_format,
                            description, file_path, file_size, file_hash,
                            parameters, quantization, architecture, license,
                            author, created_at, updated_at, tags, capabilities,
                            requirements, performance_metrics, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        model_metadata.model_id,
                        model_metadata.name,
                        model_metadata.version,
                        model_metadata.model_type.value,
                        model_metadata.model_format.value,
                        model_metadata.description,
                        model_metadata.file_path,
                        model_metadata.file_size,
                        model_metadata.file_hash,
                        model_metadata.parameters,
                        model_metadata.quantization,
                        model_metadata.architecture,
                        model_metadata.license,
                        model_metadata.author,
                        model_metadata.created_at.isoformat(),
                        model_metadata.updated_at.isoformat(),
                        self._serialize_list(model_metadata.tags),
                        self._serialize_list(model_metadata.capabilities),
                        self._serialize_dict(model_metadata.requirements),
                        self._serialize_dict(model_metadata.performance_metrics),
                        model_metadata.status.value
                    ))
                    
                    # Add version record
                    conn.execute('''
                        INSERT INTO model_versions (
                            model_id, version, file_path, file_hash,
                            created_at, is_active
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        model_metadata.model_id,
                        model_metadata.version,
                        model_metadata.file_path,
                        model_metadata.file_hash,
                        model_metadata.created_at.isoformat(),
                        True
                    ))
                
                logger.info(f"Model registered: {model_metadata.model_id} v{model_metadata.version}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        try:
            with sqlite3.connect(self.registry_db) as conn:
                cursor = conn.execute('''
                    SELECT * FROM models WHERE model_id = ?
                ''', (model_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))
                
                return ModelMetadata(
                    model_id=data['model_id'],
                    name=data['name'],
                    version=data['version'],
                    model_type=ModelType(data['model_type']),
                    model_format=ModelFormat(data['model_format']),
                    description=data['description'],
                    file_path=data['file_path'],
                    file_size=data['file_size'],
                    file_hash=data['file_hash'],
                    parameters=data['parameters'],
                    quantization=data['quantization'],
                    architecture=data['architecture'],
                    license=data['license'],
                    author=data['author'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at']),
                    tags=self._deserialize_list(data['tags']),
                    capabilities=self._deserialize_list(data['capabilities']),
                    requirements=self._deserialize_dict(data['requirements']),
                    performance_metrics=self._deserialize_dict(data['performance_metrics']),
                    status=ModelStatus(data['status'])
                )
                
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    async def list_models(self, 
                         model_type: Optional[ModelType] = None,
                         status: Optional[ModelStatus] = None,
                         tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        try:
            query = "SELECT * FROM models WHERE 1=1"
            params = []
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            models = []
            with sqlite3.connect(self.registry_db) as conn:
                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                
                for row in cursor.fetchall():
                    data = dict(zip(columns, row))
                    
                    # Tag filtering
                    if tags:
                        model_tags = self._deserialize_list(data['tags'])
                        if not any(tag in model_tags for tag in tags):
                            continue
                    
                    model = ModelMetadata(
                        model_id=data['model_id'],
                        name=data['name'],
                        version=data['version'],
                        model_type=ModelType(data['model_type']),
                        model_format=ModelFormat(data['model_format']),
                        description=data['description'],
                        file_path=data['file_path'],
                        file_size=data['file_size'],
                        file_hash=data['file_hash'],
                        parameters=data['parameters'],
                        quantization=data['quantization'],
                        architecture=data['architecture'],
                        license=data['license'],
                        author=data['author'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        updated_at=datetime.fromisoformat(data['updated_at']),
                        tags=self._deserialize_list(data['tags']),
                        capabilities=self._deserialize_list(data['capabilities']),
                        requirements=self._deserialize_dict(data['requirements']),
                        performance_metrics=self._deserialize_dict(data['performance_metrics']),
                        status=ModelStatus(data['status'])
                    )
                    models.append(model)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        try:
            with sqlite3.connect(self.registry_db) as conn:
                cursor = conn.execute('''
                    SELECT * FROM model_versions 
                    WHERE model_id = ?
                    ORDER BY created_at DESC
                ''', (model_id,))
                
                versions = []
                for row in cursor.fetchall():
                    columns = [desc[0] for desc in cursor.description]
                    data = dict(zip(columns, row))
                    
                    version = ModelVersion(
                        version=data['version'],
                        model_id=data['model_id'],
                        file_path=data['file_path'],
                        file_hash=data['file_hash'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        changelog=data.get('changelog', ''),
                        is_active=bool(data['is_active'])
                    )
                    versions.append(version)
                
                return versions
                
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        try:
            with sqlite3.connect(self.registry_db) as conn:
                cursor = conn.execute('''
                    UPDATE models SET status = ?, updated_at = ?
                    WHERE model_id = ?
                ''', (status.value, datetime.now(timezone.utc).isoformat(), model_id))
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            return False
    
    async def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """Delete a model from registry"""
        try:
            with self._lock:
                # Get model metadata
                model = await self.get_model(model_id)
                if not model:
                    logger.warning(f"Model not found: {model_id}")
                    return False
                
                # Remove files if requested
                if remove_files:
                    try:
                        file_path = Path(model.file_path)
                        if file_path.exists():
                            file_path.unlink()
                        
                        # Remove model directory if empty
                        model_dir = self.models_dir / model_id
                        if model_dir.exists() and not any(model_dir.iterdir()):
                            model_dir.rmdir()
                            
                    except Exception as e:
                        logger.warning(f"Error removing files: {e}")
                
                # Remove from database
                with sqlite3.connect(self.registry_db) as conn:
                    conn.execute('DELETE FROM model_versions WHERE model_id = ?', (model_id,))
                    conn.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
                
                logger.info(f"Model deleted: {model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    async def search_models(self, query: str, 
                          model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """Search models by name, description, or tags"""
        try:
            sql_query = '''
                SELECT * FROM models 
                WHERE (name LIKE ? OR description LIKE ? OR tags LIKE ?)
            '''
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if model_type:
                sql_query += " AND model_type = ?"
                params.append(model_type.value)
            
            models = []
            with sqlite3.connect(self.registry_db) as conn:
                cursor = conn.execute(sql_query, params)
                columns = [desc[0] for desc in cursor.description]
                
                for row in cursor.fetchall():
                    data = dict(zip(columns, row))
                    
                    model = ModelMetadata(
                        model_id=data['model_id'],
                        name=data['name'],
                        version=data['version'],
                        model_type=ModelType(data['model_type']),
                        model_format=ModelFormat(data['model_format']),
                        description=data['description'],
                        file_path=data['file_path'],
                        file_size=data['file_size'],
                        file_hash=data['file_hash'],
                        parameters=data['parameters'],
                        quantization=data['quantization'],
                        architecture=data['architecture'],
                        license=data['license'],
                        author=data['author'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        updated_at=datetime.fromisoformat(data['updated_at']),
                        tags=self._deserialize_list(data['tags']),
                        capabilities=self._deserialize_list(data['capabilities']),
                        requirements=self._deserialize_dict(data['requirements']),
                        performance_metrics=self._deserialize_dict(data['performance_metrics']),
                        status=ModelStatus(data['status'])
                    )
                    models.append(model)
            
            return models
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    async def validate_model_integrity(self, model_id: str) -> bool:
        """Validate model file integrity"""
        try:
            model = await self.get_model(model_id)
            if not model:
                return False
            
            file_path = Path(model.file_path)
            if not file_path.exists():
                await self.update_model_status(model_id, ModelStatus.MISSING)
                return False
            
            # Verify file hash
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != model.file_hash:
                await self.update_model_status(model_id, ModelStatus.CORRUPTED)
                return False
            
            # Update status to available if checks pass
            await self.update_model_status(model_id, ModelStatus.AVAILABLE)
            return True
            
        except Exception as e:
            logger.error(f"Error validating model integrity: {e}")
            await self.update_model_status(model_id, ModelStatus.CORRUPTED)
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            with sqlite3.connect(self.registry_db) as conn:
                # Total models
                cursor = conn.execute('SELECT COUNT(*) FROM models')
                total_models = cursor.fetchone()[0]
                
                # Models by type
                cursor = conn.execute('''
                    SELECT model_type, COUNT(*) 
                    FROM models 
                    GROUP BY model_type
                ''')
                models_by_type = dict(cursor.fetchall())
                
                # Models by status
                cursor = conn.execute('''
                    SELECT status, COUNT(*) 
                    FROM models 
                    GROUP BY status
                ''')
                models_by_status = dict(cursor.fetchall())
                
                # Storage usage
                cursor = conn.execute('SELECT SUM(file_size) FROM models')
                total_storage = cursor.fetchone()[0] or 0
                
                return {
                    'total_models': total_models,
                    'models_by_type': models_by_type,
                    'models_by_status': models_by_status,
                    'total_storage_bytes': total_storage,
                    'storage_path': str(self.storage_path)
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)

def create_model_registry(storage_path: str = None) -> ModelRegistry:
    """Factory function to create model registry"""
    return ModelRegistry(storage_path)