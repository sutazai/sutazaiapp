#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_downloader.py - Enterprise-grade model downloading and management system for SutazAI

This module provides automatic model downloading, version tracking, and configuration
for enterprise deployments on Dell PowerEdge R720 with E5-2640 CPUs.
"""

import os
import json
import logging
import hashlib
import sqlite3
import subprocess
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logger = logging.getLogger("sutazai.model_downloader")

# Constants
DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "models"
)
DEFAULT_DATABASE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "model_registry.db"
)
OLLAMA_MODELS = {
    "llama3-70b": {
        "model_name": "llama3:70b",
        "size_gb": 39.2,
        "memory_req_gb": 48,
        "source": "ollama",
        "quantization": "Q4_0",
        "sha256": None,  # Will be updated on first download
    },
    "llama3-8b": {
        "model_name": "llama3:8b",
        "size_gb": 4.8,
        "memory_req_gb": 8,
        "source": "ollama",
        "quantization": "Q4_0",
        "sha256": None,
    },
    "mistral-7b": {
        "model_name": "mistral:7b",
        "size_gb": 4.1,
        "memory_req_gb": 8,
        "source": "ollama",
        "quantization": "Q4_0",
        "sha256": None,
    },
    "llama3-chatqa": {
        "model_name": "llama3-chatqa",
        "size_gb": 4.7,
        "memory_req_gb": 8,
        "source": "ollama",
        "quantization": "Q4_0",
        "sha256": None,
    },
    "deepseek-coder": {
        "model_name": "deepseek-coder",
        "size_gb": 0.8,
        "memory_req_gb": 4,
        "source": "ollama",
        "quantization": "Q4_0",
        "sha256": None,
    },
}

# Startup and shutdown scripts
OLLAMA_START_SCRIPT = "/opt/sutazaiapp/bin/start_all.sh"
OLLAMA_STOP_SCRIPT = "/opt/sutazaiapp/bin/stop_all.sh"


@dataclass
class ModelVersion:
    """Model version information"""

    model_id: str
    version: str
    path: str
    download_date: datetime
    size_bytes: int
    sha256: str
    is_active: bool = True
    config: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Enterprise model registry for version tracking and management"""

    def __init__(self, db_path: str = DEFAULT_DATABASE_PATH):
        """Initialize the model registry with the database path"""
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create the database and tables if they don't exist"""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                download_date TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                config TEXT,
                UNIQUE(model_id, version)
            )
            """)
            conn.commit()

    def register_model(self, model: ModelVersion) -> Optional[int]:
        """Register a model version in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            INSERT OR REPLACE INTO models 
            (model_id, version, path, download_date, size_bytes, sha256, is_active, config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model.model_id,
                    model.version,
                    model.path,
                    model.download_date.isoformat(),
                    model.size_bytes,
                    model.sha256,
                    1 if model.is_active else 0,
                    json.dumps(model.config) if model.config else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_model(
        self, model_id: str, version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get a model by ID and optionally version"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if version:
                cursor.execute(
                    """
                SELECT * FROM models WHERE model_id = ? AND version = ?
                """,
                    (model_id, version),
                )
            else:
                cursor.execute(
                    """
                SELECT * FROM models WHERE model_id = ? AND is_active = 1
                ORDER BY download_date DESC LIMIT 1
                """,
                    (model_id,),
                )

            row = cursor.fetchone()
            if row:
                return ModelVersion(
                    model_id=row["model_id"],
                    version=row["version"],
                    path=row["path"],
                    download_date=datetime.fromisoformat(row["download_date"]),
                    size_bytes=row["size_bytes"],
                    sha256=row["sha256"],
                    is_active=bool(row["is_active"]),
                    config=json.loads(row["config"]) if row["config"] else None,
                )
            return None

    def list_models(self) -> List[str]:
        """List all model IDs in the registry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT model_id FROM models")
            return [row[0] for row in cursor.fetchall()]

    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a specific model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM models WHERE model_id = ? ORDER BY download_date DESC",
                (model_id,),
            )
            return [
                ModelVersion(
                    model_id=row["model_id"],
                    version=row["version"],
                    path=row["path"],
                    download_date=datetime.fromisoformat(row["download_date"]),
                    size_bytes=row["size_bytes"],
                    sha256=row["sha256"],
                    is_active=bool(row["is_active"]),
                    config=json.loads(row["config"]) if row["config"] else None,
                )
                for row in cursor.fetchall()
            ]

    def set_active_version(self, model_id: str, version: str) -> bool:
        """Set a specific version as the active version for a model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Set all versions of this model to inactive
            cursor.execute(
                "UPDATE models SET is_active = 0 WHERE model_id = ?", (model_id,)
            )

            # Set the specified version to active
            cursor.execute(
                """
            UPDATE models SET is_active = 1 
            WHERE model_id = ? AND version = ?
            """,
                (model_id, version),
            )

            conn.commit()
            return cursor.rowcount > 0

    def remove_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Remove a model from the registry, optionally a specific version"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if version:
                cursor.execute(
                    "DELETE FROM models WHERE model_id = ? AND version = ?",
                    (model_id, version),
                )
            else:
                cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))

            conn.commit()
            return cursor.rowcount > 0

    def update_model_config(
        self, model_id: str, version: str, config: Dict[str, Any]
    ) -> bool:
        """Update the configuration for a model version"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            UPDATE models SET config = ?
            WHERE model_id = ? AND version = ?
            """,
                (json.dumps(config), model_id, version),
            )
            conn.commit()
            return cursor.rowcount > 0


class EnterpriseModelDownloader:
    """Enterprise-grade model downloader using Ollama"""

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        registry: Optional[ModelRegistry] = None,
    ):
        """Initialize the model downloader"""
        self.models_dir = models_dir
        self.registry = registry or ModelRegistry()

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Check if ollama is installed
        try:
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("Ollama is not installed or not in PATH. Please install Ollama first.")

    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def download_from_ollama(self, model_name: str) -> bool:
        """Download a model using Ollama CLI"""
        try:
            logger.info(f"Downloading {model_name} using Ollama")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Successfully downloaded {model_name}")
            logger.debug(result.stdout)
            
            # Ensure the model is included in the startup script
            self._update_startup_scripts(model_name)
            
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to download model {model_name} using Ollama: {e}")
            if hasattr(e, 'stderr'):
                logger.error(f"Ollama error: {e.stderr}")
            return False

    def _update_startup_scripts(self, model_name: str) -> None:
        """Update startup and shutdown scripts to include the model"""
        try:
            # Ensure start script exists
            if not os.path.exists(os.path.dirname(OLLAMA_START_SCRIPT)):
                os.makedirs(os.path.dirname(OLLAMA_START_SCRIPT), exist_ok=True)
            
            # Create or update start script
            if not os.path.exists(OLLAMA_START_SCRIPT):
                with open(OLLAMA_START_SCRIPT, 'w') as f:
                    f.write("#!/bin/bash\n\n")
                    f.write("# Auto-generated by SutazAI Model Downloader\n")
                    f.write("# Start all Ollama models\n\n")
                    f.write(f"ollama serve &\n")
                    f.write(f"sleep 5\n")
                    f.write(f"ollama run {model_name} &\n")
                os.chmod(OLLAMA_START_SCRIPT, 0o755)
            else:
                # Check if model is already in the script
                with open(OLLAMA_START_SCRIPT, 'r') as f:
                    content = f.read()
                
                if f"ollama run {model_name}" not in content:
                    with open(OLLAMA_START_SCRIPT, 'a') as f:
                        f.write(f"ollama run {model_name} &\n")
            
            # Create or update stop script
            if not os.path.exists(OLLAMA_STOP_SCRIPT):
                with open(OLLAMA_STOP_SCRIPT, 'w') as f:
                    f.write("#!/bin/bash\n\n")
                    f.write("# Auto-generated by SutazAI Model Downloader\n")
                    f.write("# Stop all Ollama processes\n\n")
                    f.write("pkill -f 'ollama'\n")
                os.chmod(OLLAMA_STOP_SCRIPT, 0o755)
            
            logger.info(f"Updated startup/shutdown scripts to include {model_name}")
        except Exception as e:
            logger.error(f"Failed to update startup scripts: {e}")

    def get_model(
        self, model_id: str, force_download: bool = False, version: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a model by ID, downloading it if necessary.

        Args:
            model_id: Identifier for the model (e.g., "llama3-70b")
            force_download: Whether to force a new download even if it exists
            version: Specific version to get, or None for the latest

        Returns:
            Path to the model, or None if failed
        """
        # Check if model exists in registry
        if not force_download:
            model_version = self.registry.get_model(model_id, version)
            if model_version:
                logger.info(
                    f"Using existing model: {model_id} ({model_version.version})"
                )
                return model_version.path

        # Look up model in catalog
        if model_id not in OLLAMA_MODELS:
            logger.error(f"Model {model_id} not found in catalog")
            return None

        model_info = OLLAMA_MODELS[model_id]
        
        # Download using Ollama
        if model_info["source"] == "ollama":
            if not self.download_from_ollama(model_info["model_name"]):
                logger.error(f"Failed to download {model_id}")
                return None
            
            # Ollama stores models in its own directory, so we'll just reference it
            # The actual path is managed by Ollama
            ollama_path = f"ollama://{model_info['model_name']}"
            
            # Get version information
            effective_version = datetime.now().strftime("%Y%m%d")
            if version:
                effective_version = version

            # Register in registry
            model_entry = ModelVersion(
                model_id=model_id,
                version=effective_version,
                path=ollama_path,
                download_date=datetime.now(),
                size_bytes=int(float(model_info["size_gb"]) * 1024 * 1024 * 1024),  # Cast to float first, then int
                sha256="managed_by_ollama",  # Ollama manages integrity
                is_active=True,
                config={
                    "model_name": model_info["model_name"],
                    "quantization": model_info["quantization"],
                    "memory_req_gb": model_info["memory_req_gb"],
                },
            )

            self.registry.register_model(model_entry)
            logger.info(f"Registered {model_id} ({effective_version}) in registry")

            return ollama_path
        else:
            logger.error(f"Unsupported source type: {model_info['source']}")
            return None

    def get_compatible_model(self, available_memory_gb: float) -> Optional[str]:
        """
        Get the largest model that fits in the available memory.

        Args:
            available_memory_gb: Available memory in GB

        Returns:
            Model ID of a compatible model, or None if none fits
        """
        compatible_models: List[Tuple[str, float]] = []

        for model_id, info in OLLAMA_MODELS.items():
            memory_req = info.get("memory_req_gb")
            # Check if memory_req is a number and meets criteria
            # Cast memory_req to float for comparison
            if isinstance(memory_req, (int, float)) and float(memory_req) <= available_memory_gb:
                compatible_models.append((model_id, float(memory_req)))

        if not compatible_models:
            logger.warning(
                f"No models compatible with {available_memory_gb}GB available memory"
            )
            return None

        # Sort by memory requirements (descending) and return the largest
        compatible_models.sort(key=lambda x: x[1], reverse=True)
        return compatible_models[0][0]

    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """
        Delete a model from Ollama and registry.

        Args:
            model_id: ID of the model to delete
            version: Specific version to delete, or None for all versions

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_id not in OLLAMA_MODELS:
                logger.error(f"Model {model_id} not found in catalog")
                return False
                
            model_info = OLLAMA_MODELS[model_id]
            model_name = model_info["model_name"]
            
            # Delete from Ollama
            try:
                subprocess.run(
                    ["ollama", "rm", model_name],
                    check=True,
                    capture_output=True
                )
                logger.info(f"Deleted {model_name} from Ollama")
            except subprocess.SubprocessError as e:
                logger.error(f"Failed to delete model {model_name} from Ollama: {e}")
                return False
            
            # Delete from registry
            if version:
                self.registry.remove_model(model_id, version)
            else:
                self.registry.remove_model(model_id)
                
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False


def get_available_memory_gb() -> float:
    """Get available system memory in GB"""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        logger.warning("psutil not installed, using default memory estimation")
        return 8.0  # Default assumption


def get_optimal_model_for_e5_2640() -> str:
    """Get optimal model for Dell PowerEdge R720 with E5-2640 CPUs"""
    available_memory = get_available_memory_gb()

    # Account for E5-2640 limitations - typically 64-128GB RAM on R720
    # Recommend model based on typical deployment
    if available_memory >= 50:
        return "llama3-70b"
    elif available_memory >= 16:
        return "llama3-8b"
    else:
        return "mistral-7b"


def ensure_model_downloaded(model_id: str, force: bool = False) -> Optional[str]:
    """
    Enterprise-level function to ensure a model is downloaded and ready to use.

    Args:
        model_id: ID of the model to download
        force: Whether to force a fresh download

    Returns:
        Path to the model, or None if failed
    """
    downloader = EnterpriseModelDownloader()
    return downloader.get_model(model_id, force_download=force)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Enterprise Model Downloader")
    parser.add_argument("--model", type=str, help="Model ID to download")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--force", action="store_true", help="Force download even if model exists"
    )
    args = parser.parse_args()

    downloader = EnterpriseModelDownloader()

    if args.list:
        print("Available models:")
        for model_id, info in OLLAMA_MODELS.items():
            print(
                f"  {model_id}: {info['model_name']} ({info['size_gb']:.1f} GB)"
            )

        print("\nDownloaded models:")
        for model_id in downloader.registry.list_models():
            versions = downloader.registry.list_versions(model_id)
            if versions:
                active = next((v for v in versions if v.is_active), None)
                if active:
                    print(
                        f"  {model_id}: {active.version} - {active.path} ({float(active.size_bytes) / (1024**3):.1f} GB)"
                    )
                else:
                    print(
                        f"  {model_id}: {len(versions)} versions available (none active)"
                    )

    elif args.model:
        print(f"Downloading {args.model}...")
        path = downloader.get_model(args.model, force_download=args.force)
        if path:
            print(f"Model downloaded to: {path}")
        else:
            print(f"Failed to download {args.model}")

    else:
        optimal_model = get_optimal_model_for_e5_2640()
        print(f"Recommended model for E5-2640: {optimal_model}")
        print("Use --model MODEL_ID to download a specific model")
        print("Use --list to view available models")
