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
import concurrent.futures
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import requests
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download, login, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Set up logging
logger = logging.getLogger("sutazai.model_downloader")

# Constants
DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "models"
)
DEFAULT_DATABASE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "model_registry.db"
)
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for parallel download
DEFAULT_MAX_WORKERS = 4
HUGGINGFACE_MODELS = {
    "llama3-70b": {
        "repo_id": "TheBloke/Llama-3-70B-GGUF",
        "filename": "llama-3-70b.Q4_0.gguf",
        "size_gb": 39.2,
        "memory_req_gb": 48,
        "source": "huggingface",
        "quantization": "Q4_0",
        "sha256": None,  # Will be updated on first download
    },
    "llama3-8b": {
        "repo_id": "TheBloke/Llama-3-8B-GGUF",
        "filename": "llama-3-8b.Q4_0.gguf",
        "size_gb": 4.8,
        "memory_req_gb": 8,
        "source": "huggingface",
        "quantization": "Q4_0",
        "sha256": None,
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-v0.1-GGUF",
        "filename": "mistral-7b-v0.1.Q4_0.gguf",
        "size_gb": 4.1,
        "memory_req_gb": 8,
        "source": "huggingface",
        "quantization": "Q4_0",
        "sha256": None,
    },
}


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
    config: Dict[str, Any] = None


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

    def register_model(self, model: ModelVersion) -> int:
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
    """Enterprise-grade model downloader with parallel processing and integrity checking"""

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        registry: Optional[ModelRegistry] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        hf_token: Optional[str] = None,
    ):
        """Initialize the model downloader"""
        self.models_dir = models_dir
        self.registry = registry or ModelRegistry()
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.hf_token = hf_token

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize HuggingFace if available
        if HF_AVAILABLE and hf_token:
            try:
                login(token=hf_token)
                logger.info("Logged into Hugging Face Hub")
            except Exception as e:
                logger.warning(f"Failed to log into Hugging Face Hub: {e}")

    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _download_chunk(
        self,
        url: str,
        start_byte: int,
        end_byte: int,
        output_file: str,
        position: int,
        progress_bar: tqdm,
    ) -> bool:
        """Download a specific chunk of a file"""
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_file, "r+b") as f:
                f.seek(position)
                f.write(response.content)

            progress_bar.update(end_byte - start_byte + 1)
            return True
        except Exception as e:
            logger.error(f"Failed to download chunk {start_byte}-{end_byte}: {e}")
            return False

    def download_file_parallel(self, url: str, output_path: str) -> bool:
        """Download a file in parallel chunks for improved performance"""
        try:
            # Get file size
            response = requests.head(url, timeout=10)
            response.raise_for_status()
            file_size = int(response.headers.get("content-length", 0))

            if file_size == 0:
                logger.error("Failed to determine file size")
                return False

            # Create progress bar
            progress_bar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {os.path.basename(output_path)}",
            )

            # Create empty file of the required size
            with open(output_path, "wb") as f:
                f.seek(file_size - 1)
                f.write(b"\0")

            # Calculate chunk sizes
            chunk_size = self.chunk_size
            num_chunks = file_size // chunk_size + (
                1 if file_size % chunk_size > 0 else 0
            )

            # Download chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = []
                for i in range(num_chunks):
                    start_byte = i * chunk_size
                    end_byte = min(start_byte + chunk_size - 1, file_size - 1)
                    position = start_byte

                    futures.append(
                        executor.submit(
                            self._download_chunk,
                            url,
                            start_byte,
                            end_byte,
                            output_path,
                            position,
                            progress_bar,
                        )
                    )

                # Check if all chunks were downloaded successfully
                for future in concurrent.futures.as_completed(futures):
                    if not future.result():
                        logger.error("Failed to download one or more chunks")
                        return False

            progress_bar.close()
            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False

    def download_from_huggingface(
        self, repo_id: str, filename: str, output_dir: str, revision: str = "main"
    ) -> Optional[str]:
        """Download a model from the Hugging Face Hub"""
        if not HF_AVAILABLE:
            logger.error("huggingface_hub package not installed")
            return None

        try:
            logger.info(f"Downloading {filename} from {repo_id}")

            # Get the download URL
            api = HfApi()
            model_info = api.model_info(repo_id, revision=revision)
            file_info = next(
                (f for f in model_info.siblings if f.rfilename == filename), None
            )

            if file_info is None:
                logger.error(f"File {filename} not found in {repo_id}")
                return None

            # Create full output path
            output_path = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)

            # Download the file
            if self.download_file_parallel(file_info.download_url, output_path):
                logger.info(f"Successfully downloaded {filename}")
                return output_path
            else:
                # Fallback to standard HF download
                logger.info("Falling back to standard Hugging Face download")
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                    token=self.hf_token,
                )

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.error(f"Repository or revision not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            return None

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
            Path to the model file, or None if failed
        """
        # Check if model exists in registry
        if not force_download:
            model_version = self.registry.get_model(model_id, version)
            if model_version and os.path.exists(model_version.path):
                logger.info(
                    f"Using existing model: {model_id} ({model_version.version})"
                )
                return model_version.path

        # Look up model in catalog
        if model_id not in HUGGINGFACE_MODELS:
            logger.error(f"Model {model_id} not found in catalog")
            return None

        model_info = HUGGINGFACE_MODELS[model_id]
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Download based on source
        if model_info["source"] == "huggingface":
            output_path = self.download_from_huggingface(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                output_dir=model_dir,
            )

            if not output_path:
                logger.error(f"Failed to download {model_id}")
                return None

            # Calculate SHA256 for integrity verification
            sha256 = self._calculate_sha256(output_path)

            # If we have a known SHA256, verify it
            if model_info["sha256"] and sha256 != model_info["sha256"]:
                logger.error(
                    f"SHA256 mismatch for {model_id}. Expected {model_info['sha256']}, got {sha256}"
                )
                # Consider renaming or removing the downloaded file
                return None
            elif not model_info["sha256"]:
                # Update the catalog with the SHA256
                model_info["sha256"] = sha256
                logger.info(f"Updated SHA256 for {model_id}: {sha256}")

            # Get version information
            model_version = datetime.now().strftime("%Y%m%d")
            if version:
                model_version = version

            # Register in registry
            model_entry = ModelVersion(
                model_id=model_id,
                version=model_version,
                path=output_path,
                download_date=datetime.now(),
                size_bytes=os.path.getsize(output_path),
                sha256=sha256,
                is_active=True,
                config={
                    "repo_id": model_info["repo_id"],
                    "filename": model_info["filename"],
                    "quantization": model_info["quantization"],
                    "memory_req_gb": model_info["memory_req_gb"],
                },
            )

            self.registry.register_model(model_entry)
            logger.info(f"Registered {model_id} ({model_version}) in registry")

            return output_path
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
        compatible_models = []

        for model_id, info in HUGGINGFACE_MODELS.items():
            # Add a safety margin - model needs ~1.2x its size in memory
            if info["memory_req_gb"] <= available_memory_gb:
                compatible_models.append((model_id, info["memory_req_gb"]))

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
        Delete a model from disk and registry.

        Args:
            model_id: ID of the model to delete
            version: Specific version to delete, or None for all versions

        Returns:
            True if successful, False otherwise
        """
        try:
            if version:
                # Delete specific version
                model_version = self.registry.get_model(model_id, version)
                if model_version and os.path.exists(model_version.path):
                    os.remove(model_version.path)
                    self.registry.remove_model(model_id, version)
                    logger.info(f"Deleted {model_id} ({version})")
                    return True
                return False
            else:
                # Delete all versions
                versions = self.registry.list_versions(model_id)
                if not versions:
                    return False

                for version in versions:
                    if os.path.exists(version.path):
                        os.remove(version.path)

                # Delete directory if empty
                model_dir = os.path.join(self.models_dir, model_id)
                if os.path.exists(model_dir) and not os.listdir(model_dir):
                    os.rmdir(model_dir)

                self.registry.remove_model(model_id)
                logger.info(f"Deleted all versions of {model_id}")
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
    parser.add_argument("--token", type=str, help="Hugging Face token")
    args = parser.parse_args()

    downloader = EnterpriseModelDownloader(hf_token=args.token)

    if args.list:
        print("Available models:")
        for model_id, info in HUGGINGFACE_MODELS.items():
            print(
                f"  {model_id}: {info['repo_id']} - {info['filename']} ({info['size_gb']:.1f} GB)"
            )

        print("\nDownloaded models:")
        for model_id in downloader.registry.list_models():
            versions = downloader.registry.list_versions(model_id)
            if versions:
                active = next((v for v in versions if v.is_active), None)
                if active:
                    print(
                        f"  {model_id}: {active.version} - {active.path} ({active.size_bytes / (1024**3):.1f} GB)"
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
