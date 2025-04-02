#!/usr/bin/env python3
"""
SutazAI Model Setup Script

This script automates the downloading and setup of AI models required
for the SutazAI AGI/ASI system.
"""

import os
import json
import argparse
import hashlib
import logging
import requests
import tarfile
import zipfile
import shutil
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# Helper function for safe extraction
def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/model_setup.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ModelSetup")

# Configuration
DEFAULT_CONFIG_PATH = "config/models.json"
DEFAULT_MODELS_DIR = "data/models"
CHUNK_SIZE = 8192 * 1024  # 8MB chunks for downloading


class ModelSetup:
    """
    Handles downloading and setting up AI models for the SutazAI system.
    """

    def __init__(
        self,
        config_path=DEFAULT_CONFIG_PATH,
        models_dir=DEFAULT_MODELS_DIR,
        max_workers=4,
    ):
        """
        Initialize the model setup.

        Args:
            config_path: Path to the model configuration file
            models_dir: Directory to store downloaded models
            max_workers: Maximum number of concurrent downloads
        """
        self.config_path = config_path
        self.models_dir = Path(models_dir)
        self.max_workers = max_workers

        # Load configuration
        self.config = self._load_config()

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model setup initialized with config: {config_path}")
        logger.info(f"Models will be stored in: {models_dir}")

    def _load_config(self):
        """Load model configuration from file or create default"""
        if not os.path.exists(self.config_path):
            logger.info(
                f"Config file not found at {self.config_path}, creating default"
            )
            self._create_default_config()

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                logger.info(
                    f"Loaded configuration with {len(config.get('models', []))} models"
                )
                return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _create_default_config(self):
        """Create a default configuration file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Default model configuration
        default_config = {
            "models": [
                {
                    "id": "llama3-8b",
                    "name": "Llama 3 8B",
                    "type": "llm",
                    "url": "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/model.gguf",
                    "version": "Q4_K_M",
                    "sha256": "84a0aeea14cbba7e24e591b8c36faf7798f2dc460f353acf9af51e3d4d450f8b",
                    "size_gb": 4.8,
                    "quantized": True,
                    "local_path": "llama3/llama-3-8b-instruct-q4_k_m.gguf",
                    "requires_auth": True,
                },
                {
                    "id": "deepseek-coder",
                    "name": "DeepSeek Coder",
                    "type": "code-llm",
                    "url": "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
                    "version": "Q4_K_M",
                    "sha256": "af1a9539fc8f45e977d9e8c9148d26b05bda069a939175650fa22aa5f39a4fd3",
                    "size_gb": 3.8,
                    "quantized": True,
                    "local_path": "deepseek/deepseek-coder-6.7b-instruct-q4_k_m.gguf",
                    "requires_auth": True,
                },
                {
                    "id": "gpt4all-falcon",
                    "name": "GPT4All Falcon",
                    "type": "llm",
                    "url": "https://gpt4all.io/models/gguf/falcon-7b.Q4_0.gguf",
                    "version": "Q4_0",
                    "sha256": "d62b42a1d22fd3e5af8e1a760da4162eea64cb4387f9b78acff3aeb322817645",
                    "size_gb": 3.9,
                    "quantized": True,
                    "local_path": "gpt4all/falcon-7b-q4_0.gguf",
                    "requires_auth": False,
                },
                {
                    "id": "mxbai-embed",
                    "name": "MxBai Embeddings",
                    "type": "embeddings",
                    "url": "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/model.safetensors",
                    "sha256": "4909ea1af34108c9a76f9b2ec37b9cb642419125404754df03024b9c2b1acfd5",
                    "size_gb": 0.67,
                    "quantized": False,
                    "local_path": "embeddings/mxbai-embed-large-v1.safetensors",
                    "requires_auth": True,
                },
                {
                    "id": "whisper-small",
                    "name": "Whisper Small",
                    "type": "speech",
                    "url": "https://huggingface.co/guillaumekln/faster-whisper-small/resolve/main/model.bin",
                    "sha256": "9e9d5fd9fd7231ba8d5c0e70fc457b6b3d81d07e28d891d9b71782684e7f0eae",
                    "size_gb": 0.47,
                    "quantized": False,
                    "local_path": "speech/whisper-small.bin",
                    "requires_auth": False,
                },
            ],
            "huggingface_token": "",
            "download_options": {
                "concurrent_downloads": 2,
                "retry_attempts": 3,
                "retry_delay_seconds": 5,
                "verify_checksum": True,
            },
        }

        # Write default config
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"Created default configuration at {self.config_path}")
        return default_config

    def check_disk_space(self):
        """
        Check if there's enough disk space for all models.

        Returns:
            tuple: (space_required_gb, space_available_gb, is_sufficient)
        """

        # Calculate total size required
        space_required = sum(
            model.get("size_gb", 0) for model in self.config.get("models", [])
        )

        # Get available space
        try:
            total, used, free = shutil.disk_usage(self.models_dir)
            free_gb = free / (1024**3)

            # Add 10% buffer
            space_required_with_buffer = space_required * 1.1

            logger.info(
                f"Disk space required: {space_required_with_buffer:.2f} GB, available: {free_gb:.2f} GB"
            )

            return (
                space_required_with_buffer,
                free_gb,
                free_gb >= space_required_with_buffer,
            )
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            # If we can't check, assume there's enough space
            return space_required, 0, True

    def verify_file_hash(self, file_path, expected_hash):
        """
        Verify a file's SHA-256 hash.

        Args:
            file_path: Path to the file to verify
            expected_hash: Expected SHA-256 hash

        Returns:
            bool: True if hash matches, False otherwise
        """
        if not expected_hash:
            logger.warning(f"No hash provided for {file_path}, skipping verification")
            return True

        logger.info(f"Verifying hash for {file_path}")
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read and update hash in chunks
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            file_hash = sha256_hash.hexdigest()
            if file_hash == expected_hash:
                logger.info(f"Hash verification successful for {file_path}")
                return True
            else:
                logger.error(
                    f"Hash mismatch for {file_path}. Expected {expected_hash}, got {file_hash}"
                )
                return False
        except Exception as e:
            logger.error(f"Error verifying hash: {str(e)}")
            return False

    def download_file(self, url, output_path, model_info, show_progress=True):
        """
        Download a file with progress tracking.

        Args:
            url: URL to download
            output_path: Path to save the file
            model_info: Model information dictionary
            show_progress: Whether to show a progress bar

        Returns:
            bool: True if download successful, False otherwise
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Downloading {model_info['name']} from {url} to {output_path}")

        # Prepare headers
        headers = {}
        if model_info.get("requires_auth", False) and "huggingface.co" in url:
            token = self.config.get("huggingface_token", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                logger.warning("HuggingFace token required but not provided in config")

        # Download file
        retry_attempts = self.config.get("download_options", {}).get(
            "retry_attempts", 3
        )
        retry_delay = self.config.get("download_options", {}).get(
            "retry_delay_seconds", 5
        )

        for attempt in range(retry_attempts):
            try:
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(output_path, "wb") as f:
                    if show_progress:
                        # Use tqdm for progress bar
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc=model_info["name"],
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # No progress bar
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)

                # Verify hash if provided
                if self.config.get("download_options", {}).get(
                    "verify_checksum", True
                ) and model_info.get("sha256"):
                    if not self.verify_file_hash(output_path, model_info["sha256"]):
                        logger.error(
                            f"Hash verification failed for {output_path}, deleting file"
                        )
                        os.remove(output_path)
                        return False

                logger.info(f"Download completed: {model_info['name']}")
                return True

            except Exception as e:
                logger.error(
                    f"Download failed (attempt {attempt + 1}/{retry_attempts}): {str(e)}"
                )
                if attempt < retry_attempts - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Download failed after {retry_attempts} attempts")
                    # Clean up partial download
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return False

    def process_downloaded_file(self, file_path, model_info):
        """
        Process a downloaded file (extract archives if needed).

        Args:
            file_path: Path to the downloaded file
            model_info: Model information dictionary

        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            file_path_str = str(file_path)  # Ensure string path
            # Check if file is an archive that needs extraction
            if file_path_str.endswith(".tar.gz") or file_path_str.endswith(".tgz"):
                extract_dir = os.path.dirname(file_path)
                logger.info(
                    f"Safely extracting tar archive: {file_path} to {extract_dir}"
                )

                with tarfile.open(file_path_str, "r:gz") as tar:
                    for member in tar.getmembers():
                        member_path = os.path.join(extract_dir, member.name)
                        # Check for path traversal vulnerability
                        if not is_within_directory(extract_dir, member_path):
                            logger.warning(
                                f"Skipping potentially unsafe tar member: {member.name}"
                            )
                            continue
                        tar.extract(member, path=extract_dir)

                logger.info(f"Extraction completed for {file_path}")

                # Delete the archive after extraction
                os.remove(file_path)
                logger.info(f"Deleted archive after extraction: {file_path}")
                return True

            elif file_path_str.endswith(".zip"):
                extract_dir = os.path.dirname(file_path)
                logger.info(
                    f"Safely extracting zip archive: {file_path} to {extract_dir}"
                )

                with zipfile.ZipFile(file_path_str, "r") as zip_ref:
                    for member in zip_ref.infolist():
                        member_path = os.path.join(extract_dir, member.filename)
                        # Check for path traversal vulnerability
                        if not is_within_directory(extract_dir, member_path):
                            logger.warning(
                                f"Skipping potentially unsafe zip member: {member.filename}"
                            )
                            continue
                        zip_ref.extract(member, path=extract_dir)

                logger.info(f"Extraction completed for {file_path}")

                # Delete the archive after extraction
                os.remove(file_path)
                logger.info(f"Deleted archive after extraction: {file_path}")
                return True

            # No processing needed
            return True

        except Exception as e:
            logger.error(f"Error processing downloaded file {file_path}: {str(e)}")
            return False

    def download_model(self, model_info):
        """
        Download and process a single model.

        Args:
            model_info: Model information dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        model_id = model_info["id"]
        model_name = model_info["name"]
        logger.info(f"Setting up model: {model_name} ({model_id})")

        # Get local path
        local_path = self.models_dir / model_info["local_path"]

        # Check if model already exists
        if local_path.exists():
            logger.info(f"Model {model_name} already exists at {local_path}")

            # Verify hash if needed
            if self.config.get("download_options", {}).get(
                "verify_checksum", True
            ) and model_info.get("sha256"):
                if self.verify_file_hash(local_path, model_info["sha256"]):
                    logger.info(f"Model {model_name} hash verification successful")
                    return True
                else:
                    logger.warning(
                        f"Model {model_name} hash verification failed, will redownload"
                    )
                    os.remove(local_path)
            else:
                # Skip hash verification
                return True

        # Download the model
        success = self.download_file(model_info["url"], local_path, model_info)

        if success:
            # Process the downloaded file if needed
            return self.process_downloaded_file(local_path, model_info)

        return False

    def download_all_models(self, selected_models=None):
        """
        Download all models or a selected subset.

        Args:
            selected_models: List of model IDs to download, or None for all

        Returns:
            dict: Results with model IDs as keys and success status as values
        """
        models = self.config.get("models", [])

        # Filter models if selection provided
        if selected_models:
            models = [m for m in models if m["id"] in selected_models]
            logger.info(f"Selected {len(models)} models for download")

        # Check disk space
        space_required, space_available, is_sufficient = self.check_disk_space()

        if not is_sufficient:
            logger.error(
                f"Insufficient disk space! Required: {space_required:.2f} GB, Available: {space_available:.2f} GB"
            )
            return {model["id"]: False for model in models}

        # Determine number of concurrent downloads
        concurrent_downloads = min(
            self.max_workers,
            self.config.get("download_options", {}).get("concurrent_downloads", 2),
        )

        # Download models using thread pool
        results = {}

        if concurrent_downloads > 1 and len(models) > 1:
            logger.info(
                f"Downloading {len(models)} models with {concurrent_downloads} concurrent workers"
            )

            with ThreadPoolExecutor(max_workers=concurrent_downloads) as executor:
                future_to_model = {
                    executor.submit(self.download_model, model): model
                    for model in models
                }

                for future in future_to_model:
                    model = future_to_model[future]
                    try:
                        success = future.result()
                        results[model["id"]] = success
                    except Exception as e:
                        logger.error(f"Error downloading model {model['id']}: {str(e)}")
                        results[model["id"]] = False
        else:
            # Sequential download
            logger.info(f"Downloading {len(models)} models sequentially")

            for model in models:
                try:
                    success = self.download_model(model)
                    results[model["id"]] = success
                except Exception as e:
                    logger.error(f"Error downloading model {model['id']}: {str(e)}")
                    results[model["id"]] = False

        # Log results
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Downloaded {successful}/{len(models)} models successfully")

        return results

    def update_model_config(self, model_id, updates):
        """
        Update configuration for a specific model.

        Args:
            model_id: ID of the model to update
            updates: Dictionary of updates to apply

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find the model in config
            for i, model in enumerate(self.config.get("models", [])):
                if model["id"] == model_id:
                    # Update model configuration
                    for key, value in updates.items():
                        self.config["models"][i][key] = value

                    # Save updated config
                    with open(self.config_path, "w") as f:
                        json.dump(self.config, f, indent=2)

                    logger.info(f"Updated configuration for model {model_id}")
                    return True

            logger.error(f"Model {model_id} not found in configuration")
            return False

        except Exception as e:
            logger.error(f"Error updating model config: {str(e)}")
            return False

    def list_models(self, check_files=True):
        """
        List all models in the configuration with their status.

        Args:
            check_files: Whether to check if model files exist

        Returns:
            list: List of model dictionaries with status information
        """
        models = []

        for model in self.config.get("models", []):
            # Fix: Check if model is a dictionary before calling copy()
            if isinstance(model, dict):
                model_info = model.copy()
            else:
                # Handle case where model might be a string (model ID)
                logger.warning(f"Unexpected model format: {model}, expected dictionary")
                model_info = {"id": str(model), "name": str(model), "local_path": ""}

            if check_files:
                # Check if model file exists
                if model_info.get("local_path"):
                    local_path = self.models_dir / model_info["local_path"]
                    model_info["downloaded"] = local_path.exists()

                    # Check if hash is valid (if file exists and hash is provided)
                    if model_info["downloaded"] and model_info.get("sha256"):
                        model_info["hash_valid"] = self.verify_file_hash(
                            local_path, model_info["sha256"]
                        )
                    else:
                        model_info["hash_valid"] = None
                else:
                    model_info["downloaded"] = False
                    model_info["hash_valid"] = None

            models.append(model_info)

        return models

    def generate_model_config(self, output_path=None):
        """
        Generate a model configuration file for the application based on downloaded models.

        Args:
            output_path: Path to save the configuration, or None to use default

        Returns:
            str: Path to the generated configuration file
        """
        if output_path is None:
            output_path = "config/model_manager_config.json"

        # Get downloaded models
        downloaded_models = self.list_models(check_files=True)

        # Create configuration structure
        app_config = {
            "models": [],
            "default_model": "llama3-8b",
            "max_loaded_models": 2,
            "memory_management": {
                "max_memory_gb": 16,
                "offload_unused_after_minutes": 30,
            },
        }

        # Add downloaded models to configuration
        for model in downloaded_models:
            if model.get("downloaded", False):
                model_config = {
                    "id": model["id"],
                    "name": model["name"],
                    "type": model["type"],
                    "path": str(self.models_dir / model["local_path"]),
                    "context_window": 4096,
                }

                # Add type-specific configuration
                if model["type"] == "llm":
                    model_config["temperature"] = 0.7
                    model_config["max_tokens"] = 2048
                    model_config["load_in_4bit"] = model.get("quantized", False)
                elif model["type"] == "embeddings":
                    model_config["dimensions"] = 768
                    model_config["normalize"] = True

                app_config["models"].append(model_config)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write configuration
        with open(output_path, "w") as f:
            json.dump(app_config, f, indent=2)

        logger.info(f"Generated application model configuration at {output_path}")
        return output_path


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="SutazAI Model Setup Script")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory to store models",
    )
    parser.add_argument(
        "--model", type=str, nargs="*", help="Specific model IDs to download"
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate application model configuration",
    )
    parser.add_argument(
        "--output-config", type=str, help="Output path for generated configuration"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent downloads",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify downloaded models"
    )
    parser.add_argument("--token", type=str, help="HuggingFace API token")

    args = parser.parse_args()

    # Create model setup instance
    setup = ModelSetup(args.config, args.models_dir, args.max_workers)

    # Update HuggingFace token if provided
    if args.token:
        setup.config["huggingface_token"] = args.token
        with open(setup.config_path, "w") as f:
            json.dump(setup.config, f, indent=2)
        logger.info("Updated HuggingFace token in configuration")

    # List models
    if args.list:
        models = setup.list_models()
        print("\nAvailable models:")
        print("-" * 80)
        print(
            f"{'ID':<15} {'Name':<25} {'Type':<10} {'Size (GB)':<10} {'Downloaded':<12} {'Hash Valid'}"
        )
        print("-" * 80)

        for model in models:
            print(
                f"{model['id']:<15} {model['name']:<25} {model['type']:<10} {model.get('size_gb', 0):<10.2f} "
                f"{str(model.get('downloaded', False)):<12} {str(model.get('hash_valid', 'N/A'))}"
            )

        print("\n")
        return

    # Verify downloaded models
    if args.verify:
        models = setup.list_models()
        print("\nVerifying downloaded models:")

        for model in models:
            if model.get("downloaded", False):
                local_path = Path(args.models_dir) / model["local_path"]
                valid = setup.verify_file_hash(local_path, model.get("sha256", ""))
                print(
                    f"{model['id']:<15} {model['name']:<25} {'VALID' if valid else 'INVALID'}"
                )
            else:
                print(f"{model['id']:<15} {model['name']:<25} NOT DOWNLOADED")

        print("\n")
        return

    # Generate application model configuration
    if args.generate_config:
        output_path = args.output_config or "config/model_manager_config.json"
        setup.generate_model_config(output_path)
        print(f"Generated application model configuration at {output_path}")
        return

    # Download models
    if args.model:
        # Download specific models
        print(f"Downloading selected models: {', '.join(args.model)}")
        results = setup.download_all_models(args.model)
    else:
        # Download all models
        print("Downloading all models")
        results = setup.download_all_models()

    # Print results
    print("\nDownload results:")
    for model_id, success in results.items():
        print(f"{model_id}: {'Success' if success else 'Failed'}")

    # Generate application configuration
    if all(results.values()):
        # Only generate if all downloads successful
        setup.generate_model_config()

    print("\nModel setup completed")


if __name__ == "__main__":
    main()
