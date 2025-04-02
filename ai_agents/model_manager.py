#!/usr/bin/env python3
"""
Model Management System

This module handles the loading, initialization, and inference with various
AI models specified in the system configuration.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import threading
import time
import subprocess
import torch
import faiss
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger("ModelManager")


class ModelType(Enum):
    """Types of AI models supported by the system"""

    LLM = "large_language_model"
    VECTOR_DB = "vector_database"
    CODE_MODEL = "code_model"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding_model"


class ModelManager:
    """
    Manages the loading, versioning, and execution of AI models

    This class provides a unified interface for working with different
    model types and frameworks, handling model loading, unloading,
    caching, and inference.
    """

    def __init__(
        self,
        models_dir: str = "data/models",
        config_path: str = "config/models.json",
        cache_dir: str = "data/model_cache",
        max_memory_gb: int = 16,
        device: str = "auto",
    ):
        """
        Initialize the model manager

        Args:
            models_dir: Directory for storing model files
            config_path: Path to model configuration file
            cache_dir: Directory for model caching
            max_memory_gb: Maximum memory to use for models
            device: Device to run models on ('cpu', 'cuda', or 'auto')
        """
        self.models_dir = models_dir
        self.config_path = config_path
        self.cache_dir = cache_dir
        self.max_memory_gb = max_memory_gb

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize model registry
        self.models = {}
        self.loaded_models = {}
        self.model_locks = {}

        # Load model configurations
        self._load_model_configs()

    def _load_model_configs(self):
        """Load model configurations from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.models = config.get("models", {})
                    logger.debug(f"[_load_model_configs] Loaded models config: {json.dumps(self.models, indent=2)}")
                    logger.info(f"Loaded {len(self.models)} model configurations")
            else:
                # Create default configuration
                self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading model configurations: {str(e)}")
            # Create default configuration as fallback
            self._create_default_config()

    def _create_default_config(self):
        """Create default model configuration file"""
        default_config = {
            "models": {
                # LLM Models
                "llama3-8b": {
                    "type": "large_language_model",
                    "framework": "llama.cpp",
                    "path": "models/llama-3-8b-instruct.gguf",
                    "description": "Llama 3 8B instruct model",
                    "parameters": {
                        "context_length": 4096,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    "memory_gb": 6,
                    "quantization": "Q4_K_M",
                },
                "gpt4all-falcon": {
                    "type": "large_language_model",
                    "framework": "gpt4all",
                    "path": "models/gpt4all-falcon-q4.bin",
                    "description": "GPT4All Falcon Quantized",
                    "parameters": {"context_length": 2048, "temperature": 0.7},
                    "memory_gb": 4,
                    "quantization": "q4_0",
                },
                "deepseek-coder": {
                    "type": "code_model",
                    "framework": "deepseek",
                    "path": "models/deepseek-coder-33b-instruct.Q4_K_M.gguf",
                    "description": "DeepSeek Coder 33B for code generation",
                    "parameters": {
                        "context_length": 16384,
                        "temperature": 0.2,
                        "top_p": 0.95,
                    },
                    "memory_gb": 12,
                    "quantization": "Q4_K_M",
                },
                # Vector Databases
                "chroma-store": {
                    "type": "vector_database",
                    "framework": "chromadb",
                    "path": "data/vectors/chroma",
                    "description": "ChromaDB vector store",
                    "parameters": {
                        "collection_name": "main_store",
                        "embedding_function": "sentence-transformers",
                    },
                    "memory_gb": 2,
                },
                "faiss-index": {
                    "type": "vector_database",
                    "framework": "faiss",
                    "path": "data/vectors/faiss",
                    "description": "FAISS vector index",
                    "parameters": {"index_type": "IndexFlatL2", "dimension": 768},
                    "memory_gb": 1,
                },
            }
        }

        # Save default configuration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        self.models = default_config["models"]
        logger.debug(f"[_create_default_config] Created default models config: {json.dumps(self.models, indent=2)}")
        logger.info(
            f"Created default model configuration with {len(self.models)} models"
        )

    def load_model(self, model_id: str) -> bool:
        """
        Load a model into memory, downloading if necessary.

        Args:
            model_id: ID of the model to load

        Returns:
            Boolean indicating success
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry. Available: {list(self.models.keys())}")
            return False

        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return True

        # Create lock for this model if it doesn't exist
        if model_id not in self.model_locks:
            self.model_locks[model_id] = threading.Lock()

        # Acquire lock to prevent concurrent loading
        with self.model_locks[model_id]:
            # Re-check if model got loaded while waiting for the lock
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} was loaded by another thread while waiting.")
                return True

            try:
                logger.debug(f"[load_model] Attempting to load '{model_id}'. Config being used: {json.dumps(self.models.get(model_id, {}), indent=2)}")
                config = self.models[model_id]
                model_type = config.get("type", "unknown")
                framework = config.get("framework")

                # --- Handle Ollama type separately FIRST ---
                if model_type == "ollama":
                    if framework == "langchain":
                        ollama_params = config.get("parameters", {})
                        ollama_model_name = ollama_params.get("model")
                        base_url = ollama_params.get("base_url", "http://localhost:11434")
                        if not ollama_model_name:
                            raise ValueError(f"Missing 'model' parameter for Ollama config: {model_id}")

                        logger.info(f"Creating ChatOllama instance for model '{ollama_model_name}' at {base_url}")
                        # Ensure ChatOllama is imported
                        try:
                            from langchain_community.chat_models import ChatOllama
                        except ImportError:
                             logger.error("ChatOllama not available. Please install langchain-community.")
                             raise

                        instance = ChatOllama(
                            model=ollama_model_name,
                            base_url=base_url,
                            temperature=ollama_params.get("temperature", 0.7),
                            top_p=ollama_params.get("top_p", 1.0),
                            num_ctx=ollama_params.get("num_ctx", None),
                            keep_alive="5m" # Keep alive longer for potentially slow models
                        )
                        # Test connection/model availability
                        try:
                            logger.info(f"Testing connection to Ollama model '{ollama_model_name}'...")
                            # Use a simple, non-streaming invoke for testing
                            test_result = instance.invoke("Hi")
                            logger.info(f"Successfully connected to Ollama model '{ollama_model_name}'. Test response: {test_result}")
                        except Exception as ollama_err:
                            logger.error(f"Failed to connect/invoke Ollama model '{ollama_model_name}' at {base_url}: {ollama_err}")
                            # Optionally: check if ollama service is running locally?
                            raise ConnectionError(f"Ollama connection/invocation failed: {ollama_err}")

                        # Store the instance
                        self.loaded_models[model_id] = {
                             "instance": instance,
                             "type": model_type,
                             "framework": framework,
                             "loaded_at": time.time(),
                         }
                        logger.info(f"Successfully loaded Ollama model {model_id}")
                        return True # Successfully loaded Ollama model

                    else:
                        # If Ollama type but not langchain framework
                         logger.error(f"Framework '{framework}' not supported for Ollama type yet.")
                         return False

                # --- Handle other model types that REQUIRE a path ---
                else:
                    model_path = config.get("path", "") # Path for local models
                    if not model_path:
                         logger.error(f"Model type '{model_type}' requires a 'path' in config, but none found for {model_id}.")
                         return False

                    full_path = os.path.join(self.models_dir, model_path)

                    # Path Check and Download for non-Ollama models
                    logger.debug(f"[DEBUG] Checking path for non-Ollama: models_dir='{self.models_dir}', model_path='{model_path}', full_path='{full_path}'")

                    if not os.path.exists(full_path):
                        logger.warning(f"Model file not found: {full_path}. Attempting download.")
                        download_success = self.download_model(model_id)
                        if not download_success:
                            logger.error(f"Failed to download model {model_id}. Cannot load.")
                            return False
                        # Verify again after download attempt
                        if not os.path.exists(full_path):
                            logger.error(f"Model file {full_path} still not found after download attempt.")
                            return False
                        logger.info(f"Model {model_id} downloaded successfully to {full_path}")
                    else:
                        logger.debug(f"Model file found locally: {full_path}")

                    # Load model based on type and framework (using the now guaranteed full_path)
                    if model_type == "large_language_model":
                        if framework == "llama.cpp":
                            self._load_llama_cpp_model(model_id, config, full_path)
                        elif framework == "gpt4all":
                            self._load_gpt4all_model(model_id, config, full_path)
                        # Removed Ollama case here as it's handled above
                        else:
                            logger.error(f"Unsupported LLM framework: {framework}")
                            return False

                    elif model_type == "code_model":
                        if framework == "deepseek":
                            self._load_deepseek_model(model_id, config, full_path)
                        else:
                            logger.error(f"Unsupported code model framework: {framework}")
                            return False

                    elif model_type == "vector_database":
                        # Vector DBs might use path differently (as dir, not file)
                        if framework == "chromadb":
                            self._load_chromadb(model_id, config) # Assumes path is handled inside
                        elif framework == "faiss":
                             self._load_faiss(model_id, config) # Assumes path is handled inside
                        else:
                            logger.error(f"Unsupported vector database: {framework}")
                            return False
                    else:
                         # Should not be reached if Ollama is handled above
                        logger.error(f"Unhandled model type after path check: {model_type}")
                        return False

                    # If we reach here for non-Ollama types, the corresponding _load_* function
                    # should have populated self.loaded_models if successful.
                    # We need to ensure the load function raises an exception on failure
                    # or we need to check self.loaded_models here.
                    # Assuming _load_* functions raise Exception on failure:
                    logger.info(f"Successfully loaded non-Ollama model {model_id}")
                    return True


            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # Ensure model is not marked as loaded on failure
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]
                return False
            finally:
                # Release lock once loading is complete or failed
                # The 'with' statement handles this automatically
                pass

    def _load_llama_cpp_model(self, model_id: str, config: Dict[str, Any], model_full_path: str):
        """Load a llama.cpp model from a specific path."""
        try:
            from llama_cpp import Llama

            params = config.get("parameters", {})
            n_ctx = params.get("context_length", 2048)
            n_gpu_layers = -1 if self.device == "cuda" else 0

            # Load the model using the provided full path
            llama_model = Llama(
                model_path=model_full_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
            )

            self.loaded_models[model_id] = {
                "instance": llama_model,
                "type": "large_language_model",
                "framework": "llama.cpp",
                "loaded_at": time.time(),
            }

            logger.info(f"Loaded llama.cpp model {model_id} from {model_full_path}")

        except Exception as e:
            logger.error(f"Failed to load llama.cpp model {model_id}: {str(e)}")
            raise

    def _load_gpt4all_model(self, model_id: str, config: Dict[str, Any], model_full_path: str):
        """Load a GPT4All model from a specific path."""
        try:
            from gpt4all import GPT4All

            # Load the model using the provided full path
            gpt4all_model = GPT4All(model_full_path)

            self.loaded_models[model_id] = {
                "instance": gpt4all_model,
                "type": "large_language_model",
                "framework": "gpt4all",
                "loaded_at": time.time(),
            }

            logger.info(f"Loaded GPT4All model {model_id} from {model_full_path}")

        except Exception as e:
            logger.error(f"Failed to load GPT4All model {model_id}: {str(e)}")
            raise

    def _load_deepseek_model(self, model_id: str, config: Dict[str, Any], model_full_path: str):
        """Load a DeepSeek model (using llama.cpp) from a specific path."""
        try:
            from llama_cpp import Llama

            params = config.get("parameters", {})
            n_ctx = params.get("context_length", 16384)
            n_gpu_layers = -1 if self.device == "cuda" else 0

            # Load the model using the provided full path
            deepseek_model = Llama(
                model_path=model_full_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
            )

            self.loaded_models[model_id] = {
                "instance": deepseek_model,
                "type": "code_model",
                "framework": "deepseek",
                "loaded_at": time.time(),
            }

            logger.info(f"Loaded DeepSeek model {model_id} from {model_full_path}")

        except Exception as e:
            logger.error(f"Failed to load DeepSeek model {model_id}: {str(e)}")
            raise

    def _load_chromadb(self, model_id: str, config: Dict[str, Any]):
        """Load a ChromaDB vector store"""
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            store_path = os.path.join(self.models_dir, config["path"])
            os.makedirs(store_path, exist_ok=True)

            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=store_path, settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            collection_name = config.get("parameters", {}).get(
                "collection_name", "main_collection"
            )
            embedding_model_name = config.get("parameters", {}).get(
                "embedding_function", "all-MiniLM-L6-v2"
            )

            # Load embedding model
            embedding_model = SentenceTransformer(embedding_model_name)

            def embedding_function(texts):
                return embedding_model.encode(texts).tolist()

            # Create or get collection
            collection = client.get_or_create_collection(
                name=collection_name, embedding_function=embedding_function
            )

            self.loaded_models[model_id] = {
                "instance": {
                    "client": client,
                    "collection": collection,
                    "embedding_model": embedding_model,
                },
                "type": "vector_database",
                "framework": "chromadb",
                "loaded_at": time.time(),
            }

            logger.info(f"Loaded ChromaDB vector store {model_id}")

        except Exception as e:
            logger.error(f"Failed to load ChromaDB {model_id}: {str(e)}")
            raise

    def _load_faiss(self, model_id: str, config: Dict[str, Any]):
        """Load a FAISS vector index"""
        try:
            import faiss

            index_path = os.path.join(self.models_dir, config["path"])
            os.makedirs(os.path.dirname(index_path), exist_ok=True)

            # Check if index exists
            index_file = os.path.join(index_path, "index.faiss")
            if os.path.exists(index_file):
                # Load existing index
                index = faiss.read_index(index_file)
                logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors")
            else:
                # Create new index
                dimension = config.get("parameters", {}).get("dimension", 768)
                index_type = config.get("parameters", {}).get(
                    "index_type", "IndexFlatL2"
                )

                if index_type == "IndexFlatL2":
                    index = faiss.IndexFlatL2(dimension)
                else:
                    logger.warning(
                        f"Unsupported index type: {index_type}, using IndexFlatL2"
                    )
                    index = faiss.IndexFlatL2(dimension)

                logger.info(f"Created new FAISS index with dimension {dimension}")

                # Save the empty index
                os.makedirs(index_path, exist_ok=True)
                faiss.write_index(index, index_file)

            self.loaded_models[model_id] = {
                "instance": {
                    "index": index,
                    "path": index_path,
                    "metadata": {},  # Will store id->metadata mapping
                },
                "type": "vector_database",
                "framework": "faiss",
                "loaded_at": time.time(),
            }

            logger.info(f"Loaded FAISS vector index {model_id}")

        except Exception as e:
            logger.error(f"Failed to load FAISS index {model_id}: {str(e)}")
            raise

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory

        Args:
            model_id: ID of the model to unload

        Returns:
            Boolean indicating success
        """
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded, cannot unload")
            return False

        with self.model_locks.get(model_id, threading.Lock()):
            try:
                model_data = self.loaded_models[model_id]

                # Framework-specific unloading
                if model_data["framework"] == "faiss":
                    # Save FAISS index before unloading
                    index = model_data["instance"]["index"]
                    path = model_data["instance"]["path"]
                    index_file = os.path.join(path, "index.faiss")
                    faiss.write_index(index, index_file)

                # Remove from loaded models
                del self.loaded_models[model_id]

                # Force garbage collection to free memory
                import gc

                gc.collect()

                if self.device == "cuda":
                    torch.cuda.empty_cache()

                logger.info(f"Unloaded model {model_id}")
                return True

            except Exception as e:
                logger.error(f"Error unloading model {model_id}: {str(e)}")
                return False

    def run_inference(
        self, model_id: str, inputs: Any, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run inference with a model

        Args:
            model_id: ID of the model to use
            inputs: Input data for the model
            parameters: Optional parameters for inference

        Returns:
            Dictionary with inference results
        """
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded, attempting to load")
            if not self.load_model(model_id):
                return {"error": f"Failed to load model {model_id}"}

        model_data = self.loaded_models[model_id]
        model_type = model_data["type"]

        try:
            # Handle different model types
            if model_type == "large_language_model" or model_type == "code_model":
                return self._run_llm_inference(model_id, inputs, parameters)
            elif model_type == "vector_database":
                return self._run_vector_db_query(model_id, inputs, parameters)
            else:
                return {"error": f"Unsupported model type: {model_type}"}

        except Exception as e:
            logger.error(f"Error running inference with model {model_id}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return {"error": f"Inference error: {str(e)}"}

    def _run_llm_inference(
        self,
        model_id: str,
        inputs: Union[str, List[Dict[str, str]]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run inference with an LLM"""
        model_data = self.loaded_models[model_id]
        framework = model_data["framework"]
        model = model_data["instance"]

        # Get parameters with defaults from model config
        model_config = self.models.get(model_id, {})
        default_params = model_config.get("parameters", {})
        # Combine default model params with runtime params
        # Runtime params take precedence
        params = {**default_params, **(parameters or {})}

        # Process inputs - Note: Formatting moved to specific inference functions
        # if isinstance(inputs, str):
        #     prompt = inputs
        # elif isinstance(inputs, list):
        #     # Assume chat format
        #     prompt = self._format_chat_messages(inputs, framework)
        # else:
        #     return {"error": "Invalid input format"}

        # Run inference based on framework
        if framework == "llama.cpp":
            # Format messages specifically for llama.cpp if needed
            if isinstance(inputs, list):
                 prompt = self._format_chat_messages(inputs, framework)
            else:
                 prompt = inputs # Assume string input is already formatted
            output = self._run_llamacpp_inference(model, prompt, params)
        elif framework == "gpt4all":
            # Format messages specifically for gpt4all if needed
            if isinstance(inputs, list):
                 prompt = self._format_chat_messages(inputs, framework)
            else:
                 prompt = inputs # Assume string input is already formatted
            output = self._run_gpt4all_inference(model, prompt, params)
        elif framework == "ollama":
            # Pass the raw inputs (str or list of dicts) to the ollama inference function
            # along with the ChatOllama instance (model)
            output = self._run_ollama_inference(model, inputs, params)
        else:
            logger.error(f"Unsupported LLM framework in _run_llm_inference: {framework}")
            return {"error": f"Unsupported framework: {framework}"}

        logger.debug(f"Raw LLM inference output for {model_id}: {output}")
        return output

    def _format_chat_messages(
        self, messages: List[Dict[str, str]], framework: str
    ) -> str:
        """Format chat messages for different frameworks"""
        if framework == "llama.cpp":
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    formatted += f"<|system|>\n{content}\n"
                elif role == "user":
                    formatted += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    formatted += f"<|assistant|>\n{content}\n"
            formatted += "<|assistant|>\n"
            return formatted

        elif framework == "gpt4all":
            # Simple formatting for GPT4All
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    formatted += f"System: {content}\n"
                elif role == "user":
                    formatted += f"User: {content}\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n"
            formatted += "Assistant: "
            return formatted

        elif framework == "ollama":
            # Ollama handles this natively
            return messages

        else:
            # Default formatting
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted += f"{role.capitalize()}: {content}\n"
            formatted += "Assistant: "
            return formatted

    def _run_llamacpp_inference(
        self, model, prompt: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference with llama.cpp model"""
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)
        top_p = params.get("top_p", 0.9)

        # Run inference
        output = model(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )

        return {
            "text": output["choices"][0]["text"],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output["choices"][0]["text"].split()),
                "total_tokens": len(prompt.split())
                + len(output["choices"][0]["text"].split()),
            },
        }

    def _run_gpt4all_inference(
        self, model, prompt: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run inference with GPT4All model"""
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)

        # Run inference
        output = model.generate(prompt, max_tokens=max_tokens, temp=temperature)

        return {
            "text": output,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt.split()) + len(output.split()),
            },
        }

    def _run_ollama_inference(
        self,
        model_instance: ChatOllama,
        inputs: Union[str, List[Dict[str, str]]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run inference with Ollama model using the ChatOllama instance."""
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)
        num_ctx = params.get("num_ctx", None)

        try:
            # Prepare arguments for invoke
            invoke_args = {
                "temperature": temperature,
                "top_p": top_p,
            }
            if num_ctx is not None:
                invoke_args["num_ctx"] = num_ctx

            # Prepare input format for ChatOllama
            if isinstance(inputs, str):
                langchain_input = inputs
            elif isinstance(inputs, list):
                langchain_messages = []
                for msg in inputs:
                    role = msg.get("role", "user").lower()
                    content = msg.get("content", "")
                    if role == "system":
                        langchain_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        langchain_messages.append(AIMessage(content=content))
                    else:
                        langchain_messages.append(HumanMessage(content=content))
                langchain_input = langchain_messages
            else:
                logger.error(f"Invalid input type for Ollama inference: {type(inputs)}")
                return {"error": "Invalid input type for Ollama"}

            # Run inference using the ChatOllama instance
            logger.debug(f"Invoking ChatOllama model with args: {invoke_args}, input type: {type(langchain_input)}")
            response = model_instance.invoke(langchain_input, **invoke_args)

            # Extract results
            text_output = response.content
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            # Try to extract token usage from response metadata
            if response.response_metadata:
                 usage["prompt_tokens"] = response.response_metadata.get("prompt_eval_count", 0)
                 # Ollama API seems to use 'eval_count' for completion tokens in chat
                 # and generate endpoints, but langchain might map differently.
                 # Checking common keys.
                 completion_tokens = response.response_metadata.get("eval_count", 0)
                 if completion_tokens == 0:
                     completion_tokens = response.response_metadata.get("completion_tokens", 0)
                 usage["completion_tokens"] = completion_tokens
                 usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

                 # Log if usage data seems incomplete
                 if usage["prompt_tokens"] == 0 and usage["completion_tokens"] == 0:
                    logger.warning(f"Could not extract token usage from Ollama response metadata: {response.response_metadata}")

            return {
                "text": text_output,
                "usage": usage,
            }

        except Exception as e:
            logger.error(f"Error during Ollama inference: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Ollama inference failed: {str(e)}"}

    def _run_vector_db_query(
        self,
        model_id: str,
        query: Union[str, List[float], Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query a vector database"""
        model_data = self.loaded_models[model_id]
        framework = model_data["framework"]
        instance = model_data["instance"]

        params = parameters or {}
        k = params.get("k", 5)  # Number of results

        if framework == "chromadb":
            return self._query_chromadb(instance, query, k)
        elif framework == "faiss":
            return self._query_faiss(instance, query, k)
        else:
            return {"error": f"Unsupported vector database: {framework}"}

    def _query_chromadb(
        self, instance: Dict[str, Any], query: Union[str, List[float]], k: int
    ) -> Dict[str, Any]:
        """Query ChromaDB"""
        collection = instance["collection"]
        embedding_model = instance["embedding_model"]

        # Convert query to embedding if it's a string
        if isinstance(query, str):
            query_embedding = embedding_model.encode(query).tolist()
        else:
            query_embedding = query

        # Run query
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0] if "documents" in results else [],
            "metadatas": results["metadatas"][0] if "metadatas" in results else [],
            "distances": results["distances"][0],
        }

    def _query_faiss(
        self, instance: Dict[str, Any], query: Union[str, List[float]], k: int
    ) -> Dict[str, Any]:
        """Query FAISS index"""
        import numpy as np

        index = instance["index"]
        metadata = instance["metadata"]

        # Convert query to numpy array
        if isinstance(query, str):
            # Need to convert string to embedding first
            # This would require an embedding model
            return {
                "error": "String queries not supported for FAISS without embedding model"
            }
        elif isinstance(query, list):
            query_vector = np.array([query], dtype=np.float32)
        else:
            return {"error": "Invalid query format for FAISS"}

        # Run query
        distances, indices = index.search(query_vector, k)

        # Get metadata for results
        result_ids = [str(idx) for idx in indices[0] if idx >= 0]
        result_metadatas = [
            metadata.get(str(idx), {}) for idx in indices[0] if idx >= 0
        ]

        return {
            "ids": result_ids,
            "metadatas": result_metadatas,
            "distances": distances[0].tolist(),
        }

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return self.models

    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """List currently loaded models"""
        result = {}
        for model_id, data in self.loaded_models.items():
            result[model_id] = {
                "type": data["type"],
                "framework": data["framework"],
                "loaded_at": data["loaded_at"],
            }
        return result

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        return self.models.get(model_id)

    def add_model(self, model_id: str, model_config: Dict[str, Any]) -> bool:
        """
        Add a new model to the registry

        Args:
            model_id: Unique identifier for the model
            model_config: Configuration dictionary for the model

        Returns:
            Boolean indicating success
        """
        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists in registry")
            return False

        self.models[model_id] = model_config

        # Save updated configuration
        self._save_config()

        logger.info(f"Added model {model_id} to registry")
        return True

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the registry

        Args:
            model_id: ID of the model to remove

        Returns:
            Boolean indicating success
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return False

        # Unload if loaded
        if model_id in self.loaded_models:
            self.unload_model(model_id)

        # Remove from registry
        del self.models[model_id]

        # Save updated configuration
        self._save_config()

        logger.info(f"Removed model {model_id} from registry")
        return True

    def _save_config(self):
        """Save model configuration to file"""
        config = {"models": self.models}
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def download_model(self, model_id: str) -> bool:
        """
        Download a model file using Hugging Face Hub.

        Args:
            model_id: ID of the model (must exist in self.models)

        Returns:
            Boolean indicating success
        """
        if model_id not in self.models:
            logger.error(f"Cannot download: Model {model_id} not found in registry.")
            return False

        model_config = self.models[model_id]
        repo_id = model_config.get("repo_id")
        filename = model_config.get("path") # Use path as filename
        quantization = model_config.get("quantization") # Optional: For selecting specific files

        if not repo_id:
            logger.error(f"Cannot download {model_id}: Missing 'repo_id' in configuration.")
            return False
        if not filename:
             logger.error(f"Cannot download {model_id}: Missing 'path' (filename) in configuration.")
             return False

        # Construct the target path
        local_dir = self.models_dir
        target_path = os.path.join(local_dir, filename)
        os.makedirs(local_dir, exist_ok=True)

        logger.info(f"Attempting to download model '{model_id}' ({filename}) from repo '{repo_id}' to '{target_path}'")

        try:
            # Use hf_hub_download
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False, # Avoid symlinks, download directly
                # token=os.getenv("HF_TOKEN"), # Optional: Use token if needed for private repos
                cache_dir=self.cache_dir # Use specified cache directory
            )

            # Verify file exists after download
            if os.path.exists(target_path):
                 logger.info(f"Successfully downloaded model {model_id} to {target_path}")
                 return True
            else:
                 # This case might happen if hf_hub_download puts it in cache only
                 # Let's try a basic check if the cache dir logic works
                 # NOTE: This basic check might not be robust depending on hf_hub behavior
                 cached_path_guess = os.path.join(self.cache_dir, f"models--{repo_id.replace('/', '--')}", "snapshots")
                 # A more robust check would involve inspecting hf_hub_download's return value or cache structure
                 logger.warning(f"Download reported success, but file not found at target {target_path}. Checking cache area {cached_path_guess}...")
                 # Basic check - if *any* file was downloaded recently in the likely cache subdir
                 found_in_cache = False
                 if os.path.exists(cached_path_guess):
                      for root, _, files in os.walk(cached_path_guess):
                           for file in files:
                                if file == filename:
                                     # If found in cache, maybe copy it to target? Or adjust loading logic?
                                     # For now, just log it.
                                     logger.info(f"Found {filename} in cache, but not at target path. Manual copy might be needed or adjust loading logic.")
                                     # Returning False as it's not where load_model expects it.
                                     return False
                 logger.error(f"Download failed: File {filename} not found at target or in expected cache location after hf_hub_download.")
                 return False


        except RepositoryNotFoundError:
            logger.error(f"Download failed: Repository '{repo_id}' not found on Hugging Face Hub.")
            return False
        except EntryNotFoundError:
             logger.error(f"Download failed: File '{filename}' not found in repository '{repo_id}'. Check quantization/filename in config.")
             return False
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_model_capabilities(self, model_id: str) -> List[str]:
        """Get the capabilities of a specific model."""
        config = self.get_model_info(model_id)
        return config.get("capabilities", []) if config else []

    def cleanup(self):
        """Perform cleanup actions, e.g., unloading models (if applicable)."""
        # Currently, models are loaded on demand and might be unloaded automatically
        # or managed by the underlying frameworks (like LangChain/Ollama).
        # Add specific unloading logic here if needed in the future.
        logger.info("Model manager cleanup.")
        # Example: If models were stored in self.loaded_models dict:
        # for model_id, model_instance in self.loaded_models.items():
        #     logger.info(f"Unloading model {model_id}...")
        #     # Add specific unloading logic based on model type
        # self.loaded_models.clear()


if __name__ == "__main__":
    # Placeholder for original testing/example code that might have been here
    pass
    # Example:
    # print("Testing Model Manager...")
    # manager = ModelManager()
    # print("Available models:", manager.list_models())
    # print("Attempting to load llama3-8b...")
    # success = manager.load_model("llama3-8b")
    # print(f"Load successful: {success}")
    # if success:
    #     print("Loaded models:", manager.list_loaded_models())
