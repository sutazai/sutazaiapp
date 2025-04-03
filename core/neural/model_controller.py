#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_controller.py - API controller for the SutazAI model management system

This module provides a simplified API for applications to interact with the
model management system, providing a stable interface for model downloading,
optimization, and inference specifically optimized for Dell PowerEdge R720.
"""

import time
import logging
import threading
from typing import Dict, Optional, Any

# Import model manager components
try:
    from core.neural.model_manager import ModelManager

    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    logging.warning(
        "model_manager module not found, SutazAI model management features unavailable"
    )

try:
    from core.neural.llama_utils import get_optimized_model, create_llama_prompt

    LLAMA_UTILS_AVAILABLE = True
except ImportError:
    LLAMA_UTILS_AVAILABLE = False
    logging.warning("llama_utils module not found, Llama model support unavailable")

# Set up logging
logger = logging.getLogger("sutazai.model_controller")


class ModelController:
    """
    High-level controller for SutazAI model management system providing
    a simplified API for applications to interact with models.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "ModelController":
        """Get or create singleton instance of ModelController"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)
            return cls._instance

    def __init__(
        self,
        auto_initialize: bool = True,
        auto_download: bool = True,
        auto_optimize: bool = True,
    ):
        """Initialize the model controller"""
        self.initialized = False
        self.auto_download = auto_download
        self.auto_optimize = auto_optimize
        self._model_manager = None
        self._loaded_models = {}

        # Auto-initialize if requested
        if auto_initialize:
            self.initialize()

    def initialize(self) -> bool:
        """Initialize the controller and underlying model manager"""
        if self.initialized:
            return True

        if not MODEL_MANAGER_AVAILABLE:
            logger.error(
                "Cannot initialize controller - model_manager module not available"
            )
            return False

        try:
            # Initialize model manager
            self._model_manager = ModelManager(
                auto_download=self.auto_download, auto_optimize=self.auto_optimize
            )

            self.initialized = True
            logger.info("Model controller initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing model controller: {e}")
            return False

    def ensure_initialized(self) -> bool:
        """Ensure controller is initialized before operation"""
        if not self.initialized:
            return self.initialize()
        return True

    def get_model(
        self, model_id: Optional[str] = None, force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Get a model, automatically downloading and optimizing if necessary

        Args:
            model_id: The model ID to get, or None for the recommended model
            force_reload: Whether to force reload the model even if already loaded

        Returns:
            Dictionary with the model and status information
        """
        if not self.ensure_initialized():
            return {"success": False, "error": "Controller not initialized"}

        try:
            # Get recommended model if none specified
            if model_id is None:
                model_id = self._model_manager.get_recommended_model()
                logger.info(f"Using recommended model: {model_id}")

            # Check if model is already loaded and we don't need to reload
            if model_id in self._loaded_models and not force_reload:
                logger.info(f"Using already loaded model: {model_id}")
                return {
                    "success": True,
                    "model": self._loaded_models[model_id]["model"],
                    "model_id": model_id,
                    "cached": True,
                }

            # Get model info from manager
            model_info = self._model_manager.get_model(
                model_id=model_id, optimize=self.auto_optimize, wait=True
            )

            if not model_info.get("success", False):
                logger.error(
                    f"Failed to get model {model_id}: {model_info.get('error')}"
                )
                return {"success": False, "error": model_info.get("error")}

            # Load the model
            model_result = self._load_model(model_id, model_info)

            if not model_result.get("success", False):
                return model_result

            # Cache the loaded model
            self._loaded_models[model_id] = {
                "model": model_result["model"],
                "loaded_at": time.time(),
                "info": model_info,
            }

            return {
                "success": True,
                "model": model_result["model"],
                "model_id": model_id,
                "model_info": model_info,
                "cached": False,
            }

        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return {"success": False, "error": str(e)}

    def _load_model(self, model_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a model based on its type

        Args:
            model_id: The model ID
            model_info: Model information from the manager

        Returns:
            Dictionary with the loaded model and status
        """
        # Determine model type from path or ID
        model_path = model_info.get("model_path")
        optimized_path = model_info.get("optimized_path")

        if not model_path:
            return {"success": False, "error": f"No model path found for {model_id}"}

        # Use optimized path if available, otherwise use original
        path_to_use = optimized_path if optimized_path else model_path

        # Check file extension or model ID to determine type
        if (
            model_path.endswith(".gguf")
            or "llama" in model_id.lower()
            or "mistral" in model_id.lower()
        ):
            # For Llama models
            if not LLAMA_UTILS_AVAILABLE:
                return {"success": False, "error": "Llama utils not available"}

            try:
                # Load Llama model
                if optimized_path and optimized_path.endswith(".json"):
                    # This is a config file
                    model = get_optimized_model(config_path=optimized_path)
                else:
                    model = get_optimized_model(path_to_use)

                return {"success": True, "model": model, "model_type": "llama"}

            except Exception as e:
                logger.error(f"Error loading Llama model {model_id}: {e}")
                return {"success": False, "error": str(e)}
        else:
            # For transformer models
            # NOTE: This is a placeholder for transformer model loading
            logger.warning(
                f"Transformer model loading not yet implemented for {model_id}"
            )
            return {
                "success": False,
                "error": "Transformer model loading not implemented",
                "model_type": "transformer",
            }

    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using a model

        Args:
            prompt: The prompt text
            model_id: Optional model ID, or None for recommended model
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters to pass to the model

        Returns:
            Dictionary with generation results
        """
        if not self.ensure_initialized():
            return {"success": False, "error": "Controller not initialized"}

        try:
            # Get the model
            model_result = self.get_model(model_id)

            if not model_result.get("success", False):
                return model_result

            model = model_result["model"]
            model_id = model_result["model_id"]

            # Prepare prompt based on model type
            if (
                isinstance(model_result.get("model_type"), str)
                and model_result["model_type"] == "llama"
            ):
                # For Llama models, format with system prompt if provided
                if system_prompt:
                    formatted_prompt = create_llama_prompt(system_prompt, prompt)
                else:
                    formatted_prompt = prompt

                # Generate text
                start_time = time.time()
                result = model.generate(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                generation_time = time.time() - start_time

                # Format the result
                return {
                    "success": True,
                    "model_id": model_id,
                    "response": result["choices"][0]["text"],
                    "generation_time_sec": generation_time,
                    "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": result.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                    "model_type": "llama",
                }
            else:
                # Placeholder for transformer models
                return {
                    "success": False,
                    "error": "Generation not implemented for this model type",
                    "model_id": model_id,
                }

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"success": False, "error": str(e)}

    def list_models(self) -> Dict[str, Any]:
        """
        List all available models

        Returns:
            Dictionary with list of models and status
        """
        if not self.ensure_initialized():
            return {"success": False, "error": "Controller not initialized"}

        try:
            models = self._model_manager.list_models(include_details=True)

            # Filter and format the results
            formatted_models = []
            for model in models:
                if "error" in model:
                    continue

                formatted_model = {
                    "id": model["model_id"],
                    "version": model["version"],
                    "size_mb": model.get("size_mb", 0),
                    "available": model.get("exists", False),
                    "optimized": model.get("optimized", False),
                    "loaded": model["model_id"] in self._loaded_models,
                }

                if "metrics" in model:
                    formatted_model["performance"] = {
                        "tokens_per_second": model["metrics"].get("tokens_per_second"),
                        "success_rate": model["metrics"].get("success_rate", 1.0),
                    }

                formatted_models.append(formatted_model)

            return {
                "success": True,
                "models": formatted_models,
                "loaded_models": list(self._loaded_models.keys()),
                "count": len(formatted_models),
            }

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"success": False, "error": str(e)}

    def get_recommended_model(self) -> Dict[str, Any]:
        """
        Get the recommended model for this system

        Returns:
            Dictionary with recommended model ID and status
        """
        if not self.ensure_initialized():
            return {"success": False, "error": "Controller not initialized"}

        try:
            model_id = self._model_manager.get_recommended_model()

            return {
                "success": True,
                "model_id": model_id,
                "is_loaded": model_id in self._loaded_models,
            }

        except Exception as e:
            logger.error(f"Error getting recommended model: {e}")
            return {"success": False, "error": str(e)}

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """
        Unload a model to free memory

        Args:
            model_id: The model ID to unload

        Returns:
            Dictionary with unload status
        """
        if not self.ensure_initialized():
            return {"success": False, "error": "Controller not initialized"}

        try:
            if model_id not in self._loaded_models:
                return {"success": False, "error": f"Model {model_id} not loaded"}

            # Get the model object
            model_obj = self._loaded_models[model_id]["model"]

            # Call __del__ method to clean up resources
            if hasattr(model_obj, "__del__"):
                model_obj.__del__()

            # Remove from loaded models
            del self._loaded_models[model_id]

            return {"success": True, "model_id": model_id}

        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return {"success": False, "error": str(e)}

    def cleanup(self) -> Dict[str, Any]:
        """
        Clean up all resources

        Returns:
            Dictionary with cleanup status
        """
        try:
            # Unload all models
            for model_id in list(self._loaded_models.keys()):
                self.unload_model(model_id)

            # Clean up model manager
            if self._model_manager:
                self._model_manager.cleanup()

            self.initialized = False

            return {"success": True}

        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
            return {"success": False, "error": str(e)}


# Simple API function for easy access
def get_controller() -> ModelController:
    """Get the singleton model controller instance"""
    return ModelController.get_instance()


def generate_text(
    prompt: str,
    model_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate text using a model (simplified API)

    Args:
        prompt: The prompt text
        model_id: Optional model ID, or None for recommended model
        system_prompt: Optional system instructions
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with generation results
    """
    controller = get_controller()
    return controller.generate(
        prompt=prompt,
        model_id=model_id,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )


# CLI functionality
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="SutazAI Model Controller")

    # Commands
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument(
        "--recommended", action="store_true", help="Get recommended model"
    )
    parser.add_argument("--generate", type=str, help="Generate text from prompt")
    parser.add_argument(
        "--system-prompt", type=str, help="System instructions for generation"
    )
    parser.add_argument("--model", type=str, help="Model ID to use")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )

    args = parser.parse_args()

    # Create controller
    controller = get_controller()

    try:
        # Handle commands
        if args.list:
            result = controller.list_models()

            if result.get("success"):
                print(f"Available models ({result['count']}):")
                for model in result["models"]:
                    status = "✓" if model["available"] else "✗"
                    loaded = "[loaded]" if model["loaded"] else ""
                    opt = "[optimized]" if model["optimized"] else ""
                    print(f"  {status} {model['id']} {loaded} {opt}")

                    if "performance" in model:
                        perf = model["performance"]
                        if perf.get("tokens_per_second"):
                            print(
                                f"    Speed: {perf['tokens_per_second']:.1f} tokens/sec"
                            )

                print(
                    f"\nLoaded models: {', '.join(result['loaded_models']) if result['loaded_models'] else 'None'}"
                )
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

        elif args.recommended:
            result = controller.get_recommended_model()

            if result.get("success"):
                print(f"Recommended model: {result['model_id']}")
                print(f"Status: {'Loaded' if result['is_loaded'] else 'Not loaded'}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

        elif args.generate:
            print(f"Generating text with prompt: {args.generate}")
            result = controller.generate(
                prompt=args.generate,
                model_id=args.model,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            if result.get("success"):
                print("\nGenerated text:")
                print("-------------------")
                print(result["response"])
                print("-------------------")
                print(f"Model: {result['model_id']}")
                print(f"Time: {result['generation_time_sec']:.2f} seconds")
                if "completion_tokens" in result:
                    print(
                        f"Tokens: {result['completion_tokens']} (completion) + {result['prompt_tokens']} (prompt)"
                    )
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

        else:
            parser.print_help()

    finally:
        # Clean up resources
        controller.cleanup()
