import logging
from typing import Dict, Any, Optional, List
import ollama

from sutazai_agi.core.config_loader import get_setting

logger = logging.getLogger(__name__)

# Cache loaded models/clients to avoid reloading
_ollama_client: Optional[ollama.Client] = None

def get_ollama_client() -> ollama.Client:
    """Initializes and returns the Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        base_url = get_setting("ollama_base_url", "http://127.0.0.1:11434")
        logger.info(f"Initializing Ollama client with base URL: {base_url}")
        try:
            # Check connectivity first
            _ollama_client = ollama.Client(host=base_url)
            _ollama_client.list() # Test connection
            logger.info("Ollama client initialized and connected successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize or connect Ollama client at {base_url}: {e}", exc_info=True)
            # Potentially raise an exception or handle gracefully
            raise ConnectionError(f"Could not connect to Ollama at {base_url}. Is the server running?") from e
    return _ollama_client

class LLMInterface:
    """Provides a unified interface for interacting with different LLMs, currently focusing on Ollama."""

    def __init__(self):
        self.client = get_ollama_client()
        # Load default parameters from settings
        self.default_model_params = get_setting("model_parameters", {})
        logger.info(f"LLMInterface initialized. Default params: {self.default_model_params}")

    def list_available_models(self) -> List[str]:
        """Lists models available through the Ollama server."""
        try:
            models_info = self.client.list()
            return [model['name'] for model in models_info.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}", exc_info=True)
            return []

    def generate(self, 
                 prompt: str, 
                 model: Optional[str] = None, 
                 system_message: Optional[str] = None, 
                 stream: bool = False, 
                 **kwargs) -> Dict[str, Any]: # Or potentially Iterator[Dict] if stream=True
        """Generates text using the specified Ollama model.

        Args:
            prompt: The user prompt.
            model: The model name (e.g., 'llama2'). Defaults to 'default_llm_model' from settings.
            system_message: An optional system message.
            stream: Whether to stream the response chunk by chunk.
            **kwargs: Additional Ollama parameters (e.g., temperature, top_k) to override defaults.

        Returns:
            A dictionary containing the response, or an iterator if streaming.
            Example non-streaming response format from ollama client:
            {
                'model': 'llama2:latest',
                'created_at': '2023-12-12T14:13:43.416702Z',
                'response': 'The sky is blue because of Rayleigh scattering...',
                'done': True, 
                # ... other stats like total_duration, eval_count etc.
            }
        """
        target_model = model or get_setting("default_llm_model", "llama2")
        
        # Combine default params with call-specific kwargs
        params = {**self.default_model_params, **kwargs}

        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': prompt})
        
        logger.debug(f"Generating text with model '{target_model}', stream={stream}, params={params}")
        
        try:
            response = self.client.chat(
                model=target_model,
                messages=messages,
                stream=stream,
                options=params # Pass parameters via options dict
            )
            logger.debug(f"Received response (or stream initiator) from model '{target_model}'")
            return response
        except Exception as e:
            logger.error(f"Error during generation with model '{target_model}': {e}", exc_info=True)
            # Return a structured error response
            return {
                'error': str(e),
                'model': target_model,
                'done': True,
                'response': None
            }

    def generate_embedding(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """Generates embeddings for the given text using the specified Ollama model.
        
        Args:
            text: The text to embed.
            model: The embedding model name. Defaults to 'default_embedding_model' from settings.
            
        Returns:
            A list of floats representing the embedding, or None if an error occurs.
        """
        target_model = model or get_setting("default_embedding_model")
        if not target_model:
            logger.error("No embedding model specified in settings or arguments.")
            return None
            
        logger.debug(f"Generating embedding for text using model '{target_model}'")
        try:
            result = self.client.embeddings(model=target_model, prompt=text)
            embedding = result.get("embedding")
            if embedding:
                logger.debug(f"Successfully generated embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.error(f"Ollama response for embedding did not contain 'embedding' key. Result: {result}")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding with model '{target_model}': {e}", exc_info=True)
            return None

# --- Global LLM Interface Instance --- 
_llm_interface: Optional[LLMInterface] = None

def get_llm_interface() -> LLMInterface:
    """Returns a singleton instance of the LLMInterface."""
    global _llm_interface
    if _llm_interface is None:
        try:
            _llm_interface = LLMInterface()
        except ConnectionError:
             # Handle case where Ollama isn't running during initial import/setup
             logger.critical("LLMInterface could not be initialized. Ensure Ollama server is running.")
             # Depending on desired behavior, could raise here or allow graceful failure later
             _llm_interface = None # Mark as failed initialization 
             # Re-raise or handle appropriately elsewhere
             raise 
        except Exception as e:
             logger.critical(f"Unexpected error initializing LLMInterface: {e}", exc_info=True)
             _llm_interface = None
             raise
    # If initialization failed previously, return None or raise error
    if _llm_interface is None:
        raise RuntimeError("LLMInterface initialization failed previously. Cannot provide interface.")
    return _llm_interface

# Example Usage:
# if __name__ == '__main__':
#     try:
#         llm_interface = get_llm_interface()
#         print("Available models:", llm_interface.list_available_models())
        
#         print("\nGenerating response:")
#         response = llm_interface.generate(prompt="Why is the sky blue?")
#         print(response.get('response'))

#         print("\nGenerating embedding:")
#         embedding = llm_interface.generate_embedding(text="This is a test sentence.")
#         if embedding:
#             print(f"Embedding dimension: {len(embedding)}")
#             print(f"First few dimensions: {embedding[:5]}")
#         else:
#             print("Failed to generate embedding.")

#     except ConnectionError as e:
#         print(f"Connection Error: {e}")
#     except Exception as e:
#         print(f"An error occurred: {e}") 