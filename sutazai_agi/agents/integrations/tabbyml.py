import logging
import httpx # Using httpx for async requests
from typing import Optional, Dict, Any

from sutazai_agi.core.config_loader import get_setting

logger = logging.getLogger(__name__)

class TabbyMLClient:
    """Client for interacting with a self-hosted TabbyML code completion server."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or get_setting("tabbyml_api_url")
        if not self.base_url:
            logger.error("TabbyML API URL is not configured (checked settings key 'tabbyml_api_url'). Client will not function.")
            # Consider raising an error or having a disabled state
            raise ValueError("TabbyML base URL not configured.")
            
        self.completion_url = f"{self.base_url.rstrip('/')}/v1/completions"
        # Use an async client for non-blocking IO
        self.client = httpx.AsyncClient(timeout=10.0) # 10 second timeout
        logger.info(f"TabbyMLClient initialized for URL: {self.base_url}")

    async def check_connection(self) -> bool:
        """Checks if the TabbyML server is reachable (basic check)."""
        if not self.base_url:
            return False
        try:
            # A simple health check endpoint might be better if TabbyML provides one
            # For now, just try reaching the base URL
            response = await self.client.get(self.base_url)
            response.raise_for_status() # Raise exception for 4xx/5xx status codes
            logger.info(f"Successfully connected to TabbyML server at {self.base_url}")
            return True
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to TabbyML server at {self.base_url}: {e}")
            return False
        except httpx.HTTPStatusError as e:
             logger.error(f"TabbyML server at {self.base_url} returned status {e.response.status_code}")
             return False # Server reached but returned error status
        except Exception as e:
             logger.error(f"An unexpected error occurred checking TabbyML connection: {e}", exc_info=True)
             return False

    async def get_completion(self, 
                             language: str, 
                             prompt: str, 
                             suffix: Optional[str] = None, 
                             segments: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Requests code completion from the TabbyML server.

        Args:
            language: The programming language of the code.
            prompt: The code prefix for which completion is requested.
            suffix: Optional code suffix following the completion point.
            segments: Optional dictionary containing prefix and suffix. If provided, overrides prompt/suffix args.

        Returns:
            A dictionary containing the completion choices, or None if an error occurs.
            Example response structure (based on typical completion APIs):
            {
                "id": "cmpl-xxxx",
                "choices": [
                    { "index": 0, "text": " completion text..." }
                ]
            }
        """
        if not self.base_url:
            logger.error("Cannot get completion, TabbyMLClient not configured.")
            return None

        payload = {
            "language": language,
            "prompt": prompt
        }
        if suffix:
             payload["suffix"] = suffix
        if segments:
             payload["segments"] = segments # TabbyML specific? Check API docs

        logger.debug(f"Requesting TabbyML completion. Language: {language}, Prompt length: {len(prompt)}")
        
        try:
            response = await self.client.post(self.completion_url, json=payload)
            response.raise_for_status()
            completion_data = response.json()
            logger.debug(f"Received TabbyML completion: {completion_data}")
            return completion_data
        except httpx.RequestError as e:
            logger.error(f"Request to TabbyML server failed: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"TabbyML server returned error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred getting TabbyML completion: {e}", exc_info=True)
            return None

# Example usage (for testing)
async def _test_tabby():
     # Ensure TABBYML_API_URL is set in settings.yaml or .env
     # Example: tabbyml_api_url: http://localhost:8080
     logging.basicConfig(level=logging.INFO)
     try:
         client = TabbyMLClient()
         connected = await client.check_connection()
         if connected:
             completion = await client.get_completion(language="python", prompt="def hello_world():\n    ")
             if completion:
                 print("Completion received:", completion)
             else:
                 print("Failed to get completion.")
         else:
              print("Could not connect to TabbyML server.")
     except ValueError as e:
         print(f"Configuration error: {e}")
     except Exception as e:
          print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(_test_tabby()) 