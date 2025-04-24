import logging
import os
import asyncio
import httpx
import json
from typing import Dict, Any

# Import shared components
from sutazai_agi.core.config_loader import get_setting

logger = logging.getLogger(__name__)

# Configuration for AICI controller
AICI_CONTROLLER_URL = get_setting("code_tools.aici_controller_url", "http://127.0.0.1:8080/v1")
AICI_API_KEY = get_setting("code_tools.aici_api_key", None) # If required

async def run_aici_controller(
    prompt: str,
    controller_module_id: str, # The WASM module ID
    controller_arg: Any, # Arguments for the controller module
    agent_config: Dict[str, Any],
    timeout: int = 120 # Default timeout in seconds
) -> Dict[str, Any]:
    """Interacts with an AICI controller to generate constrained text.

    Args:
        prompt: The base prompt for the LLM.
        controller_module_id: The identifier for the AICI WASM controller module.
        controller_arg: The argument(s) to pass to the AICI controller module.
                       Usually a JSON-serializable object (dict, list, str, etc.).
        agent_config: The configuration dictionary for this specific agent instance.
        timeout: Request timeout in seconds.

    Returns:
        A dictionary containing the status and the AICI controller's response.
    """
    agent_name = agent_config.get("name", "AICI Runner")
    logger.info(f"Attempting AICI controller interaction for agent '{agent_name}'...")

    if not AICI_CONTROLLER_URL:
        logger.error("AICI Controller URL ('code_tools.aici_controller_url') is not configured.")
        return {"status": "error", "message": "AICI controller URL not configured."}

    endpoint = f"{AICI_CONTROLLER_URL.rstrip('/')}/run"

    payload = {
        "controller": controller_module_id,
        "controller_arg": controller_arg, # Must be JSON-serializable
        "prompt": prompt,
        # Add other parameters as needed by the AICI API (e.g., sampling params)
        # "sampling_params": {
        #     "max_tokens": 200,
        #     "temperature": 0.5
        # }
    }

    headers = {
        "Content-Type": "application/json",
    }
    if AICI_API_KEY:
        headers["Authorization"] = f"Bearer {AICI_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.debug(f"Sending request to AICI: {endpoint} with payload: {json.dumps(payload)[:200]}...")
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)

            result = response.json()
            logger.info(f"AICI controller '{controller_module_id}' execution successful for '{agent_name}'.")
            # The structure of `result` depends heavily on the AICI controller implementation
            # It likely contains the generated text, logs, etc.
            return {"status": "success", "output": result}

    except httpx.TimeoutException:
        logger.error(f"Request to AICI controller timed out after {timeout} seconds.")
        return {"status": "error", "message": f"AICI request timed out ({timeout}s)."}
    except httpx.RequestError as e:
        logger.error(f"Error requesting AICI controller at '{endpoint}': {e}", exc_info=True)
        # Provide more context if possible (e.g., connection error)
        error_context = str(e)
        if isinstance(e, httpx.ConnectError):
            error_context = f"Connection error to {e.request.url}"
        return {"status": "error", "message": f"AICI request failed: {error_context}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"AICI controller returned an error: {e.response.status_code} - {e.response.text[:200]}")
        return {
            "status": "error", 
            "message": f"AICI controller error: {e.response.status_code}", 
            "output": e.response.text
        }
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from AICI controller.")
        return {"status": "error", "message": "Invalid JSON response from AICI controller."}
    except Exception as e:
        logger.error(f"Unexpected error during AICI interaction for '{agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected AICI interaction error: {e}"} 