import yaml
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---

_settings: Optional[Dict[str, Any]] = None
_agents_config: Optional[Dict[str, Any]] = None

# Define default paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SETTINGS_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
DEFAULT_AGENTS_PATH = os.path.join(PROJECT_ROOT, "config", "agents.yaml")

def _load_config_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads a YAML configuration file."""
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {file_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def load_settings(settings_path: str = DEFAULT_SETTINGS_PATH) -> Dict[str, Any]:
    """Loads the main settings configuration."""
    global _settings
    if _settings is None:
        _settings = _load_config_file(settings_path)
        if _settings is None:
            _settings = {} # Return empty dict if loading failed
            logger.warning("Settings configuration could not be loaded. Using defaults where possible.")
    return _settings

def load_agents_config(agents_path: str = DEFAULT_AGENTS_PATH) -> Dict[str, Any]:
    """Loads the agents configuration."""
    global _agents_config
    if _agents_config is None:
        _agents_config = _load_config_file(agents_path)
        if _agents_config is None:
            _agents_config = {"agents": [], "tools": {}} # Return empty structure
            logger.warning("Agents configuration could not be loaded.")
    return _agents_config

def get_setting(key: str, default: Any = None) -> Any:
    """Retrieves a specific setting by key, loading settings if necessary."""
    settings = load_settings()
    # Allow nested key access using dot notation, e.g., "vector_store.provider"
    keys = key.split('.')
    value = settings
    try:
        for k in keys:
            if isinstance(value, dict):
                value = value[k]
            else:
                # Handle cases where intermediate keys might not exist
                logger.debug(f"Intermediate key '{k}' not found for setting '{key}'.")
                return default
        # Override with environment variables if they exist (e.g., SUTAZAI_JWT_SECRET_KEY)
        env_var_name = f"SUTAZAI_{key.replace('.', '_').upper()}"
        env_value = os.getenv(env_var_name)
        if env_value is not None:
            logger.info(f"Overriding setting '{key}' with environment variable {env_var_name}.")
            # Attempt basic type casting based on default or existing value
            if isinstance(value, bool):
                return env_value.lower() in ['true', '1', 'yes']
            elif isinstance(value, int):
                return int(env_value)
            elif isinstance(value, float):
                return float(env_value)
            # Add more type conversions if needed (e.g., lists from comma-separated strings)
            return env_value
        return value
    except (KeyError, TypeError):
        logger.debug(f"Setting key '{key}' not found in configuration.")
        return default

def get_agent_config(agent_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the configuration for a specific agent by name."""
    agents_config = load_agents_config()
    for agent in agents_config.get("agents", []):
        if agent.get("name") == agent_name:
            return agent
    logger.warning(f"Configuration for agent '{agent_name}' not found.")
    return None

def get_all_agent_configs() -> List[Dict[str, Any]]:
    """Returns the list of all agent configurations."""
    agents_config = load_agents_config()
    return agents_config.get("agents", [])

def get_tool_config(tool_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the configuration for a specific tool by name."""
    agents_config = load_agents_config()
    return agents_config.get("tools", {}).get(tool_name)

# Example Usage (can be removed or put under if __name__ == '__main__'):
# if __name__ == '__main__':
#     settings = load_settings()
#     agents_cfg = load_agents_config()
#     print("--- Settings ---")
#     print(settings)
#     print("\n--- Agents Config ---")
#     print(agents_cfg)
#     print(f"\nOllama Base URL: {get_setting('ollama_base_url')}")
#     print(f"Default LLM: {get_setting('default_llm_model', 'Not Set')}")
#     print(f"LangChain Agent Config: {get_agent_config('LangChain Chat Agent')}")
#     print(f"Search Tool Config: {get_tool_config('search_local_docs')}")
#     print(f"Non-existent setting: {get_setting('non_existent.key', 'Default Value')}") 