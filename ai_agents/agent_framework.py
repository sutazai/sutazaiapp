#!/usr/bin/env python3
"""
Agent Framework

This module provides a unified framework for integrating various agent types
from different frameworks (AutoGPT, LocalAGI, LangChain, etc.) into the AGI/ASI system.
It provides standardized interfaces and coordination mechanisms for different agent types.
"""

import os
import json
import logging
import threading
import uuid
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import math # For calculator tool
import re
import pathlib

# Assuming ModelManager is correctly imported and typed
from ai_agents.model_manager import ModelManager
# Assuming default_api tool exists in the execution context
# from your_api_module import default_api # Placeholder if needed

# Langchain specific imports - handle potential absence gracefully if not always needed
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate
    from langchain.tools import Tool
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define dummy types if LangChain isn't installed, to prevent NameErrors later
    AgentExecutor = type("AgentExecutor", (object,), {})
    Tool = type("Tool", (object,), {})
    ConversationBufferMemory = type("ConversationBufferMemory", (object,), {})
    # Can't log here reliably as logger might not be configured yet
    # logger.warning("LangChain components not found. LangChain agents will not be available.")


# Configure logging
LOG_FILE = "logs/agent_framework.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("AgentFramework")

if not LANGCHAIN_AVAILABLE:
     logger.warning("LangChain components not found during import. LangChain agents will not be available.")


# --- Enums ---

class AgentFrameworkType(Enum):
    """Types of agent frameworks supported"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    LOCALAGI = "localagi"
    AUTOGEN = "autogen"
    CREWAI = "crewai" # Placeholder
    CUSTOM = "custom"


class AgentCapability(Enum):
    """Capabilities that agents can have"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    TOOL_USE = "tool_use"
    WEB_SEARCH = "web_search"
    PLANNING = "planning"
    FILE_OPERATIONS = "file_operations"
    MEMORY = "memory"
    REASONING = "reasoning"
    DOCUMENT_PROCESSING = "document_processing"
    MULTIMODAL = "multimodal" # Placeholder
    COLLABORATION = "collaboration" # Placeholder
    CODE_UNDERSTANDING = "code_understanding"

class AgentState(Enum):
    """Possible states of an agent instance"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"

# --- Dataclasses ---

@dataclass
class AgentTool:
    """Represents a tool that an agent can use"""
    name: str
    description: str
    # Restore: function is required
    function: Callable[..., Dict[str, Any]] 
    parameters: Dict[str, Any] = field(default_factory=dict) # Describes expected parameters for the tool
    requires_capability: List[AgentCapability] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for API responses"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_capability": [cap.value for cap in self.requires_capability],
        }

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str # Unique identifier/name for this agent configuration
    description: str
    framework: AgentFrameworkType
    capabilities: List[AgentCapability]
    model_id: str # Identifier for the model used (handled by ModelManager)
    tools: List[str] = field(default_factory=list) # List of tool names this agent can use
    parameters: Dict[str, Any] = field(default_factory=dict) # Default parameters for the agent/framework

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "framework": self.framework.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "model_id": self.model_id,
            "tools": self.tools,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create AgentConfig from dictionary representation"""
        try:
            # Perform validation during creation
            framework_val = data["framework"]
            capabilities_val = data["capabilities"]

            if not isinstance(framework_val, str):
                 raise ValueError("framework must be a string")
            if not isinstance(capabilities_val, list):
                 raise ValueError("capabilities must be a list")

            # Ensure model_id is present and a string
            if "model_id" not in data or not isinstance(data["model_id"], str) or not data["model_id"]:
                 raise ValueError("model_id is required and must be a non-empty string")


            # Validate framework enum
            try:
                 framework_enum = AgentFrameworkType(framework_val)
            except ValueError:
                 raise ValueError(f"Invalid framework type: {framework_val}. Valid types are: {[e.value for e in AgentFrameworkType]}")

            # Validate capabilities enum
            capabilities_enum = []
            for cap in capabilities_val:
                 if not isinstance(cap, str):
                      raise ValueError(f"Capability must be a string, got: {cap}")
                 try:
                      capabilities_enum.append(AgentCapability(cap))
                 except ValueError:
                      raise ValueError(f"Invalid capability: {cap}. Valid capabilities are: {[e.value for e in AgentCapability]}")

            tools_list = data.get("tools", [])
            if not isinstance(tools_list, list) or not all(isinstance(t, str) for t in tools_list):
                 raise ValueError("tools must be a list of strings")

            params_dict = data.get("parameters", {})
            if not isinstance(params_dict, dict):
                 raise ValueError("parameters must be a dictionary")


            return cls(
                name=data["name"],
                description=data["description"],
                framework=framework_enum,
                capabilities=capabilities_enum,
                model_id=data["model_id"],
                tools=tools_list,
                parameters=params_dict,
            )
        except KeyError as ke:
            raise ValueError(f"Missing required key in agent config data: {ke}")
        except ValueError as ve:
            # Catches errors from Enum creation or explicit raises above
             raise ValueError(f"Invalid value in agent config data: {ve}")


# --- Agent Framework Class ---

class AgentFramework:
    """
    Unified framework for integrating different agent types

    Handles agent configuration, lifecycle (creation, termination),
    task execution routing, tool registration, and coordination.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config_path: str = "config/agents.json",
        tools_dir: str = "ai_agents/tools", # Currently unused, tools defined internally
        max_concurrent_agents: int = 5, # Placeholder, not enforced yet
    ):
        """
        Initialize the agent framework

        Args:
            model_manager: ModelManager instance for agent models.
            config_path: Path to agent configuration file (JSON).
            tools_dir: Directory containing agent tools (Not used currently).
            max_concurrent_agents: Maximum number of concurrent agents (Placeholder).
        """
        if not isinstance(model_manager, ModelManager):
             raise TypeError("model_manager must be an instance of ModelManager")
        self.model_manager = model_manager
        self.config_path = Path(config_path)
        self.tools_dir = Path(tools_dir)
        self.max_concurrent_agents = max_concurrent_agents # Concurrency limit not implemented

        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize agent registries and state
        self.agent_configs: Dict[str, AgentConfig] = {} # agent_id -> AgentConfig
        self.active_agents: Dict[str, Dict[str, Any]] = {} # instance_id -> {"type": str, "instance": object, "config": AgentConfig}
        self.agent_states: Dict[str, AgentState] = {} # instance_id -> AgentState
        self.agent_locks: Dict[str, threading.Lock] = {} # instance_id -> Lock (Use asyncio.Lock if fully async)

        # Initialize tool registry
        self.tools: Dict[str, AgentTool] = {} # tool_name -> AgentTool

        # Load configurations and tools
        self._load_config()
        self._load_tools() # Tools depend on self methods, load after init

        logger.info(
            f"Agent framework initialized with {len(self.agent_configs)} agent configs and {len(self.tools)} tools"
        )

    # --- Configuration Loading ---

    def _load_config(self):
        """Load agent configurations from the JSON file."""
        temp_configs: Dict[str, AgentConfig] = {}
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.debug(f"Raw config loaded from {self.config_path}")

                    agents_dict = config_data.get("agents", {})
                    if not isinstance(agents_dict, dict):
                        logger.error("Invalid config format: 'agents' key is not a dictionary.")
                        self._create_default_config() # Fallback to default
                        return

                    logger.debug(f"Found 'agents' dictionary with {len(agents_dict)} entries.")
                    successful_loads = 0
                    for agent_id, agent_data in agents_dict.items():
                        try:
                            if not isinstance(agent_data, dict):
                                logger.warning(f"Skipping invalid agent config entry for '{agent_id}': not a dictionary.")
                                continue
                            # Use agent_id from the dict key as the primary ID
                            temp_configs[agent_id] = AgentConfig.from_dict(agent_data)
                            successful_loads += 1
                        except ValueError as agent_load_error:
                            logger.error(f"Error parsing config for agent '{agent_id}': {agent_load_error}")
                        except Exception as e:
                            logger.error(f"Unexpected error loading agent '{agent_id}': {e}", exc_info=True)

                    self.agent_configs = temp_configs
                    logger.info(f"Loaded {successful_loads}/{len(agents_dict)} valid agent configurations.")
                    if successful_loads == 0 and len(agents_dict) > 0:
                         logger.warning("No valid agent configurations loaded from existing file. Consider creating defaults.")
                    elif successful_loads == 0:
                         logger.info("No agent configurations found in file.")


            else:
                logger.warning(f"Configuration file not found at {self.config_path}. Creating default config.")
                self._create_default_config() # This also loads the defaults into self.agent_configs

        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from {self.config_path}: {json_err}")
            logger.info("Creating default configuration as fallback due to JSON error.")
            self._create_default_config()
        except IOError as io_err:
            logger.error(f"Error reading configuration file {self.config_path}: {io_err}")
            # Decide how to handle this - perhaps raise an exception or create default?
            logger.info("Creating default configuration as fallback due to IO error.")
            self._create_default_config()
        except Exception as e:
            logger.error(f"Unexpected error loading agent configurations: {e}", exc_info=True)
            logger.info("Creating default configuration as fallback due to unexpected error.")
            self._create_default_config()

    def _create_default_config(self):
        """Create and save a default agent configuration file."""
        default_config_data = {
            "agents": {
                "langchain_chat_agent": { # Example ID
                    "name": "LangChain Chat Agent",
                    "description": "General purpose chat agent using LangChain ReAct and codebase tools.",
                    "framework": "langchain",
                    "capabilities": ["text_generation", "reasoning", "tool_use", "code_understanding", "file_operations"],
                    "model_id": "llama3-8b", # Default model
                    "tools": ["codebase_search", "read_file", "list_dir", "grep_search", "file_search"],
                    "parameters": {"temperature": 0.5, "max_iterations": 7},
                },
                "coder_agent": {
                    "name": "Code Assistant",
                    "description": "Generates and modifies code using custom framework.",
                    "framework": "custom",
                    "capabilities": ["code_generation", "reasoning", "file_operations"],
                    "model_id": "deepseek-coder", # Specific model for coding
                    "tools": ["read_file", "write_file"],
                    "parameters": {"temperature": 0.2, "max_tokens": 2048},
                },
                 "doc_analyzer": {
                    "name": "Document Analyzer",
                    "description": "Analyzes documents using vector search.",
                    "framework": "custom", # Example custom agent
                    "capabilities": ["document_processing", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["search_documents"], # Tool for vector DB
                    "parameters": {"temperature": 0.3},
                 }
                # Add other default agents as needed (autogpt, localagi placeholders etc.)
            }
        }

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_config_data, f, indent=2)
            logger.info(f"Created default agent configuration file at {self.config_path}")

            # Load these default configs into memory
            self.agent_configs.clear()
            for agent_id, agent_data in default_config_data["agents"].items():
                try:
                    self.agent_configs[agent_id] = AgentConfig.from_dict(agent_data)
                except ValueError as e:
                     logger.error(f"Error loading default config for '{agent_id}': {e}")

            logger.info(
                f"Loaded {len(self.agent_configs)} default agent configurations."
            )
        except IOError as io_err:
            logger.error(f"Failed to write default configuration file {self.config_path}: {io_err}")
        except Exception as e:
            logger.error(f"Unexpected error creating default configuration: {e}", exc_info=True)

    def _save_config(self):
        """Save the current agent configurations to the JSON file."""
        config_to_save = {
            "agents": {
                agent_id: config.to_dict()
                for agent_id, config in self.agent_configs.items()
            }
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Saved agent configurations to {self.config_path}")
        except IOError as io_err:
            logger.error(f"Error writing configuration file {self.config_path}: {io_err}")
        except Exception as e:
             logger.error(f"Unexpected error saving configuration: {e}", exc_info=True)

    # --- Tool Loading and Registration (Restore Backup Logic) ---

    def _load_tools(self):
        """Load available tools for agents. Tools defined internally based on backup."""
        self.tools.clear() # Ensure clean slate if reloaded

        # Restore definitions from backup, linking to self._tool_* methods
        tool_definitions = [
            AgentTool(
                name="read_file",
                description="Read the contents of a specific file within the /opt/sutazaiapp workspace. Provide start/end lines (1-indexed) or use should_read_entire_file=true.",
                function=self._tool_read_file, # RESTORED
                parameters={
                    "target_file": {"type": "string", "description": "Path to the file relative to /opt/sutazaiapp."},
                    "start_line_one_indexed": {"type": "integer", "description": "(Optional) 1-indexed start line. Default is 1."},
                    "end_line_one_indexed_inclusive": {"type": "integer", "description": "(Optional) 1-indexed end line. Default is 200 lines from start."},
                    "should_read_entire_file": {"type": "boolean", "description": "(Optional) Read the whole file, ignoring range. Default false."}
                },
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="list_dir",
                description="List contents (files and subdirectories) of a directory relative to the workspace root /opt/sutazaiapp.",
                function=self._tool_list_dir, # RESTORED
                parameters={"relative_workspace_path": {"type": "string", "description": "Directory path relative to workspace (e.g., 'ai_agents/tools'). Use '.' for root."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="codebase_search",
                description="Semantic search for code snippets within the workspace.",
                function=self._tool_codebase_search, # RESTORED
                parameters={
                    "query": {"type": "string", "description": "Natural language description of code to find."},
                    "target_directories": {"type": "array", "items": {"type": "string"}, "description": "(Optional) List of directories (glob patterns relative to workspace root) to search within. E.g., ['ai_agents/*', 'backend/routers']."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="grep_search",
                description="Fast exact text or regular expression search within files in the workspace.",
                function=self._tool_grep_search, # RESTORED
                parameters={
                    "query": {"type": "string", "description": "Regex or exact text pattern to search for."},
                    "include_pattern": {"type": "string", "description": "(Optional) Glob pattern for files to include (e.g., '*.py', 'config/*.json')."},
                    "exclude_pattern": {"type": "string", "description": "(Optional) Glob pattern for files or directories to exclude (e.g., 'logs/*', '*.tmp')."},
                    "case_sensitive": {"type": "boolean", "description": "(Optional) Perform a case-sensitive search. Default false."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING, AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="file_search",
                description="Fuzzy search for files by path name within the workspace.",
                function=self._tool_file_search, # RESTORED
                parameters={"query": {"type": "string", "description": "Partial or potentially misspelled file name or path."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
             AgentTool(
                name="write_file",
                 description="Write or overwrite content to a specific file within the /opt/sutazaiapp workspace.",
                function=self._tool_write_file, # Keep
                parameters={
                     "file_path": {"type": "string", "description": "Path to the file relative to /opt/sutazaiapp."},
                     "content": {"type": "string", "description": "The entire content to write to the file."}
                 },
                 requires_capability=[AgentCapability.FILE_OPERATIONS],
             ),
             AgentTool(
                name="execute_code",
                 description="Execute a snippet of code in a sandboxed environment (Placeholder - Currently UNSAFE and Not Implemented).",
                function=self._tool_execute_code, # Keep
                 parameters={
                     "code": {"type": "string", "description": "The code snippet to execute."},
                     "language": {"type": "string", "description": "Programming language (e.g., 'python', 'javascript')."}
                 },
                 requires_capability=[AgentCapability.CODE_EXECUTION],
             ),
             AgentTool(
                name="search_documents",
                 description="Search for relevant information across indexed documents (e.g., PDFs, text files) using vector similarity.",
                function=self._tool_search_documents, # Keep
                 parameters={
                     "query": {"type": "string", "description": "The query to search for within the documents."},
                     "n_results": {"type": "integer", "description": "Number of results to return. Default is 5."}
                 },
                 requires_capability=[AgentCapability.DOCUMENT_PROCESSING, AgentCapability.REASONING],
             ),
             AgentTool(
                 name="calculator",
                 description="Evaluate a simple mathematical expression. Supports basic arithmetic (+, -, *, /) and parentheses.",
                 function=self._tool_calculator, # Keep
                 parameters={"expression": {"type": "string", "description": "Mathematical expression string (e.g., '(2 + 3) * 4')."}},
                 requires_capability=[], # No specific capability needed, but potentially reasoning
             ),
        ]

        for tool in tool_definitions:
            self.register_tool(tool) # Use register_tool for validation

    def register_tool(self, tool: AgentTool) -> bool:
        """Register a new tool. Fails if name conflicts or function invalid."""
        if not isinstance(tool, AgentTool):
            logger.error("Invalid object passed to register_tool. Must be AgentTool.")
            return False
        if not tool.name or not isinstance(tool.name, str):
            logger.error("Tool name must be a non-empty string.")
            return False
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered. Overwriting.")
            # Allow overwriting for now

        # Restore simple validation from backup: Check if the function exists on self
        # This assumes tool.function is always provided now
        if not hasattr(self, tool.function.__name__) or not callable(getattr(self, tool.function.__name__)):
            logger.error(f"Tool function '{tool.function.__name__}' specified for tool '{tool.name}' not found or not callable on AgentFramework instance.")
            return False

        self.tools[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")
        return True

    # --- Tool Implementations (_tool_*) ---
    # Re-adding methods from backup

    def _tool_read_file(self, target_file: str, start_line_one_indexed: Optional[int] = None, end_line_one_indexed_inclusive: Optional[int] = None, should_read_entire_file: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Read file using standard Python I/O with safety checks."""
        logger.info(f"Tool: Attempting to read file '{target_file}'")
        abs_target_file = os.path.abspath(target_file)

        if not is_safe_path(abs_target_file):
            logger.warning(f"Denied unsafe file read attempt: {target_file}")
            return {"success": False, "error": "File access restricted to workspace or path is invalid."}
        
        try:
            path_obj = pathlib.Path(abs_target_file)
            if not path_obj.is_file():
                return {"success": False, "error": f"File not found or is not a regular file: {target_file}"}

            # Check file size limit before reading
            if path_obj.stat().st_size > MAX_READ_BYTES:
                 return {"success": False, "error": f"File exceeds maximum allowed size ({MAX_READ_BYTES} bytes)."}

            start = int(start_line_one_indexed) if start_line_one_indexed is not None else 1
            end = int(end_line_one_indexed_inclusive) if end_line_one_indexed_inclusive is not None else -1
            read_all = bool(should_read_entire_file)
            start = max(1, start) # Ensure start is positive
            
            lines_to_return = []
            with open(path_obj, 'r', encoding='utf-8', errors='replace') as f:
                if read_all or (start == 1 and end == -1):
                    lines_to_return = f.readlines(MAX_READ_BYTES) # Read up to limit
                    if len(lines_to_return) > MAX_READ_LINES:
                        lines_to_return = lines_to_return[:MAX_READ_LINES]
                        logger.warning(f"Read truncated to {MAX_READ_LINES} lines for {target_file}")
                else:
                    # Read specific range
                    current_line = 0
                    effective_end = float('inf') if end == -1 else end
                    line_count = 0
                    for line in f:
                        current_line += 1
                        if current_line >= start and current_line <= effective_end:
                            lines_to_return.append(line)
                            line_count += 1
                            if line_count >= MAX_READ_LINES:
                                logger.warning(f"Read range truncated to {MAX_READ_LINES} lines for {target_file}")
                                break 
                        if current_line > effective_end:
                             break # Stop reading past the requested end line
                             
            # Convert lines to string content
            content = "\n".join(lines_to_return)
            return {"success": True, "result": {"content": content}}

        except FileNotFoundError:
            logger.error(f"File not found in _tool_read_file: {target_file}")
            return {"success": False, "error": f"File not found: {target_file}"}
        except PermissionError:
             logger.error(f"Permission denied reading file: {target_file}")
             return {"success": False, "error": f"Permission denied reading file: {target_file}"}
        except ValueError as ve:
            logger.error(f"Error processing parameters for _tool_read_file: {ve}")
            return {"success": False, "error": f"Invalid parameter type: {ve}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_read_file for '{target_file}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred while reading file: {str(e)}"}

    def _tool_list_dir(self, relative_workspace_path: str) -> Dict[str, Any]:
        """Tool implementation: List directory contents using standard Python libs."""
        logger.info(f"Tool: Attempting to list directory '{relative_workspace_path}'")
        # Construct absolute path and check safety
        abs_target_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, relative_workspace_path))
        
        if not is_safe_path(abs_target_path):
             logger.warning(f"Denied unsafe directory list attempt: {relative_workspace_path}")
             return {"success": False, "error": "Directory access restricted to workspace or path is invalid."}
        
        try:
            path_obj = pathlib.Path(abs_target_path)
            if not path_obj.is_dir():
                return {"success": False, "error": f"Path is not a valid directory: {relative_workspace_path}"}

            contents = []
            for item in path_obj.iterdir():
                item_type = "dir" if item.is_dir() else "file" if item.is_file() else "other"
                contents.append({"name": item.name, "type": item_type})
                if len(contents) >= MAX_SEARCH_RESULTS * 2: # Limit results early
                    logger.warning(f"Directory listing truncated for {relative_workspace_path}")
                    break 
            
            return {"success": True, "result": {"contents": contents}}

        except FileNotFoundError:
            logger.error(f"Directory not found in _tool_list_dir: {relative_workspace_path}")
            return {"success": False, "error": f"Directory not found: {relative_workspace_path}"}
        except PermissionError:
             logger.error(f"Permission denied listing directory: {relative_workspace_path}")
             return {"success": False, "error": f"Permission denied listing directory: {relative_workspace_path}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_list_dir for '{relative_workspace_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred listing directory: {str(e)}"}

    def _tool_grep_search(self, query: str, include_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None, case_sensitive: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Basic regex search across files in the workspace."""
        logger.info(f"Tool: Performing grep search for pattern: '{query}'")
        if not query:
            return {"success": False, "error": "Empty query provided for grep search."}

        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags)
        except re.error as re_err:
            return {"success": False, "error": f"Invalid regex pattern: {re_err}"}

        # Walk through the workspace root
        file_count = 0
        match_count = 0
        try:
            for root, _, files in os.walk(WORKSPACE_ROOT):
                 # Basic include/exclude filtering (can be enhanced)
                 if exclude_pattern and re.search(exclude_pattern, root):
                      continue
                      
                 for filename in files:
                      # Basic include/exclude on filename
                      if exclude_pattern and re.search(exclude_pattern, filename):
                           continue
                      if include_pattern and not re.search(include_pattern, filename):
                           continue
                           
                      file_path = os.path.join(root, filename)
                      file_count += 1
                      # Ensure we don't search outside workspace (redundant check, but safe)
                      if not is_safe_path(file_path):
                           continue
                           
                      try:
                           # Only search text-based files (simple check, could be improved)
                           # Avoid reading huge files entirely
                           if os.path.getsize(file_path) > MAX_READ_BYTES // 10: # Limit grep search size
                                continue
                                
                           with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line_num, line in enumerate(f, 1):
                                     if regex.search(line):
                                         matches.append({
                                             "file": os.path.relpath(file_path, WORKSPACE_ROOT),
                                             "line_number": line_num,
                                             "line_content": line.strip()
                                         })
                                         match_count += 1
                                         if match_count >= MAX_SEARCH_RESULTS:
                                             logger.warning("Grep search truncated to max results.")
                                             raise StopIteration # Use exception to break out of nested loops
                      except (OSError, UnicodeDecodeError) as file_err:
                           logger.warning(f"Could not read/decode file during grep: {file_path} - {file_err}")
                      except StopIteration:
                           break # Break outer loop
                 if match_count >= MAX_SEARCH_RESULTS:
                      break

            logger.info(f"Grep search completed. Scanned {file_count} files, found {len(matches)} matches.")
            return {"success": True, "result": {"matches": matches}}

        except Exception as e:
            logger.error(f"Unexpected error during grep search for '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during grep search: {str(e)}"}

    def _tool_file_search(self, query: str) -> Dict[str, Any]:
        """Tool implementation: Basic filename search using glob."""
        logger.info(f"Tool: Performing file search for query: '{query}'")
        if not query:
            return {"success": False, "error": "Empty query provided for file search."}

        found_files = []
        match_count = 0
        try:
            # Use pathlib glob for potentially simpler pattern matching
            # Make query a glob pattern (e.g., add wildcards if not present)
            glob_query = f"**/*{query}*" # Basic substring search
            
            for path in pathlib.Path(WORKSPACE_ROOT).rglob(glob_query):
                if path.is_file(): # Only return files
                    found_files.append(os.path.relpath(str(path), WORKSPACE_ROOT))
                    match_count += 1
                    if match_count >= MAX_SEARCH_RESULTS:
                         logger.warning("File search truncated to max results.")
                         break
            
            logger.info(f"File search completed. Found {len(found_files)} matching files.")
            return {"success": True, "result": {"found_files": found_files}}

        except Exception as e:
            logger.error(f"Unexpected error during file search for '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during file search: {str(e)}"}

    def _tool_codebase_search(self, query: str, target_directories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Tool implementation: Simple text search across code files (like grep). No real semantics."""
        logger.info(f"Tool: Performing basic codebase text search for: '{query}'")
        if not query:
             return {"success": False, "error": "Empty query provided for codebase search."}
             
        include_pattern = None
        if target_directories:
             escaped_dirs = [re.escape(d.strip(os.sep)) for d in target_directories]
             # Corrected the list comprehension syntax:
             pattern_parts = [
                 f"(?:{re.escape(os.sep)}{d}{re.escape(os.sep)}|^{re.escape(os.sep)}{d}|{d}{re.escape(os.sep)}$)" 
                 for d in escaped_dirs
             ]
             include_pattern = r'|'.join(pattern_parts)
             logger.info(f"Codebase search mapped target_directories to include_pattern: {include_pattern}")
             
        # Reuse grep implementation for basic text search
        return self._tool_grep_search(query=query, include_pattern=include_pattern, case_sensitive=False)

    # --- Other Existing Tool Implementations (Keep as is if they don't use api_client) ---
    # e.g., _tool_write_file, _tool_execute_code, _tool_search_documents (if it uses model_manager), _tool_calculator

    # --- Keep existing tool implementations below this point ---
    def _tool_write_file(self, target_file: str, content: str, create_directories: Optional[bool] = True) -> Dict[str, Any]:
        """Tool implementation: Write content to a file with safety checks."""
        try:
            # Path validation
            workspace_root = Path("/opt/sutazaiapp").resolve()
            abs_path = (workspace_root / target_file).resolve()

            if workspace_root not in abs_path.parents and abs_path != workspace_root:
                 logger.warning(f"Attempt to write file outside workspace: {target_file} (Resolved: {abs_path})")
                 return {"success": False, "error": "File write access restricted to workspace."}

            target_file_str = str(abs_path)

            logger.info(f"Tool: Writing to file '{target_file_str}'")
            abs_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(abs_path, "w", encoding="utf-8") as f:
                bytes_written = f.write(content)
            logger.info(f"Successfully wrote {bytes_written} bytes to file '{target_file_str}'")
            return {"success": True, "result": {"file_path": target_file_str, "bytes_written": bytes_written}}
        except IOError as io_err:
             logger.error(f"IOError writing to file {target_file}: {io_err}")
             return {"success": False, "error": f"IOError writing file: {io_err}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_write_file for '{target_file}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred writing file: {str(e)}"}

    def _tool_execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Tool implementation: Execute code in sandbox (Placeholder - UNSAFE)."""
        logger.warning("execute_code tool is a placeholder and DOES NOT run code securely.")
        # THIS IS UNSAFE - Requires a proper sandboxing service (Docker, etc.)
        # Replace this entire block with a call to your sandbox service API
        return {
            "success": False,
            "error": "Code execution sandbox not implemented yet.",
            "result": {"output": "", "error_details": "Sandbox not implemented"}
        }

    def _tool_search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Tool implementation: Search documents in vector DB via ModelManager."""
        logger.info(f"Tool: Executing search_documents with query: '{query}', n_results: {n_results}")
        try:
            # Validate n_results
            try:
                num_results = max(1, int(n_results))
            except (ValueError, TypeError):
                 logger.warning(f"Invalid n_results '{n_results}', defaulting to 5.")
                 num_results = 5

            # Vector DB model ID (needs to be configured in ModelManager)
            vector_db_model_id = "chroma-store" # Or another ID configured for your vector DB

            logger.debug(f"Querying vector DB '{vector_db_model_id}' for '{query}' (k={num_results})")
            search_params = {"k": num_results}

            # Assuming ModelManager run_inference handles queries to vector stores
            results = self.model_manager.run_inference(
                vector_db_model_id,
                query, # Input is the query string
                parameters=search_params
            )

            # Process results (structure depends heavily on ModelManager's vector store implementation)
            if isinstance(results, dict) and "error" in results:
                 logger.error(f"Vector DB query failed via ModelManager: {results['error']}")
                 return {"success": False, "error": results['error']}

            # Example: Assuming results are returned in a list of dicts format
            if isinstance(results, list) and all(isinstance(item, dict) for item in results):
                 formatted_results = results # Use directly if already in desired format
            elif isinstance(results, dict) and "results" in results and isinstance(results["results"], list):
                 formatted_results = results["results"] # Extract if nested
            else:
                 # Handle ChromaDB-like structure if ModelManager returns that directly
                 if isinstance(results, dict) and all(k in results for k in ["ids", "documents"]):
                    formatted_results = [
                        {
                            "id": doc_id,
                            "content": doc_content if doc_content is not None else "",
                            "metadata": metadata if metadata is not None else {},
                            "distance": distance if distance is not None else float('inf'),
                        }
                        for doc_id, doc_content, metadata, distance in zip(
                            results.get("ids", [[]])[0],
                            results.get("documents", [[]])[0],
                            results.get("metadatas", [[]])[0],
                            results.get("distances", [[]])[0] # Chroma specific?
                        )
                    ]
                 else:
                      logger.error(f"Unexpected result format from vector DB via ModelManager: {type(results)}")
                      return {"success": False, "error": "Unexpected format from vector database."}

            logger.info(f"Vector DB search successful, found {len(formatted_results)} results.")
            return {"success": True, "result": {"search_results": formatted_results}}

        except Exception as e:
            logger.error(f"Unexpected error in _tool_search_documents for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during document search: {str(e)}"}

    def _tool_calculator(self, expression: str) -> Dict[str, Any]:
        """Tool implementation: Evaluate a simple mathematical expression safely."""
        allowed_chars = "0123456789+-*/(). " # Whitelist basic characters
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression."}

        try:
            # Use ast.literal_eval for safety, but it doesn't do calculations.
            # A dedicated math expression parser is better (e.g., numexpr, asteval)
            # Using restricted eval with math module enabled.
            logger.warning("Calculator tool uses restricted eval. Consider a dedicated math parser for production.")
            # Ensure safety by passing a restricted globals dictionary
            safe_globals = {"__builtins__": {}, "math": math}
            result = eval(expression, safe_globals, {})

            # Ensure result is serializable (convert Decimal, etc. if needed)
            result_str = str(result)
            logger.info(f"Tool: Calculated '{expression}' = {result_str}")
            return {"success": True, "result": {"value": result_str}}
        except SyntaxError:
            logger.error(f"Syntax error evaluating expression '{expression}'")
            return {"success": False, "error": "Invalid mathematical expression syntax."}
        except NameError as ne:
             logger.error(f"NameError evaluating expression '{expression}': {ne}")
             return {"success": False, "error": f"Invalid function or variable used: {ne}"}
        except TypeError as te:
            logger.error(f"TypeError evaluating expression '{expression}': {te}")
            return {"success": False, "error": f"Type error during calculation: {te}"}
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}", exc_info=True)
            return {"success": False, "error": f"Failed to evaluate expression: {str(e)}"}

    # --- Agent Lifecycle Management ---

    def _get_agent_config(self, agent_identifier: str) -> Optional[Tuple[str, AgentConfig]]:
        """Find agent config by ID or fallback to name."""
        if agent_identifier in self.agent_configs:
            return agent_identifier, self.agent_configs[agent_identifier]
        # Fallback: search by name if ID not found
        for agent_id, config in self.agent_configs.items():
            if config.name == agent_identifier:
                 logger.warning(f"Agent identifier '{agent_identifier}' matched by name, not ID. Found config ID: '{agent_id}'")
                 return agent_id, config
        return None

    async def create_agent(self, agent_identifier: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create an agent instance from a configuration ID or name.

        Args:
            agent_identifier: ID or Name of the agent configuration.
            parameters: Optional runtime parameters to override/add to config defaults.

        Returns:
            Instance ID (str) for the created agent, or None if creation failed.
        """
        config_result = self._get_agent_config(agent_identifier)
        if not config_result:
            logger.error(f"Agent configuration '{agent_identifier}' not found.")
            return None

        config_id, config = config_result # Use the actual config ID found
        instance_id = f"{config_id}-{uuid.uuid4().hex[:8]}"
        logger.info(f"Attempting to create agent instance '{instance_id}' from config '{config_id}'.")

        # Prepare combined parameters (config defaults + runtime overrides)
        # Start with config defaults
        combined_params = config.parameters.copy() if config.parameters else {}
        # Update/add runtime parameters if provided and valid
        if parameters:
            if isinstance(parameters, dict):
                combined_params.update(parameters)
                logger.debug(f"Merged runtime parameters for instance {instance_id}")
            else:
                logger.warning(f"Invalid runtime 'parameters' format for instance {instance_id} (expected dict), using config defaults only.")

        # Prepare lock (Use asyncio.Lock if create_method becomes async)
        self.agent_locks[instance_id] = threading.Lock()

        # Set initial state before acquiring lock
        self.agent_states[instance_id] = AgentState.INITIALIZING

        # Acquire lock for creation phase
        with self.agent_locks[instance_id]:
            try:
                # --- Call the framework-specific creation method ---
                create_method_name = f"_create_{config.framework.value}_agent"
                create_method = getattr(self, create_method_name, None)

                agent_data: Optional[Dict[str, Any]] = None
                if create_method and callable(create_method):
                    logger.info(f"Creating agent instance '{instance_id}' using framework: {config.framework.value}")
                    # Check if the creation method itself is async
                    if asyncio.iscoroutinefunction(create_method):
                        agent_data = await create_method(instance_id, config, combined_params)
                    else:
                        # Run synchronous creation method (consider executor if it blocks significantly)
                        # This might block the async event loop if `create_method` is slow!
                        # Consider `await asyncio.to_thread(create_method, instance_id, config, combined_params)`
                        agent_data = create_method(instance_id, config, combined_params)
                else:
                    logger.error(f"Unsupported agent framework '{config.framework.value}' for agent '{instance_id}'.")
                    agent_data = {"error": f"Unsupported framework: {config.framework.value}"}

                # --- Validate creation result ---
                if not isinstance(agent_data, dict) or agent_data.get("error") or not agent_data.get("instance"):
                    error_msg = agent_data.get("error", "Unknown agent creation failure") if isinstance(agent_data, dict) else "Creation method returned invalid data"
                    logger.error(f"Failed to create agent instance '{instance_id}': {error_msg}")
                    self.agent_states[instance_id] = AgentState.ERROR
                    # Clean up lock and state if creation failed within lock
                    self.agent_locks.pop(instance_id, None)
                    self.agent_states.pop(instance_id, None) # Remove state too
                    return None # Indicate failure

                # --- Store the successfully created agent instance ---
                # Ensure required keys are present
                agent_type = agent_data.get("type", config.framework.value) # Get type from result or default to config
                agent_instance = agent_data["instance"]

                self.active_agents[instance_id] = {
                    "type": agent_type,
                    "instance": agent_instance,
                    "config": config # Store the specific config used
                }
                self.agent_states[instance_id] = AgentState.IDLE # Set state to IDLE *after* success
                logger.info(f"Created and stored new instance '{instance_id}' (Type: {agent_type})")
                return instance_id # Return the ID on success

            except NotImplementedError as nie:
                # Catch errors specifically raised by placeholder create methods
                 logger.error(f"Agent creation not implemented for instance '{instance_id}' (Framework: {config.framework.value}): {nie}")
                 self.agent_states[instance_id] = AgentState.ERROR
                 self.agent_locks.pop(instance_id, None)
                 self.agent_states.pop(instance_id, None)
                 return None
            except Exception as e:
                # Catch critical errors during the creation process (e.g., within create_method)
                logger.error(f"Critical error during agent creation for '{instance_id}': {e}", exc_info=True)
                self.agent_states[instance_id] = AgentState.ERROR
                # Clean up lock and state on critical failure
                self.agent_locks.pop(instance_id, None)
                self.agent_states.pop(instance_id, None) # Remove state too
                return None # Indicate failure

    def _create_framework_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        DEPRECATED - Logic moved into create_agent.
        Factory method placeholder.
        """
        # This method's logic is now integrated directly into create_agent
        # to handle async creation methods correctly.
        logger.error("_create_framework_agent should not be called directly.")
        return {"error": "_create_framework_agent is deprecated"}


    def _create_langchain_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a LangChain agent instance (ReAct), using the model specified in AgentConfig."""
        error_return = {"type": "langchain", "instance": None, "config": config, "error": "LangChain components not available"}
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain is not installed. Cannot create LangChain agent.")
            return error_return
        error_return["error"] = None

        try:
            # --- 1. Load LLM Instance (Use model_id from agent config) ---
            model_id = config.model_id # <-- Revert: Use the ID from the agent config
            logger.info(f"Requesting model manager to load model '{model_id}' for agent '{instance_id}'.")

            if not self.model_manager.load_model(model_id): # <-- Revert: Use model_id
                 error_msg = f"Model manager failed to load model '{model_id}'. Ensure it exists in model config."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return

            loaded_model_data = self.model_manager.loaded_models.get(model_id) # <-- Revert: Use model_id
            if not loaded_model_data or "instance" not in loaded_model_data:
                 error_msg = f"Model '{model_id}' not found in loaded models after load attempt."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return
            llm = loaded_model_data["instance"] # This should load ChatOllama based on models.json
            logger.info(f"Successfully obtained LLM instance for model: {model_id}")

            # --- 2. Check if LLM instance is valid --- 
            # (Keep this check)
            if not hasattr(llm, 'invoke') and not hasattr(llm, 'ainvoke'):
                 if not callable(llm) and not hasattr(llm, 'predict'):
                     error_msg = f"Loaded model instance for '{model_id}' is not a valid LangChain LLM/ChatModel."
                     logger.error(error_msg)
                     error_return["error"] = error_msg
                     return error_return
                 else:
                     logger.warning(f"LLM for model '{model_id}' uses older interface. Compatibility might vary.")

            # --- 3. Map Tools --- 
            # (Logic remains the same)
            agent_tools = []
            missing_tool_methods = []
            if config.tools:
                for tool_name in config.tools:
                    tool_config = self.tools.get(tool_name)
                    if tool_config and hasattr(tool_config, 'function') and tool_config.function:
                        agent_tools.append(
                            Tool(
                                name=tool_config.name,
                                func=tool_config.function,
                                description=tool_config.description
                            )
                        )
                        logger.debug(f"Mapped tool '{tool_name}' ({tool_config.name})")
                    else:
                        missing_tool_methods.append(tool_name)
                        logger.warning(f"Tool implementation or definition for '{tool_name}' not found.")
                
                if not agent_tools and config.tools:
                    error_msg = f"Agent '{config.name}' requires tools, but none could be mapped. Missing: {missing_tool_methods}"
                    logger.error(error_msg)
                    error_return["error"] = error_msg
                    return error_return
                logger.info(f"Mapped {len(agent_tools)} tools for agent '{instance_id}': {[t.name for t in agent_tools]}")
            else:
                 logger.info(f"No tools configured for agent '{instance_id}'.")

            # --- 4. Create Custom ReAct Prompt (Simplified Variables) ---
            # Define a stricter prompt template manually
            # NOTE: We will only define {input} and {agent_scratchpad} as input_variables.
            # create_react_agent is expected to handle the formatting for {tools} and {tool_names}.
            react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{{tools}}

Use the following exact format STRICTLY:

Thought: (Your reasoning process for the next step)
Action: (The action to take, MUST be one of [{{tool_names}}])
Action Input: (The input to the action. Use a JSON string like {{{{ \"param\": \"value\" }}}} if multiple inputs needed, otherwise a single string. Provide an empty string \"\" if no input needed.)
Observation: (The result of the action)
... (this Thought/Action/Action Input/Observation sequence can repeat N times)
Thought: (Your reasoning process for the final answer)
Final Answer: (The final answer to the original input question)

**VERY IMPORTANT RULES:**
1.  Always start your response with 'Thought:'.
2.  Every 'Thought:' MUST be followed by an 'Action:'.
3.  Every 'Action:' MUST be followed by an 'Action Input:'.
4.  You MUST use the exact keywords 'Thought:', 'Action:', 'Action Input:', 'Observation:', 'Final Answer:'.
5.  The 'Action:' MUST be exactly one of the available tool names: [{{tool_names}}].

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

            try:
                # Define the template with ONLY the variables we explicitly pass during execution.
                prompt = PromptTemplate(
                    template=react_prompt_template,
                    # Let create_react_agent handle 'tools' and 'tool_names' internally
                    input_variables=["input", "agent_scratchpad"] 
                )
                logger.info(
                    "Created simplified prompt template for ReAct agent."
                )
            except Exception as prompt_error:
                logger.error(f"Failed to create prompt template: {prompt_error}")
                error_return["error"] = f"Prompt template error: {prompt_error}"
                return error_return

            # --- 5. Create ReAct Agent --- 
            try:
                # create_react_agent expects the prompt to potentially use 'tools' and 'tool_names'
                # but we don't list them as input_variables for the PromptTemplate object itself.
                agent = create_react_agent(llm, agent_tools, prompt)
                logger.info("Successfully created ReAct agent with simplified prompt variables")
            except Exception as agent_creation_error:
                 logger.error(f"Failed create_react_agent: {agent_creation_error}", exc_info=True)
                 error_return["error"] = f"Failed to create ReAct agent: {agent_creation_error}"
                 return error_return

            # --- 6. Create Memory --- (Keep as is, if needed)
            # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # --- 7. Create Agent Executor --- (Add handle_parsing_errors)
            try:
                max_iterations = params.get("max_iterations", 15) # Get from params or default
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=agent_tools,
                    verbose=True, # Keep verbose for debugging
                    handle_parsing_errors=True, # Crucial for attempting recovery from format errors
                    max_iterations=max_iterations,
                    # memory=memory # Add back if memory is used
                )
                logger.info(
                    f"Successfully created LangChain ReAct agent executor for instance: {instance_id}"
                )
                return {"type": "langchain", "instance": agent_executor, "config": config}
            except Exception as executor_error:
                 logger.error(f"Failed to create AgentExecutor: {executor_error}", exc_info=True)
                 error_return["error"] = f"AgentExecutor creation error: {executor_error}"
                 return error_return

        except Exception as e:
            logger.error(f"Unexpected error creating LangChain agent instance '{instance_id}': {e}", exc_info=True)
            error_return["error"] = f"Unexpected agent creation error: {str(e)}"
            return error_return

    def _create_autogpt_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an AutoGPT agent."""
        logger.warning(f"Attempted to create AutoGPT agent '{instance_id}', but it is not implemented.")
        # Raise error here, will be caught by create_agent
        raise NotImplementedError("AutoGPT agent creation is not implemented.")

    def _create_localagi_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating a LocalAGI agent."""
        logger.warning(f"Attempted to create LocalAGI agent '{instance_id}', but it is not implemented.")
        raise NotImplementedError("LocalAGI agent creation is not implemented.")

    def _create_autogen_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an Autogen agent."""
        logger.warning(f"Attempted to create Autogen agent '{instance_id}', but it is not implemented.")
        raise NotImplementedError("Autogen agent creation is not implemented.")

    def _create_custom_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Creates a custom agent instance based on BaseAgent (defined below)."""
        error_return = {"type": "custom", "instance": None, "config": config}
        logger.info(f"Attempting to create custom agent '{instance_id}'.")
        try:
            # Import BaseAgent locally to minimize potential circular dependencies
            # Ensure base_agent.py exists and is correct
            try:
                 from ai_agents.base_agent import BaseAgent
            except ImportError as import_err:
                 logger.error(f"Failed to import BaseAgent: {import_err}")
                 error_return["error"] = "BaseAgent class not found."
                 return error_return


            # --- Define the Custom Agent Implementation ---
            # This nested class is less ideal; define it at module level if possible
            class CustomAgentImpl(BaseAgent):
                """Concrete implementation for a custom agent using BaseAgent."""
                def __init__(self, config: AgentConfig, model_manager: ModelManager, tools: Dict[str, AgentTool], params: Dict[str, Any]):
                    # Pass the AgentConfig object directly to BaseAgent's __init__
                    super().__init__(config) # BaseAgent expects AgentConfig
                    self.model_manager = model_manager
                    self.available_tools = tools # Use BaseAgent's self.tools? No, keep separate.
                    self.params = params
                    # self.model_id is already set in BaseAgent.__init__
                    self.history: List[Dict[str, Any]] = [] # Initialize conversation history
                    self.instance_id = instance_id # Store instance_id for logging
                    logger.debug(f"CustomAgentImpl {self.instance_id} initialized with model {self.model_id}")

                def _initialize(self):
                    """Initialize resources needed by the custom agent."""
                    logger.info(f"Initializing CustomAgentImpl {self.instance_id} for model: {self.model_id}")
                    # Load the required model via ModelManager
                    if not self.model_manager.load_model(self.model_id):
                        raise RuntimeError(f"Failed to load model {self.model_id} via ModelManager for {self.instance_id}.")
                    logger.info(f"CustomAgentImpl {self.instance_id} initialized successfully.")
                    # No return needed by BaseAgent's current structure

                async def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
                    """Execute a task using the custom agent's logic (now async)."""
                    task_id = task.get("id", "N/A")
                    logger.info(f"CustomAgentImpl {self.instance_id} executing task: {task_id}")
                    prompt = self._build_prompt(task)
                    logger.debug(f"CustomAgentImpl {self.instance_id} prompt snippet: {prompt[:200]}...")

                    inference_params = {
                        "temperature": self.params.get("temperature", 0.7),
                        "max_tokens": self.params.get("max_tokens", 1024),
                        # Add other relevant parameters from self.params if needed
                    }

                    try:
                        # --- Await Model Inference ---
                        # Assume run_inference is awaitable or use asyncio.to_thread if it's blocking
                        # Check if model_manager has an async method, e.g., arun_inference
                        if hasattr(self.model_manager, 'arun_inference') and asyncio.iscoroutinefunction(self.model_manager.arun_inference):
                             result = await self.model_manager.arun_inference(
                                  self.model_id,
                                  prompt,
                                  parameters=inference_params,
                             )
                        elif hasattr(self.model_manager, 'run_inference'):
                             logger.warning(f"ModelManager.run_inference is synchronous for {self.instance_id}. Running in executor.")
                             loop = asyncio.get_running_loop()
                             result = await loop.run_in_executor(None,
                                  self.model_manager.run_inference, # Function to run
                                  self.model_id,
                                  prompt,
                                  inference_params # Args for the function
                             )
                        else:
                             raise NotImplementedError("ModelManager does not have a suitable run_inference or arun_inference method.")

                    except Exception as inference_e:
                         logger.error(f"ModelManager inference failed for {self.instance_id}: {inference_e}", exc_info=True)
                         raise RuntimeError(f"Inference error: {inference_e}") from inference_e


                    if isinstance(result, dict) and "error" in result:
                        error_msg = f"Inference error from ModelManager: {result['error']}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Extract output text - adjust based on actual ModelManager response structure
                    output = result.get("text", result.get("completion", "No text output from model."))
                    logger.debug(f"CustomAgentImpl {self.instance_id} raw output snippet: {output[:200]}...")

                    # Basic history update
                    instruction = task.get('instruction', task.get('input', {}).get('messages', [{}])[-1].get('content', 'N/A'))
                    self.history.append({"task": instruction, "output": output})
                    # Limit history size
                    max_history = 10
                    if len(self.history) > max_history:
                        self.history = self.history[-max_history:]

                    # Return standard structure including status
                    return {"status": "success", "output": output, "task_id": task_id}

                def _cleanup(self):
                    """Clean up resources used by this custom agent instance."""
                    logger.info(f"Cleaning up CustomAgentImpl {self.instance_id} for model: {self.model_id}")
                    # Release model? ModelManager might handle this globally. Add specific cleanup if needed.
                    # No return needed by BaseAgent's current structure

                def _build_prompt(self, task: Dict[str, Any]) -> str:
                    """Build a prompt for the custom agent task."""
                    parts = []
                    # Adapt based on expected task structure ('instruction' or nested messages)
                    instruction = task.get('instruction', task.get('input', {}).get('messages', [{}])[-1].get('content', 'No instruction provided.'))
                    parts.append(f"### Task Instruction:\n{instruction}")

                    context = task.get("context") # Optional context provided in the task
                    if context:
                        parts.append(f"### Context:\n{context}")

                    if self.history:
                        hist_str_parts = []
                        for h in reversed(self.history): # Show most recent first
                             task_snip = h['task'][:150] + "..." if len(h['task']) > 150 else h['task']
                             output_snip = h['output'][:150] + "..." if len(h['output']) > 150 else h['output']
                             hist_str_parts.append(f"Previous Task: {task_snip}\nAgent Response: {output_snip}")
                        hist_str = "\n---\n".join(hist_str_parts)
                        parts.append(f"### Recent Conversation History (Newest First):\n{hist_str}")

                    if self.config.tools:
                        tool_desc_parts = []
                        for name in self.config.tools:
                             tool = self.available_tools.get(name)
                             if tool:
                                  tool_desc_parts.append(f"- {name}: {tool.description}")
                        if tool_desc_parts:
                             tool_desc = "\n".join(tool_desc_parts)
                             parts.append(f"### Available Tools:\n{tool_desc}") # Note: Custom agent doesn't inherently know how to *use* tools yet

                    parts.append("\n### Agent Response:")
                    return "\n\n".join(parts)

            # --- End of Nested CustomAgentImpl Class ---

            # --- Continue _create_custom_agent ---
            # Map only the tools needed by this specific agent config
            agent_specific_tools = {
                tool_name: self.tools[tool_name]
                for tool_name in config.tools if tool_name in self.tools
            }
            if len(agent_specific_tools) != len(config.tools):
                 logger.warning(f"Some tools listed in config for '{instance_id}' were not found/registered: {set(config.tools) - set(agent_specific_tools)}")


            # Create the agent instance
            custom_agent_instance = CustomAgentImpl(
                config=config,
                model_manager=self.model_manager,
                tools=agent_specific_tools, # Pass the filtered tools
                params=params, # Pass combined params
            )

            # Initialize the agent (critical step - calls _initialize)
            try:
                # BaseAgent.initialize is synchronous
                custom_agent_instance.initialize()
            except Exception as init_e:
                 logger.error(f"Failed to initialize custom agent '{instance_id}': {init_e}", exc_info=True)
                 error_return["error"] = f"Initialization failed: {init_e}"
                 return error_return

            logger.info(f"Successfully created and initialized custom agent instance '{instance_id}'.")
            return {"type": "custom", "instance": custom_agent_instance, "config": config}

        except Exception as e:
            logger.error(f"Unexpected error creating custom agent '{instance_id}': {e}", exc_info=True)
            error_return["error"] = f"Unexpected error: {str(e)}"
            return error_return

    # --- Task Execution ---

    async def execute_task(self, instance_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with an agent instance (now async).
        Handles locking and state management.

        Args:
            instance_id: ID of the agent instance.
            task: Task description dictionary (structure depends on agent type/task).
                  Expected keys might include 'instruction', 'context', 'id',
                  or nested 'input' with 'messages' for chat.

        Returns:
            Result dictionary, including 'status' ('success' or 'error')
            and 'output' or 'error' message.
        """
        if instance_id not in self.active_agents:
            logger.error(f"Agent instance {instance_id} not found for task execution.")
            return {"status": "error", "error": f"Agent instance {instance_id} not found"}

        lock = self.agent_locks.get(instance_id)
        if not lock:
            logger.error(f"Lock not found for agent instance {instance_id} during task execution.")
            return {"status": "error", "error": f"Internal error: Lock missing for instance {instance_id}"}

        # Use threading.Lock for now. If fully async, switch to asyncio.Lock and 'async with'
        with lock:
            try:
                # Check state before proceeding
                current_state = self.agent_states.get(instance_id)
                if current_state != AgentState.IDLE:
                    logger.warning(f"Attempted to execute task on non-idle agent {instance_id} (State: {current_state}).")
                    return {"status": "error", "error": f"Agent instance is currently busy (state: {current_state})."}

                self.agent_states[instance_id] = AgentState.PROCESSING
                logger.info(f"Executing task {task.get('id', 'N/A')} with agent instance {instance_id}")

                # Get agent data
                agent_data = self.active_agents[instance_id]
                agent_type = agent_data["type"]
                agent_instance = agent_data["instance"] # The actual agent object (e.g., AgentExecutor, CustomAgentImpl)

                # --- Route task execution based on agent type ---
                result: Dict[str, Any] = {}
                if agent_type == "langchain":
                    result = await self._execute_langchain_task(agent_instance, task)
                elif agent_type == "custom":
                    result = await self._execute_custom_task(agent_instance, task)
                elif agent_type == "autogpt":
                    result = self._execute_autogpt_task(agent_instance, task) # Will raise NotImplementedError
                elif agent_type == "localagi":
                    result = self._execute_localagi_task(agent_instance, task) # Will raise NotImplementedError
                elif agent_type == "autogen":
                    result = self._execute_autogen_task(agent_data, task) # Passes full dict, Will raise NotImplementedError
                else:
                    result = {"status": "error", "error": f"Task execution: Unknown agent type '{agent_type}'"}
                    logger.error(result["error"])

                # --- Update state based on execution result ---
                if result.get("status") == "error":
                     self.agent_states[instance_id] = AgentState.ERROR
                     logger.error(f"Task {task.get('id', 'N/A')} failed for {instance_id}: {result.get('error', 'Unknown execution error')}")
                else:
                    # Assume success if status is not 'error'
                    self.agent_states[instance_id] = AgentState.IDLE
                    logger.info(f"Task {task.get('id', 'N/A')} completed successfully for {instance_id}")

                # Return the full result dictionary
                return result

            except NotImplementedError as nie:
                logger.error(f"Task execution not implemented for agent type of instance {instance_id}: {nie}")
                self.agent_states[instance_id] = AgentState.ERROR
                return {"status": "error", "error": str(nie), "task_id": task.get("id")}
            except Exception as e:
                # Catch unexpected errors during execution routing or within sync helpers
                logger.error(f"Unexpected error executing task with agent {instance_id}: {e}", exc_info=True)
                self.agent_states[instance_id] = AgentState.ERROR
                return {"status": "error", "error": f"Internal task execution error: {str(e)}", "task_id": task.get("id")}
            finally:
                 # Ensure state is not left as PROCESSING if an error occurred before setting IDLE/ERROR
                 current_state_after = self.agent_states.get(instance_id)
                 if current_state_after == AgentState.PROCESSING:
                      logger.warning(f"Agent {instance_id} state was PROCESSING after execute_task finished; setting to ERROR.")
                      self.agent_states[instance_id] = AgentState.ERROR


    async def _execute_langchain_task(self, agent_executor: AgentExecutor, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LangChain agent executor using arun (async)."""
        task_id = task.get("id", "N/A")
        logger.debug(f"Executing LangChain task {task_id}")
        # Extract input based on common patterns (adapt as needed)
        # Priority: task['input_str'] > last message content > task['instruction']
        input_str = ""
        if "input_str" in task:
            input_str = task["input_str"]
        elif "input" in task and "messages" in task["input"] and isinstance(task["input"]["messages"], list) and task["input"]["messages"]:
            input_str = task["input"]["messages"][-1].get("content", "")
        elif "instruction" in task:
             input_str = task["instruction"]
        elif "prompt" in task: # Fallback for simple prompts
             input_str = task["prompt"]

        if not input_str:
             logger.error(f"No valid input ('input_str', message content, or 'instruction') found for LangChain task {task_id}")
             return {"status": "error", "error": "Missing input for LangChain agent", "task_id": task_id}

        logger.debug(f"LangChain agent input for task {task_id}: '{input_str[:100]}...'")

        try:
            # Get the default inputs (tools and tool_names) that were set during agent creation
            default_inputs = getattr(agent_executor, "_default_run_inputs", {})
            
            # Get tool descriptions from the agent executor's attributes
            tool_descriptions = getattr(agent_executor, "_tool_descriptions", "")
            tool_names = getattr(agent_executor, "_tool_names", "")
            
            # Create a clean input dictionary with just the input text for memory handling
            clean_inputs = {"input": input_str}
            
            # Use the agent_executor.memory directly if possible to manage history
            if hasattr(agent_executor, "memory") and agent_executor.memory:
                logger.debug(f"Using agent's memory for task {task_id}")
            
            # Use agent_executor.ainvoke for the modern LangChain interface if available
            if hasattr(agent_executor, 'ainvoke') and asyncio.iscoroutinefunction(agent_executor.ainvoke):
                # Pass only the clean input dictionary
                response_dict = await agent_executor.ainvoke(clean_inputs)
                response = response_dict.get("output", "No output from agent invoke") # Extract output
            elif hasattr(agent_executor, 'arun') and asyncio.iscoroutinefunction(agent_executor.arun):
                logger.warning("Using deprecated arun for LangChain agent execution.")
                # Pass only the input string
                response = await agent_executor.arun(input_str)
            else:
                logger.error(f"LangChain AgentExecutor for task {task_id} has no suitable async execution method (ainvoke/arun).")
                return {"status": "error", "error": "AgentExecutor does not support async execution.", "task_id": task_id}

            logger.info(f"LangChain task {task_id} execution successful.")
            # Standardize output format
            return {"status": "success", "result": {"response": response}, "task_id": task_id} # Nest response under result
        except Exception as e:
            logger.error(f"LangChain agent execution error for task {task_id}: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "task_id": task_id}

    # Placeholder execution methods for other frameworks
    def _execute_autogpt_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("AutoGPT execution not implemented.")
        raise NotImplementedError("AutoGPT execution not implemented")

    def _execute_localagi_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("LocalAGI execution not implemented.")
        raise NotImplementedError("LocalAGI execution not implemented")

    def _execute_autogen_task(self, agent_data, task: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("Autogen execution not implemented.")
        raise NotImplementedError("Autogen execution not implemented")

    async def _execute_custom_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a custom agent (now async)."""
        task_id = task.get("id", "N/A")
        logger.debug(f"Executing Custom task {task_id}")
        try:
            # Assuming agent._execute is async def and returns a dict
            result = await agent._execute(task)
            # Ensure result dictionary has 'status'
            if not isinstance(result, dict):
                 logger.error(f"Custom agent _execute method returned non-dict: {type(result)}")
                 return {"status": "error", "error": "Custom agent returned invalid format", "task_id": task_id}

            # Standardize: Ensure 'status' key exists
            if "status" not in result:
                 result["status"] = "success" # Assume success if no error status provided

            logger.info(f"Custom task {task_id} execution status: {result['status']}")

            # Standardize: put main output under 'result.response' key if successful
            if result["status"] == "success":
                 # Extract primary output, remove from top level, place in result dict
                 output = result.pop("output", result.pop("response", "No output key found"))
                 # Create the nested 'result' dictionary if it doesn't exist
                 if "result" not in result: result["result"] = {}
                 result["result"]["response"] = output
                 return result # Return the modified dict
            else:
                # If status is error, ensure error message is present
                if "error" not in result:
                     result["error"] = "Custom agent task failed without specific error message."
                return result # Return error structure as is

        except Exception as e:
            logger.error(f"Custom agent execution error for task {task_id}: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "task_id": task_id}

    # --- Agent Termination and State ---

    async def terminate_agent(self, instance_id: str) -> bool:
        """Terminate an agent instance and clean up resources (now async)."""
        logger.info(f"Attempting to terminate agent instance '{instance_id}'.")

        # Use lock to prevent race conditions during termination
        lock = self.agent_locks.get(instance_id)
        agent_data = None
        if lock:
             # Acquire the lock. Use try_lock or timeout if needed?
             # For now, assume acquiring the lock is acceptable blocking for termination.
             async with asyncio.Lock(): # Assuming we switch to asyncio.Lock eventually
                 # Re-fetch agent_data inside lock
                 agent_data = self.active_agents.pop(instance_id, None)
                 self.agent_states.pop(instance_id, None)
                 self.agent_locks.pop(instance_id, None) # Remove lock itself after use
        else:
             # If lock doesn't exist, agent might have failed creation or already terminated
             agent_data = self.active_agents.pop(instance_id, None) # Still try removing from active list
             self.agent_states.pop(instance_id, None) # And state list
             logger.debug(f"No lock found for instance {instance_id} during termination (might be normal).")


        if not agent_data:
            logger.warning(f"Agent instance '{instance_id}' not found or already removed during termination attempt.")
            return False # Indicate instance wasn't found to terminate

        # --- Attempt agent-specific cleanup ---
        agent_instance = agent_data.get("instance")
        if agent_instance and hasattr(agent_instance, "cleanup") and callable(agent_instance.cleanup):
            try:
                logger.debug(f"Calling cleanup for agent instance '{instance_id}'.")
                # Check if cleanup is async
                if asyncio.iscoroutinefunction(agent_instance.cleanup):
                    await agent_instance.cleanup()
                else:
                    # Run sync cleanup in executor to avoid blocking the event loop
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, agent_instance.cleanup)
                logger.debug(f"Cleanup finished for agent instance '{instance_id}'.")
            except Exception as e:
                logger.error(f"Error during cleanup for agent '{instance_id}': {e}", exc_info=True)
                # Continue termination despite cleanup error

        logger.info(f"Successfully terminated agent instance '{instance_id}'.")
        return True

    def get_agent_state(self, instance_id: str) -> Optional[AgentState]:
        """Get the current state of an agent instance."""
        return self.agent_states.get(instance_id)

    # --- Listing Methods ---

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agent configurations."""
        return [config.to_dict() for config in self.agent_configs.values()]

    def list_active_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all active agent instances and their states."""
        result = {}
        # Iterate over states as the source of truth for active/potentially errored instances
        for instance_id, state in self.agent_states.items():
             agent_data = self.active_agents.get(instance_id) # May be None if creation failed after state init
             config_name = "unknown"
             agent_type = "unknown"
             if agent_data and agent_data.get("config"):
                  config_name = agent_data["config"].name
                  agent_type = agent_data.get("type", agent_data["config"].framework.value)

             result[instance_id] = {
                 "config_name": config_name,
                 "type": agent_type,
                 "state": state.value,
             }
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [tool.to_dict() for tool in self.tools.values()]

    # --- Configuration Management ---

    def add_agent_config(self, agent_id: str, config_dict: Dict[str, Any]) -> bool:
        """Add or update an agent configuration and save to file."""
        if not agent_id or not isinstance(agent_id, str):
            logger.error("Invalid agent_id provided (must be non-empty string).")
            return False
        try:
            # Validate and create AgentConfig object before storing
            config = AgentConfig.from_dict(config_dict)
            self.agent_configs[agent_id] = config
            self._save_config() # Save immediately
            logger.info(f"Added/Updated agent configuration '{agent_id}'.")
            return True
        except ValueError as ve:
            logger.error(f"Invalid configuration format for agent '{agent_id}': {ve}")
            return False
        except Exception as e:
            logger.error(f"Failed to add/update agent config '{agent_id}': {e}", exc_info=True)
            return False

    def remove_agent_config(self, agent_id: str) -> bool:
        """Remove an agent configuration. Fails if agent instance using it is active."""
        if agent_id not in self.agent_configs:
            logger.warning(f"Agent configuration '{agent_id}' not found for removal.")
            return False

        config_to_remove = self.agent_configs[agent_id]

        # Check for active instances using this specific config ID or name
        active_instance_found = False
        for instance_id, agent_data in self.active_agents.items():
             # Check if the active instance's config object is the one being removed
             if agent_data.get("config") == config_to_remove:
                  logger.warning(
                      f"Cannot remove agent configuration '{agent_id}' ('{config_to_remove.name}') as instance '{instance_id}' is active. Terminate instance first.")
                  active_instance_found = True
                  break # No need to check further

        if active_instance_found:
            return False

        # Remove from configurations
        del self.agent_configs[agent_id]

        # Save updated configuration
        self._save_config()

        logger.info(f"Removed agent configuration '{agent_id}'.")
        return True

    # --- Framework Cleanup ---

    async def cleanup(self):
        """Clean up resources used by the agent framework (now async)."""
        logger.info("Cleaning up Agent Framework...")
        active_instance_ids = list(self.active_agents.keys())
        if active_instance_ids:
             logger.info(f"Terminating {len(active_instance_ids)} active agent instances during cleanup...")
             # Use asyncio.gather to terminate concurrently
             termination_tasks = [self.terminate_agent(instance_id) for instance_id in active_instance_ids]
             results = await asyncio.gather(*termination_tasks, return_exceptions=True)
             for instance_id, result in zip(active_instance_ids, results):
                  if isinstance(result, Exception):
                       logger.error(f"Error terminating agent {instance_id} during cleanup: {result}")
                  elif not result:
                       logger.warning(f"Termination reported failure for agent {instance_id} during cleanup (might have been removed already).")
        else:
            logger.info("No active agent instances to terminate during cleanup.")


        # Clear registries after termination attempts
        self.active_agents.clear()
        self.agent_states.clear()
        self.agent_locks.clear()

        logger.info("Agent Framework cleanup complete.")


# --- Base Agent Class (Example for Custom Agents) ---
# This should ideally be in its own file (e.g., base_agent.py)
# but included here for completeness since _create_custom_agent uses it.

class BaseAgent:
    """Abstract base class for custom agents."""
    def __init__(self, config: AgentConfig):
        if not isinstance(config, AgentConfig):
             raise TypeError("config must be an AgentConfig instance.")
        self.config = config
        self.model_id = config.model_id
        # Generate instance ID here if _create_custom_agent doesn't pass one?
        # Needs consistent ID handling. For now, assume instance_id is set externally or in subclass.
        # self.instance_id = f"{config.name}-base-{uuid.uuid4().hex[:4]}"

    def initialize(self):
        """Initialize agent resources. Should be called after __init__."""
        instance_id = getattr(self, 'instance_id', 'unknown_instance') # Get ID if set
        logger.info(f"Initializing BaseAgent for {instance_id}")
        # Call concrete implementation's initialization
        init_result = self._initialize()
        logger.info(f"BaseAgent initialization complete for {instance_id}")
        return init_result # Propagate result if any

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task. Should be called by the framework."""
        instance_id = getattr(self, 'instance_id', 'unknown_instance')
        logger.info(f"Executing task via BaseAgent for {instance_id}")
        # Call concrete implementation's execution (now async)
        result = await self._execute(task)
        logger.info(f"BaseAgent execution complete for {instance_id}")
        return result

    def cleanup(self):
        """Clean up agent resources. Should be called by the framework."""
        instance_id = getattr(self, 'instance_id', 'unknown_instance')
        logger.info(f"Cleaning up BaseAgent for {instance_id}")
        # Call concrete implementation's cleanup
        cleanup_result = self._cleanup()
        logger.info(f"BaseAgent cleanup complete for {instance_id}")
        return cleanup_result

    # --- Methods for concrete classes to implement ---
    def _initialize(self):
        """Concrete agent initialization logic."""
        raise NotImplementedError

    async def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Concrete agent task execution logic (must be async)."""
        raise NotImplementedError

    def _cleanup(self):
        """Concrete agent resource cleanup logic."""
        raise NotImplementedError

# --- Constants --- (Define workspace root for safety)
WORKSPACE_ROOT = os.path.abspath("/opt/sutazaiapp")
MAX_READ_LINES = 1000 # Limit lines read by tools
MAX_READ_BYTES = 5 * 1024 * 1024 # Limit bytes read by tools (5MB)
MAX_SEARCH_RESULTS = 50 # Limit search results

# --- Helper for path validation ---
def is_safe_path(target_path: str) -> bool:
    """Checks if the path is within the defined WORKSPACE_ROOT."""
    try:
        abs_target_path = os.path.abspath(target_path)
        # Prevent path traversal and ensure it's within the workspace
        return abs_target_path.startswith(WORKSPACE_ROOT) and ".." not in pathlib.Path(target_path).parts
    except Exception as e:
        logger.warning(f"Path validation error for '{target_path}': {e}")
        return False
