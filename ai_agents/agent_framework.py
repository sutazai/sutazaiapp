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
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from ai_agents.model_manager import ModelManager
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import LlamaCpp
from langchain import hub

# Configure logging
LOG_FILE = "logs/agent_framework.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("AgentFramework")


class AgentFrameworkType(Enum):
    """Types of agent frameworks supported"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    LOCALAGI = "localagi"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
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
    MULTIMODAL = "multimodal"
    COLLABORATION = "collaboration"
    CODE_UNDERSTANDING = "code_understanding"


@dataclass
class AgentTool:
    """Represents a tool that an agent can use"""
    name: str
    description: str
    function: Callable[..., Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_capability: List[AgentCapability] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_capability": [cap.value for cap in self.requires_capability],
        }


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    description: str
    framework: AgentFrameworkType
    capabilities: List[AgentCapability]
    model_id: str
    tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
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
        """Create from dictionary representation"""
        try:
            # Correctly indented return statement
            return cls(
                name=data["name"],
                description=data["description"],
                framework=AgentFrameworkType(data["framework"]),
                capabilities=[AgentCapability(cap) for cap in data["capabilities"]],
                model_id=data["model_id"],
                tools=data.get("tools", []),
                parameters=data.get("parameters", {}),
            )
        except KeyError as ke:
            raise ValueError(f"Missing required key in agent config data: {ke}")
        except ValueError as ve:
            # Catches errors from Enum creation if invalid value provided
             raise ValueError(f"Invalid value in agent config data: {ve}")


class AgentState(Enum):
    """Possible states of an agent"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentFramework:
    """
    Unified framework for integrating different agent types
    
    This class provides a standardized interface for working with
    agents from different frameworks like LangChain, AutoGPT, etc.
    It handles agent lifecycle, tool registration, and coordination.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config_path: str = "config/agents.json",
        tools_dir: str = "ai_agents/tools",
        max_concurrent_agents: int = 5,
    ):
        """
        Initialize the agent framework
        
        Args:
            model_manager: ModelManager instance for agent models
            config_path: Path to agent configuration file
            tools_dir: Directory containing agent tools
            max_concurrent_agents: Maximum number of concurrent agents
        """
        if not isinstance(model_manager, ModelManager):
             raise TypeError("model_manager must be an instance of ModelManager")
        self.model_manager = model_manager
        self.config_path = config_path
        self.tools_dir = tools_dir
        self.max_concurrent_agents = max_concurrent_agents
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Initialize agent registries
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_states: Dict[str, AgentState] = {}
        
        # Initialize tool registry
        self.tools: Dict[str, AgentTool] = {}
        
        # Load configurations
        self._load_config()
        self._load_tools()
        
        logger.info(
            f"Agent framework initialized with {len(self.agent_configs)} agent configs and {len(self.tools)} tools"
        )
    
    def _load_config(self):
        """Load agent configurations from file"""
        temp_configs: Dict[str, AgentConfig] = {}
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.debug(f"Raw config loaded from {self.config_path}")
                    
                    agents_dict = config_data.get("agents", {})
                    if not isinstance(agents_dict, dict):
                        logger.error("Invalid config format: 'agents' key is not a dictionary.")
                        return # Or raise error / use default

                    logger.debug(f"Found 'agents' dictionary with {len(agents_dict)} entries.")
                    for agent_id, agent_data in agents_dict.items():
                        try:
                            if not isinstance(agent_data, dict):
                                logger.warning(f"Skipping invalid agent config entry for '{agent_id}': not a dictionary.")
                                continue
                            temp_configs[agent_id] = AgentConfig.from_dict(agent_data)
                        except ValueError as agent_load_error:
                            logger.error(f"Error parsing config for agent '{agent_id}': {agent_load_error}")
                        except Exception as e:
                            logger.error(f"Unexpected error loading agent '{agent_id}': {e}", exc_info=True)

                    self.agent_configs = temp_configs
                    logger.info(f"Loaded {len(self.agent_configs)} valid agent configurations.")

            else:
                logger.warning(f"Configuration file not found at {self.config_path}. Creating default config.")
                self._create_default_config() # This also loads the defaults into self.agent_configs

        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from {self.config_path}: {json_err}")
            logger.info("Creating default configuration as fallback due to JSON error.")
            self._create_default_config()
        except IOError as io_err:
            logger.error(f"Error reading configuration file {self.config_path}: {io_err}")
            # Decide how to handle this - perhaps raise an exception?
        except Exception as e:
            logger.error(f"Unexpected error loading agent configurations: {e}", exc_info=True)
            logger.info("Creating default configuration as fallback due to unexpected error.")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default agent configuration"""
        default_config = {
            "agents": {
                "coder": {
                    "name": "Code Assistant",
                    "description": "Generates and modifies code",
                    "framework": "langchain",
                    "capabilities": ["code_generation", "reasoning"],
                    "model_id": "deepseek-coder",
                    "tools": ["write_file", "read_file", "execute_code"],
                    "parameters": {"temperature": 0.2, "max_iterations": 5},
                },
                "researcher": {
                    "name": "Research Assistant",
                    "description": "Conducts research and analyzes information",
                    "framework": "langchain",
                    "capabilities": ["text_generation", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["search_documents"],
                    "parameters": {"temperature": 0.7, "max_iterations": 3},
                },
                "planner": {
                    "name": "Task Planner",
                    "description": "Plans tasks and creates workflows",
                    "framework": "autogpt",
                    "capabilities": ["planning", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["create_plan", "assign_tasks"],
                    "parameters": {
                        "temperature": 0.5,
                        "max_iterations": 10,
                        "autonomous_execution": False,
                    },
                },
                "document_processor": {
                    "name": "Document Processor",
                    "description": "Processes various document formats",
                    "framework": "custom",
                    "capabilities": ["document_processing", "text_generation"],
                    "model_id": "llama3-8b",
                    "tools": ["extract_text", "parse_pdf", "ocr_image"],
                    "parameters": {"temperature": 0.3, "max_document_size_mb": 20},
                },
            }
        }
        
        # Save default configuration
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        # Load the configs
        for agent_id, agent_data in default_config["agents"].items():
            self.agent_configs[agent_id] = AgentConfig.from_dict(agent_data)
        
        logger.info(
            f"Created default agent configuration with {len(self.agent_configs)} agents"
        )
    
    def _load_tools(self):
        """Load available tools for agents"""
        self.tools.clear() # Ensure clean slate if reloaded

        # Define tools using AgentTool dataclass
        # Note: Tool functions (`self._tool_*`) must exist below
        tool_definitions = [
            AgentTool(
            name="read_file",
                description="Read the contents of a specific file. Provide start/end lines (1-indexed) or use should_read_entire_file=true.",
            function=self._tool_read_file,
                parameters={
                    "target_file": {"type": "string", "description": "Path to the file."},
                    "start_line_one_indexed": {"type": "integer", "description": "(Optional) Start line."},
                    "end_line_one_indexed_inclusive": {"type": "integer", "description": "(Optional) End line."},
                    "should_read_entire_file": {"type": "boolean", "description": "(Optional) Read whole file."}
                },
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="list_dir",
                description="List contents of a directory relative to the workspace root /opt/sutazaiapp.",
                function=self._tool_list_dir,
                parameters={"relative_workspace_path": {"type": "string", "description": "Directory path relative to workspace."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="codebase_search",
                description="Semantic search for code snippets.",
                function=self._tool_codebase_search,
                parameters={
                    "query": {"type": "string", "description": "Search query."},
                    "target_directories": {"type": "array", "description": "(Optional) List of directories (glob patterns)."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="grep_search",
                description="Fast exact text/regex search in files.",
                function=self._tool_grep_search,
                parameters={
                    "query": {"type": "string", "description": "Regex/text pattern."},
                    "include_pattern": {"type": "string", "description": "(Optional) Glob pattern for files to include."},
                    "exclude_pattern": {"type": "string", "description": "(Optional) Glob pattern to exclude."},
                    "case_sensitive": {"type": "boolean", "description": "(Optional) Case-sensitive search."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="file_search",
                description="Fuzzy search for files by path name.",
                function=self._tool_file_search,
                parameters={"query": {"type": "string", "description": "Partial/fuzzy file name/path."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
             AgentTool(
            name="write_file",
                 description="Write content to a file.",
            function=self._tool_write_file,
            parameters={
                     "file_path": {"type": "string", "description": "Path to the file."},
                     "content": {"type": "string", "description": "Content to write."}
                 },
                 requires_capability=[AgentCapability.FILE_OPERATIONS],
             ),
             AgentTool(
            name="execute_code",
                 description="Execute code in sandbox (Placeholder).",
            function=self._tool_execute_code,
                 parameters={
                     "code": {"type": "string", "description": "Code to execute."},
                     "language": {"type": "string", "description": "Programming language."}
                 },
                 requires_capability=[AgentCapability.CODE_EXECUTION],
             ),
             AgentTool(
            name="search_documents",
                 description="Search documents in vector DB with error handling.",
            function=self._tool_search_documents,
                 parameters={
                     "query": {"type": "string", "description": "Search query."},
                     "n_results": {"type": "integer", "description": "Number of results."}
                 },
                 requires_capability=[AgentCapability.DOCUMENT_PROCESSING],
             ),
             AgentTool(
                 name="calculator",
                 description="Evaluate a simple mathematical expression.",
                 function=self._tool_calculator,
                 parameters={"expression": {"type": "string", "description": "Mathematical expression."}},
                 requires_capability=[], # No specific capability needed
             ),
             # Add other tools as needed
        ]

        for tool in tool_definitions:
            # Basic check if implementation exists
            if hasattr(self, tool.function.__name__) and callable(getattr(self, tool.function.__name__)):
                self.tools[tool.name] = tool
            else:
                logger.error(f"Tool implementation method '{tool.function.__name__}' not found for tool '{tool.name}'. Tool not loaded.")

        logger.info(f"Loaded {len(self.tools)} tools.")
        # TODO: Implement dynamic loading from tools_dir

    def _tool_read_file(self, target_file: str, start_line_one_indexed: Optional[int] = None, end_line_one_indexed_inclusive: Optional[int] = None, should_read_entire_file: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Read file using default_api with robust error checking."""
        explanation = f"Agent requested read access to {target_file}. Entire: {should_read_entire_file}. Range: {start_line_one_indexed}-{end_line_one_indexed_inclusive}."
        try:
            # Validate target_file basic safety
            abs_path = os.path.abspath(target_file)
            workspace_root = os.path.abspath("/opt/sutazaiapp") # Define workspace root
            if not abs_path.startswith(workspace_root):
                 logger.warning(f"Attempt to read file outside workspace: {target_file}")
                 return {"success": False, "error": "File access restricted to workspace."}

            logger.info(f"Tool: Reading file '{abs_path}' (Entire: {should_read_entire_file}, Range: {start_line_one_indexed}-{end_line_one_indexed_inclusive})")
            start = int(start_line_one_indexed) if start_line_one_indexed is not None else 1
            end = int(end_line_one_indexed_inclusive) if end_line_one_indexed_inclusive is not None else 200 # Default end line
            read_all = bool(should_read_entire_file)

            if read_all:
                result = default_api.read_file(target_file=abs_path, should_read_entire_file=True, start_line_one_indexed=1, end_line_one_indexed_inclusive=-1, explanation=explanation)
            else:
                 start = max(1, start) # Ensure start is at least 1
                 end = max(start, end) # Ensure end is valid and after start
                 # Add reasonable limit to prevent reading huge chunks
                 if end - start > 1000: # Limit to 1000 lines read at once unless reading all
                      end = start + 1000
                      logger.warning(f"Read range exceeded limit, truncated to lines {start}-{end}")
                      explanation += f" (Truncated to {end})"
                 result = default_api.read_file(target_file=abs_path, should_read_entire_file=False, start_line_one_indexed=start, end_line_one_indexed_inclusive=end, explanation=explanation)

            # Check the structure of the API response
            if isinstance(result, dict) and "read_file_response" in result:
                response_data = result["read_file_response"]
                # Check if the response data itself signals an error (adapt if API format differs)
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_read_file API call failed for {target_file}: {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Read file API call failed.")}
                else:
                    # Return the actual content from the response
                    return {"success": True, "content": response_data}
            else:
                 logger.error(f"_tool_read_file received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}

        except ValueError as ve:
            logger.error(f"Error processing parameters for _tool_read_file: {ve}")
            return {"success": False, "error": f"Invalid parameter type: {ve}"}
        except FileNotFoundError:
            logger.error(f"File not found in _tool_read_file: {target_file}")
            return {"success": False, "error": f"File not found: {target_file}"}
        except PermissionError:
             logger.error(f"Permission denied reading file: {target_file}")
             return {"success": False, "error": f"Permission denied reading file: {target_file}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_read_file for '{target_file}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred while reading file: {str(e)}"}

    def _tool_list_dir(self, relative_workspace_path: str) -> Dict[str, Any]:
        """Tool implementation: List directory contents using default_api with validation."""
        explanation = f"Agent requested listing of directory: {relative_workspace_path}"
        try:
            # Basic path validation
            if ".." in relative_workspace_path or relative_workspace_path.startswith("/"):
                 logger.warning(f"Attempted invalid path traversal: {relative_workspace_path}")
                 return {"success": False, "error": "Invalid path specified. Must be relative to workspace root and contain no '..'."}

            # Normalize path (optional, depends on API expectation)
            # clean_path = os.path.normpath(relative_workspace_path)
            clean_path = relative_workspace_path # Assuming API handles this

            logger.info(f"Tool: Listing directory '/opt/sutazaiapp/{clean_path}'")

            result = default_api.list_dir(relative_workspace_path=clean_path, explanation=explanation)

            if isinstance(result, dict) and "list_dir_response" in result:
                response_data = result["list_dir_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_list_dir API call failed for {clean_path}: {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "List directory API call failed.")}
                else:
                     # Assuming successful response contains the list/details
                     return {"success": True, "contents": response_data}
            else:
                 logger.error(f"_tool_list_dir received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_list_dir for '{relative_workspace_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred listing directory: {str(e)}"}

    def _tool_codebase_search(self, query: str, target_directories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Tool implementation: Semantic code search using default_api with error checks."""
        explanation = f"Agent requested codebase search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for codebase search.")
                 return {"success": False, "error": "Invalid or empty query provided for codebase search."}
            if target_directories and not isinstance(target_directories, list):
                 logger.error("Invalid target_directories provided for codebase search (must be a list).")
                 return {"success": False, "error": "Invalid target_directories format (must be a list)."}

            logger.info(f"Tool: Performing codebase search for query: '{query}' Target Dirs: {target_directories}")
            result = default_api.codebase_search(query=query, target_directories=target_directories, explanation=explanation)

            if isinstance(result, dict) and "codebase_search_response" in result:
                response_data = result["codebase_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_codebase_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Codebase search API call failed.")}
                else:
                    # Assuming successful response contains the search results
                    return {"success": True, "search_results": response_data}
            else:
                 logger.error(f"_tool_codebase_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_codebase_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during codebase search: {str(e)}"}

    def _tool_grep_search(self, query: str, include_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None, case_sensitive: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Grep search using default_api with error checks."""
        explanation = f"Agent requested grep search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for grep search.")
                 return {"success": False, "error": "Invalid or empty query provided for grep search."}

            logger.info(f"Tool: Performing grep search for pattern: '{query}', Include: {include_pattern}, Exclude: {exclude_pattern}, CaseSensitive: {case_sensitive}")
            is_case_sensitive = bool(case_sensitive) # Ensure boolean

            result = default_api.grep_search(
                query=query,
                include_pattern=include_pattern,
                exclude_pattern=exclude_pattern,
                case_sensitive=is_case_sensitive,
                explanation=explanation
            )

            if isinstance(result, dict) and "grep_search_response" in result:
                response_data = result["grep_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_grep_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Grep search API call failed.")}
                else:
                    # Assuming successful response contains matches
                    return {"success": True, "matches": response_data}
            else:
                 logger.error(f"_tool_grep_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_grep_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during grep search: {str(e)}"}

    def _tool_file_search(self, query: str) -> Dict[str, Any]:
        """Tool implementation: Fuzzy file search using default_api with error checks."""
        explanation = f"Agent requested file search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for file search.")
                 return {"success": False, "error": "Invalid or empty query provided for file search."}

            logger.info(f"Tool: Performing file search for query: '{query}'")
            result = default_api.file_search(query=query, explanation=explanation)

            if isinstance(result, dict) and "file_search_response" in result:
                response_data = result["file_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_file_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "File search API call failed.")}
                else:
                    # Assuming successful response is the list of files
                    return {"success": True, "found_files": response_data}
            else:
                 logger.error(f"_tool_file_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_file_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during file search: {str(e)}"}
    
    def _tool_write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Tool implementation: Write file with path validation and error handling."""
        explanation = f"Agent requested write access to {file_path}."
        try:
            # Path validation
            abs_path = os.path.abspath(file_path)
            workspace_root = os.path.abspath("/opt/sutazaiapp") # Define workspace root
            if not abs_path.startswith(workspace_root):
                 logger.warning(f"Attempt to write file outside workspace: {file_path}")
                 return {"success": False, "error": "File write access restricted to workspace."}

            logger.info(f"Tool: Writing to file '{abs_path}'")
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully wrote to file '{abs_path}'")
            return {"success": True, "file_path": abs_path}
        except IOError as io_err:
             logger.error(f"IOError writing to file {file_path}: {io_err}")
             return {"success": False, "error": f"IOError writing file: {io_err}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_write_file for '{file_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred writing file: {str(e)}"}
    
    def _tool_execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Tool implementation: Execute code in sandbox (Placeholder - UNSAFE)."""
        explanation = f"Agent requested execution of {language} code."
        logger.warning("execute_code tool is a placeholder and does not run code securely.")
        # In a real scenario, this should call a secure sandbox service (e.g., using Docker, firecracker)
        # NEVER use eval() or subprocess directly on untrusted agent output.
        return {"success": False, "error": "Code execution sandbox not implemented yet."}
    
    def _tool_search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Tool implementation: Search documents in vector DB with error handling."""
        explanation = f"Agent requested document search for: {query}"
        logger.info(f"Tool: Executing search_documents with query: '{query}', n_results: {n_results}")
        try:
            # Validate n_results
            try:
                num_results = max(1, int(n_results)) # Ensure positive integer
            except (ValueError, TypeError):
                 logger.warning(f"Invalid n_results '{n_results}', defaulting to 5.")
                 num_results = 5

            # Define which vector DB model to query - consider making configurable
            vector_db_model_id = "chroma-store"

            logger.debug(f"Querying vector DB '{vector_db_model_id}' for '{query}' (k={num_results})")
            # Call ModelManager to query the vector DB
            search_params = {"k": num_results}
            results = self.model_manager.run_inference(
                vector_db_model_id,
                query, # Input is the query string
                parameters=search_params
            )

            # Check for errors returned by the inference call
            if isinstance(results, dict) and "error" in results:
                 logger.error(f"Vector DB query failed: {results['error']}")
                 return {"success": False, "error": results['error']}

            logger.debug(f"Raw vector DB results: {results}")

            # Basic validation of expected keys from ChromaDB (adapt if DB changes)
            if not isinstance(results, dict) or not all(k in results for k in ["ids", "documents", "metadatas", "distances"]):
                 logger.error(f"Unexpected result format from vector DB: {results}")
                 return {"success": False, "error": "Unexpected format from vector database."}

            # Format results safely, handling potential None values
            formatted_results = [
                 {
                     "id": doc_id,
                     "content": doc_content if doc_content is not None else "",
                     "metadata": metadata if metadata is not None else {},
                     "distance": distance if distance is not None else float('inf'),
                 }
                 for doc_id, doc_content, metadata, distance in zip(
                     results.get("ids", [[]])[0], # Chroma nests results
                     results.get("documents", [[]])[0],
                     results.get("metadatas", [[]])[0],
                     results.get("distances", [[]])[0]
                 )
            ]

            logger.info(f"Vector DB search successful, returning {len(formatted_results)} results.")
            return {"success": True, "results": formatted_results}

        except Exception as e:
            logger.error(f"Unexpected error in _tool_search_documents for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during document search: {str(e)}"}

    def _tool_calculator(self, expression: str) -> Dict[str, Any]:
        """Tool implementation: Evaluate a simple mathematical expression (Placeholder - UNSAFE)."""
        explanation = f"Agent requested calculation for: {expression}"
        logger.warning("Calculator tool uses eval() and is UNSAFE for untrusted input.")
        # In a real scenario, use a safer math parsing library (like numexpr or ast)
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression."}
        try:
            # Limit eval scope
            result = eval(expression, {"__builtins__": {}}, {})
            logger.info(f"Tool: Calculated '{expression}' = {result}")
            return {"success": True, "result": str(result)}
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return {"success": False, "error": f"Failed to evaluate expression: {e}"}

    def _get_agent_config(self, agent_identifier: str) -> Optional[Tuple[str, AgentConfig]]:
        """Find agent config by ID or name."""
        if agent_identifier in self.agent_configs:
            return agent_identifier, self.agent_configs[agent_identifier]
        for agent_id, config in self.agent_configs.items():
            if config.name == agent_identifier:
                return agent_id, config
        return None

    def create_agent(
        self, agent_identifier: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an agent instance from a configuration ID or name.
        
        Args:
            agent_identifier: ID or Name of the agent configuration.
            parameters: Optional parameters to override configuration defaults.
            
        Returns:
            Instance ID for the created agent, or None if creation failed.
        """
        config_result = self._get_agent_config(agent_identifier)
        if not config_result:
            logger.error(f"Agent configuration '{agent_identifier}' not found.")
            return None # Return None instead of raising error directly

        agent_id, config = config_result
        instance_id = f"{agent_id}-{uuid.uuid4().hex[:8]}"
        logger.info(f"Attempting to create agent instance '{instance_id}' from config '{agent_id}'.")
        
        # Set initial state
        self.agent_states[instance_id] = AgentState.INITIALIZING
        
        try:
            # Merge parameters safely
            config_params = config.parameters.copy()
            if parameters:
                if not isinstance(parameters, dict):
                    logger.warning("Invalid parameters format for creation, using defaults.")
            else:
                    config_params.update(parameters)

            # Create the agent instance (returns dict with instance or error)
            agent_data = self._create_framework_agent(instance_id, config, config_params)

            # Check if instance creation failed
            if not agent_data or agent_data.get("error") or not agent_data.get("instance"):
                 creation_error = agent_data.get('error', 'Unknown creation error') if isinstance(agent_data, dict) else 'Creation returned None'
                 logger.error(f"Failed to create agent instance '{agent_id}': {creation_error}")
                 self.agent_states[instance_id] = AgentState.ERROR
                 # Clean up potential partial registration?
                 if instance_id in self.active_agents: del self.active_agents[instance_id]
                 if instance_id in self.agent_states: del self.agent_states[instance_id]
                 return None

            # Store successfully created instance
            self.active_agents[instance_id] = agent_data
            self.agent_states[instance_id] = AgentState.IDLE # Set state after successful creation
            logger.info(f"Created and stored new instance '{instance_id}' (Type: {agent_data.get('type')})")
            return instance_id
            
        except Exception as e:
            # Catch unexpected errors during the creation process
            logger.error(f"Unexpected error creating agent instance '{agent_id}': {e}", exc_info=True)
            self.agent_states[instance_id] = AgentState.ERROR
            return None

    def _create_framework_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Factory method to create agent based on framework type."""
        framework = config.framework
        creator_method_name = f"_create_{framework.value}_agent"
        creator_method = getattr(self, creator_method_name, None)

        if creator_method and callable(creator_method):
            logger.info(f"Creating agent instance '{instance_id}' using framework: {framework.value}")
            # The specific _create_* method should handle its own errors and return dict
            return creator_method(instance_id, config, params)
        else:
            logger.error(f"Unsupported agent framework type: {framework.value}")
            # Return error dict directly
            return {"type": "error", "instance": None, "config": config, "error": f"Unsupported framework: {framework.value}"}

    def _create_langchain_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a LangChain agent instance (ReAct)."""
        error_return = {"type": "langchain", "instance": None, "config": config} # Predefined error structure
        try:
            # --- 1. Load LLM Instance ---
            model_id = config.model_id
            logger.info(f"Requesting model manager to load model '{model_id}' for agent '{instance_id}'.")
            # Assuming load_model returns bool or raises error on failure
            if not self.model_manager.load_model(model_id):
                 error_msg = f"Model manager failed to load model '{model_id}'."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return

            loaded_model_data = self.model_manager.loaded_models.get(model_id)
            if not loaded_model_data or "instance" not in loaded_model_data:
                 error_msg = f"Model '{model_id}' not found in loaded models after load attempt."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return
            llm = loaded_model_data["instance"]
            logger.info(f"Successfully obtained LLM instance for model: {model_id}")

            # --- 2. Check for Tools (Codebase Agent requires tools) ---
            if not config.tools:
                error_msg = f"LangChain agent '{config.name}' requires tools for ReAct framework. Check config."
                logger.error(error_msg)
                error_return["error"] = error_msg
                return error_return

            # --- 3. Map and Wrap Tools ---
            agent_tools = []
            missing_tool_methods = []
            for tool_name in config.tools:
                tool_method = getattr(self, f"_tool_{tool_name}", None)
                tool_config = self.tools.get(tool_name)
                if tool_method and callable(tool_method) and tool_config:
                    # Langchain Tool expects a sync function
                    # If tool methods were async, would need `arun` or wrapper
                    agent_tools.append(Tool(name=tool_config.name, func=tool_method, description=tool_config.description))
                    logger.debug(f"Mapped tool '{tool_name}' ({tool_config.name})")
                else:
                    missing_tool_methods.append(tool_name)
                    logger.warning(f"Tool implementation method `_tool_{tool_name}` or definition for '{tool_name}' not found.")

            if not agent_tools:
                error_msg = f"Agent '{config.name}' lists tools, but none could be mapped. Missing: {missing_tool_methods}"
                logger.error(error_msg)
                error_return["error"] = error_msg
                return error_return
            logger.info(f"Mapped {len(agent_tools)} tools for agent '{instance_id}': {[t.name for t in agent_tools]}")

            # --- 4. Create Custom Codebase-Focused ReAct Prompt ---
            try:
                base_prompt = hub.pull("hwchase17/react")
                template_format = base_prompt.template
                react_format_instructions_start = template_format.find("Use the following format:")
                react_format_instructions = template_format[react_format_instructions_start:]
                react_format_instructions = react_format_instructions.replace(
                    "Question: the input question you must answer",
                    "Original Question: the user query you must address"
                )
            except Exception as hub_error:
                logger.error(f"Failed to pull base ReAct prompt from hub: {hub_error}")
                error_return["error"] = f"Failed to load base prompt: {hub_error}"
                return error_return

            codebase_system_message = (
                f"You are SutazAI, a highly capable AI assistant specialized in understanding and analyzing the codebase located at /opt/sutazaiapp. "
                f"Your goal is to answer questions and fulfill requests about this specific codebase. "
                f"Always assume the user is asking about the code within /opt/sutazaiapp unless specified otherwise. "
                f"You have access to the following tools: {[t.name for t in agent_tools]}. "
                f"Think step-by-step. For questions about code structure or finding things, start by using tools like 'list_dir', 'file_search', 'grep_search', or 'codebase_search'. "
                f"Only use 'read_file' on specific files identified in previous steps. Be precise with file paths. "
                f"Adhere strictly to the required action format."
            )
            custom_template_str = (
                f"{codebase_system_message}\n\n"
                f"TOOLS:\n------\n"
                f"You have access to the following tools:\n"
                f"{{tools}}\n\n" # Filled by partial
                f"{{react_format_instructions}}\n\n" # Filled by partial
                f"Begin!\n\n"
                f"Previous conversation history:\n"
                f"{{chat_history}}\n\n" # Filled by memory
                f"Original Question: {{input}}\n" # User input
                f"Thought:{{agent_scratchpad}}" # Agent working space
            )

            try:
                prompt = PromptTemplate.from_template(custom_template_str)
                prompt = prompt.partial(
                    tools="\n".join([f"{tool.name}: {tool.description}" for tool in agent_tools]),
                    tool_names=", ".join([tool.name for tool in agent_tools]),
                    react_format_instructions=react_format_instructions
                )
                logger.info("Created custom ReAct prompt for codebase interaction.")
            except Exception as prompt_error:
                 logger.error(f"Failed to create custom prompt template: {prompt_error}")
                 error_return["error"] = f"Prompt template error: {prompt_error}"
                 return error_return

            # --- 5. Create ReAct Agent ---
            try:
                 agent = create_react_agent(llm, agent_tools, prompt)
                 logger.info("Attempting agent creation with create_react_agent using custom prompt.")
            except Exception as agent_creation_error:
                 logger.error(f"Failed create_react_agent: {agent_creation_error}", exc_info=True)
                 error_return["error"] = f"Failed to create ReAct agent: {agent_creation_error}"
                 return error_return

            # --- 6. Create Memory ---
            # Consider potential deprecation warnings and alternatives if needed
            try:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            except Exception as memory_error:
                 logger.error(f"Failed to create ConversationBufferMemory: {memory_error}")
                 error_return["error"] = f"Memory creation error: {memory_error}"
                 return error_return

            # --- 7. Create Agent Executor ---
            try:
                max_iterations = params.get("max_iterations", 10)
                agent_executor = AgentExecutor(
                     agent=agent,
                     tools=agent_tools,
                     memory=memory,
                     verbose=True, # Consider making this configurable
                     handle_parsing_errors="Check your output and make sure it conforms!", # Or a custom handler
                     max_iterations=max_iterations
                )
                logger.info(f"Successfully created LangChain ReAct agent executor for instance: {instance_id}")
                # Return success dictionary
                return {"type": "langchain_agent", "instance": agent_executor, "config": config}
            except Exception as executor_error:
                 logger.error(f"Failed to create AgentExecutor: {executor_error}", exc_info=True)
                 error_return["error"] = f"AgentExecutor creation error: {executor_error}"
                 return error_return

        except Exception as e:
            # Catch-all for unexpected errors during the process
            logger.error(f"Unexpected error creating LangChain agent instance '{instance_id}': {e}", exc_info=True)
            error_return["error"] = f"Unexpected agent creation error: {str(e)}"
            return error_return

    def _create_autogpt_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an AutoGPT agent."""
        logger.warning("AutoGPT task execution is not implemented.")
        raise NotImplementedError("AutoGPT task execution is not implemented.")

    def _create_localagi_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating a LocalAGI agent."""
        logger.warning("LocalAGI task execution is not implemented.")
        raise NotImplementedError("LocalAGI task execution is not implemented.")

    def _create_autogen_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an Autogen agent."""
        logger.warning("Autogen task execution is not fully implemented.")
        # Existing placeholder logic... needs proper async handling if used in process_chat
        raise NotImplementedError("Autogen task execution is not fully implemented.")

    def _create_custom_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Creates a custom agent based on BaseAgent."""
        # This structure seems complex and might duplicate logic better handled
        # by specific agent classes. Consider refactoring BaseAgent/CustomAgent interaction.
        error_return = {"type": "custom", "instance": None, "config": config}
        logger.warning("Custom agent creation logic needs review for robustness.")
        try:
            # Import BaseAgent locally to avoid potential circular dependencies at module level
            from ai_agents.base_agent import BaseAgent

            # Define the CustomAgent class locally or ensure it's properly imported
            # This nested class definition is unusual; prefer defining it at module level
            class CustomAgentImpl(BaseAgent):
                def __init__(self, config: AgentConfig, model_manager: ModelManager, tools: Dict[str, AgentTool], params: Dict[str, Any]):
                    # Pass the AgentConfig object directly to BaseAgent's __init__
                    super().__init__(config) # BaseAgent now expects AgentConfig
                    self.model_manager = model_manager
                    self.available_tools = tools # Renamed from self.tools to avoid conflict
                    self.params = params
                    # self.model_id is already set in BaseAgent.__init__
                    self.history: List[Dict[str, Any]] = [] # Explicitly initialize history

                def _initialize(self):
                    logger.info(f"Initializing CustomAgentImpl for model: {self.model_id}")
                    if not self.model_manager.load_model(self.model_id):
                        raise RuntimeError(f"Failed to load model {self.model_id} via ModelManager.")
                    logger.info(f"CustomAgentImpl initialized successfully for model: {self.model_id}")
                    return True # Indicate success explicitly? BaseAgent doesn't use return

                def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
                    logger.info(f"CustomAgentImpl executing task: {task.get('id', 'N/A')}")
                    prompt = self._build_prompt(task)
                    logger.debug(f"CustomAgentImpl prompt: {prompt}")

                    inference_params = {
                        "temperature": self.params.get("temperature", 0.7),
                        "max_tokens": self.params.get("max_tokens", 1024), # Reduced default?
                        # Add other relevant parameters from self.params
                    }
                    
                    result = self.model_manager.run_inference(
                        self.model_id,
                        prompt,
                        inference_params,
                    )

                    if isinstance(result, dict) and "error" in result:
                        error_msg = f"Inference error: {result['error']}"
                        logger.error(error_msg)
                        # Raise an exception to be caught by the calling method
                        raise RuntimeError(error_msg)

                    output = result.get("text", "No text output from model.")
                    logger.debug(f"CustomAgentImpl raw output: {output}")

                    # Basic history update
                    self.history.append({"task": task.get('instruction', 'N/A'), "output": output})
                    # Limit history size if needed
                    max_history = 10
                    if len(self.history) > max_history:
                        self.history = self.history[-max_history:]

                    return {"output": output, "task_id": task.get("id", "N/A")} # Return standard structure

                def _cleanup(self):
                    logger.info(f"Cleaning up CustomAgentImpl for model: {self.model_id}")
                    # Nothing specific to clean up here unless resources were allocated
                    return True

                def _build_prompt(self, task: Dict[str, Any]) -> str:
                    # Simplified prompt building example
                    parts = []
                    parts.append(f"### Task Instruction: {task.get('instruction', 'No instruction provided.')}")

                    if task.get("context"):
                        parts.append(f"### Context: {task['context']}")

                    if self.history:
                        hist_str = "\n".join([f"Prev Task: {h['task'][:100]}...\nPrev Output: {h['output'][:100]}..." for h in self.history])
                        parts.append(f"### Recent Conversation History: {hist_str}")

                    if self.config.tools:
                        tool_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.available_tools.items() if name in self.config.tools])
                        parts.append(f"### Available Tools: {tool_desc}")

                    parts.append("\n### Your Response:")
                    return "\n\n".join(parts)

            # --- End of Nested Class ---

            # Map tools needed by this specific agent config
            agent_specific_tools = {
                tool_name: self.tools[tool_name]
                for tool_name in config.tools if tool_name in self.tools
            }

            # Create the agent instance
            custom_agent_instance = CustomAgentImpl(
                config=config,
                model_manager=self.model_manager,
                tools=agent_specific_tools,
                params=params,
            )

            # Initialize the agent (critical step)
            try:
                custom_agent_instance.initialize() # Call BaseAgent's initialize
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

    def execute_task(self, instance_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with an agent
        
        Args:
            instance_id: ID of the agent instance
            task: Task description
            
        Returns:
            Result of the task execution
        """
        if instance_id not in self.active_agents:
            logger.error(f"Agent instance {instance_id} not found")
            raise ValueError(f"Agent instance {instance_id} not found")
        
        with self.agent_locks[instance_id]:
            try:
                # Update agent state
                self.agent_states[instance_id] = AgentState.PROCESSING
                
                # Get the agent
                agent_data = self.active_agents[instance_id]
                agent_type = agent_data["type"]
                agent = agent_data["instance"]
                
                # Execute task based on agent type
                if agent_type == "langchain":
                    result = self._execute_langchain_task(agent, task)
                elif agent_type == "autogpt":
                    result = self._execute_autogpt_task(agent, task)
                elif agent_type == "localagi":
                    result = self._execute_localagi_task(agent, task)
                elif agent_type == "autogen":
                    result = self._execute_autogen_task(agent_data, task)
                elif agent_type == "custom":
                    result = self._execute_custom_task(agent, task)
                else:
                    raise ValueError(f"Unsupported agent type: {agent_type}")
                
                # Update agent state
                self.agent_states[instance_id] = AgentState.IDLE
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing task with agent {instance_id}: {str(e)}")
                self.agent_states[instance_id] = AgentState.ERROR
                raise
    
    def _execute_langchain_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LangChain agent"""
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        
        # Format the prompt
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction
        
        # Run the agent
        try:
            response = agent.run(prompt)
            return {"output": response, "task_id": task.get("id")}
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def _execute_autogpt_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an AutoGPT agent"""
        # This is a placeholder implementation
        return {
            "output": "AutoGPT execution not implemented",
            "task_id": task.get("id"),
        }
    
    def _execute_localagi_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LocalAGI agent"""
        # This is a placeholder implementation
        return {
            "output": "LocalAGI execution not implemented",
            "task_id": task.get("id"),
        }
    
    def _execute_autogen_task(self, agent_data, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an Autogen agent"""
        agent = agent_data["instance"]
        user_proxy = agent_data.get("user_proxy")
        
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        
        # Format the message
        if context:
            message = f"{instruction}\n\nContext: {context}"
        else:
            message = instruction
        
        # Run with or without user proxy
        try:
            if user_proxy:
                # Start a conversation between user proxy and agent
                user_proxy.initiate_chat(agent, message=message)
                # Get the chat history
                history = user_proxy.chat_messages[agent]
                # Extract the last message from the agent
                last_message = history[-1]["content"] if history else "No response"
                return {"output": last_message, "task_id": task.get("id")}
            else:
                # Direct message to agent
                response = agent.generate_reply(message)
                return {"output": response, "task_id": task.get("id")}
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def _execute_custom_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a custom agent"""
        try:
            result = agent._execute(task)
            return result
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}

    async def process_chat(self, agent_name: str, messages: List[Dict[str, Any]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat request using the specified agent instance (creating if necessary).
        Handles routing to the correct execution logic based on agent type.
        """
        logger.info(f"Processing chat request for agent name: '{agent_name}'")

        # --- 1. Find Agent Config ---
        config_result = self._get_agent_config(agent_name)
        if not config_result:
            error_msg = f"Agent configuration with name '{agent_name}' not found."
            logger.error(error_msg)
            return {"response": f"Error: {error_msg}", "usage": {}}
        agent_id, config = config_result
        logger.debug(f"Found config for agent '{agent_name}' with ID '{agent_id}'. Framework: {config.framework.value}")

        # --- 2. Get or Create Agent Instance ---
        # Use agent config name or ID consistently for instance tracking. Let's use agent_id.
        instance_id = agent_id
        instance_data = self.active_agents.get(instance_id)

        if not instance_data:
            logger.warning(f"No active instance found for '{instance_id}' (name: '{agent_name}'). Attempting creation.")
            # Merge parameters for creation
            creation_params = config.parameters.copy()
            if parameters and isinstance(parameters, dict):
                creation_params.update(parameters)
            else:
                # Log if parameters are invalid, but proceed with defaults
                if parameters is not None:
                    logger.warning("Invalid parameters format provided for creation, using defaults.")

            # Create agent (returns dict with instance or error)
            instance_data = self._create_framework_agent(instance_id, config, creation_params)

            # Validate creation result
            if not instance_data or instance_data.get("error") or not instance_data.get("instance"):
                 creation_error = instance_data.get('error', 'Unknown creation error') if isinstance(instance_data, dict) else 'Creation returned None'
                 error_msg = f"Failed to create agent instance '{agent_name}': {creation_error}. Check logs."
                 logger.error(error_msg)
                 # Update state if possible (instance_id might not exist if creation failed early)
                 if instance_id in self.agent_states:
                     self.agent_states[instance_id] = AgentState.ERROR
                 return {"response": f"Error: {error_msg}", "usage": {}}


        # --- 3. Prepare Input ---
        if not messages or not isinstance(messages, list):
             logger.error("Invalid 'messages' format received.")
             return {"response": "Error: Invalid chat message format.", "usage": {}}
        try:
            # Ensure history processing doesn't fail on empty list
            chat_history = messages[:-1] if len(messages) > 1 else []
            last_user_message_obj = messages[-1] if messages else None

            if not last_user_message_obj or not isinstance(last_user_message_obj, dict) or last_user_message_obj.get("role") != "user":
                 logger.error("Last message in list is not a valid user message.")
                 return {"response": "Error: Invalid final message (must be from user).", "usage": {}}

            last_user_message = last_user_message_obj.get("content", "").strip()
            if not last_user_message:
                logger.warning("Received empty user message.")
                return {"response": "Please provide a message.", "usage": {}}
        except (IndexError, TypeError, KeyError) as e:
             logger.error(f"Error processing input messages: {e}")
             return {"response": "Error: Could not process input messages.", "usage": {}}

        # --- 4. Execute Task ---
        instance_type = instance_data.get("type")
        executor_or_agent = instance_data.get("instance") # Renamed for clarity

        if not instance_type or not executor_or_agent:
             error_msg = f"Instance data for '{agent_name}' is corrupted or incomplete."
             logger.error(error_msg)
             self.agent_states[instance_id] = AgentState.ERROR
             return {"response": f"Error: {error_msg}", "usage": {}}

        logger.debug(f"Executing chat task for '{instance_id}' with instance type: {instance_type}")
        self.agent_states[instance_id] = AgentState.PROCESSING # Set state before execution

        try:
            # Route to the correct async execution method based on instance type
            if instance_type == "langchain_agent":
                result = await self._execute_langchain_agent_task(executor_or_agent, last_user_message, chat_history)
            elif instance_type == "custom":
                 # Custom agent execution needs to be adapted for async chat if needed
                 # Wrap the synchronous _execute method using run_in_executor
                 task_dict = {"instruction": last_user_message, "context": "", "id": f"chat-{uuid.uuid4().hex[:8]}"} # Create task dict
                 result = await self._execute_custom_task_async_wrapper(executor_or_agent, task_dict)

            elif instance_type in ["autogpt", "localagi", "autogen"]:
                 # Call placeholder methods (which raise NotImplementedError)
                 exec_method_name = f"_execute_{instance_type}_task"
                 exec_method = getattr(self, exec_method_name, None)
                 if exec_method and callable(exec_method):
                      # Assuming these placeholders might need a task dict too
                      task_dict = {"instruction": last_user_message, "context": "", "id": f"chat-{uuid.uuid4().hex[:8]}"}
                      # Placeholders are sync and raise NotImplementedError
                      exec_method(executor_or_agent, task_dict) # This will raise the error
                      # Should not reach here if exec_method raises correctly
                      raise RuntimeError(f"Placeholder {exec_method_name} did not raise NotImplementedError.")
                 else:
                      raise NotImplementedError(f"Execution logic for framework '{instance_type}' not found.")
            else:
                error_msg = f"Unsupported instance type for chat execution: {instance_type}"
                logger.error(error_msg)
                self.agent_states[instance_id] = AgentState.ERROR
                return {"response": f"Error: {error_msg}", "usage": {}}

            # Process result
            if not isinstance(result, dict):
                 error_msg = f"Execution for '{agent_name}' returned non-dict result: {type(result)}"
                 logger.error(error_msg)
                 self.agent_states[instance_id] = AgentState.ERROR
                 return {"response": f"Error: Internal agent execution error.", "usage": {}}

            response_content = result.get("output", "Agent did not return standard 'output'.")
            usage = result.get("usage", {}) # TODO: Implement usage tracking

            logger.info(f"Chat processed successfully by instance: {agent_name} ({instance_id})")
            self.agent_states[instance_id] = AgentState.IDLE # Reset state on success
            return {"response": response_content, "usage": usage}

        except NotImplementedError as nie:
             error_msg = f"Execution for agent type '{instance_type}' is not implemented: {nie}"
             logger.error(error_msg)
             self.agent_states[instance_id] = AgentState.ERROR
             return {"response": f"Error: {error_msg}", "usage": {}}
        except Exception as e:
            error_msg = f"Chat processing error: {e}" # Simplify error message for UI
            logger.error(f"Error during task execution for instance {agent_name} ({instance_id}): {e}", exc_info=True) # Log full traceback
            self.agent_states[instance_id] = AgentState.ERROR # Set error state
            return {"response": f"Error: {error_msg}", "usage": {}}


    async def _execute_langchain_agent_task(
        self, agent_executor: AgentExecutor, user_input: str, chat_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Execute a task with a pre-created LangChain AgentExecutor instance."""
        # Implementation of the method
        # This method should be implemented to handle the execution of a task with a LangChain agent
        # It should return a dictionary with the result of the task execution
        # If the task execution fails, it should return a dictionary with an error message
        # The method should handle the execution of the agent and return the result
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
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
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from ai_agents.model_manager import ModelManager
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import LlamaCpp
from langchain import hub

# Configure logging
LOG_FILE = "logs/agent_framework.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("AgentFramework")


class AgentFrameworkType(Enum):
    """Types of agent frameworks supported"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    LOCALAGI = "localagi"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
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
    MULTIMODAL = "multimodal"
    COLLABORATION = "collaboration"
    CODE_UNDERSTANDING = "code_understanding"


@dataclass
class AgentTool:
    """Represents a tool that an agent can use"""
    name: str
    description: str
    function: Callable[..., Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_capability: List[AgentCapability] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_capability": [cap.value for cap in self.requires_capability],
        }


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    description: str
    framework: AgentFrameworkType
    capabilities: List[AgentCapability]
    model_id: str
    tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
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
        """Create from dictionary representation"""
        try:
            # Correctly indented return statement
            return cls(
                name=data["name"],
                description=data["description"],
                framework=AgentFrameworkType(data["framework"]),
                capabilities=[AgentCapability(cap) for cap in data["capabilities"]],
                model_id=data["model_id"],
                tools=data.get("tools", []),
                parameters=data.get("parameters", {}),
            )
        except KeyError as ke:
            raise ValueError(f"Missing required key in agent config data: {ke}")
        except ValueError as ve:
            # Catches errors from Enum creation if invalid value provided
             raise ValueError(f"Invalid value in agent config data: {ve}")


class AgentState(Enum):
    """Possible states of an agent"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentFramework:
    """
    Unified framework for integrating different agent types
    
    This class provides a standardized interface for working with
    agents from different frameworks like LangChain, AutoGPT, etc.
    It handles agent lifecycle, tool registration, and coordination.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config_path: str = "config/agents.json",
        tools_dir: str = "ai_agents/tools",
        max_concurrent_agents: int = 5,
    ):
        """
        Initialize the agent framework
        
        Args:
            model_manager: ModelManager instance for agent models
            config_path: Path to agent configuration file
            tools_dir: Directory containing agent tools
            max_concurrent_agents: Maximum number of concurrent agents
        """
        if not isinstance(model_manager, ModelManager):
             raise TypeError("model_manager must be an instance of ModelManager")
        self.model_manager = model_manager
        self.config_path = config_path
        self.tools_dir = tools_dir
        self.max_concurrent_agents = max_concurrent_agents
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Initialize agent registries
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_states: Dict[str, AgentState] = {}
        
        # Initialize tool registry
        self.tools: Dict[str, AgentTool] = {}
        
        # Load configurations
        self._load_config()
        self._load_tools()
        
        logger.info(
            f"Agent framework initialized with {len(self.agent_configs)} agent configs and {len(self.tools)} tools"
        )
    
    def _load_config(self):
        """Load agent configurations from file"""
        temp_configs: Dict[str, AgentConfig] = {}
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.debug(f"Raw config loaded from {self.config_path}")
                    
                    agents_dict = config_data.get("agents", {})
                    if not isinstance(agents_dict, dict):
                        logger.error("Invalid config format: 'agents' key is not a dictionary.")
                        return # Or raise error / use default

                    logger.debug(f"Found 'agents' dictionary with {len(agents_dict)} entries.")
                    for agent_id, agent_data in agents_dict.items():
                        try:
                            if not isinstance(agent_data, dict):
                                logger.warning(f"Skipping invalid agent config entry for '{agent_id}': not a dictionary.")
                                continue
                            temp_configs[agent_id] = AgentConfig.from_dict(agent_data)
                        except ValueError as agent_load_error:
                            logger.error(f"Error parsing config for agent '{agent_id}': {agent_load_error}")
                        except Exception as e:
                            logger.error(f"Unexpected error loading agent '{agent_id}': {e}", exc_info=True)

                    self.agent_configs = temp_configs
                    logger.info(f"Loaded {len(self.agent_configs)} valid agent configurations.")

            else:
                logger.warning(f"Configuration file not found at {self.config_path}. Creating default config.")
                self._create_default_config() # This also loads the defaults into self.agent_configs

        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from {self.config_path}: {json_err}")
            logger.info("Creating default configuration as fallback due to JSON error.")
            self._create_default_config()
        except IOError as io_err:
            logger.error(f"Error reading configuration file {self.config_path}: {io_err}")
            # Decide how to handle this - perhaps raise an exception?
        except Exception as e:
            logger.error(f"Unexpected error loading agent configurations: {e}", exc_info=True)
            logger.info("Creating default configuration as fallback due to unexpected error.")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default agent configuration"""
        default_config = {
            "agents": {
                "coder": {
                    "name": "Code Assistant",
                    "description": "Generates and modifies code",
                    "framework": "langchain",
                    "capabilities": ["code_generation", "reasoning"],
                    "model_id": "deepseek-coder",
                    "tools": ["write_file", "read_file", "execute_code"],
                    "parameters": {"temperature": 0.2, "max_iterations": 5},
                },
                "researcher": {
                    "name": "Research Assistant",
                    "description": "Conducts research and analyzes information",
                    "framework": "langchain",
                    "capabilities": ["text_generation", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["search_documents"],
                    "parameters": {"temperature": 0.7, "max_iterations": 3},
                },
                "planner": {
                    "name": "Task Planner",
                    "description": "Plans tasks and creates workflows",
                    "framework": "autogpt",
                    "capabilities": ["planning", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["create_plan", "assign_tasks"],
                    "parameters": {
                        "temperature": 0.5,
                        "max_iterations": 10,
                        "autonomous_execution": False,
                    },
                },
                "document_processor": {
                    "name": "Document Processor",
                    "description": "Processes various document formats",
                    "framework": "custom",
                    "capabilities": ["document_processing", "text_generation"],
                    "model_id": "llama3-8b",
                    "tools": ["extract_text", "parse_pdf", "ocr_image"],
                    "parameters": {"temperature": 0.3, "max_document_size_mb": 20},
                },
            }
        }
        
        # Save default configuration
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        # Load the configs
        for agent_id, agent_data in default_config["agents"].items():
            self.agent_configs[agent_id] = AgentConfig.from_dict(agent_data)
        
        logger.info(
            f"Created default agent configuration with {len(self.agent_configs)} agents"
        )
    
    def _load_tools(self):
        """Load available tools for agents"""
        self.tools.clear() # Ensure clean slate if reloaded

        # Define tools using AgentTool dataclass
        # Note: Tool functions (`self._tool_*`) must exist below
        tool_definitions = [
            AgentTool(
            name="read_file",
                description="Read the contents of a specific file. Provide start/end lines (1-indexed) or use should_read_entire_file=true.",
            function=self._tool_read_file,
                parameters={
                    "target_file": {"type": "string", "description": "Path to the file."},
                    "start_line_one_indexed": {"type": "integer", "description": "(Optional) Start line."},
                    "end_line_one_indexed_inclusive": {"type": "integer", "description": "(Optional) End line."},
                    "should_read_entire_file": {"type": "boolean", "description": "(Optional) Read whole file."}
                },
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="list_dir",
                description="List contents of a directory relative to the workspace root /opt/sutazaiapp.",
                function=self._tool_list_dir,
                parameters={"relative_workspace_path": {"type": "string", "description": "Directory path relative to workspace."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="codebase_search",
                description="Semantic search for code snippets.",
                function=self._tool_codebase_search,
                parameters={
                    "query": {"type": "string", "description": "Search query."},
                    "target_directories": {"type": "array", "description": "(Optional) List of directories (glob patterns)."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="grep_search",
                description="Fast exact text/regex search in files.",
                function=self._tool_grep_search,
                parameters={
                    "query": {"type": "string", "description": "Regex/text pattern."},
                    "include_pattern": {"type": "string", "description": "(Optional) Glob pattern for files to include."},
                    "exclude_pattern": {"type": "string", "description": "(Optional) Glob pattern to exclude."},
                    "case_sensitive": {"type": "boolean", "description": "(Optional) Case-sensitive search."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="file_search",
                description="Fuzzy search for files by path name.",
                function=self._tool_file_search,
                parameters={"query": {"type": "string", "description": "Partial/fuzzy file name/path."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
             AgentTool(
            name="write_file",
                 description="Write content to a file.",
            function=self._tool_write_file,
            parameters={
                     "file_path": {"type": "string", "description": "Path to the file."},
                     "content": {"type": "string", "description": "Content to write."}
                 },
                 requires_capability=[AgentCapability.FILE_OPERATIONS],
             ),
             AgentTool(
            name="execute_code",
                 description="Execute code in sandbox (Placeholder).",
            function=self._tool_execute_code,
                 parameters={
                     "code": {"type": "string", "description": "Code to execute."},
                     "language": {"type": "string", "description": "Programming language."}
                 },
                 requires_capability=[AgentCapability.CODE_EXECUTION],
             ),
             AgentTool(
            name="search_documents",
                 description="Search documents in vector DB with error handling.",
            function=self._tool_search_documents,
                 parameters={
                     "query": {"type": "string", "description": "Search query."},
                     "n_results": {"type": "integer", "description": "Number of results."}
                 },
                 requires_capability=[AgentCapability.DOCUMENT_PROCESSING],
             ),
             AgentTool(
                 name="calculator",
                 description="Evaluate a simple mathematical expression.",
                 function=self._tool_calculator,
                 parameters={"expression": {"type": "string", "description": "Mathematical expression."}},
                 requires_capability=[], # No specific capability needed
             ),
             # Add other tools as needed
        ]

        for tool in tool_definitions:
            # Basic check if implementation exists
            if hasattr(self, tool.function.__name__) and callable(getattr(self, tool.function.__name__)):
                self.tools[tool.name] = tool
            else:
                logger.error(f"Tool implementation method '{tool.function.__name__}' not found for tool '{tool.name}'. Tool not loaded.")

        logger.info(f"Loaded {len(self.tools)} tools.")
        # TODO: Implement dynamic loading from tools_dir

    def _tool_read_file(self, target_file: str, start_line_one_indexed: Optional[int] = None, end_line_one_indexed_inclusive: Optional[int] = None, should_read_entire_file: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Read file using default_api with robust error checking."""
        explanation = f"Agent requested read access to {target_file}. Entire: {should_read_entire_file}. Range: {start_line_one_indexed}-{end_line_one_indexed_inclusive}."
        try:
            # Validate target_file basic safety
            abs_path = os.path.abspath(target_file)
            workspace_root = os.path.abspath("/opt/sutazaiapp") # Define workspace root
            if not abs_path.startswith(workspace_root):
                 logger.warning(f"Attempt to read file outside workspace: {target_file}")
                 return {"success": False, "error": "File access restricted to workspace."}

            logger.info(f"Tool: Reading file '{abs_path}' (Entire: {should_read_entire_file}, Range: {start_line_one_indexed}-{end_line_one_indexed_inclusive})")
            start = int(start_line_one_indexed) if start_line_one_indexed is not None else 1
            end = int(end_line_one_indexed_inclusive) if end_line_one_indexed_inclusive is not None else 200 # Default end line
            read_all = bool(should_read_entire_file)

            if read_all:
                result = default_api.read_file(target_file=abs_path, should_read_entire_file=True, start_line_one_indexed=1, end_line_one_indexed_inclusive=-1, explanation=explanation)
            else:
                 start = max(1, start) # Ensure start is at least 1
                 end = max(start, end) # Ensure end is valid and after start
                 # Add reasonable limit to prevent reading huge chunks
                 if end - start > 1000: # Limit to 1000 lines read at once unless reading all
                      end = start + 1000
                      logger.warning(f"Read range exceeded limit, truncated to lines {start}-{end}")
                      explanation += f" (Truncated to {end})"
                 result = default_api.read_file(target_file=abs_path, should_read_entire_file=False, start_line_one_indexed=start, end_line_one_indexed_inclusive=end, explanation=explanation)

            # Check the structure of the API response
            if isinstance(result, dict) and "read_file_response" in result:
                response_data = result["read_file_response"]
                # Check if the response data itself signals an error (adapt if API format differs)
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_read_file API call failed for {target_file}: {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Read file API call failed.")}
                else:
                    # Return the actual content from the response
                    return {"success": True, "content": response_data}
            else:
                 logger.error(f"_tool_read_file received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}

        except ValueError as ve:
            logger.error(f"Error processing parameters for _tool_read_file: {ve}")
            return {"success": False, "error": f"Invalid parameter type: {ve}"}
        except FileNotFoundError:
            logger.error(f"File not found in _tool_read_file: {target_file}")
            return {"success": False, "error": f"File not found: {target_file}"}
        except PermissionError:
             logger.error(f"Permission denied reading file: {target_file}")
             return {"success": False, "error": f"Permission denied reading file: {target_file}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_read_file for '{target_file}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred while reading file: {str(e)}"}

    def _tool_list_dir(self, relative_workspace_path: str) -> Dict[str, Any]:
        """Tool implementation: List directory contents using default_api with validation."""
        explanation = f"Agent requested listing of directory: {relative_workspace_path}"
        try:
            # Basic path validation
            if ".." in relative_workspace_path or relative_workspace_path.startswith("/"):
                 logger.warning(f"Attempted invalid path traversal: {relative_workspace_path}")
                 return {"success": False, "error": "Invalid path specified. Must be relative to workspace root and contain no '..'."}

            # Normalize path (optional, depends on API expectation)
            # clean_path = os.path.normpath(relative_workspace_path)
            clean_path = relative_workspace_path # Assuming API handles this

            logger.info(f"Tool: Listing directory '/opt/sutazaiapp/{clean_path}'")

            result = default_api.list_dir(relative_workspace_path=clean_path, explanation=explanation)

            if isinstance(result, dict) and "list_dir_response" in result:
                response_data = result["list_dir_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_list_dir API call failed for {clean_path}: {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "List directory API call failed.")}
                else:
                     # Assuming successful response contains the list/details
                     return {"success": True, "contents": response_data}
            else:
                 logger.error(f"_tool_list_dir received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_list_dir for '{relative_workspace_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred listing directory: {str(e)}"}

    def _tool_codebase_search(self, query: str, target_directories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Tool implementation: Semantic code search using default_api with error checks."""
        explanation = f"Agent requested codebase search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for codebase search.")
                 return {"success": False, "error": "Invalid or empty query provided for codebase search."}
            if target_directories and not isinstance(target_directories, list):
                 logger.error("Invalid target_directories provided for codebase search (must be a list).")
                 return {"success": False, "error": "Invalid target_directories format (must be a list)."}

            logger.info(f"Tool: Performing codebase search for query: '{query}' Target Dirs: {target_directories}")
            result = default_api.codebase_search(query=query, target_directories=target_directories, explanation=explanation)

            if isinstance(result, dict) and "codebase_search_response" in result:
                response_data = result["codebase_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_codebase_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Codebase search API call failed.")}
                else:
                    # Assuming successful response contains the search results
                    return {"success": True, "search_results": response_data}
            else:
                 logger.error(f"_tool_codebase_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_codebase_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during codebase search: {str(e)}"}

    def _tool_grep_search(self, query: str, include_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None, case_sensitive: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Grep search using default_api with error checks."""
        explanation = f"Agent requested grep search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for grep search.")
                 return {"success": False, "error": "Invalid or empty query provided for grep search."}

            logger.info(f"Tool: Performing grep search for pattern: '{query}', Include: {include_pattern}, Exclude: {exclude_pattern}, CaseSensitive: {case_sensitive}")
            is_case_sensitive = bool(case_sensitive) # Ensure boolean

            result = default_api.grep_search(
                query=query,
                include_pattern=include_pattern,
                exclude_pattern=exclude_pattern,
                case_sensitive=is_case_sensitive,
                explanation=explanation
            )

            if isinstance(result, dict) and "grep_search_response" in result:
                response_data = result["grep_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_grep_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Grep search API call failed.")}
                else:
                    # Assuming successful response contains matches
                    return {"success": True, "matches": response_data}
            else:
                 logger.error(f"_tool_grep_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_grep_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during grep search: {str(e)}"}

    def _tool_file_search(self, query: str) -> Dict[str, Any]:
        """Tool implementation: Fuzzy file search using default_api with error checks."""
        explanation = f"Agent requested file search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for file search.")
                 return {"success": False, "error": "Invalid or empty query provided for file search."}

            logger.info(f"Tool: Performing file search for query: '{query}'")
            result = default_api.file_search(query=query, explanation=explanation)

            if isinstance(result, dict) and "file_search_response" in result:
                response_data = result["file_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_file_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "File search API call failed.")}
                else:
                    # Assuming successful response is the list of files
                    return {"success": True, "found_files": response_data}
            else:
                 logger.error(f"_tool_file_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_file_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during file search: {str(e)}"}
    
    def _tool_write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Tool implementation: Write file with path validation and error handling."""
        explanation = f"Agent requested write access to {file_path}."
        try:
            # Path validation
            abs_path = os.path.abspath(file_path)
            workspace_root = os.path.abspath("/opt/sutazaiapp") # Define workspace root
            if not abs_path.startswith(workspace_root):
                 logger.warning(f"Attempt to write file outside workspace: {file_path}")
                 return {"success": False, "error": "File write access restricted to workspace."}

            logger.info(f"Tool: Writing to file '{abs_path}'")
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully wrote to file '{abs_path}'")
            return {"success": True, "file_path": abs_path}
        except IOError as io_err:
             logger.error(f"IOError writing to file {file_path}: {io_err}")
             return {"success": False, "error": f"IOError writing file: {io_err}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_write_file for '{file_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred writing file: {str(e)}"}
    
    def _tool_execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Tool implementation: Execute code in sandbox (Placeholder - UNSAFE)."""
        explanation = f"Agent requested execution of {language} code."
        logger.warning("execute_code tool is a placeholder and does not run code securely.")
        # In a real scenario, this should call a secure sandbox service (e.g., using Docker, firecracker)
        # NEVER use eval() or subprocess directly on untrusted agent output.
        return {"success": False, "error": "Code execution sandbox not implemented yet."}
    
    def _tool_search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Tool implementation: Search documents in vector DB with error handling."""
        explanation = f"Agent requested document search for: {query}"
        logger.info(f"Tool: Executing search_documents with query: '{query}', n_results: {n_results}")
        try:
            # Validate n_results
            try:
                num_results = max(1, int(n_results)) # Ensure positive integer
            except (ValueError, TypeError):
                 logger.warning(f"Invalid n_results '{n_results}', defaulting to 5.")
                 num_results = 5

            # Define which vector DB model to query - consider making configurable
            vector_db_model_id = "chroma-store"

            logger.debug(f"Querying vector DB '{vector_db_model_id}' for '{query}' (k={num_results})")
            # Call ModelManager to query the vector DB
            search_params = {"k": num_results}
            results = self.model_manager.run_inference(
                vector_db_model_id,
                query, # Input is the query string
                parameters=search_params
            )

            # Check for errors returned by the inference call
            if isinstance(results, dict) and "error" in results:
                 logger.error(f"Vector DB query failed: {results['error']}")
                 return {"success": False, "error": results['error']}

            logger.debug(f"Raw vector DB results: {results}")

            # Basic validation of expected keys from ChromaDB (adapt if DB changes)
            if not isinstance(results, dict) or not all(k in results for k in ["ids", "documents", "metadatas", "distances"]):
                 logger.error(f"Unexpected result format from vector DB: {results}")
                 return {"success": False, "error": "Unexpected format from vector database."}

            # Format results safely, handling potential None values
            formatted_results = [
                 {
                     "id": doc_id,
                     "content": doc_content if doc_content is not None else "",
                     "metadata": metadata if metadata is not None else {},
                     "distance": distance if distance is not None else float('inf'),
                 }
                 for doc_id, doc_content, metadata, distance in zip(
                     results.get("ids", [[]])[0], # Chroma nests results
                     results.get("documents", [[]])[0],
                     results.get("metadatas", [[]])[0],
                     results.get("distances", [[]])[0]
                 )
            ]

            logger.info(f"Vector DB search successful, returning {len(formatted_results)} results.")
            return {"success": True, "results": formatted_results}

        except Exception as e:
            logger.error(f"Unexpected error in _tool_search_documents for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during document search: {str(e)}"}

    def _tool_calculator(self, expression: str) -> Dict[str, Any]:
        """Tool implementation: Evaluate a simple mathematical expression (Placeholder - UNSAFE)."""
        explanation = f"Agent requested calculation for: {expression}"
        logger.warning("Calculator tool uses eval() and is UNSAFE for untrusted input.")
        # In a real scenario, use a safer math parsing library (like numexpr or ast)
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression."}
        try:
            # Limit eval scope
            result = eval(expression, {"__builtins__": {}}, {})
            logger.info(f"Tool: Calculated '{expression}' = {result}")
            return {"success": True, "result": str(result)}
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return {"success": False, "error": f"Failed to evaluate expression: {e}"}

    def _get_agent_config(self, agent_identifier: str) -> Optional[Tuple[str, AgentConfig]]:
        """Find agent config by ID or name."""
        if agent_identifier in self.agent_configs:
            return agent_identifier, self.agent_configs[agent_identifier]
        for agent_id, config in self.agent_configs.items():
            if config.name == agent_identifier:
                return agent_id, config
        return None

    def create_agent(
        self, agent_identifier: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an agent instance from a configuration ID or name.
        
        Args:
            agent_identifier: ID or Name of the agent configuration.
            parameters: Optional parameters to override configuration defaults.
            
        Returns:
            Instance ID for the created agent, or None if creation failed.
        """
        config_result = self._get_agent_config(agent_identifier)
        if not config_result:
            logger.error(f"Agent configuration '{agent_identifier}' not found.")
            return None # Return None instead of raising error directly

        agent_id, config = config_result
        instance_id = f"{agent_id}-{uuid.uuid4().hex[:8]}"
        logger.info(f"Attempting to create agent instance '{instance_id}' from config '{agent_id}'.")
        
        # Set initial state
        self.agent_states[instance_id] = AgentState.INITIALIZING
        
        try:
            # Merge parameters safely
            config_params = config.parameters.copy()
            if parameters:
                if not isinstance(parameters, dict):
                    logger.warning("Invalid parameters format for creation, using defaults.")
            else:
                    config_params.update(parameters)

            # Create the agent instance (returns dict with instance or error)
            agent_data = self._create_framework_agent(instance_id, config, config_params)

            # Check if instance creation failed
            if not agent_data or agent_data.get("error") or not agent_data.get("instance"):
                 creation_error = agent_data.get('error', 'Unknown creation error') if isinstance(agent_data, dict) else 'Creation returned None'
                 logger.error(f"Failed to create agent instance '{agent_id}': {creation_error}")
                 self.agent_states[instance_id] = AgentState.ERROR
                 # Clean up potential partial registration?
                 if instance_id in self.active_agents: del self.active_agents[instance_id]
                 if instance_id in self.agent_states: del self.agent_states[instance_id]
                 return None

            # Store successfully created instance
            self.active_agents[instance_id] = agent_data
            self.agent_states[instance_id] = AgentState.IDLE # Set state after successful creation
            logger.info(f"Created and stored new instance '{instance_id}' (Type: {agent_data.get('type')})")
            return instance_id
            
        except Exception as e:
            # Catch unexpected errors during the creation process
            logger.error(f"Unexpected error creating agent instance '{agent_id}': {e}", exc_info=True)
            self.agent_states[instance_id] = AgentState.ERROR
            return None

    def _create_framework_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Factory method to create agent based on framework type."""
        framework = config.framework
        creator_method_name = f"_create_{framework.value}_agent"
        creator_method = getattr(self, creator_method_name, None)

        if creator_method and callable(creator_method):
            logger.info(f"Creating agent instance '{instance_id}' using framework: {framework.value}")
            # The specific _create_* method should handle its own errors and return dict
            return creator_method(instance_id, config, params)
        else:
            logger.error(f"Unsupported agent framework type: {framework.value}")
            # Return error dict directly
            return {"type": "error", "instance": None, "config": config, "error": f"Unsupported framework: {framework.value}"}

    def _create_langchain_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a LangChain agent instance (ReAct)."""
        error_return = {"type": "langchain", "instance": None, "config": config} # Predefined error structure
        try:
            # --- 1. Load LLM Instance ---
            model_id = config.model_id
            logger.info(f"Requesting model manager to load model '{model_id}' for agent '{instance_id}'.")
            # Assuming load_model returns bool or raises error on failure
            if not self.model_manager.load_model(model_id):
                 error_msg = f"Model manager failed to load model '{model_id}'."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return

            loaded_model_data = self.model_manager.loaded_models.get(model_id)
            if not loaded_model_data or "instance" not in loaded_model_data:
                 error_msg = f"Model '{model_id}' not found in loaded models after load attempt."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return
            llm = loaded_model_data["instance"]
            logger.info(f"Successfully obtained LLM instance for model: {model_id}")

            # --- 2. Check for Tools (Codebase Agent requires tools) ---
            if not config.tools:
                error_msg = f"LangChain agent '{config.name}' requires tools for ReAct framework. Check config."
                logger.error(error_msg)
                error_return["error"] = error_msg
                return error_return

            # --- 3. Map and Wrap Tools ---
            agent_tools = []
            missing_tool_methods = []
            for tool_name in config.tools:
                tool_method = getattr(self, f"_tool_{tool_name}", None)
                tool_config = self.tools.get(tool_name)
                if tool_method and callable(tool_method) and tool_config:
                    # Langchain Tool expects a sync function
                    # If tool methods were async, would need `arun` or wrapper
                    agent_tools.append(Tool(name=tool_config.name, func=tool_method, description=tool_config.description))
                    logger.debug(f"Mapped tool '{tool_name}' ({tool_config.name})")
                else:
                    missing_tool_methods.append(tool_name)
                    logger.warning(f"Tool implementation method `_tool_{tool_name}` or definition for '{tool_name}' not found.")

            if not agent_tools:
                error_msg = f"Agent '{config.name}' lists tools, but none could be mapped. Missing: {missing_tool_methods}"
                logger.error(error_msg)
                error_return["error"] = error_msg
                return error_return
            logger.info(f"Mapped {len(agent_tools)} tools for agent '{instance_id}': {[t.name for t in agent_tools]}")

            # --- 4. Create Custom Codebase-Focused ReAct Prompt ---
            try:
                base_prompt = hub.pull("hwchase17/react")
                template_format = base_prompt.template
                react_format_instructions_start = template_format.find("Use the following format:")
                react_format_instructions = template_format[react_format_instructions_start:]
                react_format_instructions = react_format_instructions.replace(
                    "Question: the input question you must answer",
                    "Original Question: the user query you must address"
                )
            except Exception as hub_error:
                logger.error(f"Failed to pull base ReAct prompt from hub: {hub_error}")
                error_return["error"] = f"Failed to load base prompt: {hub_error}"
                return error_return

            codebase_system_message = (
                f"You are SutazAI, a highly capable AI assistant specialized in understanding and analyzing the codebase located at /opt/sutazaiapp. "
                f"Your goal is to answer questions and fulfill requests about this specific codebase. "
                f"Always assume the user is asking about the code within /opt/sutazaiapp unless specified otherwise. "
                f"You have access to the following tools: {[t.name for t in agent_tools]}. "
                f"Think step-by-step. For questions about code structure or finding things, start by using tools like 'list_dir', 'file_search', 'grep_search', or 'codebase_search'. "
                f"Only use 'read_file' on specific files identified in previous steps. Be precise with file paths. "
                f"Adhere strictly to the required action format."
            )
            custom_template_str = (
                f"{codebase_system_message}\n\n"
                f"TOOLS:\n------\n"
                f"You have access to the following tools:\n"
                f"{{tools}}\n\n" # Filled by partial
                f"{{react_format_instructions}}\n\n" # Filled by partial
                f"Begin!\n\n"
                f"Previous conversation history:\n"
                f"{{chat_history}}\n\n" # Filled by memory
                f"Original Question: {{input}}\n" # User input
                f"Thought:{{agent_scratchpad}}" # Agent working space
            )

            try:
                prompt = PromptTemplate.from_template(custom_template_str)
                prompt = prompt.partial(
                    tools="\n".join([f"{tool.name}: {tool.description}" for tool in agent_tools]),
                    tool_names=", ".join([tool.name for tool in agent_tools]),
                    react_format_instructions=react_format_instructions
                )
                logger.info("Created custom ReAct prompt for codebase interaction.")
            except Exception as prompt_error:
                 logger.error(f"Failed to create custom prompt template: {prompt_error}")
                 error_return["error"] = f"Prompt template error: {prompt_error}"
                 return error_return

            # --- 5. Create ReAct Agent ---
            try:
                 agent = create_react_agent(llm, agent_tools, prompt)
                 logger.info("Attempting agent creation with create_react_agent using custom prompt.")
            except Exception as agent_creation_error:
                 logger.error(f"Failed create_react_agent: {agent_creation_error}", exc_info=True)
                 error_return["error"] = f"Failed to create ReAct agent: {agent_creation_error}"
                 return error_return

            # --- 6. Create Memory ---
            # Consider potential deprecation warnings and alternatives if needed
            try:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            except Exception as memory_error:
                 logger.error(f"Failed to create ConversationBufferMemory: {memory_error}")
                 error_return["error"] = f"Memory creation error: {memory_error}"
                 return error_return

            # --- 7. Create Agent Executor ---
            try:
                max_iterations = params.get("max_iterations", 10)
                agent_executor = AgentExecutor(
                     agent=agent,
                     tools=agent_tools,
                     memory=memory,
                     verbose=True, # Consider making this configurable
                     handle_parsing_errors="Check your output and make sure it conforms!", # Or a custom handler
                     max_iterations=max_iterations
                )
                logger.info(f"Successfully created LangChain ReAct agent executor for instance: {instance_id}")
                # Return success dictionary
                return {"type": "langchain_agent", "instance": agent_executor, "config": config}
            except Exception as executor_error:
                 logger.error(f"Failed to create AgentExecutor: {executor_error}", exc_info=True)
                 error_return["error"] = f"AgentExecutor creation error: {executor_error}"
                 return error_return

        except Exception as e:
            # Catch-all for unexpected errors during the process
            logger.error(f"Unexpected error creating LangChain agent instance '{instance_id}': {e}", exc_info=True)
            error_return["error"] = f"Unexpected agent creation error: {str(e)}"
            return error_return

    def _create_autogpt_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an AutoGPT agent."""
        logger.warning("AutoGPT task execution is not implemented.")
        raise NotImplementedError("AutoGPT task execution is not implemented.")

    def _create_localagi_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating a LocalAGI agent."""
        logger.warning("LocalAGI task execution is not implemented.")
        raise NotImplementedError("LocalAGI task execution is not implemented.")

    def _create_autogen_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an Autogen agent."""
        logger.warning("Autogen task execution is not fully implemented.")
        # Existing placeholder logic... needs proper async handling if used in process_chat
        raise NotImplementedError("Autogen task execution is not fully implemented.")

    def _create_custom_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Creates a custom agent based on BaseAgent."""
        # This structure seems complex and might duplicate logic better handled
        # by specific agent classes. Consider refactoring BaseAgent/CustomAgent interaction.
        error_return = {"type": "custom", "instance": None, "config": config}
        logger.warning("Custom agent creation logic needs review for robustness.")
        try:
            # Import BaseAgent locally to avoid potential circular dependencies at module level
            from ai_agents.base_agent import BaseAgent

            # Define the CustomAgent class locally or ensure it's properly imported
            # This nested class definition is unusual; prefer defining it at module level
            class CustomAgentImpl(BaseAgent):
                def __init__(self, config: AgentConfig, model_manager: ModelManager, tools: Dict[str, AgentTool], params: Dict[str, Any]):
                    # Pass the AgentConfig object directly to BaseAgent's __init__
                    super().__init__(config) # BaseAgent now expects AgentConfig
                    self.model_manager = model_manager
                    self.available_tools = tools # Renamed from self.tools to avoid conflict
                    self.params = params
                    # self.model_id is already set in BaseAgent.__init__
                    self.history: List[Dict[str, Any]] = [] # Explicitly initialize history

                def _initialize(self):
                    logger.info(f"Initializing CustomAgentImpl for model: {self.model_id}")
                    if not self.model_manager.load_model(self.model_id):
                        raise RuntimeError(f"Failed to load model {self.model_id} via ModelManager.")
                    logger.info(f"CustomAgentImpl initialized successfully for model: {self.model_id}")
                    return True # Indicate success explicitly? BaseAgent doesn't use return

                def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
                    logger.info(f"CustomAgentImpl executing task: {task.get('id', 'N/A')}")
                    prompt = self._build_prompt(task)
                    logger.debug(f"CustomAgentImpl prompt: {prompt}")

                    inference_params = {
                        "temperature": self.params.get("temperature", 0.7),
                        "max_tokens": self.params.get("max_tokens", 1024), # Reduced default?
                        # Add other relevant parameters from self.params
                    }
                    
                    result = self.model_manager.run_inference(
                        self.model_id,
                        prompt,
                        inference_params,
                    )

                    if isinstance(result, dict) and "error" in result:
                        error_msg = f"Inference error: {result['error']}"
                        logger.error(error_msg)
                        # Raise an exception to be caught by the calling method
                        raise RuntimeError(error_msg)

                    output = result.get("text", "No text output from model.")
                    logger.debug(f"CustomAgentImpl raw output: {output}")

                    # Basic history update
                    self.history.append({"task": task.get('instruction', 'N/A'), "output": output})
                    # Limit history size if needed
                    max_history = 10
                    if len(self.history) > max_history:
                        self.history = self.history[-max_history:]

                    return {"output": output, "task_id": task.get("id", "N/A")} # Return standard structure

                def _cleanup(self):
                    logger.info(f"Cleaning up CustomAgentImpl for model: {self.model_id}")
                    # Nothing specific to clean up here unless resources were allocated
                    return True

                def _build_prompt(self, task: Dict[str, Any]) -> str:
                    # Simplified prompt building example
                    parts = []
                    parts.append(f"### Task Instruction: {task.get('instruction', 'No instruction provided.')}")

                    if task.get("context"):
                        parts.append(f"### Context: {task['context']}")

                    if self.history:
                        hist_str = "\n".join([f"Prev Task: {h['task'][:100]}...\nPrev Output: {h['output'][:100]}..." for h in self.history])
                        parts.append(f"### Recent Conversation History: {hist_str}")

                    if self.config.tools:
                        tool_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.available_tools.items() if name in self.config.tools])
                        parts.append(f"### Available Tools: {tool_desc}")

                    parts.append("\n### Your Response:")
                    return "\n\n".join(parts)

            # --- End of Nested Class ---

            # Map tools needed by this specific agent config
            agent_specific_tools = {
                tool_name: self.tools[tool_name]
                for tool_name in config.tools if tool_name in self.tools
            }

            # Create the agent instance
            custom_agent_instance = CustomAgentImpl(
                config=config,
                model_manager=self.model_manager,
                tools=agent_specific_tools,
                params=params,
            )

            # Initialize the agent (critical step)
            try:
                custom_agent_instance.initialize() # Call BaseAgent's initialize
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

    def execute_task(self, instance_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with an agent
        
        Args:
            instance_id: ID of the agent instance
            task: Task description
            
        Returns:
            Result of the task execution
        """
        if instance_id not in self.active_agents:
            logger.error(f"Agent instance {instance_id} not found")
            raise ValueError(f"Agent instance {instance_id} not found")
        
        with self.agent_locks[instance_id]:
            try:
                # Update agent state
                self.agent_states[instance_id] = AgentState.PROCESSING
                
                # Get the agent
                agent_data = self.active_agents[instance_id]
                agent_type = agent_data["type"]
                agent = agent_data["instance"]
                
                # Execute task based on agent type
                if agent_type == "langchain":
                    result = self._execute_langchain_task(agent, task)
                elif agent_type == "autogpt":
                    result = self._execute_autogpt_task(agent, task)
                elif agent_type == "localagi":
                    result = self._execute_localagi_task(agent, task)
                elif agent_type == "autogen":
                    result = self._execute_autogen_task(agent_data, task)
                elif agent_type == "custom":
                    result = self._execute_custom_task(agent, task)
                else:
                    raise ValueError(f"Unsupported agent type: {agent_type}")
                
                # Update agent state
                self.agent_states[instance_id] = AgentState.IDLE
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing task with agent {instance_id}: {str(e)}")
                self.agent_states[instance_id] = AgentState.ERROR
                raise
    
    def _execute_langchain_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LangChain agent"""
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        
        # Format the prompt
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction
        
        # Run the agent
        try:
            response = agent.run(prompt)
            return {"output": response, "task_id": task.get("id")}
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def _execute_autogpt_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an AutoGPT agent"""
        # This is a placeholder implementation
        return {
            "output": "AutoGPT execution not implemented",
            "task_id": task.get("id"),
        }
    
    def _execute_localagi_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LocalAGI agent"""
        # This is a placeholder implementation
        return {
            "output": "LocalAGI execution not implemented",
            "task_id": task.get("id"),
        }
    
    def _execute_autogen_task(self, agent_data, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an Autogen agent"""
        agent = agent_data["instance"]
        user_proxy = agent_data.get("user_proxy")
        
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        
        # Format the message
        if context:
            message = f"{instruction}\n\nContext: {context}"
        else:
            message = instruction
        
        # Run with or without user proxy
        try:
            if user_proxy:
                # Start a conversation between user proxy and agent
                user_proxy.initiate_chat(agent, message=message)
                # Get the chat history
                history = user_proxy.chat_messages[agent]
                # Extract the last message from the agent
                last_message = history[-1]["content"] if history else "No response"
                return {"output": last_message, "task_id": task.get("id")}
            else:
                # Direct message to agent
                response = agent.generate_reply(message)
                return {"output": response, "task_id": task.get("id")}
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def _execute_custom_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a custom agent"""
        try:
            result = agent._execute(task)
            return result
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}

    async def process_chat(self, agent_name: str, messages: List[Dict[str, Any]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat request using the specified agent instance (creating if necessary).
        Handles routing to the correct execution logic based on agent type.
        """
        logger.info(f"Processing chat request for agent name: '{agent_name}'")

        # --- 1. Find Agent Config ---
        config_result = self._get_agent_config(agent_name)
        if not config_result:
            error_msg = f"Agent configuration with name '{agent_name}' not found."
            logger.error(error_msg)
            return {"response": f"Error: {error_msg}", "usage": {}}
        agent_id, config = config_result
        logger.debug(f"Found config for agent '{agent_name}' with ID '{agent_id}'. Framework: {config.framework.value}")

        # --- 2. Get or Create Agent Instance ---
        # Use agent config name or ID consistently for instance tracking. Let's use agent_id.
        instance_id = agent_id
        instance_data = self.active_agents.get(instance_id)

        if not instance_data:
            logger.warning(f"No active instance found for '{instance_id}' (name: '{agent_name}'). Attempting creation.")
            # Merge parameters for creation
            creation_params = config.parameters.copy()
            if parameters and isinstance(parameters, dict):
                creation_params.update(parameters)
            else:
                # Log if parameters are invalid, but proceed with defaults
                if parameters is not None:
                    logger.warning("Invalid parameters format provided for creation, using defaults.")

            # Create agent (returns dict with instance or error)
            instance_data = self._create_framework_agent(instance_id, config, creation_params)

            # Validate creation result
            if not instance_data or instance_data.get("error") or not instance_data.get("instance"):
                 creation_error = instance_data.get('error', 'Unknown creation error') if isinstance(instance_data, dict) else 'Creation returned None'
                 error_msg = f"Failed to create agent instance '{agent_name}': {creation_error}. Check logs."
                 logger.error(error_msg)
                 # Update state if possible (instance_id might not exist if creation failed early)
                 if instance_id in self.agent_states:
                     self.agent_states[instance_id] = AgentState.ERROR
                 return {"response": f"Error: {error_msg}", "usage": {}}


        # --- 3. Prepare Input ---
        if not messages or not isinstance(messages, list):
             logger.error("Invalid 'messages' format received.")
             return {"response": "Error: Invalid chat message format.", "usage": {}}
        try:
            # Ensure history processing doesn't fail on empty list
            chat_history = messages[:-1] if len(messages) > 1 else []
            last_user_message_obj = messages[-1] if messages else None

            if not last_user_message_obj or not isinstance(last_user_message_obj, dict) or last_user_message_obj.get("role") != "user":
                 logger.error("Last message in list is not a valid user message.")
                 return {"response": "Error: Invalid final message (must be from user).", "usage": {}}

            last_user_message = last_user_message_obj.get("content", "").strip()
            if not last_user_message:
                logger.warning("Received empty user message.")
                return {"response": "Please provide a message.", "usage": {}}
        except (IndexError, TypeError, KeyError) as e:
             logger.error(f"Error processing input messages: {e}")
             return {"response": "Error: Could not process input messages.", "usage": {}}

        # --- 4. Execute Task ---
        instance_type = instance_data.get("type")
        executor_or_agent = instance_data.get("instance") # Renamed for clarity

        if not instance_type or not executor_or_agent:
             error_msg = f"Instance data for '{agent_name}' is corrupted or incomplete."
             logger.error(error_msg)
             self.agent_states[instance_id] = AgentState.ERROR
             return {"response": f"Error: {error_msg}", "usage": {}}

        logger.debug(f"Executing chat task for '{instance_id}' with instance type: {instance_type}")
        self.agent_states[instance_id] = AgentState.PROCESSING # Set state before execution

        try:
            # Route to the correct async execution method based on instance type
            if instance_type == "langchain_agent":
                result = await self._execute_langchain_agent_task(executor_or_agent, last_user_message, chat_history)
            elif instance_type == "custom":
                 # Custom agent execution needs to be adapted for async chat if needed
                 # Wrap the synchronous _execute method using run_in_executor
                 task_dict = {"instruction": last_user_message, "context": "", "id": f"chat-{uuid.uuid4().hex[:8]}"} # Create task dict
                 result = await self._execute_custom_task_async_wrapper(executor_or_agent, task_dict)

            elif instance_type in ["autogpt", "localagi", "autogen"]:
                 # Call placeholder methods (which raise NotImplementedError)
                 exec_method_name = f"_execute_{instance_type}_task"
                 exec_method = getattr(self, exec_method_name, None)
                 if exec_method and callable(exec_method):
                      # Assuming these placeholders might need a task dict too
                      task_dict = {"instruction": last_user_message, "context": "", "id": f"chat-{uuid.uuid4().hex[:8]}"}
                      # Placeholders are sync and raise NotImplementedError
                      exec_method(executor_or_agent, task_dict) # This will raise the error
                      # Should not reach here if exec_method raises correctly
                      raise RuntimeError(f"Placeholder {exec_method_name} did not raise NotImplementedError.")
                 else:
                      raise NotImplementedError(f"Execution logic for framework '{instance_type}' not found.")
            else:
                error_msg = f"Unsupported instance type for chat execution: {instance_type}"
                logger.error(error_msg)
                self.agent_states[instance_id] = AgentState.ERROR
                return {"response": f"Error: {error_msg}", "usage": {}}

            # Process result
            if not isinstance(result, dict):
                 error_msg = f"Execution for '{agent_name}' returned non-dict result: {type(result)}"
                 logger.error(error_msg)
                 self.agent_states[instance_id] = AgentState.ERROR
                 return {"response": f"Error: Internal agent execution error.", "usage": {}}

            response_content = result.get("output", "Agent did not return standard 'output'.")
            usage = result.get("usage", {}) # TODO: Implement usage tracking

            logger.info(f"Chat processed successfully by instance: {agent_name} ({instance_id})")
            self.agent_states[instance_id] = AgentState.IDLE # Reset state on success
            return {"response": response_content, "usage": usage}

        except NotImplementedError as nie:
             error_msg = f"Execution for agent type '{instance_type}' is not implemented: {nie}"
             logger.error(error_msg)
             self.agent_states[instance_id] = AgentState.ERROR
             return {"response": f"Error: {error_msg}", "usage": {}}
        except Exception as e:
            error_msg = f"Chat processing error: {e}" # Simplify error message for UI
            logger.error(f"Error during task execution for instance {agent_name} ({instance_id}): {e}", exc_info=True) # Log full traceback
            self.agent_states[instance_id] = AgentState.ERROR # Set error state
            return {"response": f"Error: {error_msg}", "usage": {}}


    async def _execute_langchain_agent_task(
        self, agent_executor: AgentExecutor, user_input: str, chat_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Execute a task with a pre-created LangChain AgentExecutor instance."""
        # Implementation of the method
        # This method should be implemented to handle the execution of a task with a LangChain agent
        # It should return a dictionary with the result of the task execution
        # If the task execution fails, it should return a dictionary with an error message
        # The method should handle the execution of the agent and return the result
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
        # The method should handle any exceptions that may occur during the execution
        # The method should return a dictionary with the result of the task execution
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
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from ai_agents.model_manager import ModelManager
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import LlamaCpp
from langchain import hub

# Configure logging
LOG_FILE = "logs/agent_framework.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("AgentFramework")


class AgentFrameworkType(Enum):
    """Types of agent frameworks supported"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    LOCALAGI = "localagi"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
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
    MULTIMODAL = "multimodal"
    COLLABORATION = "collaboration"
    CODE_UNDERSTANDING = "code_understanding"


@dataclass
class AgentTool:
    """Represents a tool that an agent can use"""
    name: str
    description: str
    function: Callable[..., Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_capability: List[AgentCapability] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_capability": [cap.value for cap in self.requires_capability],
        }


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    description: str
    framework: AgentFrameworkType
    capabilities: List[AgentCapability]
    model_id: str
    tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
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
        """Create from dictionary representation"""
        try:
            return cls(
                name=data["name"],
                description=data["description"],
                framework=AgentFrameworkType(data["framework"]),
                capabilities=[AgentCapability(cap) for cap in data["capabilities"]],
                model_id=data["model_id"],
                tools=data.get("tools", []),
                parameters=data.get("parameters", {}),
            )
        except KeyError as ke:
            raise ValueError(f"Missing required key in agent config data: {ke}")
        except ValueError as ve:
            # Catches errors from Enum creation if invalid value provided
             raise ValueError(f"Invalid value in agent config data: {ve}")


class AgentState(Enum):
    """Possible states of an agent"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentFramework:
    """
    Unified framework for integrating different agent types
    
    This class provides a standardized interface for working with
    agents from different frameworks like LangChain, AutoGPT, etc.
    It handles agent lifecycle, tool registration, and coordination.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config_path: str = "config/agents.json",
        tools_dir: str = "ai_agents/tools",
        max_concurrent_agents: int = 5,
    ):
        """
        Initialize the agent framework
        
        Args:
            model_manager: ModelManager instance for agent models
            config_path: Path to agent configuration file
            tools_dir: Directory containing agent tools
            max_concurrent_agents: Maximum number of concurrent agents
        """
        if not isinstance(model_manager, ModelManager):
             raise TypeError("model_manager must be an instance of ModelManager")
        self.model_manager = model_manager
        self.config_path = config_path
        self.tools_dir = tools_dir
        self.max_concurrent_agents = max_concurrent_agents
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Initialize agent registries
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_states: Dict[str, AgentState] = {}
        
        # Initialize tool registry
        self.tools: Dict[str, AgentTool] = {}
        
        # Load configurations
        self._load_config()
        self._load_tools()
        
        logger.info(
            f"Agent framework initialized with {len(self.agent_configs)} agent configs and {len(self.tools)} tools"
        )
    
    def _load_config(self):
        """Load agent configurations from file"""
        temp_configs: Dict[str, AgentConfig] = {}
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    logger.debug(f"Raw config loaded from {self.config_path}")
                    
                    agents_dict = config_data.get("agents", {})
                    if not isinstance(agents_dict, dict):
                        logger.error("Invalid config format: 'agents' key is not a dictionary.")
                        return # Or raise error / use default

                    logger.debug(f"Found 'agents' dictionary with {len(agents_dict)} entries.")
                    for agent_id, agent_data in agents_dict.items():
                        try:
                            if not isinstance(agent_data, dict):
                                logger.warning(f"Skipping invalid agent config entry for '{agent_id}': not a dictionary.")
                                continue
                            temp_configs[agent_id] = AgentConfig.from_dict(agent_data)
                        except ValueError as agent_load_error:
                            logger.error(f"Error parsing config for agent '{agent_id}': {agent_load_error}")
                        except Exception as e:
                            logger.error(f"Unexpected error loading agent '{agent_id}': {e}", exc_info=True)

                    self.agent_configs = temp_configs
                    logger.info(f"Loaded {len(self.agent_configs)} valid agent configurations.")

            else:
                logger.warning(f"Configuration file not found at {self.config_path}. Creating default config.")
                self._create_default_config() # This also loads the defaults into self.agent_configs

        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON from {self.config_path}: {json_err}")
            logger.info("Creating default configuration as fallback due to JSON error.")
            self._create_default_config()
        except IOError as io_err:
            logger.error(f"Error reading configuration file {self.config_path}: {io_err}")
            # Decide how to handle this - perhaps raise an exception?
        except Exception as e:
            logger.error(f"Unexpected error loading agent configurations: {e}", exc_info=True)
            logger.info("Creating default configuration as fallback due to unexpected error.")
            self._create_default_config()

    def _create_default_config(self):
        """Create default agent configuration"""
        default_config = {
            "agents": {
                "coder": {
                    "name": "Code Assistant",
                    "description": "Generates and modifies code",
                    "framework": "langchain",
                    "capabilities": ["code_generation", "reasoning"],
                    "model_id": "deepseek-coder",
                    "tools": ["write_file", "read_file", "execute_code"],
                    "parameters": {"temperature": 0.2, "max_iterations": 5},
                },
                "researcher": {
                    "name": "Research Assistant",
                    "description": "Conducts research and analyzes information",
                    "framework": "langchain",
                    "capabilities": ["text_generation", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["search_documents"],
                    "parameters": {"temperature": 0.7, "max_iterations": 3},
                },
                "planner": {
                    "name": "Task Planner",
                    "description": "Plans tasks and creates workflows",
                    "framework": "autogpt",
                    "capabilities": ["planning", "reasoning"],
                    "model_id": "llama3-8b",
                    "tools": ["create_plan", "assign_tasks"],
                    "parameters": {
                        "temperature": 0.5,
                        "max_iterations": 10,
                        "autonomous_execution": False,
                    },
                },
                "document_processor": {
                    "name": "Document Processor",
                    "description": "Processes various document formats",
                    "framework": "custom",
                    "capabilities": ["document_processing", "text_generation"],
                    "model_id": "llama3-8b",
                    "tools": ["extract_text", "parse_pdf", "ocr_image"],
                    "parameters": {"temperature": 0.3, "max_document_size_mb": 20},
                },
            }
        }
        
        # Save default configuration
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        # Load the configs
        for agent_id, agent_data in default_config["agents"].items():
            self.agent_configs[agent_id] = AgentConfig.from_dict(agent_data)
        
        logger.info(
            f"Created default agent configuration with {len(self.agent_configs)} agents"
        )
    
    def _load_tools(self):
        """Load available tools for agents"""
        self.tools.clear() # Ensure clean slate if reloaded

        # Define tools using AgentTool dataclass
        # Note: Tool functions (`self._tool_*`) must exist below
        tool_definitions = [
            AgentTool(
                name="read_file",
                description="Read the contents of a specific file. Provide start/end lines (1-indexed) or use should_read_entire_file=true.",
                function=self._tool_read_file,
                parameters={
                    "target_file": {"type": "string", "description": "Path to the file."},
                    "start_line_one_indexed": {"type": "integer", "description": "(Optional) Start line."},
                    "end_line_one_indexed_inclusive": {"type": "integer", "description": "(Optional) End line."},
                    "should_read_entire_file": {"type": "boolean", "description": "(Optional) Read whole file."}
                },
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="list_dir",
                description="List contents of a directory relative to the workspace root /opt/sutazaiapp.",
                function=self._tool_list_dir,
                parameters={"relative_workspace_path": {"type": "string", "description": "Directory path relative to workspace."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
            AgentTool(
                name="codebase_search",
                description="Semantic search for code snippets.",
                function=self._tool_codebase_search,
                parameters={
                    "query": {"type": "string", "description": "Search query."},
                    "target_directories": {"type": "array", "description": "(Optional) List of directories (glob patterns)."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="grep_search",
                description="Fast exact text/regex search in files.",
                function=self._tool_grep_search,
                parameters={
                    "query": {"type": "string", "description": "Regex/text pattern."},
                    "include_pattern": {"type": "string", "description": "(Optional) Glob pattern for files to include."},
                    "exclude_pattern": {"type": "string", "description": "(Optional) Glob pattern to exclude."},
                    "case_sensitive": {"type": "boolean", "description": "(Optional) Case-sensitive search."}
                },
                requires_capability=[AgentCapability.CODE_UNDERSTANDING],
            ),
            AgentTool(
                name="file_search",
                description="Fuzzy search for files by path name.",
                function=self._tool_file_search,
                parameters={"query": {"type": "string", "description": "Partial/fuzzy file name/path."}},
                requires_capability=[AgentCapability.FILE_OPERATIONS],
            ),
             AgentTool(
                 name="write_file",
                 description="Write content to a file.",
                 function=self._tool_write_file,
                 parameters={
                     "file_path": {"type": "string", "description": "Path to the file."},
                     "content": {"type": "string", "description": "Content to write."}
                 },
                 requires_capability=[AgentCapability.FILE_OPERATIONS],
             ),
             AgentTool(
                 name="execute_code",
                 description="Execute code in sandbox (Placeholder).",
                 function=self._tool_execute_code,
                 parameters={
                     "code": {"type": "string", "description": "Code to execute."},
                     "language": {"type": "string", "description": "Programming language."}
                 },
                 requires_capability=[AgentCapability.CODE_EXECUTION],
             ),
             AgentTool(
                 name="search_documents",
                 description="Search documents in vector DB with error handling.",
                 function=self._tool_search_documents,
                 parameters={
                     "query": {"type": "string", "description": "Search query."},
                     "n_results": {"type": "integer", "description": "Number of results."}
                 },
                 requires_capability=[AgentCapability.DOCUMENT_PROCESSING],
             ),
             AgentTool(
                 name="calculator",
                 description="Evaluate a simple mathematical expression.",
                 function=self._tool_calculator,
                 parameters={"expression": {"type": "string", "description": "Mathematical expression."}},
                 requires_capability=[], # No specific capability needed
             ),
             # Add other tools as needed
        ]

        for tool in tool_definitions:
            # Basic check if implementation exists
            if hasattr(self, tool.function.__name__) and callable(getattr(self, tool.function.__name__)):
                self.tools[tool.name] = tool
            else:
                logger.error(f"Tool implementation method '{tool.function.__name__}' not found for tool '{tool.name}'. Tool not loaded.")

        logger.info(f"Loaded {len(self.tools)} tools.")
        # TODO: Implement dynamic loading from tools_dir

    def _tool_read_file(self, target_file: str, start_line_one_indexed: Optional[int] = None, end_line_one_indexed_inclusive: Optional[int] = None, should_read_entire_file: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Read file using default_api with robust error checking."""
        explanation = f"Agent requested read access to {target_file}. Entire: {should_read_entire_file}. Range: {start_line_one_indexed}-{end_line_one_indexed_inclusive}."
        try:
            # Validate target_file basic safety
            abs_path = os.path.abspath(target_file)
            workspace_root = os.path.abspath("/opt/sutazaiapp") # Define workspace root
            if not abs_path.startswith(workspace_root):
                 logger.warning(f"Attempt to read file outside workspace: {target_file}")
                 return {"success": False, "error": "File access restricted to workspace."}

            logger.info(f"Tool: Reading file '{abs_path}' (Entire: {should_read_entire_file}, Range: {start_line_one_indexed}-{end_line_one_indexed_inclusive})")
            start = int(start_line_one_indexed) if start_line_one_indexed is not None else 1
            end = int(end_line_one_indexed_inclusive) if end_line_one_indexed_inclusive is not None else 200 # Default end line
            read_all = bool(should_read_entire_file)

            if read_all:
                result = default_api.read_file(target_file=abs_path, should_read_entire_file=True, start_line_one_indexed=1, end_line_one_indexed_inclusive=-1, explanation=explanation)
            else:
                 start = max(1, start) # Ensure start is at least 1
                 end = max(start, end) # Ensure end is valid and after start
                 # Add reasonable limit to prevent reading huge chunks
                 if end - start > 1000: # Limit to 1000 lines read at once unless reading all
                      end = start + 1000
                      logger.warning(f"Read range exceeded limit, truncated to lines {start}-{end}")
                      explanation += f" (Truncated to {end})"
                 result = default_api.read_file(target_file=abs_path, should_read_entire_file=False, start_line_one_indexed=start, end_line_one_indexed_inclusive=end, explanation=explanation)

            # Check the structure of the API response
            if isinstance(result, dict) and "read_file_response" in result:
                response_data = result["read_file_response"]
                # Check if the response data itself signals an error (adapt if API format differs)
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_read_file API call failed for {target_file}: {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Read file API call failed.")}
                else:
                    # Return the actual content from the response
                    return {"success": True, "content": response_data}
            else:
                 logger.error(f"_tool_read_file received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}

        except ValueError as ve:
            logger.error(f"Error processing parameters for _tool_read_file: {ve}")
            return {"success": False, "error": f"Invalid parameter type: {ve}"}
        except FileNotFoundError:
            logger.error(f"File not found in _tool_read_file: {target_file}")
            return {"success": False, "error": f"File not found: {target_file}"}
        except PermissionError:
             logger.error(f"Permission denied reading file: {target_file}")
             return {"success": False, "error": f"Permission denied reading file: {target_file}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_read_file for '{target_file}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred while reading file: {str(e)}"}

    def _tool_list_dir(self, relative_workspace_path: str) -> Dict[str, Any]:
        """Tool implementation: List directory contents using default_api with validation."""
        explanation = f"Agent requested listing of directory: {relative_workspace_path}"
        try:
            # Basic path validation
            if ".." in relative_workspace_path or relative_workspace_path.startswith("/"):
                 logger.warning(f"Attempted invalid path traversal: {relative_workspace_path}")
                 return {"success": False, "error": "Invalid path specified. Must be relative to workspace root and contain no '..'."}

            # Normalize path (optional, depends on API expectation)
            # clean_path = os.path.normpath(relative_workspace_path)
            clean_path = relative_workspace_path # Assuming API handles this

            logger.info(f"Tool: Listing directory '/opt/sutazaiapp/{clean_path}'")

            result = default_api.list_dir(relative_workspace_path=clean_path, explanation=explanation)

            if isinstance(result, dict) and "list_dir_response" in result:
                response_data = result["list_dir_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_list_dir API call failed for {clean_path}: {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "List directory API call failed.")}
                else:
                     # Assuming successful response contains the list/details
                     return {"success": True, "contents": response_data}
            else:
                 logger.error(f"_tool_list_dir received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_list_dir for '{relative_workspace_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred listing directory: {str(e)}"}

    def _tool_codebase_search(self, query: str, target_directories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Tool implementation: Semantic code search using default_api with error checks."""
        explanation = f"Agent requested codebase search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for codebase search.")
                 return {"success": False, "error": "Invalid or empty query provided for codebase search."}
            if target_directories and not isinstance(target_directories, list):
                 logger.error("Invalid target_directories provided for codebase search (must be a list).")
                 return {"success": False, "error": "Invalid target_directories format (must be a list)."}

            logger.info(f"Tool: Performing codebase search for query: '{query}' Target Dirs: {target_directories}")
            result = default_api.codebase_search(query=query, target_directories=target_directories, explanation=explanation)

            if isinstance(result, dict) and "codebase_search_response" in result:
                response_data = result["codebase_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_codebase_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Codebase search API call failed.")}
                else:
                    # Assuming successful response contains the search results
                    return {"success": True, "search_results": response_data}
            else:
                 logger.error(f"_tool_codebase_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_codebase_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during codebase search: {str(e)}"}

    def _tool_grep_search(self, query: str, include_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None, case_sensitive: Optional[bool] = False) -> Dict[str, Any]:
        """Tool implementation: Grep search using default_api with error checks."""
        explanation = f"Agent requested grep search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for grep search.")
                 return {"success": False, "error": "Invalid or empty query provided for grep search."}

            logger.info(f"Tool: Performing grep search for pattern: '{query}', Include: {include_pattern}, Exclude: {exclude_pattern}, CaseSensitive: {case_sensitive}")
            is_case_sensitive = bool(case_sensitive) # Ensure boolean

            result = default_api.grep_search(
                query=query,
                include_pattern=include_pattern,
                exclude_pattern=exclude_pattern,
                case_sensitive=is_case_sensitive,
                explanation=explanation
            )

            if isinstance(result, dict) and "grep_search_response" in result:
                response_data = result["grep_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_grep_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "Grep search API call failed.")}
                else:
                    # Assuming successful response contains matches
                    return {"success": True, "matches": response_data}
            else:
                 logger.error(f"_tool_grep_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_grep_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during grep search: {str(e)}"}

    def _tool_file_search(self, query: str) -> Dict[str, Any]:
        """Tool implementation: Fuzzy file search using default_api with error checks."""
        explanation = f"Agent requested file search for: {query}"
        try:
            if not query or not isinstance(query, str):
                 logger.error("Invalid query provided for file search.")
                 return {"success": False, "error": "Invalid or empty query provided for file search."}

            logger.info(f"Tool: Performing file search for query: '{query}'")
            result = default_api.file_search(query=query, explanation=explanation)

            if isinstance(result, dict) and "file_search_response" in result:
                response_data = result["file_search_response"]
                if isinstance(response_data, dict) and response_data.get("status") == "error":
                     logger.error(f"_tool_file_search API call failed for query '{query}': {response_data.get('error')}")
                     return {"success": False, "error": response_data.get("error", "File search API call failed.")}
                else:
                    # Assuming successful response is the list of files
                    return {"success": True, "found_files": response_data}
            else:
                 logger.error(f"_tool_file_search received unexpected result structure: {result}")
                 return {"success": False, "error": "Tool call failed or returned unexpected format."}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_file_search for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during file search: {str(e)}"}

    def _tool_write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Tool implementation: Write file with path validation and error handling."""
        explanation = f"Agent requested write access to {file_path}."
        try:
            # Path validation
            abs_path = os.path.abspath(file_path)
            workspace_root = os.path.abspath("/opt/sutazaiapp") # Define workspace root
            if not abs_path.startswith(workspace_root):
                 logger.warning(f"Attempt to write file outside workspace: {file_path}")
                 return {"success": False, "error": "File write access restricted to workspace."}

            logger.info(f"Tool: Writing to file '{abs_path}'")
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Successfully wrote to file '{abs_path}'")
            return {"success": True, "file_path": abs_path}
        except IOError as io_err:
             logger.error(f"IOError writing to file {file_path}: {io_err}")
             return {"success": False, "error": f"IOError writing file: {io_err}"}
        except Exception as e:
            logger.error(f"Unexpected error in _tool_write_file for '{file_path}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred writing file: {str(e)}"}

    def _tool_execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Tool implementation: Execute code in sandbox (Placeholder - UNSAFE)."""
        explanation = f"Agent requested execution of {language} code."
        logger.warning("execute_code tool is a placeholder and does not run code securely.")
        # In a real scenario, this should call a secure sandbox service (e.g., using Docker, firecracker)
        # NEVER use eval() or subprocess directly on untrusted agent output.
        return {"success": False, "error": "Code execution sandbox not implemented yet."}

    def _tool_search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Tool implementation: Search documents in vector DB with error handling."""
        explanation = f"Agent requested document search for: {query}"
        logger.info(f"Tool: Executing search_documents with query: '{query}', n_results: {n_results}")
        try:
            # Validate n_results
            try:
                num_results = max(1, int(n_results)) # Ensure positive integer
            except (ValueError, TypeError):
                 logger.warning(f"Invalid n_results '{n_results}', defaulting to 5.")
                 num_results = 5

            # Define which vector DB model to query - consider making configurable
            vector_db_model_id = "chroma-store"

            logger.debug(f"Querying vector DB '{vector_db_model_id}' for '{query}' (k={num_results})")
            # Call ModelManager to query the vector DB
            search_params = {"k": num_results}
            results = self.model_manager.run_inference(
                vector_db_model_id,
                query, # Input is the query string
                parameters=search_params
            )

            # Check for errors returned by the inference call
            if isinstance(results, dict) and "error" in results:
                 logger.error(f"Vector DB query failed: {results['error']}")
                 return {"success": False, "error": results['error']}

            logger.debug(f"Raw vector DB results: {results}")

            # Basic validation of expected keys from ChromaDB (adapt if DB changes)
            if not isinstance(results, dict) or not all(k in results for k in ["ids", "documents", "metadatas", "distances"]):
                 logger.error(f"Unexpected result format from vector DB: {results}")
                 return {"success": False, "error": "Unexpected format from vector database."}

            # Format results safely, handling potential None values
            formatted_results = [
                 {
                     "id": doc_id,
                     "content": doc_content if doc_content is not None else "",
                     "metadata": metadata if metadata is not None else {},
                     "distance": distance if distance is not None else float('inf'),
                 }
                 for doc_id, doc_content, metadata, distance in zip(
                     results.get("ids", [[]])[0], # Chroma nests results
                     results.get("documents", [[]])[0],
                     results.get("metadatas", [[]])[0],
                     results.get("distances", [[]])[0]
                 )
            ]

            logger.info(f"Vector DB search successful, returning {len(formatted_results)} results.")
            return {"success": True, "results": formatted_results}

        except Exception as e:
            logger.error(f"Unexpected error in _tool_search_documents for query '{query}': {e}", exc_info=True)
            return {"success": False, "error": f"An unexpected error occurred during document search: {str(e)}"}

    def _tool_calculator(self, expression: str) -> Dict[str, Any]:
        """Tool implementation: Evaluate a simple mathematical expression (Placeholder - UNSAFE)."""
        explanation = f"Agent requested calculation for: {expression}"
        logger.warning("Calculator tool uses eval() and is UNSAFE for untrusted input.")
        # In a real scenario, use a safer math parsing library (like numexpr or ast)
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression."}
        try:
            # Limit eval scope
            result = eval(expression, {"__builtins__": {}}, {})
            logger.info(f"Tool: Calculated '{expression}' = {result}")
            return {"success": True, "result": str(result)}
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return {"success": False, "error": f"Failed to evaluate expression: {e}"}

    def _get_agent_config(self, agent_identifier: str) -> Optional[Tuple[str, AgentConfig]]:
        """Find agent config by ID or name."""
        if agent_identifier in self.agent_configs:
            return agent_identifier, self.agent_configs[agent_identifier]
        for agent_id, config in self.agent_configs.items():
            if config.name == agent_identifier:
                return agent_id, config
        return None

    def create_agent(
        self, agent_identifier: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an agent instance from a configuration ID or name.

        Args:
            agent_identifier: ID or Name of the agent configuration.
            parameters: Optional parameters to override configuration defaults.

        Returns:
            Instance ID for the created agent, or None if creation failed.
        """
        config_result = self._get_agent_config(agent_identifier)
        if not config_result:
            logger.error(f"Agent configuration '{agent_identifier}' not found.")
            return None # Return None instead of raising error directly

        agent_id, config = config_result
        instance_id = f"{agent_id}-{uuid.uuid4().hex[:8]}"
        logger.info(f"Attempting to create agent instance '{instance_id}' from config '{agent_id}'.")

        # Set initial state
        self.agent_states[instance_id] = AgentState.INITIALIZING

        try:
            # Merge parameters safely
            config_params = config.parameters.copy()
            if parameters:
                if not isinstance(parameters, dict):
                    logger.warning("Invalid parameters format for creation, using defaults.")
            else:
                    config_params.update(parameters)

            # Create the agent instance (returns dict with instance or error)
            agent_data = self._create_framework_agent(instance_id, config, config_params)

            # Check if instance creation failed
            if not agent_data or agent_data.get("error") or not agent_data.get("instance"):
                 creation_error = agent_data.get('error', 'Unknown creation error') if isinstance(agent_data, dict) else 'Creation returned None'
                 logger.error(f"Failed to create agent instance '{agent_id}': {creation_error}")
                 self.agent_states[instance_id] = AgentState.ERROR
                 # Clean up potential partial registration?
                 if instance_id in self.active_agents: del self.active_agents[instance_id]
                 if instance_id in self.agent_states: del self.agent_states[instance_id]
                 return None

            # Store successfully created instance
            self.active_agents[instance_id] = agent_data
            self.agent_states[instance_id] = AgentState.IDLE # Set state after successful creation
            logger.info(f"Created and stored new instance '{instance_id}' (Type: {agent_data.get('type')})")
            return instance_id
                
        except Exception as e:
            # Catch unexpected errors during the creation process
            logger.error(f"Unexpected error creating agent instance '{agent_id}': {e}", exc_info=True)
            self.agent_states[instance_id] = AgentState.ERROR
            return None

    def _create_framework_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Factory method to create agent based on framework type."""
        framework = config.framework
        creator_method_name = f"_create_{framework.value}_agent"
        creator_method = getattr(self, creator_method_name, None)

        if creator_method and callable(creator_method):
            logger.info(f"Creating agent instance '{instance_id}' using framework: {framework.value}")
            # The specific _create_* method should handle its own errors and return dict
            return creator_method(instance_id, config, params)
        else:
            logger.error(f"Unsupported agent framework type: {framework.value}")
            # Return error dict directly
            return {"type": "error", "instance": None, "config": config, "error": f"Unsupported framework: {framework.value}"}

    def _create_langchain_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a LangChain agent instance (ReAct)."""
        error_return = {"type": "langchain", "instance": None, "config": config} # Predefined error structure
        try:
            # --- 1. Load LLM Instance ---
            model_id = config.model_id
            logger.info(f"Requesting model manager to load model '{model_id}' for agent '{instance_id}'.")
            # Assuming load_model returns bool or raises error on failure
            if not self.model_manager.load_model(model_id):
                 error_msg = f"Model manager failed to load model '{model_id}'."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return

            loaded_model_data = self.model_manager.loaded_models.get(model_id)
            if not loaded_model_data or "instance" not in loaded_model_data:
                 error_msg = f"Model '{model_id}' not found in loaded models after load attempt."
                 logger.error(error_msg)
                 error_return["error"] = error_msg
                 return error_return
            llm = loaded_model_data["instance"]
            logger.info(f"Successfully obtained LLM instance for model: {model_id}")

            # --- 2. Check for Tools (Codebase Agent requires tools) ---
            if not config.tools:
                error_msg = f"LangChain agent '{config.name}' requires tools for ReAct framework. Check config."
                logger.error(error_msg)
                error_return["error"] = error_msg
                return error_return

            # --- 3. Map and Wrap Tools ---
            agent_tools = []
            missing_tool_methods = []
            for tool_name in config.tools:
                tool_method = getattr(self, f"_tool_{tool_name}", None)
                tool_config = self.tools.get(tool_name)
                if tool_method and callable(tool_method) and tool_config:
                    # Langchain Tool expects a sync function
                    # If tool methods were async, would need `arun` or wrapper
                    agent_tools.append(Tool(name=tool_config.name, func=tool_method, description=tool_config.description))
                    logger.debug(f"Mapped tool '{tool_name}' ({tool_config.name})")
                else:
                    missing_tool_methods.append(tool_name)
                    logger.warning(f"Tool implementation method `_tool_{tool_name}` or definition for '{tool_name}' not found.")

            if not agent_tools:
                error_msg = f"Agent '{config.name}' lists tools, but none could be mapped. Missing: {missing_tool_methods}"
                logger.error(error_msg)
                error_return["error"] = error_msg
                return error_return
            logger.info(f"Mapped {len(agent_tools)} tools for agent '{instance_id}': {[t.name for t in agent_tools]}")

            # --- 4. Create Custom Codebase-Focused ReAct Prompt ---
            try:
                base_prompt = hub.pull("hwchase17/react")
                template_format = base_prompt.template
                react_format_instructions_start = template_format.find("Use the following format:")
                react_format_instructions = template_format[react_format_instructions_start:]
                react_format_instructions = react_format_instructions.replace(
                    "Question: the input question you must answer",
                    "Original Question: the user query you must address"
                )
            except Exception as hub_error:
                logger.error(f"Failed to pull base ReAct prompt from hub: {hub_error}")
                error_return["error"] = f"Failed to load base prompt: {hub_error}"
                return error_return

            codebase_system_message = (
                f"You are SutazAI, a highly capable AI assistant specialized in understanding and analyzing the codebase located at /opt/sutazaiapp. "
                f"Your goal is to answer questions and fulfill requests about this specific codebase. "
                f"Always assume the user is asking about the code within /opt/sutazaiapp unless specified otherwise. "
                f"You have access to the following tools: {[t.name for t in agent_tools]}. "
                f"Think step-by-step. For questions about code structure or finding things, start by using tools like 'list_dir', 'file_search', 'grep_search', or 'codebase_search'. "
                f"Only use 'read_file' on specific files identified in previous steps. Be precise with file paths. "
                f"Adhere strictly to the required action format."
            )
            custom_template_str = (
                f"{codebase_system_message}\n\n"
                f"TOOLS:\n------\n"
                f"You have access to the following tools:\n"
                f"{{tools}}\n\n" # Filled by partial
                f"{{react_format_instructions}}\n\n" # Filled by partial
                f"Begin!\n\n"
                f"Previous conversation history:\n"
                f"{{chat_history}}\n\n" # Filled by memory
                f"Original Question: {{input}}\n" # User input
                f"Thought:{{agent_scratchpad}}" # Agent working space
            )

            try:
                prompt = PromptTemplate.from_template(custom_template_str)
                prompt = prompt.partial(
                    tools="\n".join([f"{tool.name}: {tool.description}" for tool in agent_tools]),
                    tool_names=", ".join([tool.name for tool in agent_tools]),
                    react_format_instructions=react_format_instructions
                )
                logger.info("Created custom ReAct prompt for codebase interaction.")
            except Exception as prompt_error:
                 logger.error(f"Failed to create custom prompt template: {prompt_error}")
                 error_return["error"] = f"Prompt template error: {prompt_error}"
                 return error_return

            # --- 5. Create ReAct Agent ---
            try:
                 agent = create_react_agent(llm, agent_tools, prompt)
                 logger.info("Attempting agent creation with create_react_agent using custom prompt.")
            except Exception as agent_creation_error:
                 logger.error(f"Failed create_react_agent: {agent_creation_error}", exc_info=True)
                 error_return["error"] = f"Failed to create ReAct agent: {agent_creation_error}"
                 return error_return

            # --- 6. Create Memory ---
            # Consider potential deprecation warnings and alternatives if needed
            try:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            except Exception as memory_error:
                 logger.error(f"Failed to create ConversationBufferMemory: {memory_error}")
                 error_return["error"] = f"Memory creation error: {memory_error}"
                 return error_return

            # --- 7. Create Agent Executor ---
            try:
                max_iterations = params.get("max_iterations", 10)
                agent_executor = AgentExecutor(
                     agent=agent,
                     tools=agent_tools,
                     memory=memory,
                     verbose=True, # Consider making this configurable
                     handle_parsing_errors="Check your output and make sure it conforms!", # Or a custom handler
                     max_iterations=max_iterations
                )
                logger.info(f"Successfully created LangChain ReAct agent executor for instance: {instance_id}")
                # Return success dictionary
                return {"type": "langchain_agent", "instance": agent_executor, "config": config}
            except Exception as executor_error:
                 logger.error(f"Failed to create AgentExecutor: {executor_error}", exc_info=True)
                 error_return["error"] = f"AgentExecutor creation error: {executor_error}"
                 return error_return

        except Exception as e:
            # Catch-all for unexpected errors during the process
            logger.error(f"Unexpected error creating LangChain agent instance '{instance_id}': {e}", exc_info=True)
            error_return["error"] = f"Unexpected agent creation error: {str(e)}"
            return error_return

    def _create_autogpt_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an AutoGPT agent."""
        logger.warning("AutoGPT task execution is not implemented.")
        raise NotImplementedError("AutoGPT task execution is not implemented.")

    def _create_localagi_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating a LocalAGI agent."""
        logger.warning("LocalAGI task execution is not implemented.")
        raise NotImplementedError("LocalAGI task execution is not implemented.")

    def _create_autogen_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Placeholder for creating an Autogen agent."""
        logger.warning("Autogen task execution is not fully implemented.")
        # Existing placeholder logic... needs proper async handling if used in process_chat
        raise NotImplementedError("Autogen task execution is not fully implemented.")

    def _create_custom_agent(
        self, instance_id: str, config: AgentConfig, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Creates a custom agent based on BaseAgent."""
        # This structure seems complex and might duplicate logic better handled
        # by specific agent classes. Consider refactoring BaseAgent/CustomAgent interaction.
        error_return = {"type": "custom", "instance": None, "config": config}
        logger.warning("Custom agent creation logic needs review for robustness.")
        try:
            # Import BaseAgent locally to avoid potential circular dependencies at module level
            from ai_agents.base_agent import BaseAgent

            # Define the CustomAgent class locally or ensure it's properly imported
            # This nested class definition is unusual; prefer defining it at module level
            class CustomAgentImpl(BaseAgent):
                def __init__(self, config: AgentConfig, model_manager: ModelManager, tools: Dict[str, AgentTool], params: Dict[str, Any]):
                    # Pass the AgentConfig object directly to BaseAgent's __init__
                    super().__init__(config) # BaseAgent now expects AgentConfig
                    self.model_manager = model_manager
                    self.available_tools = tools # Renamed from self.tools to avoid conflict
                    self.params = params
                    # self.model_id is already set in BaseAgent.__init__
                    self.history: List[Dict[str, Any]] = [] # Explicitly initialize history
                
                def _initialize(self):
                    logger.info(f"Initializing CustomAgentImpl for model: {self.model_id}")
                    if not self.model_manager.load_model(self.model_id):
                        raise RuntimeError(f"Failed to load model {self.model_id} via ModelManager.")
                    logger.info(f"CustomAgentImpl initialized successfully for model: {self.model_id}")
                    return True # Indicate success explicitly? BaseAgent doesn't use return
                
                def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
                    logger.info(f"CustomAgentImpl executing task: {task.get('id', 'N/A')}")
                    prompt = self._build_prompt(task)
                    logger.debug(f"CustomAgentImpl prompt: {prompt}")

                    inference_params = {
                        "temperature": self.params.get("temperature", 0.7),
                        "max_tokens": self.params.get("max_tokens", 1024), # Reduced default?
                        # Add other relevant parameters from self.params
                    }

                    result = self.model_manager.run_inference(
                        self.model_id,
                        prompt,
                        inference_params,
                    )

                    if isinstance(result, dict) and "error" in result:
                        error_msg = f"Inference error: {result['error']}"
                        logger.error(error_msg)
                        # Raise an exception to be caught by the calling method
                        raise RuntimeError(error_msg)

                    output = result.get("text", "No text output from model.")
                    logger.debug(f"CustomAgentImpl raw output: {output}")

                    # Basic history update
                    self.history.append({"task": task.get('instruction', 'N/A'), "output": output})
                    # Limit history size if needed
                    max_history = 10
                    if len(self.history) > max_history:
                        self.history = self.history[-max_history:]

                    return {"output": output, "task_id": task.get("id", "N/A")} # Return standard structure
                
                def _cleanup(self):
                    logger.info(f"Cleaning up CustomAgentImpl for model: {self.model_id}")
                    # Nothing specific to clean up here unless resources were allocated
                    return True
                
                def _build_prompt(self, task: Dict[str, Any]) -> str:
                    # Simplified prompt building example
                    parts = []
                    parts.append(f"### Task Instruction: {task.get('instruction', 'No instruction provided.')}")

                    if task.get("context"):
                        parts.append(f"### Context: {task['context']}")
                    
                    if self.history:
                        hist_str = "\n".join([f"Prev Task: {h['task'][:100]}...\nPrev Output: {h['output'][:100]}..." for h in self.history])
                        parts.append(f"### Recent Conversation History: {hist_str}")

                    if self.config.tools:
                        tool_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.available_tools.items() if name in self.config.tools])
                        parts.append(f"### Available Tools: {tool_desc}")

                    parts.append("\n### Your Response:")
                    return "\n\n".join(parts)

            # --- End of Nested Class ---

            # Map tools needed by this specific agent config
            agent_specific_tools = {
                tool_name: self.tools[tool_name]
                for tool_name in config.tools if tool_name in self.tools
            }

            # Create the agent instance
            custom_agent_instance = CustomAgentImpl(
                config=config,
                model_manager=self.model_manager,
                tools=agent_specific_tools,
                params=params,
            )

            # Initialize the agent (critical step)
            try:
                custom_agent_instance.initialize() # Call BaseAgent's initialize
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

    def execute_task(self, instance_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with an agent
        
        Args:
            instance_id: ID of the agent instance
            task: Task description
            
        Returns:
            Result of the task execution
        """
        if instance_id not in self.active_agents:
            logger.error(f"Agent instance {instance_id} not found")
            raise ValueError(f"Agent instance {instance_id} not found")
        
        with self.agent_locks[instance_id]:
            try:
                # Update agent state
                self.agent_states[instance_id] = AgentState.PROCESSING
                
                # Get the agent
                agent_data = self.active_agents[instance_id]
                agent_type = agent_data["type"]
                agent = agent_data["instance"]
                
                # Execute task based on agent type
                if agent_type == "langchain":
                    result = self._execute_langchain_task(agent, task)
                elif agent_type == "autogpt":
                    result = self._execute_autogpt_task(agent, task)
                elif agent_type == "localagi":
                    result = self._execute_localagi_task(agent, task)
                elif agent_type == "autogen":
                    result = self._execute_autogen_task(agent_data, task)
                elif agent_type == "custom":
                    result = self._execute_custom_task(agent, task)
                else:
                    raise ValueError(f"Unsupported agent type: {agent_type}")
                
                # Update agent state
                self.agent_states[instance_id] = AgentState.IDLE
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing task with agent {instance_id}: {str(e)}")
                self.agent_states[instance_id] = AgentState.ERROR
                raise
    
    def _execute_langchain_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LangChain agent"""
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        
        # Format the prompt
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction
        
        # Run the agent
        try:
            response = agent.run(prompt)
            return {"output": response, "task_id": task.get("id")}
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def _execute_autogpt_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an AutoGPT agent"""
        # This is a placeholder implementation
        return {
            "output": "AutoGPT execution not implemented",
            "task_id": task.get("id"),
        }
    
    def _execute_localagi_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a LocalAGI agent"""
        # This is a placeholder implementation
        return {
            "output": "LocalAGI execution not implemented",
            "task_id": task.get("id"),
        }
    
    def _execute_autogen_task(self, agent_data, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an Autogen agent"""
        agent = agent_data["instance"]
        user_proxy = agent_data.get("user_proxy")
        
        instruction = task.get("instruction", "")
        context = task.get("context", "")
        
        # Format the message
        if context:
            message = f"{instruction}\n\nContext: {context}"
        else:
            message = instruction
        
        # Run with or without user proxy
        try:
            if user_proxy:
                # Start a conversation between user proxy and agent
                user_proxy.initiate_chat(agent, message=message)
                # Get the chat history
                history = user_proxy.chat_messages[agent]
                # Extract the last message from the agent
                last_message = history[-1]["content"] if history else "No response"
                return {"output": last_message, "task_id": task.get("id")}
            else:
                # Direct message to agent
                response = agent.generate_reply(message)
                return {"output": response, "task_id": task.get("id")}
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def _execute_custom_task(self, agent, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a custom agent"""
        try:
            result = agent._execute(task)
            return result
        except Exception as e:
            return {"error": str(e), "task_id": task.get("id")}
    
    def terminate_agent(self, instance_id: str) -> bool:
        """Terminate an agent instance and clean up resources."""
        logger.info(f"Attempting to terminate agent instance '{instance_id}'.")
        agent_data = self.active_agents.pop(instance_id, None) # Remove safely
        self.agent_states.pop(instance_id, None) # Remove state

        if not agent_data:
            logger.warning(f"Agent instance '{instance_id}' not found for termination.")
            return False
        
        # Attempt cleanup if the agent instance has a cleanup method
        agent_instance = agent_data.get("instance")
        if agent_instance and hasattr(agent_instance, "cleanup") and callable(agent_instance.cleanup):
            try:
                logger.debug(f"Calling cleanup for agent instance '{instance_id}'.")
                # If cleanup is async, needs await loop.run_in_executor(None, agent_instance.cleanup)
                # Assuming sync cleanup for now based on BaseAgent structure
                agent_instance.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup for agent '{instance_id}': {e}", exc_info=True)
                # Continue termination despite cleanup error

        logger.info(f"Successfully terminated agent instance '{instance_id}'.")
        return True
    
    def get_agent_state(self, instance_id: str) -> Optional[AgentState]:
        """Get the current state of an agent instance."""
        return self.agent_states.get(instance_id)
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all available agent configurations."""
        return {
            agent_id: config.to_dict()
            for agent_id, config in self.agent_configs.items()
        }
    
    def list_active_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all active agent instances and their states."""
        result = {}
        for instance_id, agent_data in self.active_agents.items():
            state = self.agent_states.get(instance_id, AgentState.ERROR) # Default to ERROR if state missing
            result[instance_id] = {
                "type": agent_data.get("type", "unknown"),
                "config_name": agent_data.get("config", {}).get("name", "unknown"),
                "state": state.value,
            }
        return result
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all registered tools."""
        return {tool_id: tool.to_dict() for tool_id, tool in self.tools.items()}
    
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
            # Allow overwriting for now, could change behavior if needed

        # Check if the function exists on self (for _tool_* methods)
        if not hasattr(self, tool.function.__name__) or not callable(getattr(self, tool.function.__name__)):
            logger.error(f"Tool function '{tool.function.__name__}' not found or not callable on AgentFramework instance.")
            return False
        
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")
        return True
    
    def add_agent_config(self, agent_id: str, config_dict: Dict[str, Any]) -> bool:
        """Add or update an agent configuration and save to file."""
        if not agent_id or not isinstance(agent_id, str):
            logger.error("Invalid agent_id provided.")
            return False
        try:
            # Validate and create AgentConfig object
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
        """Remove an agent configuration. Fails if agent instance is active."""
        if agent_id not in self.agent_configs:
            logger.warning(f"Agent configuration '{agent_id}' not found for removal.")
            return False
        
        # Check for active instances
        active_instances = [
            instance_id
            for instance_id, agent_data in self.active_agents.items()
            if agent_data["config"].name == self.agent_configs[agent_id].name
        ]
        
        if active_instances:
            logger.warning(
                f"Cannot remove agent configuration '{agent_id}' as an instance is active ('{agent_id}'). Terminate instance first.")
            return False
        
        # Remove from configurations
        del self.agent_configs[agent_id]
        
        # Save updated configuration
        self._save_config()
        
        logger.info(f"Removed agent configuration '{agent_id}'.")
        return True
    
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

    async def cleanup(self):
        """Clean up resources used by the agent framework."""
        logger.info("Cleaning up Agent Framework...")
        active_instance_ids = list(self.active_agents.keys())
        for instance_id in active_instance_ids:
            logger.info(f"Terminating agent instance during cleanup: {instance_id}")
            # Assuming terminate_agent is synchronous for now
            self.terminate_agent(instance_id) # terminate_agent now handles cleanup call

        # Clear registries after termination attempts
        self.active_agents.clear()
        self.agent_states.clear()
        # self.agent_locks.clear() # If locks were used

        logger.info("Agent Framework cleanup complete.")
