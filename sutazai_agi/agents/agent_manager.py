import logging
from typing import Dict, Any, Optional, List, Type

from sutazai_agi.core.config_loader import load_agent_config, get_setting
from sutazai_agi.agents.tool_library import ToolLibrary, _TOOL_IMPLEMENTATIONS, _not_implemented_tool
from sutazai_agi.memory.vector_store import VectorStoreInterface
from sutazai_agi.models.llm_interface import LLMInterface
from sutazai_agi.core.ethical_verifier import EthicalVerifier # Import new verifier

# Import agent execution functions
from sutazai_agi.agents.impl.langchain_agent import execute_langchain_task
from sutazai_agi.agents.impl.autogen_agent import execute_autogen_task
from sutazai_agi.agents.impl.localagi_agent import execute_localagi_task
from sutazai_agi.agents.integrations.autogpt import run_autogpt
from sutazai_agi.agents.integrations.gpt_engineer import run_gpt_engineer
from sutazai_agi.agents.integrations.agentzero import run_agentzero # New
from sutazai_agi.agents.integrations.skyvern import run_skyvern # New
from sutazai_agi.agents.integrations.aider import run_aider # New
# Add imports for other agent types if they have dedicated execution functions

logger = logging.getLogger(__name__)

# Mapping from agent type (in config) to the function that executes it
_AGENT_EXECUTION_MAP = {
    "langchain": execute_langchain_task,
    "autogen": execute_autogen_task,
    "localagi": execute_localagi_task,
    "autogpt": run_autogpt, # Now mapped
    "gpt-engineer": run_gpt_engineer, # Now mapped
    "agentzero": run_agentzero, # Now mapped
    "skyvern": run_skyvern, # Now mapped
    "aider": run_aider, # Now mapped
    # Add other agent types here
}

class AgentManager:
    """Manages the lifecycle and execution of different AI agents."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        vector_store: VectorStoreInterface,
        tool_library: ToolLibrary,
        ethical_verifier: EthicalVerifier # Inject verifier
    ):
        """Initializes the AgentManager.

        Args:
            llm_interface: Interface for interacting with LLMs.
            vector_store: Interface for interacting with the vector store.
            tool_library: Library containing available tools.
            ethical_verifier: Instance for performing safety checks.
        """
        self.llm_interface = llm_interface
        self.vector_store = vector_store
        self.tool_library = tool_library
        self.ethical_verifier = ethical_verifier # Store verifier
        self.agents_config = load_agent_config()
        logger.info(f"AgentManager initialized with {len(self.agents_config)} agent configurations.")

    def list_agents(self) -> List[Dict[str, Any]]:
        """Lists available agent configurations."""
        return list(self.agents_config.values())

    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Gets the configuration for a specific agent."""
        return self.agents_config.get(agent_name)

    async def execute_task(
        self,
        agent_name: str,
        user_input: str,
        session_id: Optional[str] = None # Optional session ID for context/memory
    ) -> Dict[str, Any]:
        """Executes a task using the specified agent.

        Args:
            agent_name: The name of the agent to use (must match config/agents.yaml).
            user_input: The user's input or task description.
            session_id: Optional session identifier for stateful operations.

        Returns:
            A dictionary containing the execution result (status, output/message).
        """
        agent_config = self.get_agent_config(agent_name)
        if not agent_config:
            logger.error(f"Agent '{agent_name}' not found in configuration.")
            return {"status": "error", "message": f"Agent '{agent_name}' not found."}

        agent_type = agent_config.get("type")
        if not agent_type:
            logger.error(f"Agent '{agent_name}' configuration missing 'type'.")
            return {"status": "error", "message": f"Agent '{agent_name}' configuration invalid."}

        execution_func = _AGENT_EXECUTION_MAP.get(agent_type)
        if not execution_func:
            logger.error(f"Execution function not found for agent type '{agent_type}' (Agent: '{agent_name}').")
            return {"status": "error", "message": f"Unsupported agent type '{agent_type}'."}

        # Prepare tools available to this agent
        available_tool_names = agent_config.get("tools", [])
        available_tools = self.tool_library.get_tools_by_name(available_tool_names)

        logger.info(f"Executing task for agent '{agent_name}' (Type: {agent_type}) with {len(available_tools)} tools.")
        try:
            # Pass necessary components to the execution function
            # The signature of execution functions needs to be standardized or handled here
            # Common args likely include: agent_config, user_input, available_tools, llm_interface, vector_store
            if agent_type in ["langchain", "autogen"]:
                # These might need LLM/vector store directly
                result = await execution_func(
                    agent_config=agent_config,
                    user_input=user_input,
                    available_tools=available_tools,
                    llm_interface=self.llm_interface,
                    vector_store=self.vector_store
                )
            elif agent_type in ["localagi"]:
                 # LocalAGI might need tools passed differently or not at all if subprocess
                 result = await execution_func(
                    agent_config=agent_config,
                    user_input=user_input,
                    available_tools=available_tools # Pass for consistency, even if unused by subprocess impl
                 )
            else:
                 # Subprocess-based agents (AutoGPT, GPT-Engineer, AgentZero etc.)
                 # Assume they primarily need config and input, timeout can be passed via config
                 timeout = agent_config.get("timeout_seconds") # Allow overriding timeout per agent
                 result = await execution_func(
                     agent_config=agent_config,
                     user_input=user_input,
                     timeout=timeout
                     # Pass other specific args if needed (e.g., files_to_edit for Aider)
                 )

        except Exception as e:
            logger.error(f"Error executing task for agent '{agent_name}': {e}", exc_info=True)
            return {"status": "error", "message": f"Agent execution failed: {e}"}

        # Perform ethical check on the final output/result
        output_content = result.get("output")
        if isinstance(output_content, str):
             is_safe = await self.ethical_verifier.check(content=output_content, agent_name=agent_name)
             if not is_safe:
                  logger.warning(f"Ethical verifier blocked output from agent '{agent_name}'.")
                  # Decide how to handle blocked output (return generic message, etc.)
                  return {"status": "error",
                          "message": "Output blocked by safety policy.",
                          "original_result": result # Optionally include original for admin review
                         }

        logger.info(f"Task execution completed for agent '{agent_name}'. Status: {result.get('status')}")
        return result