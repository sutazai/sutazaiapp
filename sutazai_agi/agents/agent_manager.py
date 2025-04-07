import logging
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator

from sutazai_agi.core.config_loader import get_all_agent_configs, get_agent_config
from sutazai_agi.core.ethical_verifier import get_verifier
from sutazai_agi.models.llm_interface import get_llm_interface
from sutazai_agi.memory.vector_store import get_vector_store
from .tool_library import load_tool # Assuming tools are loaded from tool_library

# Import specific agent implementations
from .impl.langchain_agent import execute_langchain_task, stream_langchain_task # Import stream version
# from .impl.autogen_agent import execute_autogen_task
# from .impl.autogpt_agent import execute_autogpt_task
# ... import others as they are implemented

logger = logging.getLogger(__name__)

class AgentManager:
    """Manages the lifecycle and execution of different AI agents."""

    def __init__(self):
        logger.info("Initializing AgentManager...")
        self.agent_configs = get_all_agent_configs()
        self.available_agents = {cfg['name']: cfg for cfg in self.agent_configs if cfg.get('enabled', False)}
        self.verifier = get_verifier()
        # Eagerly initialize LLM and Vector Store to catch connection errors early
        try:
            self.llm_interface = get_llm_interface()
            self.vector_store = get_vector_store()
            logger.info(f"AgentManager initialized with {len(self.available_agents)} enabled agents.")
        except Exception as e:
             logger.critical(f"Failed to initialize core components (LLM/VectorStore) for AgentManager: {e}", exc_info=True)
             # Depending on design, either raise here or handle missing components during execution
             self.available_agents = {} # No agents can run without core components
             raise RuntimeError("AgentManager could not initialize core dependencies.") from e

    def list_enabled_agents(self) -> List[Dict[str, Any]]:
        """Returns configuration details of enabled agents."""
        return list(self.available_agents.values())

    async def execute_task(self, agent_name: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a task using the specified agent.

        Handles non-streaming requests.
        For streaming, the caller should use specific stream_task methods if available.

        Args:
            agent_name: The name of the agent to use (must be enabled).
            task_input: A dictionary containing the input for the agent's task 
                        (e.g., {'query': 'user message'} or {'goal': 'complex objective'}).

        Returns:
            A dictionary containing the agent's result or an error message.
            Example: {'status': 'success', 'output': 'Agent response...'} or
                     {'status': 'error', 'message': 'Agent failed...'}
        """
        logger.info(f"Received task for agent '{agent_name}'. Input keys: {list(task_input.keys())}")

        if agent_name not in self.available_agents:
            logger.error(f"Agent '{agent_name}' is not available or not enabled.")
            return {"status": "error", "message": f"Agent '{agent_name}' not found or disabled."}

        agent_config = self.available_agents[agent_name]
        agent_type = agent_config.get("type")

        # Check if streaming is requested via task_input (though this method shouldn't handle it)
        is_streaming = task_input.get("stream", False)
        if is_streaming:
            # This standard execute_task is NOT for streaming.
            # Caller (API endpoint) should detect stream=True and call a dedicated streaming function.
            logger.error(f"execute_task called for agent '{agent_name}' with stream=True. This is not supported here.")
            return {"status": "error", "message": "Streaming tasks should be initiated via a dedicated streaming endpoint/method."}

        # --- Ethical Pre-Check (Optional - depends on policy) ---
        # Could add a check here based on task_input if needed, 
        # though checks usually happen per-tool call within the agent execution.

        # --- Agent Execution Logic (Non-Streaming) ---
        try:
            # --- Load Tools for the Agent --- 
            tools_dict: Dict[str, Callable] = {}
            configured_tool_names = agent_config.get("tools", [])
            if configured_tool_names:
                logger.debug(f"Loading tools for agent '{agent_name}': {configured_tool_names}")
                for tool_name in configured_tool_names:
                    tool_func = load_tool(tool_name) # Get the actual function/class based on implementation name in agents.yaml tools section
                    if tool_func:
                        # Use the tool name from the agent's config as the key
                        tools_dict[tool_name] = tool_func 
                    else:
                        logger.warning(f"Tool '{tool_name}' configured for agent '{agent_name}' not found in tool library. Skipping.")
            else:
                logger.debug(f"No tools configured for agent '{agent_name}'")

            if agent_type == "langchain":
                # Call the non-streaming, async-compatible function
                # Assuming execute_langchain_task is now properly async or handles its own loop
                result = await self._execute_langchain_non_streaming(agent_config, task_input, tools_dict)
            
            elif agent_type == "autogen":
                # TODO: Implement async execution if needed
                logger.warning(f"Execution logic for agent type '{agent_type}' is not yet implemented.")
                result = {"status": "error", "message": f"Agent type '{agent_type}' execution not implemented."}
            
            elif agent_type == "autogpt":
                # TODO: Implement async execution if needed
                logger.warning(f"Execution logic for agent type '{agent_type}' is not yet implemented.")
                result = {"status": "error", "message": f"Agent type '{agent_type}' execution not implemented."}
            
            # Add async elif blocks for other agent types as needed
            
            else:
                logger.error(f"Unknown agent type '{agent_type}' configured for agent '{agent_name}'.")
                result = {"status": "error", "message": f"Unknown agent type: {agent_type}"}

            # --- Ethical Post-Check --- 
            # Check the final output if the agent succeeded
            if result.get("status") == "success" and "output" in result:
                 final_output = result["output"]
                 # Ensure output is string for simple check, adapt if complex objects
                 if not isinstance(final_output, str):
                      try:
                           final_output_str = str(final_output)
                      except: 
                           final_output_str = "[Non-string output]"
                 else:
                      final_output_str = final_output
                 
                 if not self.verifier.check_output(agent_name, final_output_str):
                      logger.warning(f"Final output from agent '{agent_name}' blocked by ethical verifier.")
                      # Decide how to handle blocked output (return error, generic message, etc.)
                      return {"status": "error", "message": "Output blocked by ethical policy."}

            logger.info(f"Task execution finished for agent '{agent_name}'. Status: {result.get('status')}")
            return result

        except Exception as e:
            logger.error(f"An unexpected error occurred during '{agent_name}' execution: {e}", exc_info=True)
            return {"status": "error", "message": f"Internal server error during agent execution: {e}"}

    # --- Add Helper for specific agent calls to keep execute_task cleaner --- 
    async def _execute_langchain_non_streaming(self, agent_config, task_input, tools_dict):
        # This ensures we call the correct non-streaming function from langchain_agent
        return execute_langchain_task( # No await needed IF execute_langchain_task handles its async call internally
            agent_config=agent_config,
            task_input=task_input,
            llm_interface=self.llm_interface, 
            vector_store=self.vector_store, 
            available_tools=tools_dict,
            verifier=self.verifier
        )

    # --- Add a specific method for streaming LangChain tasks --- 
    async def stream_langchain_task_manager( 
        self, 
        agent_name: str, 
        task_input: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Initiates and yields results from a streaming LangChain task."""
        logger.info(f"Initiating stream for agent '{agent_name}'.")
        if agent_name not in self.available_agents:
            logger.error(f"Agent '{agent_name}' is not available or not enabled for streaming.")
            yield {"type": "error", "message": f"Agent '{agent_name}' not found or disabled."}
            return

        agent_config = self.available_agents[agent_name]
        agent_type = agent_config.get("type")

        if agent_type != "langchain":
            logger.error(f"Streaming only implemented for 'langchain' type, agent '{agent_name}' is '{agent_type}'")
            yield {"type": "error", "message": f"Streaming not supported for agent type '{agent_type}'."}
            return

        # Load tools for the agent (same logic as execute_task)
        tools_dict: Dict[str, Callable] = {}
        configured_tool_names = agent_config.get("tools", [])
        if configured_tool_names:
            for tool_name in configured_tool_names:
                tool_func = load_tool(tool_name)
                if tool_func:
                    tools_dict[tool_name] = tool_func
                else:
                    logger.warning(f"Tool '{tool_name}' for agent '{agent_name}' not found. Skipping for stream.")

        # Call the dedicated streaming function from the implementation
        try:
            async for chunk in stream_langchain_task(
                agent_config=agent_config,
                task_input=task_input,
                llm_interface=self.llm_interface, 
                vector_store=self.vector_store, 
                available_tools=tools_dict,
                verifier=self.verifier
            ):
                yield chunk # Yield the structured chunk directly
        except Exception as e:
            logger.error(f"Unexpected error during '{agent_name}' streaming setup or execution: {e}", exc_info=True)
            yield {"type": "error", "message": f"Internal server error during agent stream: {e}"}

    def get_available_agent_names(self) -> List[str]:
        """Returns a list of names of all configured and enabled agents."""
        return [agent_config['name'] for agent_config in self.agent_configs if agent_config.get('enabled', True)]

# --- Global Agent Manager Instance --- 
_agent_manager: Optional[AgentManager] = None

def get_agent_manager() -> AgentManager:
    """Returns a singleton instance of the AgentManager."""
    global _agent_manager
    if _agent_manager is None:
        try:
             _agent_manager = AgentManager()
        except RuntimeError as e:
             logger.critical(f"AgentManager initialization failed: {e}. Agents will not be available.")
             _agent_manager = None # Ensure it stays None if init fails
             raise # Re-raise critical error
        except Exception as e:
             logger.critical(f"Unexpected error initializing AgentManager: {e}", exc_info=True)
             _agent_manager = None
             raise

    if _agent_manager is None:
         raise RuntimeError("AgentManager initialization failed previously. Cannot provide manager.")
         
    return _agent_manager

# Example Usage (needs adaptation for async execute_task):
# async def main_example():
#     try:
#         manager = get_agent_manager()
#         print("Enabled Agents:", [agent['name'] for agent in manager.list_enabled_agents()])
#         
#         # Example non-streaming task
#         task = {"query": "What is ChromaDB? Summarize it briefly.", "stream": False}
#         result = await manager.execute_task("LangChain Chat Agent", task)
#         print("\nNon-Streaming Task Result:", result)

#         # Example streaming task
#         stream_task = {"query": "Tell me a short story about a robot.", "stream": True}
#         print("\nStreaming Task Result:")
#         async for chunk in manager.stream_langchain_task_manager("LangChain Chat Agent", stream_task):
#             print(chunk)
#             
#     except RuntimeError as e:
#          print(f"Error: {e}")
#     except Exception as e:
#          print(f"An unexpected error occurred: {e}")

# if __name__ == '__main__':
#     import asyncio
#     asyncio.run(main_example()) 