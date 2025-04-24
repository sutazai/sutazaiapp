import logging
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Union
import asyncio

# LangChain imports (ensure these are installed)
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder
from langchain.tools import Tool, tool # Allow using @tool decorator if needed later
# Import standard file management and shell tools
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain_community.tools import ShellTool # Use ShellTool for command execution
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig # Import RunnableConfig
from langchain_core.exceptions import OutputParserException # For handling parsing errors
from langchain.tools.render import render_text_description # <-- Import render_text_description
from langchain_core.messages import BaseMessage # <-- Import BaseMessage

from sutazai_agi.models.llm_interface import LLMInterface
from sutazai_agi.memory.vector_store import VectorStoreInterface
from sutazai_agi.core.ethical_verifier import EthicalVerifier
from sutazai_agi.core.config_loader import get_setting, get_tool_config

logger = logging.getLogger(__name__)

# Add a custom memory class with async support for newer LangChain versions
class AsyncCompatibleMemory(ConversationBufferWindowMemory):
    """Memory class that works with both sync and async LangChain code."""
    
    async def aget_messages(self):
        """Get the messages stored in the memory asynchronously."""
        return self.chat_memory.messages
    
    async def asave_context(self, inputs, outputs):
        """Save context from this conversation to buffer asynchronously."""
        # The base class save_context is synchronous, so we run it in default executor
        # This might not be truly non-blocking if base save_context is heavy IO
        # A truly async memory store (like Redis) would be better for production
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.save_context, inputs, outputs)
        # return self.save_context(inputs, outputs) # Previous potentially blocking way
    
    async def aclear(self):
        """Clear memory contents asynchronously."""
        # Similar to asave_context, run sync clear in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.clear)
        # return self.clear() # Previous potentially blocking way
        
    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        """Add messages to the memory asynchronously."""
        # Run sync add_messages in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.chat_memory.add_messages, messages)

# In-memory store for conversation histories (replace with persistent store like Redis/DB for production)
_chat_memory_store = {}

def get_session_history(session_id: str) -> AsyncCompatibleMemory:
    """Returns the conversation memory for a specific session, creating it if it doesn't exist."""
    if session_id not in _chat_memory_store:
        # Create memory with k=5 history window
        _chat_memory_store[session_id] = AsyncCompatibleMemory(
            k=5, 
            memory_key="chat_history", 
            input_key="input", # Explicitly define input key
            output_key="output", # Explicitly define output key (even if not directly used by final answer)
            return_messages=True
        )
    return _chat_memory_store[session_id]

def _wrap_tool_for_langchain(tool_func: Callable, tool_name: str, verifier: EthicalVerifier, agent_name: str) -> Tool:
    """Wraps a standard Python function tool into a Langchain Tool, adding ethical verification."""
    # Fetch the description from the tool's config or use the docstring as fallback
    tool_config = get_tool_config(tool_name)
    description = tool_config.get("description") if tool_config else None
    if not description:
        description = getattr(tool_func, '__doc__', f"Tool {tool_name}: No description provided.")

    # Define the function that LangChain AgentExecutor will call
    async def verified_tool_func_async(*args, **kwargs):
        # Perform ethical check before execution
        tool_config = get_tool_config(tool_name)
        parameter_names = list(tool_config.get("parameters", {}).keys()) if tool_config else []
        parameters = kwargs if kwargs else (dict(zip(parameter_names, args)) if args and parameter_names else {})
        if not parameters and args and len(parameter_names) == 1:
            parameters[parameter_names[0]] = args[0]

        logger.debug(f"Langchain Tool '{tool_name}' invoked async with args: {args}, kwargs: {kwargs}. Parsed params: {parameters}")
        # Use the synchronous check_action method, as acheck_action doesn't exist
        if verifier.check_action(agent_name=agent_name, tool_name=tool_name, parameters=parameters):
            try:
                # Assume the underlying tool_func might be sync or async
                # LangChain typically handles awaiting if tool_func is async
                result = tool_func(*args, **kwargs)
                # If tool_func itself is an awaitable, LangChain should handle it.
                # If not, we might need to run it in an executor if it's blocking sync code.
                # For now, assume LangChain handles it or tools are quick.
                
                # Await the result ONLY if it's actually awaitable (a coroutine)
                if asyncio.iscoroutine(result):
                    result = await result
                    
                logger.debug(f"Tool '{tool_name}' executed successfully. Result: {str(result)[:100]}...")
                return result
            except Exception as e:
                logger.error(f"Error during async execution of tool '{tool_name}': {e}", exc_info=True)
                return f"Error: Tool '{tool_name}' failed during execution: {e}"
        else:
            logger.warning(f"Action blocked by ethical verifier for tool '{tool_name}'")
            return f"Error: Action blocked by ethical policy for tool '{tool_name}'."

    # For non-async LangChain tool usage (might still be needed? Check AgentExecutor)
    def verified_tool_func_sync(*args, **kwargs):
        tool_config = get_tool_config(tool_name)
        parameter_names = list(tool_config.get("parameters", {}).keys()) if tool_config else []
        parameters = kwargs if kwargs else (dict(zip(parameter_names, args)) if args and parameter_names else {})
        if not parameters and args and len(parameter_names) == 1:
            parameters[parameter_names[0]] = args[0]

        logger.debug(f"Langchain Tool '{tool_name}' invoked sync with args: {args}, kwargs: {kwargs}. Parsed params: {parameters}")
        # Note: Calling async verifier from sync code is tricky. 
        # This sync wrapper might need careful handling or removal if everything is async.
        # For now, assume it's not used or verifier has a sync check method.
        if verifier.check_action(agent_name=agent_name, tool_name=tool_name, parameters=parameters): # Assuming sync check exists
            try:
                result = tool_func(*args, **kwargs)
                logger.debug(f"Tool '{tool_name}' executed successfully. Result: {str(result)[:100]}...")
                return result
            except Exception as e:
                logger.error(f"Error during sync execution of tool '{tool_name}': {e}", exc_info=True)
                return f"Error: Tool '{tool_name}' failed during execution: {e}"
        else:
            logger.warning(f"Action blocked by ethical verifier for tool '{tool_name}'")
            return f"Error: Action blocked by ethical policy for tool '{tool_name}'."

    # Prefer the async version for LangChain's async operations
    return Tool(
        name=tool_name,
        func=verified_tool_func_sync, # Main function (check if AgentExecutor uses this or coroutine)
        coroutine=verified_tool_func_async, # Async version for ainvoke/astream
        description=description
        # Add args_schema here if needed for more complex tools
    )

def _setup_langchain_executor(
    agent_config: Dict[str, Any],
    llm_interface: LLMInterface, # Keeping interface for potential future use
    vector_store: VectorStoreInterface, # Keeping interface for potential future use
    available_tools: Dict[str, Callable], # Custom tools loaded by AgentManager
    verifier: EthicalVerifier
) -> RunnableWithMessageHistory:
    """Sets up the LangChain agent, executor, and history runnable."""
    agent_name = agent_config.get("name", "Unknown LangChain Agent")
    configured_tool_names = agent_config.get("tools", []) # Get the original list of tool names from config
    
    # 1. Get LLM Configuration
    model_name = agent_config.get("model") or get_setting("default_llm_model")
    model_params = {**get_setting("model_parameters", {}), **agent_config.get("model_params", {})} # Merge params
    llm = ChatOllama(
        model=model_name,
        base_url=get_setting("ollama_base_url"), 
        temperature=model_params.get("temperature", 0.7),
        top_k=model_params.get("top_k", 40),
        top_p=model_params.get("top_p", 0.9),
        # Add other compatible parameters from model_params
    )
    logger.debug(f"Initialized ChatOllama for agent '{agent_name}' with model '{model_name}'")

    # 2. Prepare Tools
    langchain_tools = []
    # Add custom tools passed from AgentManager/config first
    if available_tools:
        for tool_name, tool_func in available_tools.items():
            # Check if this custom tool was actually configured for this agent
            if tool_name in configured_tool_names:
                if callable(tool_func):
                    wrapped_tool = _wrap_tool_for_langchain(tool_func, tool_name, verifier, agent_name)
                    langchain_tools.append(wrapped_tool)
                else:
                    logger.warning(f"Custom tool implementation '{tool_name}' for agent '{agent_name}' is not callable. Skipping.")
            # else: Tool loaded by library but not configured for this agent, ignore.
    
    # Conditionally add standard LangChain tools if requested in config
    standard_tools_added = []
    try:
        if "ReadFileTool" in configured_tool_names:
            langchain_tools.append(ReadFileTool())
            standard_tools_added.append("ReadFileTool")
        if "WriteFileTool" in configured_tool_names:
            # Consider security implications of WriteFileTool
            # Ensure agent_filesystem_root_dir is enforced if possible, though WriteFileTool might not respect it directly
            langchain_tools.append(WriteFileTool())
            standard_tools_added.append("WriteFileTool")
        if "ListDirectoryTool" in configured_tool_names:
            langchain_tools.append(ListDirectoryTool())
            standard_tools_added.append("ListDirectoryTool")
        if "ShellTool" in configured_tool_names:
            # VERY DANGEROUS - ensure ethical verifier is robust or disable this tool by default
            shell_tool = ShellTool(ask_human_input=False) 
            # Enhancing description might help LLM use it correctly
            shell_tool.description = shell_tool.description + f" Use this tool to execute shell commands. Input must be a valid command string. Example: ls -l" 
            langchain_tools.append(shell_tool)
            standard_tools_added.append("ShellTool")

        if standard_tools_added:
             logger.debug(f"Added standard tools based on config for agent '{agent_name}': {standard_tools_added}")

    except Exception as e:
        logger.error(f"Failed to instantiate standard tools requested by config for agent '{agent_name}': {e}", exc_info=True)

    if not langchain_tools:
        logger.warning(f"No tools available or configured for LangChain agent '{agent_name}'")
    else:
        logger.info(f"Final tools prepared for LangChain agent '{agent_name}': {[t.name for t in langchain_tools]}")

    # --- Agent Type Handling --- 
    lc_config = agent_config.get("langchain_config", {})
    # Default to 'openai-tools' if not specified, as it's generally more robust
    agent_type = lc_config.get("agent_type", "openai-tools") 
    prompt_template = None
    agent = None

    # --- Setup for openai-tools agent (Recommended) ---
    if agent_type == "openai-tools":
        # This agent type uses LLM's built-in tool calling capabilities (if available) 
        # or simulates it. It generally requires a specific prompt structure.
        try:
            # Format tool names for inclusion in the prompt
            tool_names_str = ", ".join([t.name for t in langchain_tools]) if langchain_tools else "No tools available"
            system_prompt_text = agent_config.get("system_prompt", f"You are a helpful assistant named {agent_name}. You have access to the following tools: {tool_names_str}")
            
            # Define the prompt structure for tool calling
            # Ensure 'agent_scratchpad' is handled correctly by create_tool_calling_agent
            prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_text),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"), # Crucial placeholder for intermediate steps
                ])
            
            # Create the agent using the specific constructor
            # Note: Even with ChatOllama, create_tool_calling_agent structures the interaction
            # for tool use, relying on the LLM's ability to generate tool calls in its response
            # based on the prompt and tool descriptions.
            agent = create_tool_calling_agent(llm, langchain_tools, prompt_template)
            logger.info(f"Created agent '{agent_name}' with type: {agent_type}")
        except Exception as e: # Catch errors during openai-tools setup
            logger.error(f"Failed to create '{agent_type}' agent for '{agent_name}': {e}. Falling back.", exc_info=True)
            agent_type = "react" # Fallback if setup fails
            agent = None
            prompt_template = None

    # --- Setup for structured-chat agent ---
    elif agent_type == "structured-chat":
        # This agent type uses specific prompt formatting and output parsing 
        # suitable for models fine-tuned for function/tool calling in a structured way.
        try:
            from langchain import hub # Import hub
            # Pull a standard structured chat prompt from the hub
            # This prompt is expected to have placeholders like {tools} and {tool_names}
            prompt_template = hub.pull("hwchase17/structured-chat-agent") 
            
            # --- Input variables check (for debugging/understanding) ---
            # You can inspect the required variables if needed:
            # logger.debug(f"Structured chat prompt input variables: {prompt_template.input_variables}")
            # Expected variables typically include: input, agent_scratchpad, chat_history, tools, tool_names
            
            # No need to format tool names manually here, create_structured_chat_agent handles it
            # when used with a compatible prompt from the hub.
            
            agent = create_structured_chat_agent(llm, langchain_tools, prompt_template)
            logger.info(f"Created agent '{agent_name}' with type: {agent_type}")
        except Exception as e: # Catch errors during structured-chat setup
            logger.error(f"Failed to create '{agent_type}' agent for '{agent_name}': {e}. Falling back.", exc_info=True)
            agent_type = "react" # Fallback if setup fails
            agent = None
            prompt_template = None

    # --- Fallback to ReAct agent if others fail or specified ---
    # Check if we need to fallback OR if react was explicitly chosen
    if agent is None or agent_type == "react": 
        logger.warning(f"Using ReAct agent setup for agent '{agent_name}'. Either specified, or previous setup failed.")
        agent_type = "react" # Ensure type reflects reality
        try:
            from langchain import hub
            from langchain.agents.format_scratchpad import format_log_to_str
            from langchain.agents.output_parsers import ReActSingleInputOutputParser
            from langchain.tools.render import render_text_description
            
            # Pull a standard ReAct prompt
            prompt_template = hub.pull("hwchase17/react-chat") # Ensure this prompt is suitable
            
            # Render tools for the ReAct prompt
            tools_string = render_text_description(langchain_tools)
            tool_names_string = ", ".join([t.name for t in langchain_tools])
            
            # Create the ReAct agent runnable
            # Note: Ensure all required variables ('tools', 'tool_names', 'agent_scratchpad', 'input', 'chat_history') 
            # are correctly populated and expected by the prompt.
            agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_log_to_str(x.get("intermediate_steps", [])), # Use .get for safety
                    "chat_history": lambda x: x.get("chat_history", []), # Use .get for safety
                    "tools": lambda x: tools_string,
                    "tool_names": lambda x: tool_names_string,
                }
                | prompt_template
                | llm
                | ReActSingleInputOutputParser()
            )
            logger.info(f"Created agent '{agent_name}' with type: react (fallback or specified)")
        except Exception as e:
            logger.critical(f"FATAL: Failed to create fallback ReAct agent for '{agent_name}': {e}", exc_info=True)
            # If even ReAct fails, we cannot proceed
            raise RuntimeError(f"Could not create any agent executor for {agent_name}") from e
        
    # 4. Create Agent Executor
    # Ensure agent is not None here (ReAct fallback should have been created)
    if agent is None:
         raise RuntimeError(f"Agent creation failed unexpectedly for {agent_name}")
        
    agent_executor = AgentExecutor(
        agent=agent,
        tools=langchain_tools, 
        verbose=lc_config.get("verbose", get_setting("agent_verbose_logging", False)),
        max_iterations=lc_config.get("max_iterations", 5),
        handle_parsing_errors=True # Add robust parsing error handling
    )
    logger.debug(f"Created AgentExecutor for agent '{agent_name}'")

    # 5. Setup Runnable with Message History
    agent_executor_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history, # Function to retrieve memory per session_id
        input_messages_key="input", # The key for the main input message(s)
        history_messages_key="chat_history", # The key for the memory messages
    )

    return agent_executor_with_history

async def execute_langchain_task(
    agent_config: Dict[str, Any],
    task_input: Dict[str, Any],
    llm_interface: LLMInterface,
    vector_store: VectorStoreInterface,
    available_tools: Dict[str, Callable],
    verifier: EthicalVerifier
) -> Dict[str, Any]:
    """Executes a task using the specified LangChain agent configuration (non-streaming)."""
    agent_name = agent_config.get("name", "Unknown LangChain Agent")
    session_id = task_input.get("session_id", "default_session")
    messages = task_input.get("messages", [])

    if not messages or not isinstance(messages, list):
        logger.error(f"No valid 'messages' list found in task_input for agent '{agent_name}'")
        return {"status": "error", "message": "Input requires a list of messages."}

    # Extract the latest message content for the 'input' key
    last_user_message_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message_content = msg.get("content", "")
            break
            
    if not last_user_message_content:
        logger.error(f"No user message content found in the messages list for agent '{agent_name}'")
        return {"status": "error", "message": "No user message content found."}

    logger.info(f"Executing non-streaming task for agent '{agent_name}' in session '{session_id}'")
    
    try:
        # Setup the agent executor with history
        agent_executor_with_history = _setup_langchain_executor(
            agent_config, llm_interface, vector_store, available_tools, verifier
        )
        
        # Prepare the input dictionary for the executor
        executor_input = {"input": last_user_message_content}
        
        # Define the RunnableConfig with the session_id
        config = RunnableConfig(configurable={"session_id": session_id})
        
        # Invoke the agent asynchronously
        result = await agent_executor_with_history.ainvoke(executor_input, config=config)

        # --- Log the raw result structure --- 
        logger.debug(f"Raw agent execution result for '{agent_name}': {result}")

        # --- Ethical Post-Check (Check final output) --- 
        # Extract the likely output string, handling potential dictionary structure
        if isinstance(result, dict):
            # Try common keys, default to the 'output' key if present, otherwise stringify the dict
            final_output = result.get("output") or result.get("final_response_to_human") or result.get("answer") or str(result) 
        elif isinstance(result, str):
            final_output = result
        else:
            final_output = str(result) # Fallback for other types
            
        # Ensure final_output is a string for the verifier and return value
        if not isinstance(final_output, str):
             final_output = str(final_output)
             logger.warning(f"Agent '{agent_name}' output was not a string, converted: {final_output[:100]}...")

        # Use the synchronous check_output method as acheck_output doesn't exist
        if not verifier.check_output(agent_name, final_output):
            logger.warning(f"Final output from agent '{agent_name}' blocked by ethical verifier.")
            return {"status": "error", "message": "Output blocked by ethical policy."}

        logger.info(f"Agent '{agent_name}' execution successful. Output length: {len(final_output)}")
        return {"status": "success", "output": final_output}

    except OutputParserException as ope:
        logger.error(f"Output parsing error for agent '{agent_name}': {ope}", exc_info=True)
        # Try to return the raw output if available
        raw_output = getattr(ope, 'llm_output', str(ope))
        return {"status": "error", "message": f"Agent failed to parse output: {raw_output}"}
    except Exception as e:
        logger.error(f"Error during agent execution for '{agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"Agent execution failed: {e}"}

async def stream_langchain_task(
    agent_config: Dict[str, Any],
    task_input: Dict[str, Any],
    llm_interface: LLMInterface,
    vector_store: VectorStoreInterface,
    available_tools: Dict[str, Callable],
    verifier: EthicalVerifier
) -> AsyncGenerator[Dict[str, Any], None]:
    """Executes a task using the specified LangChain agent configuration (streaming)."""
    agent_name = agent_config.get("name", "Unknown LangChain Agent")
    session_id = task_input.get("session_id", "default_session")
    messages = task_input.get("messages", [])

    if not messages or not isinstance(messages, list):
        logger.error(f"No valid 'messages' list found in task_input for streaming agent '{agent_name}'")
        yield {"type": "error", "message": "Input requires a list of messages.", "agent_name": agent_name}
        return

    last_user_message_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message_content = msg.get("content", "")
            break
            
    if not last_user_message_content:
        logger.error(f"No user message content found for streaming agent '{agent_name}'")
        yield {"type": "error", "message": "No user message content found.", "agent_name": agent_name}
        return

    logger.info(f"Initiating streaming task for agent '{agent_name}' in session '{session_id}'")

    try:
        # Setup the agent executor with history
        agent_executor_with_history = _setup_langchain_executor(
            agent_config, llm_interface, vector_store, available_tools, verifier
        )
        
        # Prepare the input dictionary for the executor
        executor_input = {"input": last_user_message_content}
        
        # Define the RunnableConfig with the session_id
        config = RunnableConfig(configurable={"session_id": session_id})
        
        # Stream the agent execution
        async for chunk in agent_executor_with_history.astream(executor_input, config=config):
            # Process different types of chunks from the stream
            # (This logic depends heavily on the agent type: ReAct, OpenAI Functions/Tools, etc.)
            # The following is a general structure, adjust keys based on actual output
            
            # Example for ReAct/Structured Chat style agents:
            if "actions" in chunk: # Agent decides to take an action
                for action in chunk["actions"]:
                    # Optional: Add ethical check before yielding tool call?
                    logger.debug(f"Agent '{agent_name}' requesting tool: {action.tool} with input: {action.tool_input}")
                    yield {"type": "tool_call", "tool_name": action.tool, "tool_input": action.tool_input, "agent_name": agent_name}
            elif "steps" in chunk: # Observation/result from a tool
                 for step in chunk["steps"]:
                     # Optional: Add ethical check on tool output before yielding?
                     logger.debug(f"Agent '{agent_name}' received tool result: {str(step.observation)[:100]}...")
                     yield {"type": "tool_end", "tool_name": "UnknownTool", "tool_output": step.observation, "agent_name": agent_name}
            elif "messages" in chunk: # Intermediate message (e.g. AI thinking or using tool)
                 # Extract content if it's a message chunk
                 for message in chunk["messages"]:
                      if hasattr(message, 'content') and message.content:
                           # Only yield if it's a content delta, not just history message
                           # This part needs careful checking based on LangChain stream output structure
                           pass # Avoid yielding intermediate AI messages unless desired
            elif "output" in chunk: # Final answer chunk
                 final_output = chunk.get("output", "")
                 if final_output:
                     # Check output ethically before yielding
                     if not await verifier.acheck_output(agent_name, final_output):
                          logger.warning(f"Streaming output from '{agent_name}' blocked by ethical verifier.")
                          yield {"type": "error", "message": "Output blocked by ethical policy.", "agent_name": agent_name}
                          # Decide if we should break or continue if stream provides more chunks
                          break 
                     else:
                          logger.debug(f"Agent '{agent_name}' yielding final content delta: {final_output[:100]}...")
                          yield {"type": "content_delta", "content": final_output, "agent_name": agent_name}
            # Add handling for other potential keys like 'intermediate_steps' if needed
            
            # If the agent uses the newer tool calling standard (like openai-tools)
            # The chunk structure might be different. Inspect the chunks for keys like
            # 'tool_calls', 'content', etc.
            # Example adaptation:
            # if "tool_calls" in chunk and chunk["tool_calls"]:
            #     for tool_call in chunk["tool_calls"]:
            #         yield {"type": "tool_call", "tool_name": tool_call['name'], "tool_input": tool_call['args'], "agent_name": agent_name}
            # elif "content" in chunk and chunk["content"]:
            #     yield {"type": "content_delta", "content": chunk["content"], "agent_name": agent_name}

        # After the loop, signal completion
        logger.info(f"Agent '{agent_name}' streaming finished.")
        yield {"type": "finish", "reason": "stop", "agent_name": agent_name} # Assume stop, reason might come from stream

    except Exception as e:
        logger.error(f"Error during agent streaming for '{agent_name}': {e}", exc_info=True)
        yield {"type": "error", "message": f"Agent streaming failed: {e}", "agent_name": agent_name} 