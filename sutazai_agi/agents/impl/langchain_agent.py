import logging
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Union

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
from langchain_core.exceptions import OutputParserException # For handling parsing errors

from sutazai_agi.models.llm_interface import LLMInterface
from sutazai_agi.memory.vector_store import VectorStoreInterface
from sutazai_agi.core.ethical_verifier import EthicalVerifier
from sutazai_agi.core.config_loader import get_setting, get_tool_config

logger = logging.getLogger(__name__)

# In-memory store for conversation histories (replace with persistent store like Redis/DB for production)
_chat_memory_store = {}

def get_session_history(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _chat_memory_store:
        # Create memory with k=5 history window
        _chat_memory_store[session_id] = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", return_messages=True
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
    def verified_tool_func(*args, **kwargs):
        # Assume args[0] is the primary input if args exist, else use kwargs
        # This might need adjustment depending on how LangChain passes args/kwargs
        tool_config = get_tool_config(tool_name)
        parameter_names = list(tool_config.get("parameters", {}).keys()) if tool_config else []
        parameters = kwargs if kwargs else (dict(zip(parameter_names, args)) if args and parameter_names else {})

        # If only one arg expected and passed, map it correctly if possible
        if not parameters and args and len(parameter_names) == 1:
            parameters[parameter_names[0]] = args[0]

        logger.debug(f"Langchain Tool '{tool_name}' invoked with args: {args}, kwargs: {kwargs}. Parsed params: {parameters}")
        
        # Perform ethical check before execution
        if verifier.check_action(agent_name=agent_name, tool_name=tool_name, parameters=parameters):
            try:
                result = tool_func(*args, **kwargs)
                logger.debug(f"Tool '{tool_name}' executed successfully. Result: {str(result)[:100]}...")
                return result
            except Exception as e:
                logger.error(f"Error during execution of tool '{tool_name}': {e}", exc_info=True)
                return f"Error: Tool '{tool_name}' failed during execution: {e}"
        else:
            logger.warning(f"Action blocked by ethical verifier for tool '{tool_name}'")
            return f"Error: Action blocked by ethical policy for tool '{tool_name}'."

    return Tool(
        name=tool_name,
        func=verified_tool_func, # Use the wrapper function
        description=description
        # Add args_schema here if needed for more complex tools
    )

async def _execute_or_stream_langchain_agent(
    agent_config: Dict[str, Any],
    task_input: Dict[str, Any],
    llm_interface: LLMInterface,
    vector_store: VectorStoreInterface,
    available_tools: Dict[str, Callable],
    verifier: EthicalVerifier
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """Internal function to handle both streaming and non-streaming execution."""
    agent_name = agent_config.get("name", "Unknown LangChain Agent")
    stream = task_input.get("stream", False)
    session_id = task_input.get("session_id", "default_session")
    input_query = task_input.get("query") or task_input.get("messages", [{}])[-1].get("content") # Adapt input source

    if not input_query:
        raise ValueError("Missing input query/message for agent task.")

    # --- Steps 1-6: LLM, Tools, Prompt, Agent, Executor, History (mostly unchanged) ---
    # (Copy relevant setup code from original execute_langchain_task here)
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
    # Add tools passed from AgentManager/config first
    if available_tools:
        for tool_name, tool_func in available_tools.items():
            if callable(tool_func):
                # Use the existing wrapper which includes ethical checks
                wrapped_tool = _wrap_tool_for_langchain(tool_func, tool_name, verifier, agent_name)
                langchain_tools.append(wrapped_tool)
            else:
                logger.warning(f"Tool '{tool_name}' for agent '{agent_name}' is not callable. Skipping.")
    
    # Add standard coding/filesystem tools
    # These are instantiated directly as they are LangChain Tool objects
    # Note: The ethical implications of these tools, especially ShellTool, should be carefully managed.
    # The existing ethical verifier might need adjustments or these tools might need specific checks.
    try:
        # Determine root directory if needed, default to current working directory or a configured safe path
        # For security, consider restricting the root_dir or using ask_human_input=True for ShellTool
        fs_root_dir = get_setting("agent_filesystem_root_dir", None) # Example: Load from config or default
        
        read_file_tool = ReadFileTool() # Can add root_dir=fs_root_dir if configured
        write_file_tool = WriteFileTool() # Can add root_dir=fs_root_dir if configured
        list_directory_tool = ListDirectoryTool() # Can add root_dir=fs_root_dir if configured
        
        # Shell tool is powerful and potentially dangerous - use with extreme caution.
        # Consider adding ask_human_input=True for safety in many environments.
        shell_tool = ShellTool(ask_human_input=False) # Set ask_human_input=True for interactive approval
        # Add a specific description to guide the LLM
        shell_tool.description = shell_tool.description + f" args: {shell_tool.args}"
        # Make sure commands are non-interactive (e.g., use 'cat' instead of 'less').

        langchain_tools.extend([
            read_file_tool,
            write_file_tool,
            list_directory_tool,
            shell_tool # Add the shell tool
        ])
        logger.debug(f"Added standard file system and shell tools for agent '{agent_name}'")

    except Exception as e:
        logger.error(f"Failed to instantiate standard tools for agent '{agent_name}': {e}", exc_info=True)
        # Decide if this should prevent agent execution or just log the error

    if not langchain_tools:
        logger.warning(f"No tools available or configured for LangChain agent '{agent_name}'")
    else:
        logger.debug(f"Total tools prepared for LangChain agent '{agent_name}': {len(langchain_tools)}")

    # 3. Define Agent Prompt
    lc_config = agent_config.get("langchain_config", {})
    agent_type = lc_config.get("agent_type", "react")
    prompt_template = None
    if agent_type == "react":
        from langchain import hub
        prompt_template = hub.pull("hwchase17/react-chat")
    elif agent_type == "structured-chat":
        system_prompt = lc_config.get("system_prompt", f"You are {agent_name}.")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
    # Add other agent_types if needed
    else:
        logger.warning(f"Unknown agent_type '{agent_type}'. Using default structured chat prompt.")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are {agent_name}."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
    if not prompt_template:
        raise RuntimeError(f"Failed to create prompt template for agent_type '{agent_type}'")

    # 4. Create the Agent Runnable
    if agent_type == "react":
        lc_agent = create_react_agent(llm, langchain_tools, prompt_template)
    elif agent_type == "structured-chat":
        lc_agent = create_structured_chat_agent(llm, langchain_tools, prompt_template)
    # Add elif for other types like create_tool_calling_agent
    else:
        raise ValueError(f"Agent creation function for type '{agent_type}' not implemented.")

    # 5. Create Agent Executor
    agent_executor = AgentExecutor(
        agent=lc_agent,
        tools=langchain_tools,
        verbose=lc_config.get("verbose", False), # Default verbose to False for cleaner logs
        max_iterations=lc_config.get("max_iterations", 5),
        handle_parsing_errors=True
    )
    
    # 6. Add Memory
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    logger.debug(f"RunnableWithMessageHistory created for session '{session_id}'")

    # --- 7. Execute or Stream --- 
    invocation_input = {"input": input_query}
    invocation_config = {"configurable": {"session_id": session_id}}

    if stream:
        logger.info(f"Streaming LangChain agent '{agent_name}' with input: '{str(input_query)[:100]}...'")
        # Return the async generator directly
        async def stream_wrapper() -> AsyncGenerator[Dict[str, Any], None]:
            final_output_chunks = []
            try:
                async for chunk in agent_with_chat_history.astream(invocation_input, config=invocation_config):
                    # Process the chunk - structure depends on what .astream yields
                    # Example: Check for output token chunks
                    # The exact structure needs inspection/debugging based on LangChain version and agent type
                    # Common structures might involve keys like 'actions', 'steps', 'messages', 'output'
                    output_chunk = chunk.get("output")
                    if output_chunk:
                        final_output_chunks.append(output_chunk)
                        yield {"type": "content_delta", "content": output_chunk}
                    # Add handling for tool calls/starts/ends if needed for OpenAI format

                # Yield final assembled output if needed (or handle finish reason)
                yield {"type": "final_output", "content": "".join(final_output_chunks)}
                yield {"type": "finish", "reason": "stop"} 

            except OutputParserException as ope:
                logger.error(f"LangChain streaming failed due to output parsing: {ope}", exc_info=True)
                yield {"type": "error", "message": f"Agent streaming failed: Could not parse LLM output: {str(ope)}"}
            except Exception as stream_error:
                logger.error(f"LangChain streaming failed: {stream_error}", exc_info=True)
                yield {"type": "error", "message": f"Agent streaming failed: {stream_error}"}

        return stream_wrapper()

    else: # Not streaming
        logger.info(f"Invoking LangChain agent '{agent_name}' with input: '{str(input_query)[:100]}...'")
        try:
            response = await agent_with_chat_history.ainvoke(invocation_input, config=invocation_config)
            output = response.get("output", "No output generated.")
            logger.info(f"LangChain agent '{agent_name}' execution completed.")
            return {"status": "success", "output": output}
        except OutputParserException as ope:
            logger.error(f"LangChain agent '{agent_name}' failed due to output parsing: {ope}", exc_info=True)
            error_detail = f"Could not parse LLM output: {str(ope)}"
            return {"status": "error", "message": f"Agent execution failed: {error_detail}"}
        except Exception as invoke_error:
            logger.error(f"LangChain agent '{agent_name}' invocation failed: {invoke_error}", exc_info=True)
            return {"status": "error", "message": f"Agent invocation failed: {invoke_error}"}

# Mark as async
async def execute_langchain_task(
    agent_config: Dict[str, Any],
    task_input: Dict[str, Any],
    llm_interface: LLMInterface,
    vector_store: VectorStoreInterface,
    available_tools: Dict[str, Callable],
    verifier: EthicalVerifier
) -> Dict[str, Any]:
    """Public function called by AgentManager. Handles non-streaming case.
    NOTE: For streaming, AgentManager should call a different function or 
    this function needs to return the generator directly, which complicates the API.
    Let's keep this for non-streaming for now. AgentManager needs adaptation for streaming.
    """
    agent_name = agent_config.get("name", "Unknown LangChain Agent")
    if task_input.get("stream", False):
        # This function currently doesn't handle returning the generator.
        # AgentManager needs modification to call _execute_or_stream_langchain_agent directly
        # for streaming requests and handle the async generator.
        logger.error("Streaming requested but execute_langchain_task called. AgentManager logic needs update.")
        return {"status": "error", "message": "Internal configuration error: Streaming not supported via this path."}
    
    logger.info(f"Executing non-streaming task with LangChain agent: {agent_name}")
    try:
        # Assuming AgentManager.execute_task now awaits this correctly.
        # This function is expected to be called via `await` from AgentManager
        result = await _execute_or_stream_langchain_agent(
            agent_config, task_input, llm_interface, vector_store, available_tools, verifier
        )

        return result # Should be the dict {"status": ..., "output": ...}
    except ValueError as ve:
        logger.error(f"Input validation error for LangChain agent '{agent_name}': {ve}", exc_info=True)
        return {"status": "error", "message": f"Agent input error: {ve}"}
    except RuntimeError as re:
        logger.error(f"Runtime error during LangChain agent '{agent_name}' setup: {re}", exc_info=True)
        return {"status": "error", "message": f"Agent setup error: {re}"}
    except Exception as e:
        logger.error(f"Unexpected error during LangChain agent '{agent_name}' execution: {e}", exc_info=True)
        return {"status": "error", "message": f"Internal error in LangChain agent: {e}"}

# Consider adding a separate function for streaming if AgentManager needs distinct calls
async def stream_langchain_task(
    agent_config: Dict[str, Any],
    task_input: Dict[str, Any],
    llm_interface: LLMInterface,
    vector_store: VectorStoreInterface,
    available_tools: Dict[str, Callable],
    verifier: EthicalVerifier
) -> AsyncGenerator[Dict[str, Any], None]:
    """Executes a task using LangChain and streams the results."""
    agent_name = agent_config.get("name", "Unknown LangChain Agent")
    logger.info(f"Executing streaming task with LangChain agent: {agent_name}")
    
    task_input["stream"] = True # Ensure stream flag is set
    
    try:
        async for chunk in _execute_or_stream_langchain_agent(
            agent_config, task_input, llm_interface, vector_store, available_tools, verifier
        ):
            yield chunk
    except ValueError as ve:
        logger.error(f"Input validation error for LangChain agent '{agent_name}': {ve}", exc_info=True)
        yield {"type": "error", "message": f"Agent input error: {ve}"}
    except RuntimeError as re:
        logger.error(f"Runtime error during LangChain agent '{agent_name}' setup: {re}", exc_info=True)
        yield {"type": "error", "message": f"Agent setup error: {re}"}
    except Exception as e:
        logger.error(f"Unexpected error during LangChain agent '{agent_name}' streaming: {e}", exc_info=True)
        yield {"type": "error", "message": f"Internal error in LangChain agent stream: {e}"} 