import logging
from typing import Callable, Any, Dict, Optional
import numexpr # Import the numexpr library

# Import tool implementation functions directly
from sutazai_agi.memory.vector_store import get_vector_store
from sutazai_agi.models.llm_interface import get_llm_interface
# Import sandbox execution when implemented
# from sutazai_agi.security.sandbox import execute_code_in_sandbox, manage_files_in_sandbox

# Import specific integration modules when implemented
# from .integrations.tabbyml import get_code_completion
# from .integrations.semgrep import scan_code_semgrep
# ... etc

from sutazai_agi.core.config_loader import get_tool_config

logger = logging.getLogger(__name__)

# --- Tool Implementations ---
# These functions perform the actual work of the tools.
# They should be robust and handle potential errors.

def search_vector_store_tool(query: str, top_k: int = 3, collection_name: Optional[str] = None, **kwargs) -> str:
    """Performs a similarity search in the vector store."""
    logger.debug(f"Executing search_vector_store tool with query: '{query}', top_k={top_k}")
    try:
        vector_store = get_vector_store()
        results = vector_store.query(query_texts=[query], n_results=top_k, collection_name=collection_name)
        if results and results['documents'] and results['documents'][0]:
            # Format results into a readable string for the LLM
            output = "Search Results:\n"
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else None
                output += f"- Document {i+1} (ID: {results['ids'][0][i]}, Distance: {distance:.4f}):\n"
                output += f"  Metadata: {metadata}\n"
                output += f"  Content: {doc[:500]}...\n" # Truncate for brevity
            return output
        else:
            return "No relevant documents found in the knowledge base."
    except Exception as e:
        logger.error(f"Error executing search_vector_store tool: {e}", exc_info=True)
        return f"Error: Could not perform search due to an internal error ({e})."

def summarize_with_llm_tool(text: str, max_length: Optional[int] = None, **kwargs) -> str:
    """Uses the LLM to summarize the provided text."""
    logger.debug(f"Executing summarize_with_llm tool. Text length: {len(text)}")
    try:
        llm_interface = get_llm_interface()
        prompt = f"Please summarize the following text concisely: {text}"
        if max_length:
            prompt += f" The summary should be approximately {max_length} words long."
        
        # Use the default LLM for summarization
        response = llm_interface.generate(prompt=prompt)
        
        if response and response.get("response") and not response.get("error"):
            return response["response"]
        else:
            error_msg = response.get('error', 'Unknown error from LLM') if response else 'No response from LLM'
            logger.error(f"LLM summarization failed: {error_msg}")
            return f"Error: Could not summarize text due to LLM failure ({error_msg})."
    except Exception as e:
        logger.error(f"Error executing summarize_with_llm tool: {e}", exc_info=True)
        return f"Error: Could not summarize text due to an internal error ({e})."

def safe_calculator_tool(expression: str, **kwargs) -> str:
    """Safely evaluates a mathematical expression using numexpr."""
    logger.debug(f"Executing safe_calculator tool with expression: '{expression}'")
    try:
        # Use numexpr for safe evaluation
        # It handles standard math operations, variables can be passed via local_dict if needed
        result = numexpr.evaluate(expression, local_dict={}) # Pass empty dict for safety
        # numexpr returns numpy types, convert to standard Python types
        return f"Result: {result.item()}"
    except KeyError as e:
        # Handle cases where numexpr doesn't recognize a function/variable
        logger.error(f"Error evaluating expression '{expression}': Unrecognized element {e}", exc_info=False) # Don't need full traceback for this
        return f"Error: Could not calculate '{expression}'. Unrecognized function or variable: {e}. Only basic math (+, -, *, /, **, parentheses) is supported."
    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {e}", exc_info=True)
        return f"Error: Could not calculate '{expression}'. Invalid expression or internal error ({e})."

# --- Placeholder Implementations for Sandboxed Tools --- 
# These need actual implementation using sutazai_agi.security.sandbox

def execute_code_in_sandbox_tool(code: str, **kwargs) -> str:
    logger.warning("execute_code_in_sandbox_tool is a placeholder and does not execute code.")
    # Replace with actual call to sandbox module, e.g.:
    # from sutazai_agi.security.sandbox import execute_code
    # return execute_code(code)
    return "Error: Code execution sandbox not implemented."

def manage_files_in_sandbox_tool(action: str, path: str, content: Optional[str] = None, **kwargs) -> str:
    logger.warning("manage_files_in_sandbox_tool is a placeholder and does not manage files.")
    # Replace with actual call to sandbox module, e.g.:
    # from sutazai_agi.security.sandbox import manage_file
    # return manage_file(action, path, content)
    return "Error: File management sandbox not implemented."

# --- Placeholder for LocalAGI memory tool --- 
def localagi_memory_tool(action: str, key: str, value: Optional[str] = None, **kwargs) -> str:
     logger.warning("localagi_memory_tool is a placeholder.")
     # Needs implementation, potentially using a simple file or small DB
     if action == 'read':
         return f"Placeholder: Value for key '{key}' not found."
     elif action == 'write':
         return f"Placeholder: Wrote value for key '{key}'."
     else:
         return "Error: Invalid action for memory tool."

# --- Tool Registry --- 
# Maps the implementation name (from agents.yaml) to the actual function
_tool_registry: Dict[str, Callable[..., Any]] = {
    "search_vector_store": search_vector_store_tool,
    "summarize_with_llm": summarize_with_llm_tool,
    "safe_calculator": safe_calculator_tool,
    "execute_code_in_sandbox": execute_code_in_sandbox_tool, # Placeholder
    "manage_files_in_sandbox": manage_files_in_sandbox_tool, # Placeholder
    "localagi_memory_tool": localagi_memory_tool, # Placeholder
    # Add other tool implementations here as they are created
    # "get_code_completion": get_code_completion_tool,
    # "scan_code_semgrep": scan_code_semgrep_tool,
}

def load_tool(tool_implementation_name: str) -> Optional[Callable[..., Any]]:
    """Loads a tool function based on its implementation name from the registry."""
    tool_func = _tool_registry.get(tool_implementation_name)
    if tool_func:
        logger.debug(f"Loaded tool implementation: {tool_implementation_name}")
        return tool_func
    else:
        logger.error(f"Tool implementation '{tool_implementation_name}' not found in registry.")
        return None

def get_tool_description(tool_name: str) -> Optional[str]:
    """Retrieves the description of a tool from the configuration."""
    config = get_tool_config(tool_name)
    return config.get("description") if config else None 