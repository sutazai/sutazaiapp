import logging
from typing import Callable, Any, Dict, Optional, List, Tuple
import numexpr # Import the numexpr library
import cv2 # OpenCV for image processing
from ultralytics import YOLO # YOLO model
import os
from duckduckgo_search import DDGS # Import DDGS
import subprocess # Needed for terminal tool
import shlex # Needed for safe splitting of terminal commands
import asyncio # For running subprocess async
import tempfile # For execute_code_in_sandbox
import shutil # For manage_files
import json # For localagi_memory_tool

# Import tool implementation functions directly
from sutazai_agi.memory.vector_store import get_vector_store
from sutazai_agi.models.llm_interface import get_llm_interface
# Import sandbox execution when implemented
# from sutazai_agi.security.sandbox import execute_code_in_sandbox, manage_files_in_sandbox

# Import specific integration modules when implemented
from .integrations.tabbyml import TabbyMLClient
from .integrations.semgrep import SemgrepAnalyzer
# from .integrations.gpt_engineer import run_gpt_engineer # No longer needed directly here
# from .integrations.aider import run_aider # No longer needed directly here

from sutazai_agi.core.config_loader import get_tool_config, get_setting, load_tool_config

# Import Code Service getter
from sutazai_agi.backend.services.code_service import get_code_service, CodeService

logger = logging.getLogger(__name__)

# --- Define a safe base path (adjust as needed) ---
# Use the configured workspace path, default to /opt/v3/workspace
SAFE_WORKSPACE = os.path.abspath(get_setting("agent_workspace", "/opt/v3/workspace")) 

if not os.path.exists(SAFE_WORKSPACE):
    try:
        os.makedirs(SAFE_WORKSPACE)
        logger.info(f"Created safe workspace directory: {SAFE_WORKSPACE}")
    except OSError as e:
         logger.error(f"Failed to create safe workspace directory {SAFE_WORKSPACE}: {e}. File operations may fail.", exc_info=True)
         # Set to None or raise error? For now, log and continue, tools will fail later.
         SAFE_WORKSPACE = None 

def _resolve_safe_path(relative_path: str) -> Optional[str]:
    """Resolves a relative path against the SAFE_WORKSPACE and checks for traversal."""
    if SAFE_WORKSPACE is None:
        logger.error("SAFE_WORKSPACE is not configured or could not be created. Path resolution failed.")
        return None
    try:
        # Normalize the path (e.g., collapses ..)
        # Ensure the input path is treated as relative to the workspace
        target_path_str = os.path.normpath(os.path.join(SAFE_WORKSPACE, os.path.normpath(f"./{relative_path}")))

        # Check if the resolved path is still within the SAFE_WORKSPACE
        if os.path.commonpath([SAFE_WORKSPACE, target_path_str]) == SAFE_WORKSPACE:
            return target_path_str
        else:
            logger.warning(f"Attempted path traversal detected: '{relative_path}' resolves outside {SAFE_WORKSPACE}")
            return None
    except Exception as e:
        logger.error(f"Error resolving safe path for '{relative_path}': {e}")
        return None

# --- Define Sandbox Path --- 
SANDBOX_WORKSPACE = os.path.join(SAFE_WORKSPACE, "sandbox_exec")
if SAFE_WORKSPACE and not os.path.exists(SANDBOX_WORKSPACE):
    try:
        os.makedirs(SANDBOX_WORKSPACE)
        logger.info(f"Created sandbox workspace directory: {SANDBOX_WORKSPACE}")
    except OSError as e:
         logger.error(f"Failed to create sandbox workspace directory {SANDBOX_WORKSPACE}: {e}. Sandbox tools may fail.", exc_info=True)
         SANDBOX_WORKSPACE = None

def _resolve_sandbox_path(relative_path: str) -> Optional[str]:
    """Resolves a relative path against the SANDBOX_WORKSPACE and checks for traversal."""
    if SANDBOX_WORKSPACE is None:
        logger.error("SANDBOX_WORKSPACE is not configured or could not be created. Path resolution failed.")
        return None
    try:
        # Normalize the path (e.g., collapses ..)
        target_path_str = os.path.normpath(os.path.join(SANDBOX_WORKSPACE, os.path.normpath(f"./{relative_path}")))
        # Check if the resolved path is still within the SANDBOX_WORKSPACE
        if os.path.commonpath([SANDBOX_WORKSPACE, target_path_str]) == SANDBOX_WORKSPACE:
            return target_path_str
        else:
            logger.warning(f"Attempted path traversal detected: '{relative_path}' resolves outside {SANDBOX_WORKSPACE}")
            return None
    except Exception as e:
        logger.error(f"Error resolving sandbox path for '{relative_path}': {e}")
        return None

# --- Define LocalAGI Memory Path --- 
LOCALAGI_MEMORY_FILE = os.path.join(SAFE_WORKSPACE, "localagi_memory.json")
if SAFE_WORKSPACE and not os.path.exists(os.path.dirname(LOCALAGI_MEMORY_FILE)):
    try:
        os.makedirs(os.path.dirname(LOCALAGI_MEMORY_FILE))
    except OSError:
        pass # Ignore if directory already exists due to race condition

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

# --- Web Search Tool ---
def web_search_tool(query: str, max_results: int = 5, **kwargs) -> str:
    """Performs a web search using DuckDuckGo and returns the results."""
    logger.debug(f"Executing web_search tool with query: '{query}', max_results={max_results}")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        
        if results:
            output = "Web Search Results:\n"
            for i, result in enumerate(results):
                output += f"- Result {i+1}:\n"
                output += f"  Title: {result.get('title', 'N/A')}\n"
                output += f"  URL: {result.get('href', 'N/A')}\n"
                output += f"  Snippet: {result.get('body', 'N/A')[:200]}...\n" # Truncate snippet
            return output
        else:
            return "No relevant web search results found."
    except Exception as e:
        logger.error(f"Error executing web_search tool: {e}", exc_info=True)
        return f"Error: Could not perform web search due to an internal error ({e})."

# --- YOLO Object Detection Tool ---
# Load the YOLO model once when the module is loaded.
# Using a smaller, faster model like 'yolov8n.pt' (nano) is recommended for CPU inference.
# If GPU is available and configured, YOLO will use it automatically.
try:
    # TODO: Make the model path/name configurable?
    _yolo_model_path = 'yolov8n.pt' 
    _yolo_model = YOLO(_yolo_model_path)
    logger.info(f"YOLO model '{_yolo_model_path}' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model from '{_yolo_model_path}': {e}", exc_info=True)
    _yolo_model = None # Set to None if loading fails

def detect_objects_in_image_tool(image_path: str, confidence_threshold: float = 0.25, **kwargs) -> str:
    """
    Detects objects in an image file using the YOLO model.

    Args:
        image_path (str): The path to the image file.
        confidence_threshold (float): Minimum confidence score for detected objects (0.0 to 1.0).

    Returns:
        str: A textual description of detected objects and their locations, or an error message.
    """
    logger.debug(f"Executing detect_objects_in_image tool on '{image_path}' with threshold {confidence_threshold}")

    if not _yolo_model:
        return "Error: YOLO model is not available. Check loading errors."

    if not os.path.exists(image_path):
        logger.error(f"Image path not found: {image_path}")
        return f"Error: Image file not found at '{image_path}'."

    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Could not read image file '{image_path}'. It might be corrupted or in an unsupported format."

        # Perform inference
        results = _yolo_model(img, conf=confidence_threshold, verbose=False) # verbose=False to reduce console spam

        if not results or not results[0].boxes:
            return "No objects detected in the image with the specified confidence."

        # Process results
        detected_objects = []
        object_names = results[0].names # Dictionary mapping class_id to name
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = object_names.get(class_id, f"Class_{class_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_objects.append(f"- {label} (confidence: {conf:.2f}) at bounding box [{x1},{y1},{x2},{y2}]")

        if detected_objects:
            return "Detected Objects:\n" + "\n".join(detected_objects)
        else:
             # This case should be caught by the earlier check, but added for safety
            return "No objects detected matching the criteria."

    except Exception as e:
        logger.error(f"Error during YOLO detection on '{image_path}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred during object detection ({e})."

# --- File System Tools ---

def read_file_tool(file_path: str, **kwargs) -> str:
    """Reads the content of a specified file within the safe workspace."""
    logger.debug(f"Executing read_file tool for path: '{file_path}'")
    safe_path = _resolve_safe_path(file_path)
    if not safe_path:
        return f"Error: Access denied. Path '{file_path}' is outside the allowed workspace."

    try:
        if not os.path.exists(safe_path):
            return f"Error: File not found at '{file_path}'."
        if not os.path.isfile(safe_path):
             return f"Error: Path '{file_path}' is not a file."

        # TODO: Add encoding detection or enforce UTF-8? Consider max file size limit.
        # Ref: https://github.com/cline/cline/issues/1251 (Potential emoji/encoding issue)
        with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Consider limiting the output size for very large files
        max_read_chars = 10000
        if len(content) > max_read_chars:
             logger.warning(f"Reading file '{file_path}', content truncated to {max_read_chars} characters.")
             return f"""File Content (truncated):
```
{content[:max_read_chars]}
```"""
        else:
            return f"""File Content:
```
{content}
```"""

    except Exception as e:
        logger.error(f"Error reading file '{file_path}': {e}", exc_info=True)
        return f"Error: Could not read file '{file_path}' due to an internal error ({e})."

def write_file_tool(file_path: str, content: str, **kwargs) -> str:
    """Writes or overwrites content to a specified file within the safe workspace."""
    logger.debug(f"Executing write_file tool for path: '{file_path}'")
    safe_path = _resolve_safe_path(file_path)
    if not safe_path:
        return f"Error: Access denied. Path '{file_path}' is outside the allowed workspace."

    try:
        # Ensure the directory exists if writing to a nested path
        dir_name = os.path.dirname(safe_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name) # Create parent directories if needed

        # TODO: Add file size limits? Backup existing file?
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to file '{file_path}'."
    except Exception as e:
        logger.error(f"Error writing to file '{file_path}': {e}", exc_info=True)
        return f"Error: Could not write to file '{file_path}' due to an internal error ({e})."

def list_directory_tool(directory_path: str = ".", **kwargs) -> str:
    """Lists the contents (files and subdirectories) of a specified directory within the safe workspace."""
    logger.debug(f"Executing list_directory tool for path: '{directory_path}'")
    safe_path = _resolve_safe_path(directory_path)
    if not safe_path:
        return f"Error: Access denied. Path '{directory_path}' is outside the allowed workspace."

    try:
        if not os.path.exists(safe_path):
            return f"Error: Directory not found at '{directory_path}'."
        if not os.path.isdir(safe_path):
            return f"Error: Path '{directory_path}' is not a directory."

        items = os.listdir(safe_path)
        if not items:
            return f"Directory '{directory_path}' is empty."

        # Format output
        output = f"Contents of directory '{directory_path}':\n"
        output += "\n".join([f"- {item}" for item in items])
        # Consider limiting the number of items listed for very large directories
        return output
    except Exception as e:
        logger.error(f"Error listing directory '{directory_path}': {e}", exc_info=True)
        return f"Error: Could not list directory '{directory_path}' due to an internal error ({e})."

# --- Terminal/Shell Tool (Restricted) ---

# Define a list of allowed commands for basic safety
ALLOWED_COMMANDS = {'ls', 'pwd', 'echo', 'cat', 'head', 'tail', 'grep'} # Example safe commands

async def terminal_tool(command: str, **kwargs) -> str:
    """Executes a shell command within the safe workspace and returns the output."""
    logger.debug(f"Executing terminal tool with command: '{command}'")
    
    # Basic security check: Prevent complex commands, sudo, etc.
    # This is NOT a complete sandbox.
    forbidden = ["sudo", "rm ", "mv ", "cp ", "|", ";", "&", ">", "<", "`", "$("]
    if any(f in command for f in forbidden):
        logger.warning(f"Potentially dangerous command blocked: {command}")
        return "Error: Command blocked due to security policy (contains forbidden characters/commands)."

    # IMPORTANT: Always execute within the SAFE_WORKSPACE
    if SAFE_WORKSPACE is None:
        logger.error("Cannot execute terminal command: SAFE_WORKSPACE not configured.")
        return "Error: Workspace not configured."

    try:
        # Use asyncio.create_subprocess_shell for non-blocking execution
        # Execute within the SAFE_WORKSPACE directory
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=SAFE_WORKSPACE # Set working directory
        )
        stdout, stderr = await process.communicate()
        
        exit_code = process.returncode
        
        output = f"Exit Code: {exit_code}\n"
        if stdout:
            output += f"--- STDOUT ---\\n{stdout.decode('utf-8', errors='ignore')}\\n"
        if stderr:
            output += f"--- STDERR ---\\n{stderr.decode('utf-8', errors='ignore')}\\n"
            
        # Limit output size
        max_output_chars = 5000
        if len(output) > max_output_chars:
            logger.warning(f"Terminal command '{command}' output truncated.")
            output = output[:max_output_chars] + "\\n... (output truncated)"
            
        return output.strip()

    except FileNotFoundError:
        logger.error(f"Command not found: {command.split()[0]}")
        return f"Error: Command not found: {command.split()[0]}."
    except Exception as e:
        logger.error(f"Error executing terminal command '{command}': {e}", exc_info=True)
        return f"Error: Failed to execute command '{command}' due to an internal error ({e})."

# --- Tool Implementations (Replacing Placeholders) ---

async def execute_code_in_sandbox_tool(code: str, language: str = "python", **kwargs) -> str:
    """Executes code (currently only Python) in a restricted environment and returns the output."""
    logger.debug(f"Executing execute_code_in_sandbox tool. Language: {language}")
    
    if SANDBOX_WORKSPACE is None:
        return "Error: Sandbox workspace not available."
        
    if language.lower() != "python":
        return f"Error: Code execution currently only supports Python, not '{language}'."

    # Create a temporary file for the code
    temp_file_path = None
    try:
        # Using NamedTemporaryFile within the SANDBOX_WORKSPACE
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False, encoding='utf-8', dir=SANDBOX_WORKSPACE) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code)
        
        logger.info(f"Executing Python code from temporary file: {temp_file_path}")
        
        # Execute the python script using subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            "python", # Command to run
            temp_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=SANDBOX_WORKSPACE # Run within the sandbox directory
        )
        # Communicate with the process asynchronously
        stdout, stderr = await process.communicate()
        
        exit_code = process.returncode
        output = f"Exit Code: {exit_code}\\n"
        if stdout:
            output += f"--- STDOUT ---\\n{stdout.decode('utf-8', errors='ignore')}\\n"
        if stderr:
            output += f"--- STDERR ---\\n{stderr.decode('utf-8', errors='ignore')}\\n"

        # Limit output size
        max_output_chars = 5000
        if len(output) > max_output_chars:
            logger.warning(f"Code execution output truncated.")
            output = output[:max_output_chars] + "\\n... (output truncated)"
            
        return output.strip()

    except Exception as e:
        logger.error(f"Error executing code in sandbox: {e}", exc_info=True)
        return f"Error: Failed to execute code due to an internal error ({e})."
    finally:
        # Ensure temporary file is deleted
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.warning(f"Could not delete temporary code execution file {temp_file_path}: {e}")

def manage_files_in_sandbox_tool(action: str, path: str, content: Optional[str] = None, **kwargs) -> str:
    """Manages files (read, write, list, mkdir, delete) strictly within the sandbox workspace."""
    logger.debug(f"Executing manage_files_in_sandbox tool. Action: {action}, Path: {path}")
    
    safe_path = _resolve_sandbox_path(path)
    if not safe_path:
        # For list action on root, allow it even if path resolves outside (e.g., path='..')
        if action == 'list' and path == '.':
             safe_path = SANDBOX_WORKSPACE
        elif action == 'mkdir' and not os.path.dirname(path): # Allow creating top-level dir
             safe_path = _resolve_sandbox_path(os.path.basename(path))
             if not safe_path:
                 return f"Error: Access denied or invalid path '{path}' for directory creation in sandbox."
        else:
            return f"Error: Access denied. Path '{path}' is outside the allowed sandbox workspace."

    action = action.lower()

    try:
        if action == "read":
            if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
                return f"Error: File not found at sandbox path '{path}'."
            with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            # Limit read size?
            return f"Content of {path}:\n{file_content[:5000]}" + ("... (truncated)" if len(file_content) > 5000 else "")
            
        elif action == "write":
            if content is None:
                return "Error: Content must be provided for write action."
            # Ensure parent directory exists
            parent_dir = os.path.dirname(safe_path)
            if not os.path.exists(parent_dir):
                 try:
                     os.makedirs(parent_dir)
                 except OSError as e:
                     return f"Error: Could not create parent directory for '{path}': {e}"
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to sandbox file '{path}'."
            
        elif action == "list":
            if not os.path.exists(safe_path) or not os.path.isdir(safe_path):
                # If path was '.', maybe SANDBOX_WORKSPACE itself failed creation
                if path == '.' and SANDBOX_WORKSPACE is None:
                     return "Error: Sandbox workspace does not exist."
                return f"Error: Directory not found at sandbox path '{path}'."
            items = os.listdir(safe_path)
            if not items:
                return f"Sandbox directory '{path}' is empty."
            return f"Contents of sandbox/{path}:\n" + "\n".join([f"- {item}" for item in items])

        elif action == "mkdir":
             if os.path.exists(safe_path):
                 return f"Error: Path '{path}' already exists in sandbox."
             os.makedirs(safe_path)
             return f"Successfully created directory '{path}' in sandbox."

        elif action == "delete" or action == "rm":
             if not os.path.exists(safe_path):
                 return f"Error: Path '{path}' not found in sandbox."
             if os.path.isdir(safe_path):
                 # Be careful with recursive delete
                 shutil.rmtree(safe_path) 
                 return f"Successfully deleted directory '{path}' from sandbox."
             else:
                 os.remove(safe_path)
                 return f"Successfully deleted file '{path}' from sandbox."
        else:
            return f"Error: Invalid action '{action}'. Allowed actions: read, write, list, mkdir, delete."

    except Exception as e:
        logger.error(f"Error managing sandbox file/dir '{path}' with action '{action}': {e}", exc_info=True)
        return f"Error performing action '{action}' on sandbox path '{path}': {e}"

def localagi_memory_tool(action: str, key: str, value: Optional[str] = None, **kwargs) -> str:
    """Reads or writes key-value pairs to a persistent JSON file acting as agent memory."""
    logger.debug(f"Executing localagi_memory_tool. Action: {action}, Key: {key}")
    
    if SAFE_WORKSPACE is None:
         return "Error: Workspace not available for LocalAGI memory."

    action = action.lower()
    memory_data = {}

    # Load existing memory data
    try:
        if os.path.exists(LOCALAGI_MEMORY_FILE):
            with open(LOCALAGI_MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"LocalAGI memory file {LOCALAGI_MEMORY_FILE} is corrupted. Starting fresh.")
        memory_data = {}
    except Exception as e:
        logger.error(f"Error loading LocalAGI memory file {LOCALAGI_MEMORY_FILE}: {e}", exc_info=True)
        return f"Error loading memory: {e}"

    # Perform action
    if action == 'read':
        retrieved_value = memory_data.get(key)
        if retrieved_value is not None:
            return f"Value for key '{key}': {retrieved_value}"
        else:
            return f"No value found for key '{key}'."

    elif action == 'write':
        if value is None:
            return "Error: Value must be provided for write action."
        memory_data[key] = value
        # Save updated memory data
        try:
            with open(LOCALAGI_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2)
            return f"Successfully wrote value for key '{key}'."
        except Exception as e:
            logger.error(f"Error writing to LocalAGI memory file {LOCALAGI_MEMORY_FILE}: {e}", exc_info=True)
            return f"Error saving memory: {e}"

    else:
        return "Error: Invalid action for memory tool. Use 'read' or 'write'."

async def deep_research_tool(topic: str, depth: str = "medium", focus_areas: Optional[str] = None, **kwargs) -> str:
    """Conducts research on a topic using internal knowledge base and LLM synthesis."""
    logger.debug(f"Executing deep_research tool on topic: '{topic}', depth: {depth}")
    
    try:
        llm_interface = get_llm_interface()
        if not llm_interface:
             return "Error: LLM Interface not available for research."
        
        # Step 1: Break down the research topic into sub-questions
        # Determine number of questions based on depth
        num_questions = 3 if depth == "brief" else (5 if depth == "medium" else 7)
        breakdown_prompt = f"""
        You are planning a research project on the topic: "{topic}"
        Research depth requested: {depth}
        {f'Specific focus areas: {focus_areas}' if focus_areas else ''}
        
        Generate {num_questions} specific and distinct sub-questions or key aspects required to understand this topic thoroughly based on the requested depth and focus.
        Output ONLY the numbered list of questions, one per line. Do not include any other text.
        """
        
        # Assuming llm_interface.generate is synchronous for now, wrap if needed
        # Or refactor llm_interface to have an async generate method
        # For now, assume synchronous call is okay within this async tool function
        # If llm_interface.generate becomes async, await it here.
        breakdown_response = llm_interface.generate(prompt=breakdown_prompt)
        
        if not breakdown_response or not breakdown_response.get("response") or breakdown_response.get("error"):
            err_msg = breakdown_response.get("error", "No response") if breakdown_response else "No response"
            logger.error(f"Failed to break down research topic '{topic}': {err_msg}")
            return f"Error: Failed to break down research topic ({err_msg})."
        
        sub_questions_text = breakdown_response["response"].strip()
        sub_questions = [q.strip() for q in sub_questions_text.split('\n') if q.strip()]
        if not sub_questions:
             logger.error(f"LLM returned empty list for sub-questions for topic '{topic}'")
             return "Error: Could not generate sub-questions for research."
             
        logger.info(f"Broke down '{topic}' into {len(sub_questions)} sub-questions: {sub_questions}")
        
        # Step 2: Search internal knowledge base for each sub-question
        all_findings = []
        vector_store = get_vector_store()
        if not vector_store:
             return "Error: Vector Store not available for research."
             
        for i, question in enumerate(sub_questions):
            # Extract just the question text (remove numbering/formatting)
            clean_question = question
            if '.' in clean_question:
                parts = clean_question.split('.', 1)
                if parts[0].isdigit():
                    clean_question = parts[1].strip()
            # Add more cleaning if needed
            
            logger.debug(f"Researching sub-question: '{clean_question}'")
            finding_summary = f"Sub-Question {i+1}: {clean_question}\\n"
            try:
                # Use the vector store tool logic directly for internal search
                search_results = vector_store.query(query_texts=[clean_question], n_results=3) # Get top 3 results
                
                if search_results and search_results['documents'] and search_results['documents'][0]:
                    finding_summary += "Relevant Information Found:\\n"
                    for j, doc in enumerate(search_results['documents'][0]):
                        metadata = search_results['metadatas'][0][j] if search_results['metadatas'] else {}
                        source = metadata.get('source', 'Unknown')
                        finding_summary += f"- From '{source}': {doc[:300]}...\\n"
                else:
                    finding_summary += "No relevant information found in the local knowledge base.\\n"
                # Append findings regardless of search result inside the try block
                all_findings.append(finding_summary) 
            except Exception as search_err:
                 logger.error(f"Error searching vector store for question '{clean_question}': {search_err}", exc_info=True)
                 finding_summary += f"Error during search: {search_err}\\n"
            # Append findings regardless of search result outside the try/except block
            all_findings.append(finding_summary)
        
        # Step 3: Synthesize all findings into a coherent report using LLM
        findings_text = "\n".join(all_findings)
        
        synthesis_prompt = f"""
        Synthesize the following research findings into a coherent report on the topic "{topic}".
        Structure the report logically, address each sub-question's findings, and provide a concluding summary.
        Ensure the report is based *only* on the information provided below.
        
        RESEARCH FINDINGS:
        ---
        {findings_text}
        ---
        
        Synthesized Report:
        """
        
        # Use LLM for synthesis
        synthesis_response = llm_interface.generate(prompt=synthesis_prompt)
        
        if synthesis_response and synthesis_response.get("response") and not synthesis_response.get("error"):
            logger.info(f"Successfully generated research report for topic: '{topic}'")
            return synthesis_response["response"]
        else:
            err_msg = synthesis_response.get("error", "No response") if synthesis_response else "No response"
            logger.error(f"LLM failed to synthesize research report for '{topic}': {err_msg}")
            # Fallback: return raw findings if synthesis fails
            return f"## Research Findings for: {topic}\\n\\nCould not synthesize report (LLM Error: {err_msg}). Raw findings follow:\\n\\n{findings_text}"
        
    except Exception as e:
        logger.error(f"Error during deep research on '{topic}': {e}", exc_info=True)
        return f"Error: Deep research process failed due to an internal error ({e})."

# --- Tools requiring external integrations ---
from .integrations.tabbyml import TabbyMLClient
from .integrations.semgrep import SemgrepAnalyzer

_tabby_client: Optional[TabbyMLClient] = None
_semgrep_analyzer: Optional[SemgrepAnalyzer] = None

def _get_tabby_client() -> Optional[TabbyMLClient]:
    global _tabby_client
    if _tabby_client is None:
        try:
            _tabby_client = TabbyMLClient() # Assumes URL is in settings
        except ValueError as e:
            logger.error(f"Failed to initialize TabbyMLClient: {e}")
            _tabby_client = None # Mark as failed
        except Exception as e:
             logger.error(f"Unexpected error initializing TabbyMLClient: {e}", exc_info=True)
             _tabby_client = None
    return _tabby_client

def _get_semgrep_analyzer() -> Optional[SemgrepAnalyzer]:
    global _semgrep_analyzer
    if _semgrep_analyzer is None:
        try:
             _semgrep_analyzer = SemgrepAnalyzer() # Assumes semgrep is in PATH
        except Exception as e:
             logger.error(f"Unexpected error initializing SemgrepAnalyzer: {e}", exc_info=True)
             _semgrep_analyzer = None
    return _semgrep_analyzer

async def get_code_completion_tool(language: str, prompt: str, suffix: Optional[str] = None, **kwargs) -> str:
    """Provides code completion suggestions using TabbyML."""
    logger.debug(f"Executing get_code_completion tool. Language: {language}, Prompt length: {len(prompt)}")
    client = _get_tabby_client()
    if not client:
        return "Error: TabbyML client is not configured or failed to initialize."
    
    completion_data = await client.get_completion(language=language, prompt=prompt, suffix=suffix)
    
    if completion_data and completion_data.get("choices"):
        # Assuming the first choice is the most relevant
        first_choice = completion_data["choices"][0]
        completion_text = first_choice.get("text", "")
        logger.info(f"TabbyML provided completion of length {len(completion_text)}")
        # Return just the completion text
        return completion_text
    elif completion_data and "error" in completion_data:
         return f"Error from TabbyML: {completion_data['error']}" 
    else:
        logger.warning("TabbyML did not return a valid completion.")
        return "Error: Could not get code completion from TabbyML."

def scan_code_semgrep_tool(code: Optional[str] = None, file_path: Optional[str] = None, language: Optional[str] = "python", config: str = "auto", **kwargs) -> str:
    """Scans a code snippet or file using Semgrep for potential issues."""
    logger.debug(f"Executing scan_code_semgrep tool. File: {file_path}, Lang: {language}, Code provided: {code is not None}")
    analyzer = _get_semgrep_analyzer()
    if not analyzer:
        return "Error: Semgrep analyzer failed to initialize."

    if code and file_path:
        return "Error: Provide either code snippet or file_path, not both."
    elif code:
        if not language:
            return "Error: Language must be specified when providing a code snippet."
        results = analyzer.analyze_code_snippet(code=code, language=language, config=config)
    elif file_path:
        # Validate file_path is within workspace before passing to analyzer?
        # Analyzer does its own check, but extra safety here might be good.
        safe_path_check = _resolve_safe_path(file_path)
        if not safe_path_check:
             return f"Error: Access denied. Path '{file_path}' is outside the allowed workspace."
        # Pass the original relative path or the resolved safe path? Let's use resolved for clarity.
        results = analyzer.analyze_file(file_path=safe_path_check, config=config)
    else:
        return "Error: Must provide either code snippet or file_path."

    if results and "error" in results:
        return f"Semgrep Error: {results['error']}" + (f" ({results['stderr']})" if results.get('stderr') else "")
    elif results and "results" in results:
        findings = results["results"]
        if not findings:
            return "Semgrep scan completed. No issues found."
        else:
            output = f"Semgrep Findings ({len(findings)}):\n"
            # Format findings for readability
            for finding in findings[:10]: # Limit number of findings shown
                output += f"- Rule: {finding['check_id']}\n"
                output += f"  Severity: {finding.get('extra', {}).get('severity', 'N/A')}\n"
                output += f"  File: {finding['path']}:{finding['start']['line']}\n"
                output += f"  Message: {finding.get('extra', {}).get('message', 'N/A')[:150]}...\n"
            if len(findings) > 10:
                 output += "... (additional findings truncated)\n"
            return output
    else:
        logger.warning(f"Semgrep returned unexpected structure: {results}")
        return "Error: Semgrep analysis finished with unexpected results."

# --- GPT-Engineer Tool ---
async def gpt_engineer_tool(
    prompt: str,
    # base_model: Optional[str] = None, # Example optional args
    # image_generation: Optional[bool] = None,
    # improve_mode: Optional[bool] = None,
    # existing_project_path: Optional[str] = None,
    timeout: int = 900,
    _agent_config: Optional[Dict[str, Any]] = None, # Injected by agent manager
    **kwargs # Catches any other args passed by LLM
) -> str:
    """
    Generates or modifies a codebase based on a natural language prompt using GPT-Engineer.

    Args:
        prompt (str): The detailed description of the desired application or changes.
        timeout (int, optional): Maximum execution time in seconds. Defaults to 900.
        # Optional args for more control (can be added):
        # base_model (str, optional): Specify the base model for GPT-Engineer (e.g., 'gpt-4', 'gpt-3.5-turbo'). Defaults to GPT-Engineer's default.
        # image_generation (bool, optional): Enable image generation features if supported. Defaults to False.
        # improve_mode (bool, optional): Set to True to modify an existing project instead of creating a new one. Defaults to False. Requires existing_project_path.
        # existing_project_path (str, optional): Path to the existing project to improve (required if improve_mode is True).

    Returns:
        str: A summary of the execution status, including the output path or error message.
    """
    logger.debug(f"Executing gpt_engineer_tool with prompt: '{prompt[:50]}...' and timeout: {timeout}")
    
    if _agent_config is None:
        logger.error("Agent configuration not provided to gpt_engineer_tool. Cannot run.")
        return "Error: Agent configuration missing. Tool cannot be executed."
        
    # Prepare arguments for run_gpt_engineer, potentially filtering from kwargs
    # For now, just pass prompt, agent_config, timeout
    run_args = {
        "prompt": prompt,
        "agent_config": _agent_config,
        "timeout": timeout
        # Add optional args here based on function signature and **kwargs if implementing them
    }
    
    try:
        # Import here to avoid potential circular dependencies at module level
        from .integrations.gpt_engineer import run_gpt_engineer 
        
        result = await run_gpt_engineer(**run_args)
        
        status = result.get("status", "error")
        message = result.get("message")
        output_path = result.get("output_path")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if status == "success":
            response = f"GPT-Engineer completed successfully. Code generated/modified at: {output_path}"
            if stderr: # Include stderr even on success, as it might contain useful logs
                response += f"\\nLog output (stderr):\\n{stderr[:500]}..." # Truncate stderr
            return response
        else:
            response = f"GPT-Engineer failed: {message}"
            if output_path: # Often the workspace path is returned on error for debugging
                response += f"\\nWorkspace/Output path for debugging: {output_path}"
            # Include logs on failure
            if stderr:
                response += f"\\nStderr Log:\\n{stderr[:1000]}..."
            elif stdout: # If no stderr, maybe stdout has info
                 response += f"\\nStdout Log:\\n{stdout[:1000]}..."
            return response

    except ImportError:
        logger.error("Failed to import run_gpt_engineer. Check the integration file.")
        return "Error: GPT-Engineer integration is not available."
    except Exception as e:
        logger.error(f"Unexpected error calling run_gpt_engineer from tool: {e}", exc_info=True)
        return f"Error: An unexpected internal error occurred while running GPT-Engineer: {e}"

# --- Central Tool Registry ---
# Maps tool names (used in agent prompts/configs) to their implementation functions and metadata.
# The metadata can include description and potentially args_schema for LangChain tools.

_TOOL_IMPLEMENTATIONS = {
    "search_vector_store": {
        "function": search_vector_store_tool,
        "description": "Performs a similarity search in the vector knowledge base to find relevant information based on the user query. Use this to answer questions based on stored documents or past conversations.",
    },
    "summarize_text": {
        "function": summarize_with_llm_tool,
        "description": "Summarizes a long piece of text using the main language model. Useful for condensing information from documents or long user inputs.",
    },
    "calculator": {
        "function": safe_calculator_tool,
        "description": "Evaluates a mathematical expression safely. Supports basic arithmetic (+, -, *, /, **). Use for calculations.",
    },
    "web_search": {
        "function": web_search_tool,
        "description": "Performs a web search using DuckDuckGo to find current information or answer questions not covered by the internal knowledge base. Use this when real-time external information is needed.",
    },
    "detect_objects_in_image": {
        "function": detect_objects_in_image_tool,
        "description": "Analyzes an image file using YOLO object detection model and returns a list of detected objects, their confidence scores, and bounding boxes. Provide the path to the image file.",
    },
    "read_file": {
        "function": read_file_tool,
        "description": "Reads the content of a file within the designated safe workspace. Provide the relative path to the file.",
    },
    "write_file": {
        "function": write_file_tool,
        "description": "Writes content to a file within the designated safe workspace. Provide the relative path to the file and the content to write. Use with caution, especially when overwriting.",
    },
    "list_directory": {
        "function": list_directory_tool,
        "description": "Lists the files and subdirectories within a specified directory in the safe workspace. Provide the relative path to the directory (default is the workspace root).",
    },
    "terminal": {
        "function": terminal_tool,
        "description": "Executes a shell command in a restricted environment within the safe workspace. Use with extreme caution. Only safe, non-interactive commands are recommended. Cannot be used for network operations.",
    },
    "execute_code": {
        "function": execute_code_in_sandbox_tool,
        "description": "Executes a snippet of code (Python by default) in a secure sandbox environment. Returns the standard output and standard error. Use this to run generated code, test snippets, or perform isolated computations.",
    },
    "manage_sandbox_files": {
        "function": manage_files_in_sandbox_tool,
        "description": "Manages files within the secure sandbox environment. Actions: 'create', 'read', 'update', 'delete'. Provide the action, relative path within the sandbox, and content (for create/update).",
    },
    "localagi_memory": {
        "function": localagi_memory_tool,
        "description": "Manages persistent key-value memory specific to the LocalAGI agent. Actions: 'read', 'write'. Provide action, key, and value (for write). Use to store or retrieve agent state or user preferences.",
    },
    "get_code_completion": {
        "function": get_code_completion_tool,
        "description": "Provides code completion suggestions using TabbyML based on the current code context (prompt) and language. Can help complete lines or functions.",
    },
    "scan_code_semgrep": {
        "function": scan_code_semgrep_tool,
        "description": "Scans a code snippet or file using Semgrep for potential bugs, security vulnerabilities, or style issues based on predefined rules. Provide either the code content or the file path.",
    },
    # --- New Code Assistant Tools ---
    "generate_codebase": {
        "function": gpt_engineer_tool,
        "description": "Generates a new codebase project based on a natural language prompt using GPT-Engineer. Provide a detailed prompt and a unique project name. Best for creating entire small projects from scratch.",
    },
    "edit_code_files": {
        "function": aider_tool,
        "description": "Edits one or more existing code files based on a natural language instruction using Aider. Provide a list of relative file paths and the detailed instruction. Can operate within a specified git repository context.",
    },
    "gpt_engineer": gpt_engineer_tool,
}

# --- Tool Loading Functions ---

def load_tool(tool_implementation_name: str) -> Optional[Callable[..., Any]]:
    """Loads the actual callable function for a given tool name."""
    tool_info = _TOOL_IMPLEMENTATIONS.get(tool_implementation_name)
    if tool_info:
        return tool_info.get("function")
    logger.warning(f"Tool implementation '{tool_implementation_name}' not found.")
    return None

def get_tool_description(tool_name: str) -> Optional[str]:
    """Gets the description for a given tool name."""
    tool_info = _TOOL_IMPLEMENTATIONS.get(tool_name)
    if tool_info:
        return tool_info.get("description")
    logger.warning(f"Tool description for '{tool_name}' not found.")
    return None

def get_available_tools_info() -> Dict[str, str]:
    """Returns a dictionary of available tool names and their descriptions."""
    return {name: data.get("description", "No description available.")
            for name, data in _TOOL_IMPLEMENTATIONS.items()}

# Example of how args_schema could be added for LangChain Tools (if needed)
# from pydantic import BaseModel, Field
# class SearchVectorStoreArgs(BaseModel):
#     query: str = Field(description="The search query text.")
#     top_k: int = Field(3, description="Number of results to return.")
#     collection_name: Optional[str] = Field(None, description="Specific collection to search in.")
#
# _TOOL_IMPLEMENTATIONS["search_vector_store"]["args_schema"] = SearchVectorStoreArgs

# Add function to get all available tool names and descriptions if needed
def get_available_tools_info() -> Dict[str, str]:
    """Returns a dictionary of available tool names and their descriptions."""
    return {name: data.get("description", "No description available.") 
            for name, data in _TOOL_IMPLEMENTATIONS.items()}

# --- GPT-Engineer Tool Description ---
_TOOL_DESCRIPTIONS = {
    "gpt_engineer": "Generates or modifies an entire codebase based on a detailed natural language prompt using the GPT-Engineer tool. Provide the 'prompt' describing the application or desired changes. This tool can take a significant amount of time to run. Returns a summary message indicating success (with output path) or failure.",
}

# Add descriptions to the _TOOL_IMPLEMENTATIONS dictionary
_TOOL_IMPLEMENTATIONS.update(_TOOL_DESCRIPTIONS) 

# --- Placeholder Tool --- 
def _not_implemented_tool(*args, **kwargs) -> Dict[str, Any]:
    """Placeholder for tools not yet implemented."""
    tool_name = kwargs.get("_tool_name", "Unknown Tool")
    logger.warning(f"Tool '{tool_name}' called but is not implemented.")
    return {"status": "error", "message": f"Tool '{tool_name}' is not implemented."}

# --- Core Tool Implementations --- 

async def search_local_docs(query: str, top_k: int = 5, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Searches the local document vector store (ChromaDB) for relevant information."""
    logger.info(f"Executing local document search for query: '{query[:50]}...' Top K: {top_k}")
    try:
        vector_store = get_vector_store()
        results = await vector_store.query(
            query_texts=[query],
            n_results=top_k,
            collection_name=collection_name or get_setting("document_processing.vector_collection", "sutazai_documents")
        )
        
        if results and results.get('documents') and results['documents'][0]:
            formatted_results = [
                {"content": doc, "metadata": meta, "score": dist}
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
            ]
            return {"status": "success", "results": formatted_results}
        else:
             return {"status": "success", "results": [], "message": "No relevant documents found."}

    except Exception as e:
        logger.error(f"Error during local document search: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to search local documents: {e}"}

async def manage_file(action: str, path: str, content: Optional[str] = None) -> Dict[str, Any]:
    """Reads, writes, or lists files within the agent's workspace."""
    allowed_actions = ["read", "write", "list_dir"]
    if action not in allowed_actions:
        return {"status": "error", "message": f"Invalid file action '{action}'. Allowed: {allowed_actions}"}

    agent_workspace = get_setting("agent_workspace", "./workspace")
    if not os.path.isabs(agent_workspace):
         agent_workspace = os.path.abspath(agent_workspace)

    if os.path.isabs(path):
        return {"status": "error", "message": "Absolute paths are not allowed."}
    
    full_path = os.path.abspath(os.path.join(agent_workspace, path))

    if os.path.commonpath([agent_workspace, full_path]) != agent_workspace:
        logger.warning(f"Attempted file access outside workspace: '{path}' resolved to '{full_path}'")
        return {"status": "error", "message": "File path is outside the allowed workspace."}

    logger.info(f"Executing file action '{action}' on path: '{full_path}' (relative: '{path}')")

    try:
        if action == "read":
            if not os.path.exists(full_path) or not os.path.isfile(full_path):
                 return {"status": "error", "message": f"File not found: {path}"}
            with open(full_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            return {"status": "success", "content": file_content}
        elif action == "write":
            if content is None:
                return {"status": "error", "message": "Content is required for write action."}
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"status": "success", "message": f"File written successfully: {path}"}
        elif action == "list_dir":
            if not os.path.exists(full_path) or not os.path.isdir(full_path):
                 return {"status": "error", "message": f"Directory not found: {path}"}
            items = os.listdir(full_path)
            dir_listing = [
                {"name": item, "type": "dir" if os.path.isdir(os.path.join(full_path, item)) else "file"}
                for item in items
            ]
            return {"status": "success", "listing": dir_listing}
    except Exception as e:
        logger.error(f"Error during file action '{action}' on '{path}': {e}", exc_info=True)
        return {"status": "error", "message": f"File operation failed: {e}"}

async def get_code_completion(code_snippet: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Gets code completions using TabbyML."""
    logger.info("Requesting code completion from TabbyML.")
    try:
        result = await run_tabby_ml_completion(code_snippet, context)
        return result
    except NameError:
        logger.error("TabbyML integration function 'run_tabby_ml_completion' not found.")
        return _not_implemented_tool(_tool_name="get_code_completion")
    except Exception as e:
        logger.error(f"Error getting code completion: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to get code completion: {e}"}

async def scan_code_semgrep(code: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Scans code or a file using Semgrep for security issues."""
    if not code and not file_path:
        return {"status": "error", "message": "Either code snippet or file_path must be provided for Semgrep scan."}
    target = f"file '{file_path}'" if file_path else "code snippet"
    logger.info(f"Requesting Semgrep scan for {target}.")
    full_file_path = None
    if file_path:
        agent_workspace = get_setting("agent_workspace", "./workspace")
        if not os.path.isabs(agent_workspace):
            agent_workspace = os.path.abspath(agent_workspace)
        if os.path.isabs(file_path):
             return {"status": "error", "message": "Absolute paths are not allowed for file_path."}
        full_file_path = os.path.abspath(os.path.join(agent_workspace, file_path))
        if os.path.commonpath([agent_workspace, full_file_path]) != agent_workspace:
             logger.warning(f"Semgrep scan requested outside workspace: '{file_path}'")
             return {"status": "error", "message": "File path is outside the allowed workspace."}
        if not os.path.exists(full_file_path) or not os.path.isfile(full_file_path):
             return {"status": "error", "message": f"File not found for Semgrep scan: {file_path}"}
    try:
        result = await run_semgrep_scan(code=code, file_path=full_file_path)
        return result
    except NameError:
        logger.error("Semgrep integration function 'run_semgrep_scan' not found.")
        return _not_implemented_tool(_tool_name="scan_code_semgrep")
    except Exception as e:
        logger.error(f"Error running Semgrep scan: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to run Semgrep scan: {e}"}

# --- Placeholder Tools for New Integrations --- 

async def execute_agentzero_task(task_description: str, config_overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """Executes a task using the AgentZero agent (Not Implemented)."""
    return _not_implemented_tool(_tool_name="execute_agentzero_task")

async def run_skyvern_automation(instruction: str, target_url: Optional[str] = None) -> Dict[str, Any]:
    """Runs a browser automation task using Skyvern (Not Implemented)."""
    return _not_implemented_tool(_tool_name="run_skyvern_automation")

async def generate_code_gptengineer(prompt: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Generates a codebase using GPT-Engineer (Not Implemented)."""
    return _not_implemented_tool(_tool_name="generate_code_gptengineer")

async def edit_code_aider(files_to_edit: List[str], instruction: str) -> Dict[str, Any]:
    """Edits code files using Aider (Not Implemented)."""
    return _not_implemented_tool(_tool_name="edit_code_aider")

# --- Tool Mapping --- 
_TOOL_IMPLEMENTATIONS: Dict[str, Callable] = {
    "search_local_documents": search_local_docs,
    "manage_file": manage_file,
    "calculator": _not_implemented_tool,
    "summarize_text": _not_implemented_tool,
    "get_code_completion": get_code_completion,
    "scan_code_semgrep": scan_code_semgrep,
    "execute_python_code": _not_implemented_tool,
    "generate_code_gptengineer": generate_code_gptengineer,
    "edit_code_aider": edit_code_aider,
    "browse_local_content": _not_implemented_tool,
    "execute_agentzero_task": execute_agentzero_task,
    "run_skyvern_automation": run_skyvern_automation,
    "localagi_memory_tool": _not_implemented_tool,
    "deep_research_tool": _not_implemented_tool,
    "financial_analysis": _not_implemented_tool,
    "get_current_time": _not_implemented_tool,
}

class ToolLibrary:
    """Provides access to available tools and their descriptions."""
    def __init__(self):
        self.tools = _TOOL_IMPLEMENTATIONS
        self.tool_descriptions = self._load_descriptions()
        logger.info(f"ToolLibrary initialized with {len(self.tools)} tools.")

    def _load_descriptions(self) -> Dict[str, str]:
        """Loads tool descriptions from config or uses docstrings."""
        descriptions = {}
        tool_config = load_tool_config()
        for name, func in self.tools.items():
            if name in tool_config:
                descriptions[name] = tool_config[name].get("description", "No description provided.")
            else:
                descriptions[name] = func.__doc__ or "No description provided."
        return descriptions

    def get_tool(self, name: str) -> Optional[Callable]:
        """Gets the implementation function for a tool by name."""
        tool_func = self.tools.get(name)
        if tool_func:
            return tool_func
        else:
            logger.warning(f"Tool implementation '{name}' not found.")
            return None

    def get_tool_description(self, name: str) -> Optional[str]:
        """Gets the description for a tool by name."""
        return self.tool_descriptions.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lists all available tools with their names and descriptions."""
        return [
            {"name": name, "description": self.get_tool_description(name)}
            for name in self.tools.keys()
        ]

    def get_tools_by_name(self, tool_names: List[str]) -> Dict[str, Callable]:
        """Gets a dictionary of tool implementations for a given list of names."""
        selected_tools = {}
        for name in tool_names:
            tool_func = self.get_tool(name)
            if tool_func:
                selected_tools[name] = tool_func
            else:
                logger.warning(f"Tool '{name}' requested but not found in library.")
        return selected_tools

# --- Global Tool Library Instance --- 
_tool_library: Optional[ToolLibrary] = None

def get_tool_library() -> ToolLibrary:
    """Returns a singleton instance of the ToolLibrary."""
    global _tool_library
    if _tool_library is None:
        _tool_library = ToolLibrary()
    return _tool_library 