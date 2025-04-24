import logging
import subprocess
import json
import tempfile
import os
import shlex
from typing import Optional, List, Dict, Any

from sutazai_agi.core.config_loader import get_setting

logger = logging.getLogger(__name__)

# Define workspace path for temp files (use configured workspace)
TEMP_WORKSPACE = os.path.abspath(get_setting("agent_workspace", "/opt/v3/workspace"))
SEMGREP_SCAN_DIR = os.path.join(TEMP_WORKSPACE, "semgrep_scan")

# Ensure the scan directory exists
os.makedirs(SEMGREP_SCAN_DIR, exist_ok=True)

class SemgrepAnalyzer:
    """Handles running Semgrep static analysis on code snippets or files."""

    def __init__(self, semgrep_path: str = "semgrep"):
        self.semgrep_cmd = semgrep_path # Allow specifying path to semgrep binary
        # Test if semgrep command exists?

    def _run_semgrep(self, target: str, config: str = "auto", output_format: str = "json") -> Optional[Dict[str, Any]]:
        """Runs the semgrep command as a subprocess."""
        command = [
            self.semgrep_cmd,
            "scan",
            f"--{output_format}", # Use specific output format flag
            "--config", config, # Use specified config (e.g., 'auto' or 'p/python')
            target # Target file or directory
        ]
        logger.debug(f"Executing Semgrep command: {' '.join(shlex.quote(c) for c in command)}")
        
        try:
            # Use timeout to prevent hangs
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False) 
            
            if result.returncode != 0:
                # Semgrep exits with specific codes for findings, error, etc.
                # Code 1 usually means findings were found.
                # Other non-zero codes might indicate actual errors.
                if result.returncode > 1: 
                     logger.error(f"Semgrep command failed with exit code {result.returncode}. Error: {result.stderr}")
                     # Return an error structure? For now, return None to indicate failure.
                     return {"error": f"Semgrep execution failed with code {result.returncode}", "stderr": result.stderr}
                else:
                     logger.info(f"Semgrep completed with exit code {result.returncode} (likely findings found).")
                         
            if not result.stdout:
                 logger.warning("Semgrep produced no output.")
                 # If exit code was 0 or 1, this might mean no findings, return empty results structure
                 if result.returncode <= 1:
                     return {"results": [], "errors": [], "version": "unknown"} 
                 else: # If exit code > 1 and no stdout, it's an error
                     return {"error": f"Semgrep failed (code {result.returncode}) with no output", "stderr": result.stderr} 
            
            # Parse JSON output
            if output_format == "json":
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse Semgrep JSON output: {json_err}. Output: {result.stdout[:500]}...")
                    return {"error": f"Failed to parse Semgrep JSON output: {json_err}"} 
            else:
                 # Handle other formats if needed, or return raw text
                 return {"raw_output": result.stdout} # Example for non-JSON

        except subprocess.TimeoutExpired:
            logger.error("Semgrep command timed out after 60 seconds.")
            return {"error": "Semgrep command timed out."} 
        except FileNotFoundError:
            logger.error(f"Semgrep command '{self.semgrep_cmd}' not found. Is Semgrep installed and in PATH?")
            return {"error": f"Semgrep command '{self.semgrep_cmd}' not found."}
        except Exception as e:
            logger.error(f"An unexpected error occurred running Semgrep: {e}", exc_info=True)
            return {"error": f"Unexpected error during Semgrep execution: {e}"}

    def analyze_code_snippet(self, code: str, language: str, config: str = "auto") -> Optional[Dict[str, Any]]:
        """Analyzes a code snippet by writing it to a temporary file."""
        # Determine file extension based on language
        # TODO: Make this mapping more robust
        extension_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp",
            "go": ".go",
            "ruby": ".rb",
            # Add more mappings as needed
        }
        extension = extension_map.get(language.lower(), ".tmp")

        # Create a temporary file in the secure scan directory
        try:
            # Using NamedTemporaryFile within the SEMGREP_SCAN_DIR
            with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False, encoding='utf-8', dir=SEMGREP_SCAN_DIR) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code)
            
            logger.debug(f"Analyzing code snippet (language: {language}) in temporary file: {temp_file_path}")
            results = self._run_semgrep(target=temp_file_path, config=config)
            
        except Exception as e:
             logger.error(f"Error managing temporary file for Semgrep snippet analysis: {e}", exc_info=True)
             results = {"error": f"Error creating/writing temporary file for analysis: {e}"} 
        finally:
            # Ensure temporary file is deleted
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Deleted temporary Semgrep file: {temp_file_path}")
                except OSError as e:
                    logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")
            
        return results

    def analyze_file(self, file_path: str, config: str = "auto") -> Optional[Dict[str, Any]]:
        """Analyzes a specific file."""
        # IMPORTANT: Ensure file_path is validated and within the allowed workspace
        # This check should ideally happen *before* calling this function (e.g., in the tool wrapper)
        # but adding a basic check here for safety.
        safe_base = os.path.abspath(get_setting("agent_workspace", "/opt/v3/workspace"))
        target_path_abs = os.path.abspath(file_path)
        if not target_path_abs.startswith(safe_base):
             logger.error(f"Attempt to scan file outside workspace denied: {file_path}")
             return {"error": "File path is outside the allowed workspace."}
             
        if not os.path.exists(target_path_abs):
            logger.error(f"File not found for Semgrep analysis: {file_path}")
            return {"error": f"File not found: {file_path}"}
            
        logger.debug(f"Analyzing file with Semgrep: {file_path}")
        return self._run_semgrep(target=target_path_abs, config=config)

# Example usage (for testing)
def _test_semgrep():
     logging.basicConfig(level=logging.INFO)
     analyzer = SemgrepAnalyzer()
     
     # Test snippet analysis
     python_code = "import os\nos.system('ls')\npassword = \"hardcoded\"" # Example vulnerable code
     print("\n--- Analyzing Snippet ---")
     results_snippet = analyzer.analyze_code_snippet(code=python_code, language="python", config="p/python")
     print(json.dumps(results_snippet, indent=2))
     
     # Test file analysis (requires a file in the workspace)
     test_file_path = os.path.join(TEMP_WORKSPACE, "test_semgrep.py")
     try:
         with open(test_file_path, "w") as f:
             f.write(python_code)
         print(f"\n--- Analyzing File: {test_file_path} ---")
         # Use relative path from workspace perspective if needed by tool interface
         relative_path = os.path.relpath(test_file_path, TEMP_WORKSPACE) 
         results_file = analyzer.analyze_file(file_path=test_file_path, config="p/python") # Pass absolute path here
         print(json.dumps(results_file, indent=2))
     except Exception as e:
         print(f"Error during file test: {e}")
     finally:
          if os.path.exists(test_file_path):
               os.remove(test_file_path)

# if __name__ == "__main__":
#     _test_semgrep() 