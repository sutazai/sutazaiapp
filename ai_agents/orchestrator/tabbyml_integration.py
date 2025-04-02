"""
SutazAI TabbyML Integration Module
Interfaces with TabbyML service for code completions and suggestions
"""

import json
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger("tabbyml_integration")


class TabbyMLClient:
    """
    Client for interacting with TabbyML service for code completions.
    TabbyML provides AI-powered code completions with local models.
    """

    def __init__(
        self, endpoint: str = "http://localhost:8080", api_key: Optional[str] = None
    ):
        """
        Initialize the TabbyML client

        Args:
            endpoint: The base URL for TabbyML API
            api_key: API key for authentication (if required)
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self._check_tabby_connection()

    def _check_tabby_connection(self):
        """Check if TabbyML service is accessible"""
        try:
            response = requests.get(f"{self.endpoint}/v1/completions/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"TabbyML service is available at {self.endpoint}")
                health_info = response.json()
                logger.info(f"TabbyML health: {health_info}")
                return True
            else:
                logger.warning(
                    f"TabbyML service returned status code {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            logger.error(f"Could not connect to TabbyML service: {str(e)}")
            return False

    def get_completion(
        self,
        code: str,
        language: str,
        cursor_position: Optional[int] = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Get code completion suggestions from TabbyML

        Args:
            code: The code snippet to complete
            language: Programming language of the code
            cursor_position: Position where completion should be inserted
                            (if None, end of code is used)
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Dictionary with completion results
        """
        if cursor_position is None:
            cursor_position = len(code)

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "language": self._normalize_language(language),
                "text": code,
                "cursor": cursor_position,
                "max_new_tokens": max_new_tokens,
            }

            logger.debug(f"Sending completion request to TabbyML: {payload}")

            response = requests.post(
                f"{self.endpoint}/v1/completions",
                headers=headers,
                json=payload,
                timeout=15,  # Increased timeout for larger completions
            )

            if response.status_code == 200:
                completion_data = response.json()
                return {
                    "success": True,
                    "completions": self._process_completion_response(completion_data),
                    "raw_response": completion_data,
                }
            else:
                logger.error(
                    f"TabbyML API error: {response.status_code} - {response.text}"
                )
                return {
                    "success": False,
                    "error": f"API returned status code {response.status_code}",
                    "message": response.text,
                }

        except requests.RequestException as e:
            logger.error(f"Error connecting to TabbyML service: {str(e)}")
            return {"success": False, "error": f"Connection error: {str(e)}"}
        except json.JSONDecodeError:
            logger.error("Failed to parse TabbyML API response as JSON")
            return {"success": False, "error": "Invalid JSON response from API"}
        except Exception as e:
            logger.error(f"Unexpected error when getting code completion: {str(e)}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def _process_completion_response(
        self, response_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process and format the completion response from TabbyML

        Args:
            response_data: Raw API response from TabbyML

        Returns:
            List of processed completion suggestions
        """
        completions = []

        if "choices" in response_data:
            for idx, choice in enumerate(response_data["choices"]):
                completion = {
                    "id": idx,
                    "text": choice.get("text", ""),
                    "score": choice.get("score", 0.0),
                }
                completions.append(completion)

        # Sort completions by score (highest first)
        completions.sort(key=lambda x: x["score"], reverse=True)

        return completions

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language name to TabbyML format

        Args:
            language: Programming language name

        Returns:
            Normalized language name for TabbyML API
        """
        language_map = {
            "py": "python",
            "python": "python",
            "js": "javascript",
            "javascript": "javascript",
            "ts": "typescript",
            "typescript": "typescript",
            "java": "java",
            "c#": "csharp",
            "csharp": "csharp",
            "cs": "csharp",
            "c": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "go": "go",
            "golang": "go",
            "rb": "ruby",
            "ruby": "ruby",
            "php": "php",
            "rs": "rust",
            "rust": "rust",
        }

        return language_map.get(language.lower(), language.lower())

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the TabbyML model

        Returns:
            Dictionary with model information
        """
        try:
            response = requests.get(f"{self.endpoint}/v1/model/info", timeout=5)

            if response.status_code == 200:
                return {"success": True, "model_info": response.json()}
            else:
                logger.error(
                    f"Error getting model info: {response.status_code} - {response.text}"
                )
                return {
                    "success": False,
                    "error": f"API returned status code {response.status_code}",
                }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_completion_with_context(
        self,
        file_content: str,
        language: str,
        cursor_offset: int,
        max_new_tokens: int = 100,
        project_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Get code completion with project context information

        Args:
            file_content: The content of the current file
            language: Programming language of the code
            cursor_offset: Cursor position in the file
            max_new_tokens: Maximum number of new tokens to generate
            project_files: Dictionary mapping file paths to content for context

        Returns:
            Dictionary with completion results
        """
        payload = {
            "language": self._normalize_language(language),
            "text": file_content,
            "cursor": cursor_offset,
            "max_new_tokens": max_new_tokens,
        }

        # Add project context if provided
        if project_files:
            context_files = []
            for file_path, content in project_files.items():
                context_files.append({"path": file_path, "content": content})
            payload["context"] = {"files": context_files}

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(
                f"{self.endpoint}/v1/completions/context",
                headers=headers,
                json=payload,
                timeout=30,  # Increased timeout for context-aware completions
            )

            if response.status_code == 200:
                completion_data = response.json()
                return {
                    "success": True,
                    "completions": self._process_completion_response(completion_data),
                    "raw_response": completion_data,
                }
            else:
                logger.error(
                    f"TabbyML API error: {response.status_code} - {response.text}"
                )
                return {
                    "success": False,
                    "error": f"API returned status code {response.status_code}",
                    "message": response.text,
                }

        except Exception as e:
            logger.error(f"Error getting context-aware completion: {str(e)}")
            return {"success": False, "error": str(e)}


# Helper functions for easier usage
def get_completion(
    code: str,
    language: str,
    cursor_position: Optional[int] = None,
    endpoint: str = "http://localhost:8080",
) -> Dict[str, Any]:
    """
    Get code completion from TabbyML

    Args:
        code: Code snippet to complete
        language: Programming language of the code
        cursor_position: Position for completion (None for end of code)
        endpoint: TabbyML service endpoint

    Returns:
        Dictionary with completion results
    """
    client = TabbyMLClient(endpoint=endpoint)
    return client.get_completion(code, language, cursor_position)


def get_best_completion(
    code: str,
    language: str,
    cursor_position: Optional[int] = None,
    endpoint: str = "http://localhost:8080",
) -> Tuple[str, float]:
    """
    Get the highest-scored completion from TabbyML

    Args:
        code: Code snippet to complete
        language: Programming language of the code
        cursor_position: Position for completion (None for end of code)
        endpoint: TabbyML service endpoint

    Returns:
        Tuple of (completion_text, confidence_score)
    """
    result = get_completion(code, language, cursor_position, endpoint)

    if result["success"] and result["completions"]:
        best_completion = result["completions"][0]
        return best_completion["text"], best_completion["score"]
    else:
        return "", 0.0


def complete_code_snippet(
    code: str,
    language: str,
    cursor_position: Optional[int] = None,
    endpoint: str = "http://localhost:8080",
) -> str:
    """
    Complete code snippet with the best suggestion from TabbyML

    Args:
        code: Code snippet to complete
        language: Programming language of the code
        cursor_position: Position for completion (None for end of code)
        endpoint: TabbyML service endpoint

    Returns:
        Completed code snippet
    """
    if cursor_position is None:
        cursor_position = len(code)

    completion, _ = get_best_completion(code, language, cursor_position, endpoint)

    if completion:
        # Insert the completion at the cursor position
        return code[:cursor_position] + completion + code[cursor_position:]
    else:
        return code
