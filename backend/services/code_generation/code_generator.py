"""
SutazAI Code Generator
Interfaces with local LLMs to generate code based on specifications
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional

from backend.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("code_generator")


class CodeGenerator:
    """
    Code generation service using local LLMs like GPT4All and DeepSeek.
    Provides methods for generating code from specifications and improving existing code.
    """

    def __init__(self):
        """Initialize the code generator with available models"""
        self.models_dir = settings.MODELS_DIR or "/opt/sutazaiapp/model_management"
        self.models = self._discover_models()
        self.default_model = settings.DEFAULT_CODE_MODEL or "gpt4all"

        # Initialize language-specific templates
        self.language_templates = {
            "python": {
                "prefix": "# Python code that implements the following specification:\n# {}\n\n",
                "suffix": "\n\n# End of code",
            },
            "javascript": {
                "prefix": "// JavaScript code that implements the following specification:\n// {}\n\n",
                "suffix": "\n\n// End of code",
            },
            "typescript": {
                "prefix": "// TypeScript code that implements the following specification:\n// {}\n\n",
                "suffix": "\n\n// End of code",
            },
            "java": {
                "prefix": "// Java code that implements the following specification:\n// {}\n\n",
                "suffix": "\n\n// End of code",
            },
            "csharp": {
                "prefix": "// C# code that implements the following specification:\n// {}\n\n",
                "suffix": "\n\n// End of code",
            },
            # Add more languages as needed
        }

        logger.info(
            f"CodeGenerator initialized with models: {list(self.models.keys())}"
        )

    def _discover_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover available LLM models in the models directory
        """
        models: Dict[str, Dict[str, Any]] = {}

        # Define known model types and their paths
        model_configs = [
            {
                "name": "gpt4all",
                "path": os.path.join(self.models_dir, "GPT4All/model.bin"),
                "type": "gpt4all",
                "handler": self._generate_with_gpt4all,
            },
            {
                "name": "deepseek-coder",
                "path": os.path.join(self.models_dir, "DeepSeek-Coder-33B/model.bin"),
                "type": "deepseek",
                "handler": self._generate_with_deepseek,
            },
        ]

        # Check which models are available
        for config in model_configs:
            # Explicitly cast path to str for os.path.exists
            model_path = str(config["path"])
            if os.path.exists(model_path):
                # Assuming config['name'] is str, but ignore potential Mypy confusion
                models[config["name"]] = { # type: ignore[index]
                    "path": model_path,
                    "type": config["type"],
                    "handler": config["handler"],
                }
                logger.info(f"Found model: {config['name']} at {config['path']}")
            else:
                logger.warning(f"Model not found: {config['name']} at {config['path']}")

        return models

    def generate_code(
        self, spec_text: str, language: str = "python", model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate code based on a specification using a local LLM

        Args:
            spec_text: Specification text describing the code to generate
            language: Programming language for code generation
            model: Model to use (defaults to the default model)

        Returns:
            Dictionary with generated code and metadata
        """
        start_time = time.time()

        # Use default model if none specified or if specified model doesn't exist
        model_name = model or self.default_model
        if model_name not in self.models:
            logger.warning(
                f"Model {model_name} not found, falling back to {self.default_model}"
            )
            model_name = self.default_model

        try:
            # Format the prompt with language-specific template
            language = language.lower()
            template = self.language_templates.get(
                language,
                {
                    "prefix": "# Code in {} that implements:\n# {}\n\n",
                    "suffix": "\n\n# End of code",
                },
            )

            # Format the template - handle the case where prefix has one or two placeholders
            prefix = template["prefix"]
            if prefix.count("{}") == 2:
                prompt = prefix.format(language, spec_text)
            else:
                prompt = prefix.format(spec_text)

            # Get the appropriate model handler
            model_config = self.models[model_name]
            handler = model_config["handler"]

            # Generate code using the handler
            generated_code = handler(prompt, language)

            # Clean up the generated code
            generated_code = self._clean_generated_code(generated_code)

            # Run static analysis on the generated code
            issues = self._analyze_code(generated_code, language)

            # Calculate generation time
            generation_time_ms = int((time.time() - start_time) * 1000)

            return {
                "language": language,
                "generated_code": generated_code,
                "issues": issues,
                "generation_time_ms": generation_time_ms,
                "model": model_name,
            }

        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            raise RuntimeError(f"Failed to generate code: {str(e)}")

    def improve_code(
        self, code: str, issues: List[str], language: str = "python"
    ) -> Dict[str, Any]:
        """
        Improve existing code based on identified issues

        Args:
            code: Original code to improve
            issues: List of issues to fix
            language: Programming language of the code

        Returns:
            Dictionary with improved code and metadata
        """
        start_time = time.time()

        try:
            # Format the prompt
            issues_text = "\n".join([f"- {issue}" for issue in issues])
            prompt = f"""Please improve the following {language} code by fixing these issues:
{issues_text}

Original code:
```{language}
{code}
```

Improved code:
```{language}
"""

            # Generate improved code using the default model
            model_config = self.models[self.default_model]
            handler = model_config["handler"]

            improved_code = handler(prompt, language)

            # Clean up the improved code
            improved_code = self._extract_code_block(improved_code, language)

            # Run static analysis on the improved code
            new_issues = self._analyze_code(improved_code, language)

            # Calculate generation time
            generation_time_ms = int((time.time() - start_time) * 1000)

            return {
                "original_code": code,
                "improved_code": improved_code,
                "fixed_issues": issues,
                "remaining_issues": new_issues,
                "generation_time_ms": generation_time_ms,
            }

        except Exception as e:
            logger.error(f"Code improvement error: {str(e)}")
            raise RuntimeError(f"Failed to improve code: {str(e)}")

    def _generate_with_gpt4all(self, prompt: str, language: str) -> str:
        """
        Generate code using GPT4All model

        This is a placeholder implementation. In a real implementation, you would:
        1. Load the GPT4All model
        2. Generate a completion
        3. Return the generated code

        For now, we'll simulate a response with a placeholder
        """
        # In a real implementation, uncomment and adapt this code:
        # from gpt4all import GPT4All
        # model_path = self.models["gpt4all"]["path"]
        # model = GPT4All(model_path)
        # response = model.generate(prompt, max_tokens=1024, temp=0.7)
        # return response

        # Placeholder implementation
        if language == "python":
            return '''def fibonacci(n):
    """Calculate the Fibonacci sequence up to n terms."""
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result'''
        elif language == "javascript":
            return """function fibonacci(n) {
    // Calculate the Fibonacci sequence up to n terms
    let a = 0, b = 1;
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(a);
        [a, b] = [b, a + b];
    }
    return result;
}"""
        else:
            return f"// Placeholder code for {language}\nconsole.log('Hello world');"

    def _generate_with_deepseek(self, prompt: str, language: str) -> str:
        """
        Generate code using DeepSeek Coder model

        This is a placeholder implementation. In a real implementation, you would:
        1. Load the DeepSeek model
        2. Generate a completion
        3. Return the generated code
        """
        # In a real implementation, you would use the appropriate library
        # Similar to GPT4All implementation but with DeepSeek-specific code

        # Placeholder implementation
        if language == "python":
            return '''def fibonacci(n: int) -> list:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    
    sequence = [0]
    if n > 1:
        sequence.append(1)
        
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    
    return sequence

# Example usage
if __name__ == "__main__":
    print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]'''
        else:
            return f"// DeepSeek placeholder for {language}\nconsole.log('Generated by DeepSeek');"

    def _clean_generated_code(self, code: str) -> str:
        """
        Clean up generated code - remove unnecessary markers or prefixes
        """
        # Remove common prefixes/suffixes that models might add
        prefixes_to_remove = [
            "```python",
            "```javascript",
            "```java",
            "```csharp",
            "```typescript",
            "```",
            "Here's",
            "Here is",
            "```bash",
            "```shell",
        ]

        suffixes_to_remove = ["```"]

        result = code.strip()

        # Remove prefixes
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix) :].strip()

        # Remove suffixes
        for suffix in suffixes_to_remove:
            if result.endswith(suffix):
                result = result[: -len(suffix)].strip()

        return result

    def _extract_code_block(self, text: str, language: str) -> str:
        """
        Extract code block from text that may contain markdown formatting
        """
        # Look for code blocks with language marker
        import re

        pattern = f"```(?:{language})?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code block found, return the original text
        return text.strip()

    def _analyze_code(self, code: str, language: str) -> List[str]:
        """
        Analyze code for potential issues using static analysis

        This is a placeholder. In a real implementation, you would:
        1. For Python, use tools like pylint, flake8, or a custom semgrep rule
        2. For JavaScript, use ESLint
        3. For other languages, use appropriate tools

        Returns:
            List of issue descriptions
        """
        # Placeholder implementation
        issues = []

        # Very basic checks
        if language == "python":
            if "import random" in code and "random.seed" not in code:
                issues.append("Random number generation without setting seed")

            if "except:" in code and "except Exception:" not in code:
                issues.append(
                    "Bare except clause - consider catching specific exceptions"
                )

            if "print(" in code and "def " in code:
                issues.append(
                    "Print statements in function body - consider using logging"
                )

        # For demonstration purposes only
        # In a real implementation, you would run actual static analysis tools

        return issues
