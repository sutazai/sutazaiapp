import os
import json
import logging
from typing import Dict, Any, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import semgrep


class CodeGenerationService:
    """
    Offline code generation service with security scanning
    """

    def __init__(
        self,
        models_dir: str = "/opt/sutazaiapp/model_management",
        output_dir: str = "/opt/sutazaiapp/generated_code",
        max_tokens: int = 2048,
    ):
        """
        Initialize CodeGenerationService

        Args:
            models_dir (str): Directory containing local LLM models
            output_dir (str): Directory to save generated code
            max_tokens (int): Maximum tokens for code generation
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.max_tokens = max_tokens

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            filename=os.path.join(output_dir, "code_generation.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )

        # Load available models
        self.available_models = self._discover_models()

    def _discover_models(self) -> Dict[str, str]:
        """
        Discover available local LLM models

        Returns:
            Dict of model names and paths
        """
        models = {}
        for model_name in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.isdir(model_path):
                models[model_name] = model_path
        return models

    def _load_model(self, model_name: str):
        """
        Load a specific model

        Args:
            model_name (str): Name of the model to load

        Returns:
            Tuple of model and tokenizer
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not found")

        model_path = self.available_models[model_name]

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)

            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def generate_code(
        self, specification: str, model_name: str = "deepseek-coder", language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code from a specification

        Args:
            specification (str): Code generation specification
            model_name (str): Name of the model to use
            language (str): Programming language

        Returns:
            Dict containing generated code and security analysis
        """
        try:
            # Load model
            model, tokenizer = self._load_model(model_name)

            # Prepare prompt
            prompt = f"""
            Generate {language} code based on the following specification:
            {specification}
            
            Ensure the code is:
            - Clean and readable
            - Follows best practices
            - Includes necessary imports and comments
            """

            # Generate code
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(inputs.input_ids, max_length=self.max_tokens, num_return_sequences=1)

            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Security scanning with Semgrep
            security_results = self._scan_code(generated_code, language)

            # Save generated code
            output_file = os.path.join(self.output_dir, f"generated_{language}_{model_name}_{hash(specification)}.py")

            with open(output_file, "w") as f:
                f.write(generated_code)

            logger.info(f"Code generated successfully using {model_name}")

            return {
                "code": generated_code,
                "security_warnings": security_results,
                "model": model_name,
                "output_file": output_file,
            }

        except Exception as e:
            logger.exception(f"Code generation error: {e}")
            return {"error": str(e), "code": None, "security_warnings": []}

    def _scan_code(self, code: str, language: str) -> List[Dict[str, Any]]:
        """
        Scan generated code for security vulnerabilities

        Args:
            code (str): Generated code to scan
            language (str): Programming language

        Returns:
            List of security warnings
        """
        try:
            # Temporary file for Semgrep scanning
            temp_file = os.path.join(self.output_dir, f"temp_scan_{hash(code)}.{language}")

            with open(temp_file, "w") as f:
                f.write(code)

            # Run Semgrep scan
            semgrep_config = os.path.join(self.models_dir, "semgrep_security_rules.yaml")

            results = semgrep.scan(temp_file, config=semgrep_config, output_format="json")

            # Clean up temporary file
            os.remove(temp_file)

            return results.get("results", [])

        except Exception as e:
            logger.error(f"Security scanning error: {e}")
            return [{"error": str(e)}]


def main():
    """
    Example usage and testing
    """
    code_gen = CodeGenerationService()

    # Example code generation
    spec = "Create a function to parse CSV files with error handling"
    result = code_gen.generate_code(spec)

    print("Generated Code:")
    print(result["code"])

    print("\nSecurity Warnings:")
    print(json.dumps(result["security_warnings"], indent=2))


if __name__ == "__main__":
    main()
