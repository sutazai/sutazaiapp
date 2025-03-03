#!/usr/bin/env python3.11
"""Code Generation Service

This module provides a secure code generation service using local LLMs with
security scanning capabilities.

This service provides:
- Local LLM-based code generation
- Security scanning of generated code
- Safe file handling and logging
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import semgrep
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class CodeGenerationService:
    """Offline code generation service with security scanning.
    
    This service provides code generation using local LLMs and security scanning
    of the generated code to identify potential vulnerabilities.
    """
    
    def __init__(
        self,
        models_dir: str = "/opt/sutazaiapp/models",
        output_dir: str = "/opt/sutazaiapp/generated_code",
        max_tokens: int = 2000,
    ):
        """Initialize the code generation service.
        
        Args:
            models_dir: Directory containing the local LLM models
            output_dir: Directory for storing generated code
            max_tokens: Maximum number of tokens to generate
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.max_tokens = max_tokens
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load available models
        self.models: Dict[str, Path] = self._discover_models()
        
        logger.info(f"Initialized CodeGenerationService with {len(self.models)} models")
        
        # Configure logger
        logging.basicConfig(
            filename=str(self.output_dir / "code_generation.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
        
    def _discover_models(self) -> Dict[str, Path]:
        """Discover available local LLM models.
        
        Returns:
            Dict of model names and paths
        """
        models: Dict[str, Path] = {}
        
        # Check if models directory exists
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return models
            
        # Discover models
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models[model_dir.name] = model_dir
                logger.info(f"Discovered model: {model_dir.name}")
                
        return models
        
    def _load_model(
        self,
        model_name: str,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of model and tokenizer
            
        Raises:
            ValueError: If model is not found
            Exception: If model loading fails
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
            
        model_path = self.models[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
    def generate_code(
        self,
        specification: str,
        model_name: str = "deepseek-coder",
        language: str = "python",
    ) -> Dict[str, Any]:
        """Generate code from a specification.
        
        Args:
            specification: Code generation specification
            model_name: Name of the model to use
            language: Programming language
            
        Returns:
            Dict containing:
            - code (str | None): Generated code if successful, None if failed
            - security_warnings (List[Dict[str, Any]]): List of security warnings
            - model (str): Name of the model used
            - output_file (Path | None): Path to the output file if successful
            - error (str | None): Error message if failed
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
            outputs = model.generate(
                inputs["input_ids"],
                max_length=self.max_tokens,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
            )
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Security scanning with Semgrep
            security_results = self._scan_code(generated_code, language)
            
            # Save generated code to file
            file_hash = hash(specification) % 10000
            output_file = self.output_dir / f"{language}_code_{file_hash}.{language}"
            with open(output_file, "w") as f:
                f.write(generated_code)
                
            logger.info(f"Code generated successfully: {output_file}")
            
            # Return results
            return {
                "code": generated_code,
                "security_warnings": security_results,
                "model": model_name,
                "output_file": output_file,
                "error": None,
            }
        except Exception as e:
            logger.exception(f"Code generation error: {e}")
            return {
                "code": None,
                "security_warnings": [],
                "model": model_name,
                "output_file": None,
                "error": str(e),
            }
            
    def _scan_code(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Scan generated code for security vulnerabilities.
        
        Args:
            code: Generated code to scan
            language: Programming language
            
        Returns:
            List of security warnings
        """
        try:
            temp_file = self.output_dir / f"temp_scan_{hash(code)}.{language}"
            semgrep_config = self.models_dir / "semgrep_security_rules.yaml"
            
            # Use context manager for temporary file
            with open(temp_file, "w") as f:
                f.write(code)
                
            # Run semgrep scan
            scan_results = semgrep.scan(
                [str(temp_file)],
                config=str(semgrep_config),
                output_format="json",
            )
            
            # Parse semgrep results
            warnings = []
            if scan_results.get("results"):
                for result in scan_results["results"]:
                    warnings.append({
                        "rule_id": result.get("rule_id"),
                        "severity": result.get("severity"),
                        "message": result.get("message"),
                        "line": result.get("start", {}).get("line"),
                    })
                    
            return warnings
        except Exception as e:
            logger.error(f"Security scanning error: {e}")
            return [{"error": str(e)}]
        finally:
            # Ensure temp file is removed even if scan fails
            if temp_file.exists():
                temp_file.unlink()


def main() -> None:
    """Example usage and testing."""
    code_gen = CodeGenerationService()
    
    # Example code generation
    spec = "Create a function to parse CSV files with error handling"
    result = code_gen.generate_code(spec)
    
    if result["error"]:
        logger.error(f"Code generation failed: {result['error']}")
        return
        
    print("Generated Code:")
    print("-----------------------------------")
    print(result["code"])
    print("-----------------------------------")
    
    if result["security_warnings"]:
        print("\nSecurity Warnings:")
        print(json.dumps(result["security_warnings"], indent=2))
    else:
        print("\nNo security warnings found.")
        
    print(f"Output saved to: {result['output_file']}")


if __name__ == "__main__":
    main()

