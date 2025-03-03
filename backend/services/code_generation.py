#!/usr/bin/env python3.11"""Code Generation ServiceThis module provides a secure code generation service using local LLMs withsecurity scanning capabilities."""import jsonimport loggingfrom pathlib import Pathfrom typing import Any, Dict, List, Optional, Tupleimport semgrepfrom loguru import loggerfrom transformers import (AutoModelForCausalLM,AutoTokenizer,PreTrainedModel,PreTrainedTokenizer,)class CodeGenerationService:    """
Offline code generation service with security scanning.
This service provides:
- Local LLM-based code generation
- Security scanning of generated code
- Safe file handling and logging
"""
def __init__(        self,
    models_dir: str = "/opt/sutazaiapp/model_management",
    output_dir: str = "/opt/sutazaiapp/generated_code",
    max_tokens: int = 2048,
    ):
    """
    Initialize CodeGenerationService.
    Args:
    models_dir: Directory containing local LLM models
    output_dir: Directory to save generated code
    max_tokens: Maximum tokens for code generation
    """
    self.models_dir = Path(models_dir)
    self.output_dir = Path(output_dir)
    self.max_tokens = max_tokens
    # Ensure output directory exists
    self.output_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    logging.basicConfig(
    filename=self.output_dir / "code_generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    )
    # Load available models
    self.available_models = self._discover_models()
    def _discover_models(self) -> Dict[str, Path]:            """
        Discover available local LLM models.
        Returns:
        Dict of model names and paths
        """
        models = {}
        for model_path in self.models_dir.iterdir():
        if model_path.is_dir():
            models[model_path.name] = model_path
            return models
        def _load_model(                self, model_name: str,
            ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
            """
            Load a specific model.
            Args:
            model_name: Name of the model to load
            Returns:
            Tuple of model and tokenizer
            Raises:
            ValueError: If model is not found
            Exception: If model loading fails
            """
            if model_name not in self.available_models:
                raise ValueError(f"Model {model_name} not found")
                model_path = self.available_models[model_name]
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModelForCausalLM.from_pretrained(str(model_path))
                    return model, tokenizer
                except Exception as e:                        logger.error(f"Failed to load model {model_name}: {e}")                        raise                        def generate_code(                            self,
                    specification: str,
                    model_name: str = "deepseek-coder",
                    language: str = "python",
                    ) -> Dict[str, Any]:
                    """
                    Generate code from a specification.
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
                        inputs.input_ids,
                        max_length=self.max_tokens,
                        num_return_sequences=1,
                        )
                        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Security scanning with Semgrep
                        security_results = self._scan_code(generated_code, language)
                        # Save generated code
                        output_file = (
                        self.output_dir
                        / f"generated_{language}_{model_name}_{hash(specification)}.py"
                        )
                        with open(output_file, "w") as f:
                        f.write(generated_code)
                        logger.info(f"Code generated successfully using {model_name}")
                        return {
                    "code": generated_code,                            "security_warnings": security_results,                            "model": model_name,                            "output_file": output_file,                            "error": None,                            }                            except Exception as e:                                logger.exception(f"Code generation error: {e}")                                return {                            "code": None,                            "security_warnings": [],                            "model": model_name,                            "output_file": None,                            "error": str(e),                            }                            def _scan_code(self, code: str, language: str) -> List[Dict[str, Any]]:                                """
                    Scan generated code for security vulnerabilities.
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
                        try:
                            # Run Semgrep scan
                            results = semgrep.scan(
                            str(temp_file),
                            config=str(semgrep_config),
                            output_format="json",
                            )
                            return results.get("results", [])
                        finally:                                        # Ensure temp file is removed even if scan fails                                        if temp_file.exists():                                            temp_file.unlink()                                            except Exception as e:                                                logger.error(f"Security scanning error: {e}")                                                return [{"error": str(e)}]                                            def main() -> None:                                                """Example usage and testing."""
                            try:
                                code_gen = CodeGenerationService()
                                # Example code generation
                                spec = "Create a function to parse CSV files with error handling"
                                result = code_gen.generate_code(spec)
                                if result["error"]:
                                    logger.error(f"Code generation failed: {result['error']}")
                                    return
                                print("Generated Code:")
                                print(result["code"])
                                print("\nOutput File:")
                                print(f"Code saved to: {result['output_file']}")
                                if result["security_warnings"]:
                                    print("\nSecurity Warnings:")
                                    print(json.dumps(result["security_warnings"], indent=2))
                                    else:
                                        print("\nNo security warnings found.")
                                        except Exception as e:
                                            logger.exception("Main execution failed")
                                            print(f"Error: {e}")
                                            if __name__ == "__main__":
                                                main()

