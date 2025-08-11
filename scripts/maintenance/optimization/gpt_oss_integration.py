#!/usr/bin/env python3
"""
GPT-OSS Integration for SutazAI
Specialized integration for GPT-OSS models with optimized prompting
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)

class GPTOSSIntegration:
    """Specialized integration for GPT-OSS models"""
    
    def __init__(self):
        self.model_manager = None  # Will be set by the main service
        self.code_templates = {
            "python": {
                "function": "def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}",
                "class": "class {name}:\n    \"\"\"{docstring}\"\"\"\n    \n    def __init__(self{params}):\n        {body}",
                "script": "#!/usr/bin/env python3\n\"\"\"\n{description}\n\"\"\"\n\n{imports}\n\n{body}"
            },
            "javascript": {
                "function": "function {name}({params}) {\n    // {description}\n    {body}\n}",
                "class": "class {name} {\n    // {description}\n    constructor({params}) {\n        {body}\n    }\n}",
                "module": "// {description}\n\n{imports}\n\n{body}"
            },
            "java": {
                "method": "public {return_type} {name}({params}) {\n    // {description}\n    {body}\n}",
                "class": "public class {name} {\n    // {description}\n    \n    public {name}({params}) {\n        {body}\n    }\n}"
            }
        }
        
        self.optimization_patterns = {
            "python": [
                {
                    "pattern": r"for i in range\(len\((.+?)\)\):",
                    "replacement": r"for i, item in enumerate(\1):",
                    "description": "Use enumerate instead of range(len())"
                },
                {
                    "pattern": r"\.append\(",
                    "check": "list_comprehension",
                    "description": "Consider list comprehension for better performance"
                }
            ]
        }
    
    async def initialize(self):
        """Initialize GPT-OSS integration"""
        logger.info("Initializing GPT-OSS integration...")
        # Pre-load GPT-OSS models if available
        logger.info("GPT-OSS integration initialized")
    
    async def generate_code(self, prompt: str, language: str = "python") -> str:
        """Generate code using GPT-OSS with optimized prompting"""
        
        # Enhance prompt for better code generation
        enhanced_prompt = self._enhance_code_prompt(prompt, language)
        
        # Use the model manager to generate (will be injected)
        if self.model_manager:
            model_name = "tinyllama"  # Use unified model
            if model_name not in self.model_manager.loaded_models:
                await self.model_manager.load_model(model_name)
            
            generated = await self.model_manager.generate(
                model_name, enhanced_prompt, 
                max_tokens=1024, temperature=0.2
            )
        else:
            # Fallback mock generation
            generated = self._mock_generate_code(prompt, language)
        
        # Post-process the generated code
        cleaned_code = self._clean_generated_code(generated, language)
        
        return cleaned_code
    
    async def complete_code(self, partial_code: str, language: str = "python") -> str:
        """Complete partial code using GPT-OSS"""
        
        completion_prompt = self._create_completion_prompt(partial_code, language)
        
        if self.model_manager:
            model_name = "tinyllama"
            if model_name not in self.model_manager.loaded_models:
                await self.model_manager.load_model(model_name)
            
            completed = await self.model_manager.generate(
                model_name, completion_prompt,
                max_tokens=512, temperature=0.1
            )
        else:
            completed = self._mock_complete_code(partial_code, language)
        
        # Merge with original code
        full_code = self._merge_completion(partial_code, completed)
        
        return full_code
    
    async def explain_code(self, code: str, language: str = "python") -> str:
        """Explain code using GPT-OSS's understanding capabilities"""
        
        explanation_prompt = self._create_explanation_prompt(code, language)
        
        if self.model_manager:
            model_name = "tinyllama"
            if model_name not in self.model_manager.loaded_models:
                await self.model_manager.load_model(model_name)
            
            explanation = await self.model_manager.generate(
                model_name, explanation_prompt,
                max_tokens=256, temperature=0.3
            )
        else:
            explanation = self._mock_explain_code(code, language)
        
        return explanation.strip()
    
    async def optimize_code(self, code: str, language: str = "python") -> str:
        """Optimize code using GPT-OSS and pattern matching"""
        
        # First, apply pattern-based optimizations
        optimized = self._apply_pattern_optimizations(code, language)
        
        # Then use AI for further optimization
        optimization_prompt = self._create_optimization_prompt(optimized, language)
        
        if self.model_manager:
            model_name = "tinyllama"
            if model_name not in self.model_manager.loaded_models:
                await self.model_manager.load_model(model_name)
            
            ai_optimized = await self.model_manager.generate(
                model_name, optimization_prompt,
                max_tokens=1024, temperature=0.2
            )
            
            optimized = self._clean_generated_code(ai_optimized, language)
        
        return optimized
    
    def _enhance_code_prompt(self, prompt: str, language: str) -> str:
        """Enhance the prompt for better code generation"""
        
        enhanced = f"""Generate high-quality {language} code for the following requirement:

Requirement: {prompt}

Please provide:
1. Clean, readable code
2. Proper error handling
3. Meaningful variable names
4. Appropriate comments
5. Follow {language} best practices

Code:
```{language}
"""
        return enhanced
    
    def _create_completion_prompt(self, partial_code: str, language: str) -> str:
        """Create a prompt for code completion"""
        
        prompt = f"""Complete the following {language} code. Continue from where it left off:

```{language}
{partial_code}"""
        
        return prompt
    
    def _create_explanation_prompt(self, code: str, language: str) -> str:
        """Create a prompt for code explanation"""
        
        prompt = f"""Explain what this {language} code does. Be clear and concise:

```{language}
{code}
```

Explanation:"""
        
        return prompt
    
    def _create_optimization_prompt(self, code: str, language: str) -> str:
        """Create a prompt for code optimization"""
        
        prompt = f"""Optimize this {language} code for better performance, readability, and maintainability:

```{language}
{code}
```

Optimized code:
```{language}
"""
        
        return prompt
    
    def _clean_generated_code(self, generated: str, language: str) -> str:
        """Clean up generated code"""
        
        # Remove markdown code blocks if present
        if "```" in generated:
            parts = generated.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Inside code block
                    # Remove language identifier
                    lines = part.strip().split('\n')
                    if lines[0].strip().lower() == language.lower():
                        generated = '\n'.join(lines[1:])
                    else:
                        generated = part.strip()
                    break
        
        # Remove common unwanted prefixes/suffixes
        generated = generated.strip()
        
        # Remove duplicate imports (for Python)
        if language.lower() == "python":
            generated = self._deduplicate_imports(generated)
        
        return generated
    
    def _deduplicate_imports(self, code: str) -> str:
        """Remove duplicate import statements in Python code"""
        lines = code.split('\n')
        seen_imports = set()
        cleaned_lines = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                if line.strip() not in seen_imports:
                    seen_imports.add(line.strip())
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _apply_pattern_optimizations(self, code: str, language: str) -> str:
        """Apply pattern-based optimizations"""
        
        if language.lower() not in self.optimization_patterns:
            return code
        
        optimized = code
        patterns = self.optimization_patterns[language.lower()]
        
        for pattern_config in patterns:
            if "pattern" in pattern_config and "replacement" in pattern_config:
                optimized = re.sub(
                    pattern_config["pattern"], 
                    pattern_config["replacement"], 
                    optimized
                )
        
        return optimized
    
    def _merge_completion(self, original: str, completion: str) -> str:
        """Merge original code with completion"""
        
        # Simple merge - in production, this would be more sophisticated
        if completion.startswith(original):
            return completion
        else:
            return original + "\n" + completion
    
    def _mock_generate_code(self, prompt: str, language: str) -> str:
        """Mock code generation when model is not available"""
        
        if language.lower() == "python":
            return f'''def generated_function():
    """
    Generated function for: {prompt}
    """
    # Implementation would go here
    result = []
    for i in range(10):
        result.append(i * 2)
    return result

# Usage example
if __name__ == "__main__":
    print(generated_function())'''
        
        elif language.lower() == "javascript":
            return f'''function generatedFunction() {{
    // Generated function for: {prompt}
    const result = [];
    for (let i = 0; i < 10; i++) {{
        result.push(i * 2);
    }}
    return result;
}}

// Usage example
console.log(generatedFunction());'''
        
        else:
            return f'// Generated {language} code for: {prompt}\n// Implementation would go here'
    
    def _mock_complete_code(self, partial: str, language: str) -> str:
        """Mock code completion when model is not available"""
        return partial + "\n    # Completion would continue here\n    pass"
    
    def _mock_explain_code(self, code: str, language: str) -> str:
        """Mock code explanation when model is not available"""
        lines = len(code.split('\n'))
        functions = len(re.findall(r'def\s+\w+', code)) if language == "python" else 0
        
        return f"This {language} code contains {lines} lines and defines {functions} function(s). It appears to implement functionality related to the given requirements."