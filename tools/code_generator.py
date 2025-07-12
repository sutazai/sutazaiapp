import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
from abc import ABC, abstractmethod

from models import model_manager
from config import config
from memory import vector_memory

logger = logging.getLogger(__name__)

@dataclass
class CodeGenerationRequest:
    """Request structure for code generation."""
    prompt: str
    language: str = "python"
    style: str = "default"
    complexity: str = "medium"
    include_tests: bool = False
    include_docs: bool = True
    generation_id: Optional[str] = None
    template_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    customizations: Optional[Dict[str, Any]] = None

@dataclass
class CodeGenerationResult:
    """Result structure for code generation."""
    generation_id: str
    generated_code: str
    language: str
    style: str
    metadata: Dict[str, Any]
    quality_score: float
    suggestions: List[str]
    tests: Optional[str] = None
    documentation: Optional[str] = None
    errors: List[str] = None

class CodeTemplate:
    """Represents a code template."""
    
    def __init__(self, name: str, language: str, category: str, template_code: str, parameters: List[str]):
        self.name = name
        self.language = language
        self.category = category
        self.template_code = template_code
        self.parameters = parameters
        self.created_at = datetime.utcnow()
    
    def render(self, parameters: Dict[str, Any]) -> str:
        """Render template with provided parameters."""
        rendered = self.template_code
        for param, value in parameters.items():
            placeholder = f"{{{param}}}"
            rendered = rendered.replace(placeholder, str(value))
        return rendered

class CodeTemplateManager:
    """Manages code templates."""
    
    def __init__(self):
        self.templates: Dict[str, CodeTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default code templates."""
        # Python templates
        self.templates["python_function"] = CodeTemplate(
            name="python_function",
            language="python",
            category="function",
            template_code="""def {function_name}({parameters}):
    \"\"\"
    {description}
    
    Args:
{args_docs}
    
    Returns:
        {return_type}: {return_description}
    \"\"\"
    {body}
    return {return_value}""",
            parameters=["function_name", "parameters", "description", "args_docs", "return_type", "return_description", "body", "return_value"]
        )
        
        self.templates["python_class"] = CodeTemplate(
            name="python_class",
            language="python",
            category="class",
            template_code="""class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_parameters}):
        \"\"\"
        Initialize {class_name}.
        \"\"\"
{init_body}
{methods}""",
            parameters=["class_name", "description", "init_parameters", "init_body", "methods"]
        )
        
        # JavaScript templates
        self.templates["js_function"] = CodeTemplate(
            name="js_function",
            language="javascript",
            category="function",
            template_code="""/**
 * {description}
 * @param {{{param_types}}} {param_names}
 * @returns {{{return_type}}} {return_description}
 */
function {function_name}({parameters}) {
{body}
    return {return_value};
}""",
            parameters=["function_name", "parameters", "description", "param_types", "param_names", "return_type", "return_description", "body", "return_value"]
        )
        
        # Add more templates as needed
    
    def get_template(self, name: str) -> Optional[CodeTemplate]:
        """Get template by name."""
        return self.templates.get(name)
    
    def list_templates(self, language: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available templates with optional filtering."""
        templates = []
        for template in self.templates.values():
            if language and template.language != language:
                continue
            if category and template.category != category:
                continue
            
            templates.append({
                "name": template.name,
                "language": template.language,
                "category": template.category,
                "parameters": template.parameters,
                "created_at": template.created_at.isoformat()
            })
        return templates
    
    def add_template(self, template: CodeTemplate):
        """Add a new template."""
        self.templates[template.name] = template

class CodeStyleManager:
    """Manages code styles and formatting rules."""
    
    def __init__(self):
        self.styles = {
            "python": {
                "default": {
                    "indentation": "    ",  # 4 spaces
                    "line_length": 88,
                    "quote_style": "double",
                    "naming_convention": "snake_case"
                },
                "pep8": {
                    "indentation": "    ",
                    "line_length": 79,
                    "quote_style": "double",
                    "naming_convention": "snake_case"
                },
                "google": {
                    "indentation": "    ",
                    "line_length": 80,
                    "quote_style": "double",
                    "naming_convention": "snake_case"
                }
            },
            "javascript": {
                "default": {
                    "indentation": "  ",  # 2 spaces
                    "line_length": 80,
                    "quote_style": "single",
                    "naming_convention": "camelCase"
                },
                "airbnb": {
                    "indentation": "  ",
                    "line_length": 100,
                    "quote_style": "single",
                    "naming_convention": "camelCase"
                }
            }
        }
    
    def get_style(self, language: str, style_name: str = "default") -> Dict[str, Any]:
        """Get style configuration for a language."""
        return self.styles.get(language, {}).get(style_name, {})
    
    def list_styles(self, language: Optional[str] = None) -> Dict[str, Any]:
        """List available styles."""
        if language:
            return {language: list(self.styles.get(language, {}).keys())}
        return {lang: list(styles.keys()) for lang, styles in self.styles.items()}

class IntelligentCodeGenerator:
    """Main code generation engine using AI models."""
    
    def __init__(self):
        self.template_manager = CodeTemplateManager()
        self.style_manager = CodeStyleManager()
        self.generation_history: Dict[str, Dict[str, Any]] = {}
    
    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        style: str = "default",
        complexity: str = "medium",
        include_tests: bool = False,
        include_docs: bool = True,
        generation_id: Optional[str] = None
    ) -> CodeGenerationResult:
        """Generate code based on prompt and specifications."""
        try:
            if not generation_id:
                from uuid import uuid4
                generation_id = str(uuid4())
            
            # Get style configuration
            style_config = self.style_manager.get_style(language, style)
            
            # Build comprehensive prompt
            system_prompt = self._build_system_prompt(language, style, complexity, include_docs, include_tests)
            full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
            
            # Generate code using AI model
            model_name = self._select_model_for_language(language)
            generation_result = await model_manager.generate_text(
                model_name=model_name,
                prompt=full_prompt,
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more consistent code
            )
            
            generated_code = generation_result.get("text", "")
            
            # Clean and format generated code
            cleaned_code = self._clean_generated_code(generated_code, language)
            
            # Generate tests if requested
            tests = None
            if include_tests:
                tests = await self._generate_tests(cleaned_code, language, prompt)
            
            # Generate documentation if requested
            documentation = None
            if include_docs:
                documentation = await self._generate_documentation(cleaned_code, language, prompt)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(cleaned_code, language)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(cleaned_code, language, quality_score)
            
            result = CodeGenerationResult(
                generation_id=generation_id,
                generated_code=cleaned_code,
                language=language,
                style=style,
                metadata={
                    "prompt": prompt,
                    "model_used": model_name,
                    "tokens_used": generation_result.get("tokens_used", 0),
                    "generation_time": generation_result.get("generation_time", 0),
                    "style_config": style_config,
                    "complexity": complexity
                },
                quality_score=quality_score,
                suggestions=suggestions,
                tests=tests,
                documentation=documentation
            )
            
            # Store in history
            self.generation_history[generation_id] = {
                "result": result,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return CodeGenerationResult(
                generation_id=generation_id or "error",
                generated_code="// Error generating code",
                language=language,
                style=style,
                metadata={"error": str(e)},
                quality_score=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def refactor_code(
        self,
        code: str,
        language: str = "python",
        goals: List[str] = None,
        preserve_functionality: bool = True,
        refactor_id: Optional[str] = None
    ) -> CodeGenerationResult:
        """Refactor existing code."""
        if goals is None:
            goals = ["readability", "performance"]
        
        try:
            if not refactor_id:
                from uuid import uuid4
                refactor_id = str(uuid4())
            
            # Build refactoring prompt
            goals_str = ", ".join(goals)
            preserve_note = "IMPORTANT: Preserve the original functionality exactly." if preserve_functionality else ""
            
            prompt = f"""Refactor the following {language} code to improve: {goals_str}.
{preserve_note}

Original code:
```{language}
{code}
```

Provide only the refactored code with improvements:"""
            
            model_name = self._select_model_for_language(language)
            generation_result = await model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            refactored_code = self._clean_generated_code(generation_result.get("text", ""), language)
            quality_score = self._calculate_quality_score(refactored_code, language)
            suggestions = self._generate_suggestions(refactored_code, language, quality_score)
            
            result = CodeGenerationResult(
                generation_id=refactor_id,
                generated_code=refactored_code,
                language=language,
                style="refactored",
                metadata={
                    "original_code": code,
                    "refactor_goals": goals,
                    "preserve_functionality": preserve_functionality,
                    "model_used": model_name
                },
                quality_score=quality_score,
                suggestions=suggestions
            )
            
            self.generation_history[refactor_id] = {
                "result": result,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return CodeGenerationResult(
                generation_id=refactor_id or "error",
                generated_code=code,  # Return original on error
                language=language,
                style="error",
                metadata={"error": str(e)},
                quality_score=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def generate_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        language: str = "python",
        customizations: Dict[str, Any] = None,
        generation_id: Optional[str] = None
    ) -> CodeGenerationResult:
        """Generate code from a template."""
        try:
            if not generation_id:
                from uuid import uuid4
                generation_id = str(uuid4())
            
            template = self.template_manager.get_template(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")
            
            # Render template with parameters
            base_code = template.render(parameters)
            
            # Apply customizations if provided
            if customizations:
                # Use AI to apply customizations
                customization_prompt = f"""Customize the following {language} code based on these requirements:
{customizations}

Base code:
```{language}
{base_code}
```

Provide the customized code:"""
                
                model_name = self._select_model_for_language(language)
                generation_result = await model_manager.generate_text(
                    model_name=model_name,
                    prompt=customization_prompt,
                    max_tokens=2000,
                    temperature=0.3
                )
                
                customized_code = self._clean_generated_code(generation_result.get("text", ""), language)
            else:
                customized_code = base_code
            
            quality_score = self._calculate_quality_score(customized_code, language)
            suggestions = self._generate_suggestions(customized_code, language, quality_score)
            
            result = CodeGenerationResult(
                generation_id=generation_id,
                generated_code=customized_code,
                language=language,
                style="template",
                metadata={
                    "template_name": template_name,
                    "parameters": parameters,
                    "customizations": customizations
                },
                quality_score=quality_score,
                suggestions=suggestions
            )
            
            self.generation_history[generation_id] = {
                "result": result,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating from template: {e}")
            return CodeGenerationResult(
                generation_id=generation_id or "error",
                generated_code="// Error generating from template",
                language=language,
                style="error",
                metadata={"error": str(e)},
                quality_score=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def optimize_code(
        self,
        code: str,
        language: str = "python",
        optimization_type: str = "performance",
        target_metrics: Dict[str, Any] = None,
        optimization_id: Optional[str] = None
    ) -> CodeGenerationResult:
        """Optimize code for specific criteria."""
        try:
            if not optimization_id:
                from uuid import uuid4
                optimization_id = str(uuid4())
            
            # Build optimization prompt
            metrics_str = ""
            if target_metrics:
                metrics_str = f"Target metrics: {target_metrics}"
            
            prompt = f"""Optimize the following {language} code for {optimization_type}.
{metrics_str}

Original code:
```{language}
{code}
```

Provide optimized code with explanatory comments:"""
            
            model_name = self._select_model_for_language(language)
            generation_result = await model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            optimized_code = self._clean_generated_code(generation_result.get("text", ""), language)
            quality_score = self._calculate_quality_score(optimized_code, language)
            suggestions = self._generate_suggestions(optimized_code, language, quality_score)
            
            result = CodeGenerationResult(
                generation_id=optimization_id,
                generated_code=optimized_code,
                language=language,
                style="optimized",
                metadata={
                    "original_code": code,
                    "optimization_type": optimization_type,
                    "target_metrics": target_metrics,
                    "model_used": model_name
                },
                quality_score=quality_score,
                suggestions=suggestions
            )
            
            self.generation_history[optimization_id] = {
                "result": result,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing code: {e}")
            return CodeGenerationResult(
                generation_id=optimization_id or "error",
                generated_code=code,
                language=language,
                style="error",
                metadata={"error": str(e)},
                quality_score=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def fix_code(
        self,
        code: str,
        language: str = "python",
        error_message: str = "",
        issue_description: str = "",
        fix_approach: str = "conservative",
        fix_id: Optional[str] = None
    ) -> CodeGenerationResult:
        """Fix code issues and bugs."""
        try:
            if not fix_id:
                from uuid import uuid4
                fix_id = str(uuid4())
            
            # Build fix prompt
            context = ""
            if error_message:
                context += f"Error message: {error_message}\n"
            if issue_description:
                context += f"Issue description: {issue_description}\n"
            
            approach_note = "Make minimal changes to fix the issue." if fix_approach == "conservative" else "Make comprehensive improvements while fixing the issue."
            
            prompt = f"""Fix the issues in the following {language} code.
{context}
Approach: {approach_note}

Code to fix:
```{language}
{code}
```

Provide the fixed code with explanatory comments:"""
            
            model_name = self._select_model_for_language(language)
            generation_result = await model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            fixed_code = self._clean_generated_code(generation_result.get("text", ""), language)
            quality_score = self._calculate_quality_score(fixed_code, language)
            suggestions = self._generate_suggestions(fixed_code, language, quality_score)
            
            result = CodeGenerationResult(
                generation_id=fix_id,
                generated_code=fixed_code,
                language=language,
                style="fixed",
                metadata={
                    "original_code": code,
                    "error_message": error_message,
                    "issue_description": issue_description,
                    "fix_approach": fix_approach,
                    "model_used": model_name
                },
                quality_score=quality_score,
                suggestions=suggestions
            )
            
            self.generation_history[fix_id] = {
                "result": result,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fixing code: {e}")
            return CodeGenerationResult(
                generation_id=fix_id or "error",
                generated_code=code,
                language=language,
                style="error",
                metadata={"error": str(e)},
                quality_score=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    def _build_system_prompt(self, language: str, style: str, complexity: str, include_docs: bool, include_tests: bool) -> str:
        """Build system prompt for code generation."""
        prompt_parts = [
            f"You are an expert {language} developer.",
            f"Generate clean, efficient, and well-structured {language} code.",
            f"Follow {style} style conventions.",
            f"Code complexity level: {complexity}."
        ]
        
        if include_docs:
            prompt_parts.append("Include comprehensive documentation and comments.")
        
        if include_tests:
            prompt_parts.append("Include unit tests for the generated code.")
        
        prompt_parts.extend([
            "Ensure code is production-ready and follows best practices.",
            "Focus on readability, maintainability, and performance."
        ])
        
        return " ".join(prompt_parts)
    
    def _select_model_for_language(self, language: str) -> str:
        """Select the best model for a given programming language."""
        # Prefer code-specific models for programming tasks
        if "deepseek" in model_manager.loaded_models:
            return "deepseek"
        elif "codellama" in model_manager.loaded_models:
            return "codellama"
        elif "llama2" in model_manager.loaded_models:
            return "llama2"
        else:
            # Return first available model
            available_models = list(model_manager.loaded_models.keys())
            return available_models[0] if available_models else "default"
    
    def _clean_generated_code(self, code: str, language: str) -> str:
        """Clean and format generated code."""
        # Remove markdown code blocks if present
        code = re.sub(rf'```{language}\n', '', code)
        code = re.sub(r'```\n?', '', code)
        code = re.sub(r'^```', '', code, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Ensure proper line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        return code
    
    async def _generate_tests(self, code: str, language: str, prompt: str) -> str:
        """Generate tests for the given code."""
        try:
            test_prompt = f"""Generate comprehensive unit tests for the following {language} code.
Original request: {prompt}

Code to test:
```{language}
{code}
```

Generate tests using appropriate testing framework:"""
            
            model_name = self._select_model_for_language(language)
            result = await model_manager.generate_text(
                model_name=model_name,
                prompt=test_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            return self._clean_generated_code(result.get("text", ""), language)
        except Exception as e:
            logger.warning(f"Error generating tests: {e}")
            return f"# Error generating tests: {str(e)}"
    
    async def _generate_documentation(self, code: str, language: str, prompt: str) -> str:
        """Generate documentation for the given code."""
        try:
            doc_prompt = f"""Generate comprehensive documentation for the following {language} code.
Original request: {prompt}

Code to document:
```{language}
{code}
```

Generate documentation including usage examples:"""
            
            model_name = self._select_model_for_language(language)
            result = await model_manager.generate_text(
                model_name=model_name,
                prompt=doc_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return result.get("text", "").strip()
        except Exception as e:
            logger.warning(f"Error generating documentation: {e}")
            return f"Error generating documentation: {str(e)}"
    
    def _calculate_quality_score(self, code: str, language: str) -> float:
        """Calculate a quality score for the generated code."""
        try:
            score = 0.0
            max_score = 100.0
            
            # Basic checks
            if code.strip():
                score += 20  # Non-empty code
            
            # Length appropriateness (not too short, not too long)
            lines = code.split('\n')
            if 5 <= len(lines) <= 100:
                score += 20
            elif len(lines) > 0:
                score += 10
            
            # Language-specific checks
            if language == "python":
                if "def " in code or "class " in code:
                    score += 20  # Has functions or classes
                if '"""' in code or "'''" in code:
                    score += 15  # Has docstrings
                if "import " in code or "from " in code:
                    score += 10  # Has imports
            elif language == "javascript":
                if "function " in code or "=>" in code:
                    score += 20  # Has functions
                if "/**" in code:
                    score += 15  # Has JSDoc comments
            
            # Indentation consistency
            if self._has_consistent_indentation(code):
                score += 15
            
            return min(score / max_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.5  # Default score
    
    def _has_consistent_indentation(self, code: str) -> bool:
        """Check if code has consistent indentation."""
        lines = [line for line in code.split('\n') if line.strip()]
        if len(lines) < 2:
            return True
        
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.append(indent)
        
        if not indents:
            return True
        
        # Check if indentation follows a pattern (multiples of 2 or 4)
        return all(indent % 2 == 0 or indent % 4 == 0 for indent in indents)
    
    def _generate_suggestions(self, code: str, language: str, quality_score: float) -> List[str]:
        """Generate improvement suggestions for the code."""
        suggestions = []
        
        if quality_score < 0.7:
            suggestions.append("Consider adding more comprehensive error handling")
            suggestions.append("Add more detailed documentation and comments")
        
        if quality_score < 0.5:
            suggestions.append("Improve code structure and organization")
            suggestions.append("Consider breaking down complex functions into smaller ones")
        
        # Language-specific suggestions
        if language == "python":
            if "def " not in code and "class " not in code:
                suggestions.append("Consider organizing code into functions or classes")
            if '"""' not in code and "'''" not in code:
                suggestions.append("Add docstrings to functions and classes")
        
        elif language == "javascript":
            if "const " not in code and "let " not in code:
                suggestions.append("Use const or let instead of var for variable declarations")
            if "/**" not in code:
                suggestions.append("Add JSDoc comments for better documentation")
        
        return suggestions
    
    async def list_templates(self, language: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available templates."""
        return self.template_manager.list_templates(language, category)
    
    async def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get template details."""
        template = self.template_manager.get_template(template_name)
        if template:
            return {
                "name": template.name,
                "language": template.language,
                "category": template.category,
                "parameters": template.parameters,
                "template_code": template.template_code,
                "created_at": template.created_at.isoformat()
            }
        return None
    
    async def get_code_styles(self, language: Optional[str] = None) -> Dict[str, Any]:
        """Get available code styles."""
        return self.style_manager.list_styles(language)
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return ["python", "javascript", "typescript", "java", "cpp", "c", "csharp", "go", "rust", "php", "ruby"]
    
    async def get_generation_history(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get generation history for a specific ID."""
        return self.generation_history.get(generation_id)

# Global instance
code_generator = IntelligentCodeGenerator()

# Convenience functions for backward compatibility
async def generate_code(**kwargs) -> CodeGenerationResult:
    """Generate code using the global code generator instance."""
    return await code_generator.generate_code(**kwargs)

async def refactor_code(**kwargs) -> CodeGenerationResult:
    """Refactor code using the global code generator instance."""
    return await code_generator.refactor_code(**kwargs)

async def generate_from_template(**kwargs) -> CodeGenerationResult:
    """Generate from template using the global code generator instance."""
    return await code_generator.generate_from_template(**kwargs)

async def optimize_code(**kwargs) -> CodeGenerationResult:
    """Optimize code using the global code generator instance."""
    return await code_generator.optimize_code(**kwargs)

async def fix_code(**kwargs) -> CodeGenerationResult:
    """Fix code using the global code generator instance."""
    return await code_generator.fix_code(**kwargs)

async def list_templates(**kwargs) -> List[Dict[str, Any]]:
    """List templates using the global code generator instance."""
    return await code_generator.list_templates(**kwargs)

async def get_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get template using the global code generator instance."""
    return await code_generator.get_template(template_name)

async def get_code_styles(**kwargs) -> Dict[str, Any]:
    """Get code styles using the global code generator instance."""
    return await code_generator.get_code_styles(**kwargs)

async def get_supported_languages() -> List[str]:
    """Get supported languages using the global code generator instance."""
    return await code_generator.get_supported_languages()

async def get_generation_history(generation_id: str) -> Optional[Dict[str, Any]]:
    """Get generation history using the global code generator instance."""
    return await code_generator.get_generation_history(generation_id)
