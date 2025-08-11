"""
Code Generator Agent - Specialized AI Agent for Code Generation
==============================================================

This agent specializes in generating code using local Ollama models.
It demonstrates how to extend the BaseAgent for specific capabilities.
"""

import asyncio
import json
import os
import tempfile
from typing import Dict, Any, List
from datetime import datetime

from agents.core.base_agent import BaseAgent, AgentMessage, AgentStatus, AgentCapability


class CodeGeneratorAgent(BaseAgent):
    """
    Specialized agent for code generation tasks
    
    Capabilities:
    - Generate code from specifications
    - Code completion and suggestions
    - Code refactoring assistance
    - Multi-language support
    """
    
    async def on_initialize(self):
        """Initialize code generator specific components"""
        self.logger.info("Initializing Code Generator Agent")
        
        # Register custom message handlers
        self.register_message_handler("generate_code", self._handle_generate_code)
        self.register_message_handler("complete_code", self._handle_complete_code)
        self.register_message_handler("refactor_code", self._handle_refactor_code)
        self.register_message_handler("explain_code", self._handle_explain_code)
        
        # Specialized capabilities
        self.add_capability(AgentCapability.CODE_GENERATION)
        self.add_capability(AgentCapability.REASONING)
        
        # Code generation settings
        self.supported_languages = [
            "python", "javascript", "typescript", "java", "cpp", "c", 
            "go", "rust", "php", "ruby", "swift", "kotlin", "scala",
            "html", "css", "sql", "bash", "dockerfile"
        ]
        
        self.code_templates = {
            "python": {
                "function": "def {function_name}({parameters}):\n    \"\"\"{docstring}\"\"\"\n    {body}",
                "class": "class {class_name}:\n    \"\"\"{docstring}\"\"\"\n    \n    def __init__(self{parameters}):\n        {body}",
                "api_endpoint": "@app.{method}('{path}')\ndef {function_name}({parameters}):\n    \"\"\"{docstring}\"\"\"\n    {body}"
            },
            "javascript": {
                "function": "function {function_name}({parameters}) {\n    // {docstring}\n    {body}\n}",
                "class": "class {class_name} {\n    // {docstring}\n    constructor({parameters}) {\n        {body}\n    }\n}",
                "async_function": "async function {function_name}({parameters}) {\n    // {docstring}\n    {body}\n}"
            }
        }
        
        self.logger.info("Code Generator Agent initialized successfully")
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation task"""
        task_type = task_data.get("task_type", "generate_code")
        
        try:
            if task_type == "generate_code":
                return await self._execute_code_generation(task_id, task_data)
            elif task_type == "complete_code":
                return await self._execute_code_completion(task_id, task_data)
            elif task_type == "refactor_code":
                return await self._execute_code_refactoring(task_id, task_data)
            elif task_type == "explain_code":
                return await self._execute_code_explanation(task_id, task_data)
            elif task_type == "review_code":
                return await self._execute_code_review(task_id, task_data)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}",
                    "task_id": task_id
                }
                
        except Exception as e:
            self.logger.error(f"Code generation task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _execute_code_generation(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code from specifications"""
        specification = task_data.get("specification", "")
        language = task_data.get("language", "python")
        code_type = task_data.get("code_type", "function")  # function, class, module, etc.
        
        if not specification:
            return {
                "success": False,
                "error": "No specification provided",
                "task_id": task_id
            }
        
        # Build prompt for code generation
        system_prompt = f"""You are an expert {language} developer. Generate clean, well-documented, and efficient code based on the given specification.

Guidelines:
- Follow {language} best practices and conventions
- Include proper error handling
- Add appropriate comments and docstrings
- Ensure code is production-ready
- Use meaningful variable and function names
- Include type hints where applicable"""

        user_prompt = f"""Generate {language} code for the following specification:

Specification: {specification}
Code Type: {code_type}
Language: {language}

Requirements:
- Generate complete, working code
- Include proper imports if needed
- Add comprehensive documentation
- Follow the specified code type structure
- Ensure the code is ready to use

Please provide only the code without additional explanation."""

        # Query the model
        generated_code = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=3000
        )
        
        if not generated_code:
            return {
                "success": False,
                "error": "Failed to generate code",
                "task_id": task_id
            }
        
        # Post-process the generated code
        cleaned_code = self._clean_generated_code(generated_code, language)
        
        # Validate the generated code if possible
        validation_result = await self._validate_code(cleaned_code, language)
        
        return {
            "success": True,
            "result": {
                "generated_code": cleaned_code,
                "language": language,
                "code_type": code_type,
                "validation": validation_result,
                "specification": specification
            },
            "task_id": task_id
        }
    
    async def _execute_code_completion(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete partial code"""
        partial_code = task_data.get("partial_code", "")
        language = task_data.get("language", "python")
        context = task_data.get("context", "")
        
        if not partial_code:
            return {
                "success": False,
                "error": "No partial code provided",
                "task_id": task_id
            }
        
        system_prompt = f"""You are an expert {language} developer. Complete the given partial code in a logical and efficient manner.

Guidelines:
- Maintain consistency with the existing code style
- Follow {language} best practices
- Complete the code functionality logically
- Add appropriate error handling
- Ensure the completion makes sense in the given context"""

        user_prompt = f"""Complete the following {language} code:

Context: {context}

Partial Code:
```{language}
{partial_code}
```

Please provide the completed code, maintaining the existing style and structure."""

        completed_code = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        if not completed_code:
            return {
                "success": False,
                "error": "Failed to complete code",
                "task_id": task_id
            }
        
        cleaned_code = self._clean_generated_code(completed_code, language)
        
        return {
            "success": True,
            "result": {
                "completed_code": cleaned_code,
                "original_code": partial_code,
                "language": language,
                "context": context
            },
            "task_id": task_id
        }
    
    async def _execute_code_refactoring(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor existing code"""
        original_code = task_data.get("original_code", "")
        language = task_data.get("language", "python")
        refactoring_goals = task_data.get("refactoring_goals", "improve readability and performance")
        
        if not original_code:
            return {
                "success": False,
                "error": "No original code provided",
                "task_id": task_id
            }
        
        system_prompt = f"""You are an expert {language} developer specializing in code refactoring. Improve the given code while maintaining its functionality.

Refactoring Goals: {refactoring_goals}

Guidelines:
- Preserve the original functionality
- Improve code readability and maintainability
- Optimize performance where possible
- Follow {language} best practices and conventions
- Remove code smells and anti-patterns
- Add proper documentation if missing"""

        user_prompt = f"""Refactor the following {language} code:

Original Code:
```{language}
{original_code}
```

Refactoring Goals: {refactoring_goals}

Please provide the refactored code with explanations of the changes made."""

        refactored_result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=3000
        )
        
        if not refactored_result:
            return {
                "success": False,
                "error": "Failed to refactor code",
                "task_id": task_id
            }
        
        # Extract code and explanation from the result
        refactored_code, explanation = self._extract_code_and_explanation(refactored_result, language)
        
        return {
            "success": True,
            "result": {
                "refactored_code": refactored_code,
                "original_code": original_code,
                "explanation": explanation,
                "language": language,
                "refactoring_goals": refactoring_goals
            },
            "task_id": task_id
        }
    
    async def _execute_code_explanation(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain existing code"""
        code_to_explain = task_data.get("code", "")
        language = task_data.get("language", "python")
        explanation_level = task_data.get("explanation_level", "intermediate")  # beginner, intermediate, advanced
        
        if not code_to_explain:
            return {
                "success": False,
                "error": "No code provided for explanation",
                "task_id": task_id
            }
        
        system_prompt = f"""You are an expert {language} developer and teacher. Explain the given code clearly and comprehensively.

Explanation Level: {explanation_level}

Guidelines:
- Provide clear, step-by-step explanations
- Explain the purpose and functionality
- Highlight important concepts and patterns
- Mention best practices and potential improvements
- Adapt the explanation to the specified level"""

        user_prompt = f"""Explain the following {language} code:

```{language}
{code_to_explain}
```

Explanation Level: {explanation_level}

Please provide a comprehensive explanation including:
1. Overall purpose and functionality
2. Step-by-step breakdown
3. Key concepts and patterns used
4. Best practices demonstrated
5. Potential improvements or considerations"""

        explanation = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2500
        )
        
        if not explanation:
            return {
                "success": False,
                "error": "Failed to generate explanation",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "code": code_to_explain,
                "explanation": explanation,
                "language": language,
                "explanation_level": explanation_level
            },
            "task_id": task_id
        }
    
    async def _execute_code_review(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for quality and best practices"""
        code_to_review = task_data.get("code", "")
        language = task_data.get("language", "python")
        review_focus = task_data.get("review_focus", "general")  # security, performance, readability, general
        
        if not code_to_review:
            return {
                "success": False,
                "error": "No code provided for review",
                "task_id": task_id
            }
        
        system_prompt = f"""You are an expert {language} code reviewer. Analyze the given code and provide constructive feedback.

Review Focus: {review_focus}

Guidelines:
- Identify issues and potential improvements
- Suggest specific fixes and optimizations
- Rate different aspects of the code
- Provide actionable recommendations
- Be constructive and helpful in your feedback"""

        user_prompt = f"""Review the following {language} code:

```{language}
{code_to_review}
```

Review Focus: {review_focus}

Please provide a comprehensive code review including:
1. Overall code quality assessment
2. Specific issues and concerns
3. Best practices compliance
4. Security considerations (if applicable)
5. Performance implications
6. Readability and maintainability
7. Recommended improvements
8. Code quality rating (1-10)"""

        review_result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500
        )
        
        if not review_result:
            return {
                "success": False,
                "error": "Failed to generate code review",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "code": code_to_review,
                "review": review_result,
                "language": language,
                "review_focus": review_focus,
                "reviewed_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    def _clean_generated_code(self, code: str, language: str) -> str:
        """Clean and format generated code"""
        # Remove markdown code block markers
        code = code.strip()
        
        # Remove code block markers
        if code.startswith(f"```{language}"):
            code = code[len(f"```{language}"):].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        
        if code.endswith("```"):
            code = code[:-3].strip()
        
        # Remove extra whitespace
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_lines.append(line.rstrip())
        
        return '\n'.join(cleaned_lines)
    
    def _extract_code_and_explanation(self, result: str, language: str) -> tuple:
        """Extract code and explanation from AI response"""
        # Try to find code blocks
        code_start = result.find(f"```{language}")
        if code_start == -1:
            code_start = result.find("```")
        
        if code_start != -1:
            code_end = result.find("```", code_start + 3)
            if code_end != -1:
                # Extract code
                code_section = result[code_start:code_end + 3]
                code = self._clean_generated_code(code_section, language)
                
                # Extract explanation (text before and after code)
                explanation_before = result[:code_start].strip()
                explanation_after = result[code_end + 3:].strip()
                explanation = f"{explanation_before}\n\n{explanation_after}".strip()
                
                return code, explanation
        
        # If no code blocks found, assume the entire response is explanation
        return "", result
    
    async def _validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Basic code validation"""
        validation_result = {
            "syntax_valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            if language == "python":
                # Basic Python syntax validation
                compile(code, '<string>', 'exec')
                validation_result["syntax_valid"] = True
            else:
                # For other languages, we'll assume syntax is valid
                # In a production system, you'd integrate proper validators
                validation_result["syntax_valid"] = True
                validation_result["suggestions"].append(f"Syntax validation not implemented for {language}")
        
        except SyntaxError as e:
            validation_result["errors"].append(f"Syntax error: {str(e)}")
        except Exception as e:
            validation_result["warnings"].append(f"Validation warning: {str(e)}")
        
        return validation_result
    
    # Message handlers
    async def _handle_generate_code(self, message: AgentMessage):
        """Handle code generation request"""
        content = message.content
        
        task_data = {
            "task_type": "generate_code",
            "specification": content.get("specification", ""),
            "language": content.get("language", "python"),
            "code_type": content.get("code_type", "function")
        }
        
        result = await self._execute_code_generation(message.id, task_data)
        
        await self.send_message(
            message.sender_id,
            "code_generation_result",
            result
        )
    
    async def _handle_complete_code(self, message: AgentMessage):
        """Handle code completion request"""
        content = message.content
        
        task_data = {
            "task_type": "complete_code",
            "partial_code": content.get("partial_code", ""),
            "language": content.get("language", "python"),
            "context": content.get("context", "")
        }
        
        result = await self._execute_code_completion(message.id, task_data)
        
        await self.send_message(
            message.sender_id,
            "code_completion_result",
            result
        )
    
    async def _handle_refactor_code(self, message: AgentMessage):
        """Handle code refactoring request"""
        content = message.content
        
        task_data = {
            "task_type": "refactor_code",
            "original_code": content.get("original_code", ""),
            "language": content.get("language", "python"),
            "refactoring_goals": content.get("refactoring_goals", "improve readability and performance")
        }
        
        result = await self._execute_code_refactoring(message.id, task_data)
        
        await self.send_message(
            message.sender_id,
            "code_refactoring_result",
            result
        )
    
    async def _handle_explain_code(self, message: AgentMessage):
        """Handle code explanation request"""
        content = message.content
        
        task_data = {
            "task_type": "explain_code",
            "code": content.get("code", ""),
            "language": content.get("language", "python"),
            "explanation_level": content.get("explanation_level", "intermediate")
        }
        
        result = await self._execute_code_explanation(message.id, task_data)
        
        await self.send_message(
            message.sender_id,
            "code_explanation_result",
            result
        )
    
    async def on_message_received(self, message: AgentMessage):
        """Handle unknown message types"""
        self.logger.warning(f"Received unknown message type: {message.message_type}")
        
        # Send error response
        await self.send_message(
            message.sender_id,
            "error",
            {
                "error": f"Unknown message type: {message.message_type}",
                "original_message_id": message.id
            }
        )
    
    async def on_shutdown(self):
        """Cleanup when shutting down"""
        self.logger.info("Code Generator Agent shutting down")
        # Perform any cleanup tasks here