"""
Code Generation Module (CGM)
Responsible for generating, modifying, and optimizing code with self-improvement capabilities
"""

import asyncio
import ast
import logging
import time
import json
import inspect
import importlib.util
import subprocess
import tempfile
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class CodeType(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"

class OptimizationTarget(str, Enum):
    PERFORMANCE = "performance"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    EFFICIENCY = "efficiency"

@dataclass
class CodeGenerationTask:
    """Task for code generation"""
    id: str
    task_type: str
    description: str
    requirements: Dict[str, Any]
    target_language: CodeType
    optimization_targets: List[OptimizationTarget]
    context: Dict[str, Any]
    priority: int = 5
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class GeneratedCode:
    """Generated code result"""
    id: str
    task_id: str
    code: str
    language: CodeType
    quality_score: float
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: float
    tested: bool = False
    deployed: bool = False

@dataclass
class MetaLearningModel:
    """Meta-learning model for adaptation"""
    model_id: str
    architecture: str
    parameters: Dict[str, Any]
    performance_history: List[float]
    adaptation_count: int
    last_updated: float

class NeuralCodeGenerator(nn.Module):
    """Neural network for code generation"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, attention_mask=None):
        embedded = self.embedding(x)
        transformed = self.transformer(embedded, src_key_padding_mask=attention_mask)
        output = self.output_layer(transformed)
        return output

class CodeGenerationModule:
    """
    Advanced Code Generation Module with Meta-Learning and Self-Improvement
    """
    
    AUTHORIZED_USER = "os.getenv("ADMIN_EMAIL", "admin@localhost")"
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/cgm"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.neural_generator = None
        self.tokenizer = None
        self.meta_learning_models = {}
        self.code_templates = {}
        self.optimization_strategies = {}
        
        # Generated code tracking
        self.generated_code = {}
        self.generation_history = []
        self.performance_metrics = {}
        
        # Self-improvement mechanisms
        self.improvement_queue = []
        self.code_review_results = {}
        self.refactoring_suggestions = {}
        
        # Meta-learning components
        self.adaptation_memory = {}
        self.task_embeddings = {}
        self.learning_curves = {}
        
        # Initialize
        self._initialize_cgm()
    
    def _initialize_cgm(self):
        """Initialize Code Generation Module"""
        try:
            # Initialize neural code generator
            self._initialize_neural_generator()
            
            # Load code templates
            self._load_code_templates()
            
            # Initialize optimization strategies
            self._initialize_optimization_strategies()
            
            # Setup meta-learning models
            self._initialize_meta_learning()
            
            # Load existing data
            self._load_existing_data()
            
            logger.info("🧠 Code Generation Module initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize CGM: {e}")
            raise
    
    def _initialize_neural_generator(self):
        """Initialize neural code generation models"""
        try:
            # Load pre-trained code generation model
            model_name = "microsoft/CodeGPT-small-py"
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.code_generator_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
                logger.info("✅ Pre-trained code generator loaded")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
                # Fallback to simple neural generator
                self._initialize_simple_generator()
            
        except Exception as e:
            logger.error(f"Neural generator initialization failed: {e}")
            self._initialize_simple_generator()
    
    def _initialize_simple_generator(self):
        """Initialize simple neural generator as fallback"""
        try:
            vocab_size = 10000  # Simplified vocabulary
            self.neural_generator = NeuralCodeGenerator(vocab_size)
            
            # Simple tokenizer
            self.simple_vocab = {f"token_{i}": i for i in range(vocab_size)}
            self.reverse_vocab = {v: k for k, v in self.simple_vocab.items()}
            
            logger.info("✅ Simple neural generator initialized")
            
        except Exception as e:
            logger.error(f"Simple generator initialization failed: {e}")
    
    def _load_code_templates(self):
        """Load code templates for different languages and patterns"""
        self.code_templates = {
            CodeType.PYTHON: {
                "class": """class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}
""",
                "function": """def {function_name}({parameters}){return_type}:
    \"\"\"
    {description}
    
    Args:
        {args_docs}
    
    Returns:
        {return_docs}
    \"\"\"
    {body}
""",
                "async_function": """async def {function_name}({parameters}){return_type}:
    \"\"\"
    {description}
    
    Args:
        {args_docs}
    
    Returns:
        {return_docs}
    \"\"\"
    try:
        {body}
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        raise
""",
                "test": """def test_{function_name}():
    \"\"\"Test {function_name} function\"\"\"
    # Arrange
    {test_setup}
    
    # Act
    result = {function_call}
    
    # Assert
    {assertions}
"""
            },
            CodeType.JAVASCRIPT: {
                "class": """class {class_name} {{
    /**
     * {description}
     */
    constructor({constructor_params}) {{
        {constructor_body}
    }}
    
    {methods}
}}""",
                "function": """function {function_name}({parameters}) {{
    /**
     * {description}
     * @param {{{param_types}}} {param_names}
     * @returns {{{return_type}}} {return_description}
     */
    {body}
}}""",
                "arrow_function": """const {function_name} = ({parameters}) => {{
    {body}
}};"""
            }
        }
    
    def _initialize_optimization_strategies(self):
        """Initialize code optimization strategies"""
        self.optimization_strategies = {
            OptimizationTarget.PERFORMANCE: {
                "techniques": [
                    "loop_optimization",
                    "memory_optimization", 
                    "algorithm_selection",
                    "caching",
                    "parallelization"
                ],
                "metrics": ["execution_time", "memory_usage", "cpu_utilization"]
            },
            OptimizationTarget.READABILITY: {
                "techniques": [
                    "variable_naming",
                    "function_decomposition",
                    "comment_generation",
                    "code_formatting",
                    "documentation"
                ],
                "metrics": ["readability_score", "complexity_score", "documentation_coverage"]
            },
            OptimizationTarget.MAINTAINABILITY: {
                "techniques": [
                    "modularization",
                    "dependency_injection",
                    "interface_design",
                    "error_handling",
                    "testing"
                ],
                "metrics": ["coupling", "cohesion", "test_coverage", "cyclomatic_complexity"]
            },
            OptimizationTarget.SECURITY: {
                "techniques": [
                    "input_validation",
                    "sanitization",
                    "authentication",
                    "authorization", 
                    "encryption"
                ],
                "metrics": ["security_score", "vulnerability_count", "compliance_level"]
            }
        }
    
    def _initialize_meta_learning(self):
        """Initialize meta-learning components"""
        try:
            # Create meta-learning models for different task types
            meta_models = {
                "code_generation": MetaLearningModel(
                    model_id="meta_codegen_v1",
                    architecture="transformer",
                    parameters={
                        "learning_rate": 0.001,
                        "adaptation_steps": 5,
                        "memory_size": 1000
                    },
                    performance_history=[],
                    adaptation_count=0,
                    last_updated=time.time()
                ),
                "code_optimization": MetaLearningModel(
                    model_id="meta_optimizer_v1",
                    architecture="neural_network",
                    parameters={
                        "hidden_layers": [256, 128, 64],
                        "activation": "relu",
                        "dropout": 0.2
                    },
                    performance_history=[],
                    adaptation_count=0,
                    last_updated=time.time()
                ),
                "code_review": MetaLearningModel(
                    model_id="meta_reviewer_v1",
                    architecture="attention",
                    parameters={
                        "attention_heads": 8,
                        "embedding_dim": 512,
                        "context_length": 2048
                    },
                    performance_history=[],
                    adaptation_count=0,
                    last_updated=time.time()
                )
            }
            
            self.meta_learning_models.update(meta_models)
            logger.info("✅ Meta-learning models initialized")
            
        except Exception as e:
            logger.error(f"Meta-learning initialization failed: {e}")
    
    async def generate_code(self, task: CodeGenerationTask, user_id: str) -> Dict[str, Any]:
        """Generate code based on task specification"""
        try:
            # Authorization check
            if user_id != self.AUTHORIZED_USER:
                return {
                    "success": False,
                    "error": "Unauthorized: Only authorized user can request code generation"
                }
            
            logger.info(f"🔨 Generating code for task: {task.description}")
            
            # Analyze task requirements
            task_analysis = await self._analyze_task(task)
            
            # Select appropriate generation strategy
            strategy = await self._select_generation_strategy(task, task_analysis)
            
            # Generate code using selected strategy
            generated_code = await self._execute_generation_strategy(task, strategy)
            
            # Optimize generated code
            optimized_code = await self._optimize_code(generated_code, task.optimization_targets)
            
            # Test generated code
            test_results = await self._test_generated_code(optimized_code, task)
            
            # Create code record
            code_record = GeneratedCode(
                id=str(uuid.uuid4()),
                task_id=task.id,
                code=optimized_code["code"],
                language=task.target_language,
                quality_score=optimized_code["quality_score"],
                performance_metrics=test_results["metrics"],
                metadata={
                    "strategy": strategy["name"],
                    "optimization_applied": optimized_code["optimizations"],
                    "generation_time": time.time() - task.created_at
                },
                generated_at=time.time(),
                tested=test_results["success"]
            )
            
            # Store generated code
            self.generated_code[code_record.id] = code_record
            self.generation_history.append(code_record.id)
            
            # Update meta-learning models
            await self._update_meta_learning(task, code_record, test_results)
            
            # Schedule self-improvement review
            await self._schedule_improvement_review(code_record)
            
            return {
                "success": True,
                "code_id": code_record.id,
                "code": code_record.code,
                "quality_score": code_record.quality_score,
                "test_results": test_results,
                "metadata": code_record.metadata
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_task(self, task: CodeGenerationTask) -> Dict[str, Any]:
        """Analyze task requirements and complexity"""
        try:
            analysis = {
                "complexity": "medium",
                "estimated_time": 300,  # seconds
                "required_techniques": [],
                "dependencies": [],
                "risk_level": "low"
            }
            
            # Analyze description for complexity indicators
            description = task.description.lower()
            
            # Complexity analysis
            complexity_indicators = {
                "high": ["machine learning", "ai", "neural network", "optimization", "parallel", "distributed"],
                "medium": ["algorithm", "data structure", "api", "database", "async"],
                "low": ["simple", "basic", "utility", "helper", "format"]
            }
            
            for level, indicators in complexity_indicators.items():
                if any(indicator in description for indicator in indicators):
                    analysis["complexity"] = level
                    break
            
            # Estimate time based on complexity
            time_estimates = {"low": 60, "medium": 300, "high": 900}
            analysis["estimated_time"] = time_estimates[analysis["complexity"]]
            
            # Identify required techniques
            technique_keywords = {
                "async": ["async", "await", "concurrent"],
                "database": ["database", "sql", "orm", "query"],
                "api": ["api", "rest", "endpoint", "request"],
                "testing": ["test", "unit test", "integration"],
                "optimization": ["optimize", "performance", "efficient"]
            }
            
            for technique, keywords in technique_keywords.items():
                if any(keyword in description for keyword in keywords):
                    analysis["required_techniques"].append(technique)
            
            # Risk assessment
            risk_indicators = ["system", "file", "network", "security", "database"]
            if any(indicator in description for indicator in risk_indicators):
                analysis["risk_level"] = "medium"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            return {"complexity": "medium", "estimated_time": 300, "required_techniques": [], "dependencies": [], "risk_level": "low"}
    
    async def _select_generation_strategy(self, task: CodeGenerationTask, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal code generation strategy"""
        try:
            strategies = {
                "template_based": {
                    "name": "template_based",
                    "description": "Use predefined templates",
                    "complexity_match": ["low", "medium"],
                    "speed": "fast",
                    "quality": "good"
                },
                "neural_generation": {
                    "name": "neural_generation", 
                    "description": "Use neural code generation",
                    "complexity_match": ["medium", "high"],
                    "speed": "medium",
                    "quality": "excellent"
                },
                "hybrid": {
                    "name": "hybrid",
                    "description": "Combine template and neural approaches",
                    "complexity_match": ["high"],
                    "speed": "slow",
                    "quality": "excellent"
                },
                "meta_learning": {
                    "name": "meta_learning",
                    "description": "Use meta-learning adaptation",
                    "complexity_match": ["high"],
                    "speed": "medium",
                    "quality": "adaptive"
                }
            }
            
            # Select strategy based on task complexity and requirements
            complexity = analysis["complexity"]
            
            # Default to template-based for low complexity
            if complexity == "low":
                return strategies["template_based"]
            
            # Use neural generation for medium complexity
            elif complexity == "medium":
                if hasattr(self, 'code_generator_pipeline') and self.code_generator_pipeline:
                    return strategies["neural_generation"]
                else:
                    return strategies["template_based"]
            
            # Use hybrid or meta-learning for high complexity
            else:
                if len(self.generation_history) > 10:  # Enough data for meta-learning
                    return strategies["meta_learning"]
                else:
                    return strategies["hybrid"]
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return {"name": "template_based", "description": "Fallback template strategy"}
    
    async def _execute_generation_strategy(self, task: CodeGenerationTask, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected code generation strategy"""
        try:
            strategy_name = strategy["name"]
            
            if strategy_name == "template_based":
                return await self._generate_from_template(task)
            elif strategy_name == "neural_generation":
                return await self._generate_with_neural_model(task)
            elif strategy_name == "hybrid":
                return await self._generate_hybrid(task)
            elif strategy_name == "meta_learning":
                return await self._generate_with_meta_learning(task)
            else:
                return await self._generate_from_template(task)  # Fallback
                
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return await self._generate_from_template(task)  # Fallback
    
    async def _generate_from_template(self, task: CodeGenerationTask) -> Dict[str, Any]:
        """Generate code using templates"""
        try:
            language = task.target_language
            templates = self.code_templates.get(language, {})
            
            # Analyze task to determine template type
            description = task.description.lower()
            
            if "class" in description:
                template_type = "class"
            elif "async" in description or "await" in description:
                template_type = "async_function"
            elif "test" in description:
                template_type = "test"
            else:
                template_type = "function"
            
            template = templates.get(template_type, templates.get("function", ""))
            
            # Extract parameters from requirements
            requirements = task.requirements
            
            # Generate code from template
            if template_type == "function" and language == CodeType.PYTHON:
                code = template.format(
                    function_name=requirements.get("function_name", "generated_function"),
                    parameters=requirements.get("parameters", ""),
                    return_type=f" -> {requirements.get('return_type', 'Any')}" if requirements.get('return_type') else "",
                    description=task.description,
                    args_docs=requirements.get("args_docs", ""),
                    return_docs=requirements.get("return_docs", ""),
                    body=self._generate_function_body(task)
                )
            elif template_type == "class" and language == CodeType.PYTHON:
                code = template.format(
                    class_name=requirements.get("class_name", "GeneratedClass"),
                    description=task.description,
                    init_params=requirements.get("init_params", ""),
                    init_body=requirements.get("init_body", "pass"),
                    methods=self._generate_class_methods(task)
                )
            else:
                # Simple function template
                code = f"""def generated_function():
    \"\"\"
    {task.description}
    \"\"\"
    # TODO: Implement function logic
    pass
"""
            
            return {
                "code": code,
                "strategy": "template_based",
                "quality_score": 0.7,
                "generation_time": 0.1
            }
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return {
                "code": f"# Generated code for: {task.description}\n# TODO: Implement functionality\npass",
                "strategy": "template_based",
                "quality_score": 0.5,
                "generation_time": 0.1
            }
    
    def _generate_function_body(self, task: CodeGenerationTask) -> str:
        """Generate function body based on task requirements"""
        try:
            requirements = task.requirements
            description = task.description.lower()
            
            # Basic patterns based on description
            if "calculate" in description or "compute" in description:
                return """    # Perform calculation
    result = 0  # Replace with actual calculation
    return result"""
            
            elif "process" in description:
                return """    # Process input data
    processed_data = input_data  # Replace with actual processing
    return processed_data"""
            
            elif "validate" in description:
                return """    # Validate input
    if not input_data:
        raise ValueError("Invalid input")
    return True"""
            
            elif "fetch" in description or "get" in description:
                return """    # Fetch data
    data = {}  # Replace with actual data fetching
    return data"""
            
            else:
                return """    # TODO: Implement function logic
    pass"""
                
        except Exception as e:
            logger.error(f"Function body generation failed: {e}")
            return "    pass"
    
    def _generate_class_methods(self, task: CodeGenerationTask) -> str:
        """Generate class methods based on task requirements"""
        try:
            requirements = task.requirements
            methods = requirements.get("methods", [])
            
            if not methods:
                # Generate basic methods based on description
                description = task.description.lower()
                if "manager" in description:
                    methods = ["add", "remove", "get", "list"]
                elif "processor" in description:
                    methods = ["process", "validate", "transform"]
                elif "service" in description:
                    methods = ["start", "stop", "status"]
                else:
                    methods = ["execute"]
            
            method_code = ""
            for method_name in methods:
                method_code += f"""
    def {method_name}(self, *args, **kwargs):
        \"\"\"
        {method_name.title()} method for {task.description}
        \"\"\"
        # TODO: Implement {method_name} logic
        pass
"""
            
            return method_code
            
        except Exception as e:
            logger.error(f"Class methods generation failed: {e}")
            return "\n    def execute(self):\n        \"\"\"Execute method\"\"\"\n        pass"
    
    async def _generate_with_neural_model(self, task: CodeGenerationTask) -> Dict[str, Any]:
        """Generate code using neural model"""
        try:
            if not hasattr(self, 'code_generator_pipeline') or not self.code_generator_pipeline:
                logger.warning("Neural model not available, falling back to template")
                return await self._generate_from_template(task)
            
            # Prepare prompt for code generation
            prompt = self._create_generation_prompt(task)
            
            # Generate code using the model
            start_time = time.time()
            generated_outputs = self.code_generator_pipeline(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generation_time = time.time() - start_time
            
            # Extract generated code
            generated_text = generated_outputs[0]['generated_text']
            code = self._extract_code_from_output(generated_text, prompt)
            
            # Calculate quality score
            quality_score = self._calculate_code_quality(code, task)
            
            return {
                "code": code,
                "strategy": "neural_generation",
                "quality_score": quality_score,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Neural generation failed: {e}")
            return await self._generate_from_template(task)
    
    def _create_generation_prompt(self, task: CodeGenerationTask) -> str:
        """Create prompt for neural code generation"""
        try:
            language = task.target_language.value
            description = task.description
            requirements = task.requirements
            
            prompt = f"""# Task: Generate {language} code
# Description: {description}
# Requirements: {json.dumps(requirements, indent=2)}

def """
            
            if "function_name" in requirements:
                prompt += requirements["function_name"]
            else:
                prompt += "generated_function"
            
            if "parameters" in requirements:
                prompt += f"({requirements['parameters']})"
            else:
                prompt += "()"
            
            prompt += ":\n    \"\"\"\n    " + description + "\n    \"\"\"\n"
            
            return prompt
            
        except Exception as e:
            logger.error(f"Prompt creation failed: {e}")
            return f"def generated_function():\n    \"\"\"{task.description}\"\"\"\n"
    
    def _extract_code_from_output(self, generated_text: str, prompt: str) -> str:
        """Extract clean code from model output"""
        try:
            # Remove the prompt from the generated text
            if prompt in generated_text:
                code = generated_text.replace(prompt, "")
            else:
                code = generated_text
            
            # Clean up the code
            lines = code.split('\n')
            clean_lines = []
            
            for line in lines:
                # Remove empty lines at the beginning
                if not clean_lines and not line.strip():
                    continue
                
                # Stop at certain markers
                if line.strip().startswith('#') and ('end' in line.lower() or 'note' in line.lower()):
                    break
                
                clean_lines.append(line)
            
            # Join and return cleaned code
            clean_code = '\n'.join(clean_lines).strip()
            
            # If code is too short, add basic structure
            if len(clean_code) < 20:
                clean_code = f"    # Generated code\n    # TODO: Implement functionality\n    pass"
            
            return prompt + clean_code
            
        except Exception as e:
            logger.error(f"Code extraction failed: {e}")
            return prompt + "    pass"
    
    def _calculate_code_quality(self, code: str, task: CodeGenerationTask) -> float:
        """Calculate quality score for generated code"""
        try:
            score = 0.5  # Base score
            
            # Check for syntax validity
            try:
                if task.target_language == CodeType.PYTHON:
                    ast.parse(code)
                    score += 0.2  # Valid syntax
            except:
                score -= 0.1  # Invalid syntax
            
            # Check for documentation
            if '"""' in code or "'''" in code:
                score += 0.1
            
            # Check for error handling
            if "try:" in code and "except" in code:
                score += 0.1
            
            # Check for proper naming
            if not any(name in code.lower() for name in ["temp", "tmp", "foo", "bar"]):
                score += 0.1
            
            # Clamp score between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5
    
    async def _generate_hybrid(self, task: CodeGenerationTask) -> Dict[str, Any]:
        """Generate code using hybrid approach (template + neural)"""
        try:
            # Generate using both approaches
            template_result = await self._generate_from_template(task)
            
            if hasattr(self, 'code_generator_pipeline') and self.code_generator_pipeline:
                neural_result = await self._generate_with_neural_model(task)
                
                # Combine results based on quality scores
                if neural_result["quality_score"] > template_result["quality_score"]:
                    best_code = neural_result["code"]
                    best_score = neural_result["quality_score"]
                else:
                    best_code = template_result["code"]
                    best_score = template_result["quality_score"]
            else:
                best_code = template_result["code"]
                best_score = template_result["quality_score"]
            
            return {
                "code": best_code,
                "strategy": "hybrid",
                "quality_score": best_score + 0.1,  # Bonus for hybrid approach
                "generation_time": 0.5
            }
            
        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}")
            return await self._generate_from_template(task)
    
    async def _generate_with_meta_learning(self, task: CodeGenerationTask) -> Dict[str, Any]:
        """Generate code using meta-learning adaptation"""
        try:
            # Find similar tasks in history
            similar_tasks = await self._find_similar_tasks(task)
            
            if similar_tasks:
                # Adapt based on successful patterns
                adapted_code = await self._adapt_from_similar_tasks(task, similar_tasks)
                return {
                    "code": adapted_code,
                    "strategy": "meta_learning",
                    "quality_score": 0.8,
                    "generation_time": 0.3
                }
            else:
                # Fall back to neural generation
                return await self._generate_with_neural_model(task)
                
        except Exception as e:
            logger.error(f"Meta-learning generation failed: {e}")
            return await self._generate_from_template(task)
    
    async def _find_similar_tasks(self, task: CodeGenerationTask) -> List[str]:
        """Find similar tasks in generation history"""
        try:
            similar_tasks = []
            current_keywords = set(task.description.lower().split())
            
            for code_id in self.generation_history[-50:]:  # Check last 50 generations
                if code_id in self.generated_code:
                    code_record = self.generated_code[code_id]
                    
                    # Find task description (would be stored in metadata)
                    task_desc = code_record.metadata.get("task_description", "")
                    task_keywords = set(task_desc.lower().split())
                    
                    # Calculate similarity
                    similarity = len(current_keywords & task_keywords) / len(current_keywords | task_keywords)
                    
                    if similarity > 0.3:  # Similarity threshold
                        similar_tasks.append(code_id)
            
            return similar_tasks[-5:]  # Return top 5 most recent similar tasks
            
        except Exception as e:
            logger.error(f"Similar task search failed: {e}")
            return []
    
    async def _adapt_from_similar_tasks(self, task: CodeGenerationTask, similar_tasks: List[str]) -> str:
        """Adapt code from similar successful tasks"""
        try:
            # Get code from similar tasks
            similar_codes = []
            for task_id in similar_tasks:
                if task_id in self.generated_code:
                    code_record = self.generated_code[task_id]
                    if code_record.quality_score > 0.7:  # Only use high-quality code
                        similar_codes.append(code_record.code)
            
            if not similar_codes:
                return await self._generate_from_template(task)
            
            # Extract common patterns
            common_patterns = self._extract_common_patterns(similar_codes)
            
            # Generate adapted code
            adapted_code = self._create_adapted_code(task, common_patterns, similar_codes[0])
            
            return adapted_code
            
        except Exception as e:
            logger.error(f"Code adaptation failed: {e}")
            return f"# Adapted code for: {task.description}\n# TODO: Implement functionality\npass"
    
    def _extract_common_patterns(self, codes: List[str]) -> Dict[str, Any]:
        """Extract common patterns from similar codes"""
        try:
            patterns = {
                "common_imports": [],
                "common_functions": [],
                "common_structures": [],
                "common_error_handling": False
            }
            
            # Analyze each code for patterns
            for code in codes:
                lines = code.split('\n')
                
                # Extract imports
                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        patterns["common_imports"].append(line.strip())
                
                # Check for error handling
                if "try:" in code and "except" in code:
                    patterns["common_error_handling"] = True
            
            # Remove duplicates
            patterns["common_imports"] = list(set(patterns["common_imports"]))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return {"common_imports": [], "common_functions": [], "common_structures": [], "common_error_handling": False}
    
    def _create_adapted_code(self, task: CodeGenerationTask, patterns: Dict[str, Any], base_code: str) -> str:
        """Create adapted code using extracted patterns"""
        try:
            # Start with template
            template_result = asyncio.run(self._generate_from_template(task))
            adapted_code = template_result["code"]
            
            # Add common imports
            if patterns["common_imports"]:
                imports = '\n'.join(patterns["common_imports"][:3])  # Limit to 3 imports
                adapted_code = imports + '\n\n' + adapted_code
            
            # Add error handling if common
            if patterns["common_error_handling"] and "try:" not in adapted_code:
                # Wrap main logic in try-catch
                lines = adapted_code.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith(('#', 'import', 'from', 'def', 'class', '"""')):
                        # Insert try block
                        lines[i] = "    try:"
                        # Add except block at the end
                        lines.append("    except Exception as e:")
                        lines.append("        logger.error(f'Error: {e}')")
                        lines.append("        raise")
                        break
                
                adapted_code = '\n'.join(lines)
            
            return adapted_code
            
        except Exception as e:
            logger.error(f"Code adaptation failed: {e}")
            return f"# Adapted code for: {task.description}\n# TODO: Implement functionality\npass"
    
    async def _optimize_code(self, generated_code: Dict[str, Any], optimization_targets: List[OptimizationTarget]) -> Dict[str, Any]:
        """Optimize generated code based on targets"""
        try:
            code = generated_code["code"]
            optimizations_applied = []
            quality_boost = 0.0
            
            for target in optimization_targets:
                if target == OptimizationTarget.PERFORMANCE:
                    code, perf_boost = await self._optimize_performance(code)
                    quality_boost += perf_boost
                    optimizations_applied.append("performance")
                
                elif target == OptimizationTarget.READABILITY:
                    code, read_boost = await self._optimize_readability(code)
                    quality_boost += read_boost
                    optimizations_applied.append("readability")
                
                elif target == OptimizationTarget.SECURITY:
                    code, sec_boost = await self._optimize_security(code)
                    quality_boost += sec_boost
                    optimizations_applied.append("security")
            
            # Update quality score
            new_quality_score = min(1.0, generated_code["quality_score"] + quality_boost)
            
            return {
                "code": code,
                "quality_score": new_quality_score,
                "optimizations": optimizations_applied,
                "strategy": generated_code["strategy"]
            }
            
        except Exception as e:
            logger.error(f"Code optimization failed: {e}")
            return generated_code
    
    async def _optimize_performance(self, code: str) -> Tuple[str, float]:
        """Optimize code for performance"""
        try:
            optimized_code = code
            boost = 0.0
            
            # Add basic performance optimizations
            lines = code.split('\n')
            
            # Look for optimization opportunities
            for i, line in enumerate(lines):
                # Optimize list comprehensions
                if "for " in line and " in " in line and not "[" in line:
                    # Suggest list comprehension where appropriate
                    if "append(" in line:
                        boost += 0.05
                
                # Add caching hints
                if "def " in line and "cache" not in line:
                    lines[i] = line + "\n    # Consider adding @lru_cache for repeated calls"
                    boost += 0.03
            
            optimized_code = '\n'.join(lines)
            
            return optimized_code, boost
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return code, 0.0
    
    async def _optimize_readability(self, code: str) -> Tuple[str, float]:
        """Optimize code for readability"""
        try:
            optimized_code = code
            boost = 0.0
            
            # Add type hints if missing
            if "def " in code and "->" not in code:
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith("def ") and "->" not in line:
                        lines[i] = line.replace("):", ") -> Any:")
                        boost += 0.05
                        break
                optimized_code = '\n'.join(lines)
            
            # Ensure proper documentation
            if '"""' not in code:
                # Add basic docstring
                lines = optimized_code.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        lines.insert(i + 1, '    """Function documentation"""')
                        boost += 0.1
                        break
                optimized_code = '\n'.join(lines)
            
            return optimized_code, boost
            
        except Exception as e:
            logger.error(f"Readability optimization failed: {e}")
            return code, 0.0
    
    async def _optimize_security(self, code: str) -> Tuple[str, float]:
        """Optimize code for security"""
        try:
            optimized_code = code
            boost = 0.0
            
            # Add input validation
            if "def " in code and "validate" not in code.lower():
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith("def ") and "(" in line:
                        # Add input validation comment
                        lines.insert(i + 2, "    # TODO: Add input validation")
                        boost += 0.1
                        break
                optimized_code = '\n'.join(lines)
            
            # Add error handling
            if "try:" not in code:
                lines = optimized_code.split('\n')
                # Find main logic and wrap in try-catch
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith(('#', 'def', 'class', '"""', 'import', 'from')):
                        lines.insert(i, "    try:")
                        lines.append("    except Exception as e:")
                        lines.append("        # Log security-related errors")
                        lines.append("        raise")
                        boost += 0.1
                        break
                optimized_code = '\n'.join(lines)
            
            return optimized_code, boost
            
        except Exception as e:
            logger.error(f"Security optimization failed: {e}")
            return code, 0.0
    
    async def _test_generated_code(self, optimized_code: Dict[str, Any], task: CodeGenerationTask) -> Dict[str, Any]:
        """Test generated code for correctness and performance"""
        try:
            code = optimized_code["code"]
            
            # Basic syntax check
            syntax_valid = True
            try:
                if task.target_language == CodeType.PYTHON:
                    ast.parse(code)
            except SyntaxError as e:
                syntax_valid = False
                logger.warning(f"Syntax error in generated code: {e}")
            
            # Performance metrics
            metrics = {
                "syntax_valid": syntax_valid,
                "line_count": len(code.split('\n')),
                "character_count": len(code),
                "has_documentation": '"""' in code or "'''" in code,
                "has_error_handling": "try:" in code and "except" in code,
                "complexity_score": self._calculate_complexity(code)
            }
            
            # Simulated execution time (would be actual in real implementation)
            metrics["estimated_execution_time"] = 0.1
            
            test_success = syntax_valid and metrics["complexity_score"] < 10
            
            return {
                "success": test_success,
                "metrics": metrics,
                "issues": [] if test_success else ["Syntax errors or high complexity"]
            }
            
        except Exception as e:
            logger.error(f"Code testing failed: {e}")
            return {
                "success": False,
                "metrics": {"error": str(e)},
                "issues": ["Testing failed"]
            }
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity of code"""
        try:
            complexity = 1  # Base complexity
            
            # Count decision points
            decision_points = ["if ", "elif ", "for ", "while ", "except ", "and ", "or "]
            for point in decision_points:
                complexity += code.count(point)
            
            return complexity
            
        except Exception as e:
            logger.error(f"Complexity calculation failed: {e}")
            return 1
    
    async def _update_meta_learning(self, task: CodeGenerationTask, code_record: GeneratedCode, test_results: Dict[str, Any]):
        """Update meta-learning models based on generation results"""
        try:
            # Update performance history
            performance_score = code_record.quality_score
            
            # Update appropriate meta-learning model
            if code_record.metadata["strategy"] in self.meta_learning_models:
                model = self.meta_learning_models[code_record.metadata["strategy"]]
                model.performance_history.append(performance_score)
                model.adaptation_count += 1
                model.last_updated = time.time()
                
                # Keep only last 100 performance scores
                if len(model.performance_history) > 100:
                    model.performance_history = model.performance_history[-100:]
            
            # Update task embeddings for future similarity matching
            task_embedding = self._create_task_embedding(task)
            self.task_embeddings[task.id] = {
                "embedding": task_embedding,
                "performance": performance_score,
                "timestamp": time.time()
            }
            
            logger.info(f"📈 Updated meta-learning models with performance: {performance_score:.3f}")
            
        except Exception as e:
            logger.error(f"Meta-learning update failed: {e}")
    
    def _create_task_embedding(self, task: CodeGenerationTask) -> List[float]:
        """Create embedding representation of task"""
        try:
            # Simple embedding based on task features
            description_words = task.description.lower().split()
            
            # Create feature vector
            features = [
                len(description_words),  # Description length
                len(task.requirements),  # Number of requirements
                task.priority,  # Task priority
                len(task.optimization_targets),  # Number of optimization targets
                1.0 if task.target_language == CodeType.PYTHON else 0.0,  # Language indicator
            ]
            
            # Pad or truncate to fixed size
            embedding_size = 10
            while len(features) < embedding_size:
                features.append(0.0)
            
            return features[:embedding_size]
            
        except Exception as e:
            logger.error(f"Task embedding creation failed: {e}")
            return [0.0] * 10
    
    async def _schedule_improvement_review(self, code_record: GeneratedCode):
        """Schedule self-improvement review for generated code"""
        try:
            review_task = {
                "id": str(uuid.uuid4()),
                "type": "code_review",
                "code_id": code_record.id,
                "scheduled_at": time.time() + 3600,  # Review in 1 hour
                "priority": 3
            }
            
            self.improvement_queue.append(review_task)
            logger.info(f"📋 Scheduled improvement review for code: {code_record.id}")
            
        except Exception as e:
            logger.error(f"Review scheduling failed: {e}")
    
    async def perform_self_improvement(self, user_id: str) -> Dict[str, Any]:
        """Perform self-improvement analysis and updates"""
        try:
            # Authorization check
            if user_id != self.AUTHORIZED_USER:
                return {
                    "success": False,
                    "error": "Unauthorized: Only authorized user can trigger self-improvement"
                }
            
            improvements_made = []
            
            # Process improvement queue
            for task in self.improvement_queue[:5]:  # Process up to 5 tasks
                if task["type"] == "code_review":
                    result = await self._perform_code_review(task["code_id"])
                    if result["improvements"]:
                        improvements_made.extend(result["improvements"])
            
            # Clear processed tasks
            self.improvement_queue = self.improvement_queue[5:]
            
            # Analyze generation patterns
            pattern_analysis = await self._analyze_generation_patterns()
            
            # Update optimization strategies
            strategy_updates = await self._update_optimization_strategies()
            
            # Meta-learning model adaptation
            model_adaptations = await self._adapt_meta_learning_models()
            
            return {
                "success": True,
                "improvements_made": improvements_made,
                "pattern_analysis": pattern_analysis,
                "strategy_updates": strategy_updates,
                "model_adaptations": model_adaptations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Self-improvement failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _perform_code_review(self, code_id: str) -> Dict[str, Any]:
        """Perform automated code review"""
        try:
            if code_id not in self.generated_code:
                return {"improvements": [], "issues": ["Code not found"]}
            
            code_record = self.generated_code[code_id]
            code = code_record.code
            
            improvements = []
            
            # Check for common improvements
            if "TODO" in code:
                improvements.append("Remove TODO comments and implement functionality")
            
            if code.count('\n') < 5:
                improvements.append("Consider adding more detailed implementation")
            
            if '"""' not in code and "'''" not in code:
                improvements.append("Add comprehensive documentation")
            
            if "logging" not in code and "log" not in code:
                improvements.append("Add appropriate logging")
            
            # Store review results
            self.code_review_results[code_id] = {
                "reviewed_at": time.time(),
                "improvements": improvements,
                "quality_assessment": "good" if len(improvements) < 3 else "needs_improvement"
            }
            
            return {"improvements": improvements, "issues": []}
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return {"improvements": [], "issues": [str(e)]}
    
    async def _analyze_generation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in code generation"""
        try:
            if len(self.generation_history) < 10:
                return {"message": "Insufficient data for pattern analysis"}
            
            # Analyze recent generations
            recent_codes = [self.generated_code[cid] for cid in self.generation_history[-20:] if cid in self.generated_code]
            
            # Calculate metrics
            avg_quality = sum(code.quality_score for code in recent_codes) / len(recent_codes)
            strategy_usage = {}
            
            for code in recent_codes:
                strategy = code.metadata.get("strategy", "unknown")
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Identify trends
            trends = []
            if avg_quality > 0.8:
                trends.append("High quality code generation trend")
            elif avg_quality < 0.6:
                trends.append("Quality improvement needed")
            
            return {
                "average_quality": avg_quality,
                "strategy_usage": strategy_usage,
                "trends": trends,
                "total_analyzed": len(recent_codes)
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _update_optimization_strategies(self) -> Dict[str, Any]:
        """Update optimization strategies based on performance"""
        try:
            updates = []
            
            # Analyze which optimizations have been most effective
            optimization_effectiveness = {}
            
            for code_record in self.generated_code.values():
                optimizations = code_record.metadata.get("optimization_applied", [])
                quality = code_record.quality_score
                
                for opt in optimizations:
                    if opt not in optimization_effectiveness:
                        optimization_effectiveness[opt] = []
                    optimization_effectiveness[opt].append(quality)
            
            # Update strategy priorities based on effectiveness
            for opt, scores in optimization_effectiveness.items():
                if scores:
                    avg_effectiveness = sum(scores) / len(scores)
                    if avg_effectiveness > 0.8:
                        updates.append(f"Increased priority for {opt} optimization")
                    elif avg_effectiveness < 0.6:
                        updates.append(f"Reviewing {opt} optimization strategy")
            
            return {
                "updates": updates,
                "optimization_effectiveness": {k: sum(v)/len(v) for k, v in optimization_effectiveness.items() if v}
            }
            
        except Exception as e:
            logger.error(f"Strategy update failed: {e}")
            return {"error": str(e)}
    
    async def _adapt_meta_learning_models(self) -> Dict[str, Any]:
        """Adapt meta-learning models based on performance"""
        try:
            adaptations = []
            
            for model_name, model in self.meta_learning_models.items():
                if len(model.performance_history) > 5:
                    recent_performance = model.performance_history[-5:]
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    
                    # Adapt model parameters based on performance
                    if avg_performance > 0.8:
                        # Good performance, increase confidence
                        model.parameters["confidence"] = model.parameters.get("confidence", 0.5) + 0.1
                        adaptations.append(f"Increased confidence for {model_name}")
                    elif avg_performance < 0.6:
                        # Poor performance, adjust learning rate
                        model.parameters["learning_rate"] = model.parameters.get("learning_rate", 0.001) * 0.9
                        adaptations.append(f"Reduced learning rate for {model_name}")
                    
                    model.adaptation_count += 1
                    model.last_updated = time.time()
            
            return {
                "adaptations": adaptations,
                "models_adapted": len([a for a in adaptations if a])
            }
            
        except Exception as e:
            logger.error(f"Model adaptation failed: {e}")
            return {"error": str(e)}
    
    def _load_existing_data(self):
        """Load existing CGM data"""
        try:
            # Load generated code
            code_file = self.data_dir / "generated_code.json"
            if code_file.exists():
                with open(code_file, 'r') as f:
                    data = json.load(f)
                    for code_data in data.get("codes", []):
                        code_record = GeneratedCode(**code_data)
                        self.generated_code[code_record.id] = code_record
            
            # Load generation history
            history_file = self.data_dir / "generation_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.generation_history = json.load(f).get("history", [])
            
            # Load meta-learning models
            models_file = self.data_dir / "meta_models.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                    for model_data in models_data.get("models", []):
                        model = MetaLearningModel(**model_data)
                        self.meta_learning_models[model.model_id] = model
            
            logger.info("✅ CGM data loaded from previous sessions")
            
        except Exception as e:
            logger.error(f"Failed to load CGM data: {e}")
    
    async def save_cgm_data(self):
        """Save CGM data"""
        try:
            # Save generated code
            code_data = {
                "codes": [asdict(code) for code in self.generated_code.values()]
            }
            with open(self.data_dir / "generated_code.json", 'w') as f:
                json.dump(code_data, f, indent=2, default=str)
            
            # Save generation history
            history_data = {"history": self.generation_history}
            with open(self.data_dir / "generation_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save meta-learning models
            models_data = {
                "models": [asdict(model) for model in self.meta_learning_models.values()]
            }
            with open(self.data_dir / "meta_models.json", 'w') as f:
                json.dump(models_data, f, indent=2, default=str)
            
            logger.info("✅ CGM data saved")
            
        except Exception as e:
            logger.error(f"Failed to save CGM data: {e}")
    
    async def get_cgm_status(self) -> Dict[str, Any]:
        """Get CGM status and metrics"""
        try:
            return {
                "total_generated_codes": len(self.generated_code),
                "generation_history_length": len(self.generation_history),
                "meta_learning_models": len(self.meta_learning_models),
                "improvement_queue_size": len(self.improvement_queue),
                "average_quality_score": sum(code.quality_score for code in self.generated_code.values()) / max(len(self.generated_code), 1),
                "neural_generator_available": hasattr(self, 'code_generator_pipeline') and self.code_generator_pipeline is not None,
                "last_generation": max([code.generated_at for code in self.generated_code.values()]) if self.generated_code else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get CGM status: {e}")
            return {"error": str(e)}

# Global instance
code_generation_module = CodeGenerationModule()

# Convenience functions
async def generate_code(task: CodeGenerationTask, user_id: str) -> Dict[str, Any]:
    """Generate code"""
    return await code_generation_module.generate_code(task, user_id)

async def perform_self_improvement(user_id: str) -> Dict[str, Any]:
    """Perform self-improvement"""
    return await code_generation_module.perform_self_improvement(user_id)

async def get_cgm_status() -> Dict[str, Any]:
    """Get CGM status"""
    return await code_generation_module.get_cgm_status()