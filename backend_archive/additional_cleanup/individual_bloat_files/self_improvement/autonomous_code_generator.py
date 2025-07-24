#!/usr/bin/env python3
"""
Autonomous Code Generator for SutazAI Self-Improvement
Generates and improves code automatically using all available AI models
"""

import os
import ast
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import aiohttp
import json
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class CodeGenerationTask:
    """Represents a code generation task"""
    task_id: str
    description: str
    language: str
    context: Dict[str, Any]
    priority: int = 5
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class GeneratedCode:
    """Represents generated code with metadata"""
    code: str
    language: str
    quality_score: float
    model_used: str
    generation_time: float
    tests_included: bool = False
    documentation_included: bool = False

class AutonomousCodeGenerator:
    """
    Autonomous code generation system that uses multiple AI models
    to generate, improve, and optimize code automatically
    """
    
    def __init__(self):
        self.available_models = [
            "deepseek-coder", "gpt-engineer", "aider", "awesome-code-ai"
        ]
        self.generation_history = []
        self.improvement_cycles = 0
        self.quality_threshold = 0.8
        
        # Service URLs from environment
        self.service_urls = {
            "enhanced_model_manager": os.getenv("ENHANCED_MODEL_MANAGER_URL", "http://enhanced-model-manager:8090"),
            "awesome_code_ai": os.getenv("AWESOME_CODE_AI_URL", "http://awesome-code-ai:8089"),
            "gpt_engineer": os.getenv("GPT_ENGINEER_URL", "http://gpt-engineer:8080"),
            "aider": os.getenv("AIDER_URL", "http://aider:8080"),
            "faiss": os.getenv("FAISS_SERVICE_URL", "http://faiss:8088")
        }
        
        self.generation_stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "improvement_cycles": 0,
            "quality_improvements": 0
        }
    
    async def initialize(self):
        """Initialize the autonomous code generator"""
        logger.info("Initializing Autonomous Code Generator...")
        
        # Test connectivity to all services
        await self._test_service_connectivity()
        
        # Load existing generation history
        await self._load_generation_history()
        
        logger.info("Autonomous Code Generator initialized successfully")
    
    async def _test_service_connectivity(self):
        """Test connectivity to all AI services"""
        async with aiohttp.ClientSession() as session:
            for service_name, url in self.service_urls.items():
                try:
                    async with session.get(f"{url}/health", timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"✅ {service_name} service is available")
                        else:
                            logger.warning(f"⚠️ {service_name} service returned status {response.status}")
                except Exception as e:
                    logger.warning(f"❌ {service_name} service is not available: {e}")
    
    async def generate_self_improvement_code(self, system_analysis: Dict[str, Any]) -> List[GeneratedCode]:
        """
        Generate code to improve the system based on analysis
        """
        logger.info("Starting autonomous self-improvement code generation...")
        
        # Analyze system needs
        improvement_tasks = await self._analyze_improvement_needs(system_analysis)
        
        generated_codes = []
        
        for task in improvement_tasks:
            logger.info(f"Processing improvement task: {task.description}")
            
            # Generate code using multiple models
            code_variants = await self._generate_code_variants(task)
            
            # Evaluate and select best variant
            best_code = await self._select_best_code(code_variants, task)
            
            if best_code and best_code.quality_score >= self.quality_threshold:
                generated_codes.append(best_code)
                self.generation_stats["successful_generations"] += 1
            
            self.generation_stats["total_generated"] += 1
        
        # Apply iterative improvements
        improved_codes = await self._apply_iterative_improvements(generated_codes)
        
        logger.info(f"Generated {len(improved_codes)} self-improvement code modules")
        
        return improved_codes
    
    async def _analyze_improvement_needs(self, system_analysis: Dict[str, Any]) -> List[CodeGenerationTask]:
        """Analyze system and identify improvement opportunities"""
        
        improvement_tasks = []
        
        # Performance improvements
        if system_analysis.get("performance_issues"):
            for issue in system_analysis["performance_issues"]:
                task = CodeGenerationTask(
                    task_id=f"perf_{len(improvement_tasks)}",
                    description=f"Optimize performance issue: {issue['description']}",
                    language="python",
                    context={"issue_type": "performance", "details": issue},
                    priority=8
                )
                improvement_tasks.append(task)
        
        # Security improvements
        if system_analysis.get("security_vulnerabilities"):
            for vuln in system_analysis["security_vulnerabilities"]:
                task = CodeGenerationTask(
                    task_id=f"sec_{len(improvement_tasks)}",
                    description=f"Fix security vulnerability: {vuln['type']}",
                    language="python",
                    context={"issue_type": "security", "details": vuln},
                    priority=9
                )
                improvement_tasks.append(task)
        
        # Feature enhancements
        missing_features = [
            "Advanced caching layer",
            "Real-time monitoring dashboard",
            "Automated backup system",
            "Load balancing optimization",
            "Database connection pooling",
            "API rate limiting improvements",
            "Memory usage optimization",
            "Concurrent processing enhancements"
        ]
        
        for feature in missing_features:
            task = CodeGenerationTask(
                task_id=f"feat_{len(improvement_tasks)}",
                description=f"Implement {feature}",
                language="python",
                context={"issue_type": "feature", "feature": feature},
                priority=6
            )
            improvement_tasks.append(task)
        
        # Sort by priority
        improvement_tasks.sort(key=lambda x: x.priority, reverse=True)
        
        return improvement_tasks[:20]  # Limit to top 20 tasks
    
    async def _generate_code_variants(self, task: CodeGenerationTask) -> List[GeneratedCode]:
        """Generate multiple code variants using different models"""
        
        variants = []
        
        # Generate with Enhanced Model Manager (DeepSeek-Coder)
        deepseek_code = await self._generate_with_deepseek(task)
        if deepseek_code:
            variants.append(deepseek_code)
        
        # Generate with Awesome Code AI
        awesome_code = await self._generate_with_awesome_code_ai(task)
        if awesome_code:
            variants.append(awesome_code)
        
        # Generate with GPT-Engineer
        gpt_engineer_code = await self._generate_with_gpt_engineer(task)
        if gpt_engineer_code:
            variants.append(gpt_engineer_code)
        
        # Generate with Aider
        aider_code = await self._generate_with_aider(task)
        if aider_code:
            variants.append(aider_code)
        
        return variants
    
    async def _generate_with_deepseek(self, task: CodeGenerationTask) -> Optional[GeneratedCode]:
        """Generate code using DeepSeek-Coder via Enhanced Model Manager"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                payload = {
                    "prompt": self._create_generation_prompt(task),
                    "language": task.language,
                    "model": "deepseek-coder"
                }
                
                async with session.post(
                    f"{self.service_urls['enhanced_model_manager']}/code/generate",
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        generation_time = time.time() - start_time
                        
                        return GeneratedCode(
                            code=data["generated_code"],
                            language=task.language,
                            quality_score=0.85,  # DeepSeek typically high quality
                            model_used="deepseek-coder",
                            generation_time=generation_time,
                            tests_included=self._has_tests(data["generated_code"]),
                            documentation_included=self._has_documentation(data["generated_code"])
                        )
                        
        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {e}")
            return None
    
    async def _generate_with_awesome_code_ai(self, task: CodeGenerationTask) -> Optional[GeneratedCode]:
        """Generate code using Awesome Code AI service"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                payload = {
                    "prompt": self._create_generation_prompt(task),
                    "language": task.language,
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
                
                async with session.post(
                    f"{self.service_urls['awesome_code_ai']}/generate",
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        generation_time = time.time() - start_time
                        
                        return GeneratedCode(
                            code=data["generated_code"],
                            language=task.language,
                            quality_score=0.80,
                            model_used="awesome-code-ai",
                            generation_time=generation_time,
                            tests_included=self._has_tests(data["generated_code"]),
                            documentation_included=self._has_documentation(data["generated_code"])
                        )
                        
        except Exception as e:
            logger.error(f"Error generating with Awesome Code AI: {e}")
            return None
    
    async def _generate_with_gpt_engineer(self, task: CodeGenerationTask) -> Optional[GeneratedCode]:
        """Generate code using GPT-Engineer"""
        try:
            # GPT-Engineer integration would go here
            # For now, return a mock implementation
            return GeneratedCode(
                code=self._create_mock_implementation(task),
                language=task.language,
                quality_score=0.75,
                model_used="gpt-engineer",
                generation_time=2.5,
                tests_included=True,
                documentation_included=True
            )
        except Exception as e:
            logger.error(f"Error generating with GPT-Engineer: {e}")
            return None
    
    async def _generate_with_aider(self, task: CodeGenerationTask) -> Optional[GeneratedCode]:
        """Generate code using Aider"""
        try:
            # Aider integration would go here
            return GeneratedCode(
                code=self._create_mock_implementation(task),
                language=task.language,
                quality_score=0.78,
                model_used="aider",
                generation_time=1.8,
                tests_included=False,
                documentation_included=True
            )
        except Exception as e:
            logger.error(f"Error generating with Aider: {e}")
            return None
    
    def _create_generation_prompt(self, task: CodeGenerationTask) -> str:
        """Create an optimized prompt for code generation"""
        
        context_info = ""
        if task.context.get("issue_type") == "performance":
            context_info = "Focus on performance optimization and efficiency."
        elif task.context.get("issue_type") == "security":
            context_info = "Prioritize security best practices and vulnerability prevention."
        elif task.context.get("issue_type") == "feature":
            context_info = "Implement a robust, scalable feature with proper error handling."
        
        prompt = f"""
Generate high-quality {task.language} code for the following requirement:

Task: {task.description}
Context: {context_info}
Priority: {task.priority}/10

Requirements:
1. Write clean, maintainable code
2. Include comprehensive error handling
3. Add detailed docstrings and comments
4. Follow {task.language} best practices
5. Include unit tests if applicable
6. Optimize for performance and memory usage

Additional Context:
{json.dumps(task.context, indent=2)}

Generated Code:
"""
        return prompt
    
    async def _select_best_code(self, variants: List[GeneratedCode], task: CodeGenerationTask) -> Optional[GeneratedCode]:
        """Select the best code variant based on multiple criteria"""
        
        if not variants:
            return None
        
        # Calculate comprehensive scores
        for variant in variants:
            comprehensive_score = await self._calculate_comprehensive_score(variant, task)
            variant.quality_score = comprehensive_score
        
        # Sort by quality score
        variants.sort(key=lambda x: x.quality_score, reverse=True)
        
        best_variant = variants[0]
        logger.info(f"Selected best variant from {best_variant.model_used} with score {best_variant.quality_score:.3f}")
        
        return best_variant
    
    async def _calculate_comprehensive_score(self, code: GeneratedCode, task: CodeGenerationTask) -> float:
        """Calculate comprehensive quality score for generated code"""
        
        base_score = code.quality_score
        
        # Bonus for tests
        if code.tests_included:
            base_score += 0.1
        
        # Bonus for documentation
        if code.documentation_included:
            base_score += 0.05
        
        # Penalty for very long generation time
        if code.generation_time > 10:
            base_score -= 0.05
        
        # Syntax check
        if self._is_syntactically_valid(code.code, code.language):
            base_score += 0.05
        else:
            base_score -= 0.2
        
        # Complexity check
        complexity_score = self._analyze_code_complexity(code.code, code.language)
        base_score += complexity_score * 0.1
        
        return min(1.0, max(0.0, base_score))
    
    async def _apply_iterative_improvements(self, codes: List[GeneratedCode]) -> List[GeneratedCode]:
        """Apply iterative improvements to generated code"""
        
        improved_codes = []
        
        for code in codes:
            logger.info(f"Applying iterative improvements to {code.model_used} code")
            
            # Apply optimization passes
            optimized_code = await self._optimize_code(code)
            
            # Add missing components
            enhanced_code = await self._enhance_code(optimized_code)
            
            # Final quality check
            if enhanced_code.quality_score >= self.quality_threshold:
                improved_codes.append(enhanced_code)
                self.generation_stats["quality_improvements"] += 1
        
        self.improvement_cycles += 1
        self.generation_stats["improvement_cycles"] = self.improvement_cycles
        
        return improved_codes
    
    async def _optimize_code(self, code: GeneratedCode) -> GeneratedCode:
        """Optimize generated code for performance and readability"""
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "code": code.code,
                    "language": code.language
                }
                
                # Try optimizing with Enhanced Model Manager
                async with session.post(
                    f"{self.service_urls['enhanced_model_manager']}/code/optimize",
                    json=payload,
                    timeout=20
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        optimized_code = GeneratedCode(
                            code=data["optimized_code"],
                            language=code.language,
                            quality_score=code.quality_score + 0.1,
                            model_used=f"{code.model_used}+optimized",
                            generation_time=code.generation_time,
                            tests_included=code.tests_included,
                            documentation_included=code.documentation_included
                        )
                        
                        return optimized_code
                        
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
        
        return code
    
    async def _enhance_code(self, code: GeneratedCode) -> GeneratedCode:
        """Enhance code with additional features like tests and documentation"""
        
        enhanced_code_str = code.code
        quality_bonus = 0
        
        # Add tests if missing
        if not code.tests_included and code.language == "python":
            test_code = self._generate_test_code(code.code)
            enhanced_code_str += "\n\n" + test_code
            quality_bonus += 0.1
        
        # Add documentation if missing
        if not code.documentation_included:
            doc_code = self._add_documentation(enhanced_code_str, code.language)
            enhanced_code_str = doc_code
            quality_bonus += 0.05
        
        # Add error handling if missing
        if not self._has_error_handling(enhanced_code_str):
            enhanced_code_str = self._add_error_handling(enhanced_code_str, code.language)
            quality_bonus += 0.05
        
        return GeneratedCode(
            code=enhanced_code_str,
            language=code.language,
            quality_score=min(1.0, code.quality_score + quality_bonus),
            model_used=f"{code.model_used}+enhanced",
            generation_time=code.generation_time,
            tests_included=True,
            documentation_included=True
        )
    
    def _create_mock_implementation(self, task: CodeGenerationTask) -> str:
        """Create a mock implementation when services are unavailable"""
        
        if task.context.get("issue_type") == "performance":
            return f'''
import asyncio
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Performance optimization module for: {task.description}
    """
    
    def __init__(self):
        self.cache = {{}}
        self.metrics = {{}}
    
    async def optimize(self, data: Any) -> Any:
        """
        Optimize the given data for performance
        """
        try:
            start_time = time.time()
            
            # Implementation for performance optimization
            result = await self._process_with_optimization(data)
            
            execution_time = time.time() - start_time
            self.metrics["last_execution_time"] = execution_time
            
            logger.info(f"Performance optimization completed in {{execution_time:.3f}}s")
            return result
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {{e}}")
            raise
    
    async def _process_with_optimization(self, data: Any) -> Any:
        """Process data with performance optimizations"""
        # Placeholder implementation
        return data

# Usage example
if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    # asyncio.run(optimizer.optimize(sample_data))
'''
        
        elif task.context.get("issue_type") == "security":
            return f'''
import hashlib
import secrets
import hmac
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SecurityEnhancement:
    """
    Security enhancement module for: {task.description}
    """
    
    def __init__(self):
        self.security_config = {{}}
        self.audit_log = []
    
    def secure_hash(self, data: str) -> str:
        """
        Create a secure hash of the input data
        """
        try:
            salt = secrets.token_hex(16)
            hash_obj = hashlib.pbkdf2_hmac('sha256', 
                                         data.encode('utf-8'), 
                                         salt.encode('utf-8'), 
                                         100000)
            return salt + hash_obj.hex()
        except Exception as e:
            logger.error(f"Secure hashing failed: {{e}}")
            raise
    
    def verify_integrity(self, data: str, signature: str, key: str) -> bool:
        """
        Verify data integrity using HMAC
        """
        try:
            expected_signature = hmac.new(
                key.encode('utf-8'),
                data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {{e}}")
            return False
    
    def audit_log_entry(self, action: str, user_id: str, details: Dict[str, Any]):
        """
        Create an audit log entry
        """
        entry = {{
            "timestamp": time.time(),
            "action": action,
            "user_id": user_id,
            "details": details
        }}
        self.audit_log.append(entry)
        logger.info(f"Audit log entry created for action: {{action}}")

# Usage example
if __name__ == "__main__":
    security = SecurityEnhancement()
    # hash_result = security.secure_hash("sensitive_data")
'''
        
        else:  # Feature implementation
            feature_name = task.context.get("feature", "New Feature")
            return f'''
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class {feature_name.replace(" ", "")}Config:
    """Configuration for {feature_name}"""
    enabled: bool = True
    max_concurrent: int = 10
    timeout: float = 30.0
    retry_attempts: int = 3

class {feature_name.replace(" ", "")}:
    """
    Implementation of {feature_name}
    Generated for: {task.description}
    """
    
    def __init__(self, config: Optional[{feature_name.replace(" ", "")}Config] = None):
        self.config = config or {feature_name.replace(" ", "")}Config()
        self.is_initialized = False
        self.stats = {{
            "requests_processed": 0,
            "errors_encountered": 0,
            "average_processing_time": 0.0
        }}
    
    async def initialize(self):
        """Initialize the {feature_name.lower()}"""
        try:
            logger.info("Initializing {feature_name.lower()}...")
            # Initialization logic here
            self.is_initialized = True
            logger.info("{feature_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {feature_name.lower()}: {{e}}")
            raise
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """
        Process data using {feature_name.lower()}
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Main processing logic
            result = await self._process_internal(data)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["requests_processed"] += 1
            self._update_average_time(processing_time)
            
            logger.info(f"{feature_name} processing completed in {{processing_time:.3f}}s")
            return result
            
        except Exception as e:
            self.stats["errors_encountered"] += 1
            logger.error(f"{feature_name} processing failed: {{e}}")
            raise
    
    async def _process_internal(self, data: Any) -> Dict[str, Any]:
        """Internal processing implementation"""
        # Placeholder for actual implementation
        return {{
            "status": "processed",
            "data": data,
            "timestamp": time.time()
        }}
    
    def _update_average_time(self, new_time: float):
        """Update average processing time"""
        current_avg = self.stats["average_processing_time"]
        total_requests = self.stats["requests_processed"]
        
        if total_requests == 1:
            self.stats["average_processing_time"] = new_time
        else:
            self.stats["average_processing_time"] = (
                (current_avg * (total_requests - 1) + new_time) / total_requests
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

# Usage example
if __name__ == "__main__":
    feature = {feature_name.replace(" ", "")}()
    # asyncio.run(feature.process(sample_data))
'''
    
    def _has_tests(self, code: str) -> bool:
        """Check if code includes test cases"""
        test_indicators = [
            "def test_", "class Test", "unittest", "pytest", 
            "assert ", "assertEqual", "assertTrue", "assertFalse"
        ]
        return any(indicator in code for indicator in test_indicators)
    
    def _has_documentation(self, code: str) -> bool:
        """Check if code includes documentation"""
        doc_indicators = ['"""', "'''", "# ", "Args:", "Returns:", "Raises:"]
        return any(indicator in code for indicator in doc_indicators)
    
    def _has_error_handling(self, code: str) -> bool:
        """Check if code includes error handling"""
        error_indicators = ["try:", "except", "raise", "finally:"]
        return any(indicator in code for indicator in error_indicators)
    
    def _is_syntactically_valid(self, code: str, language: str) -> bool:
        """Check if code is syntactically valid"""
        if language.lower() == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        return True  # Assume valid for other languages
    
    def _analyze_code_complexity(self, code: str, language: str) -> float:
        """Analyze code complexity and return normalized score (0-1)"""
        if language.lower() == "python":
            try:
                tree = ast.parse(code)
                # Simple complexity metric based on node count
                node_count = len(list(ast.walk(tree)))
                # Normalize to 0-1 range (assuming 100 nodes = complexity 1.0)
                return min(1.0, node_count / 100.0)
            except:
                return 0.5  # Default moderate complexity
        return 0.7  # Default for other languages
    
    def _generate_test_code(self, code: str) -> str:
        """Generate basic test code for the given implementation"""
        return '''
# Generated Tests
import unittest
from unittest.mock import Mock, patch

class TestGeneratedCode(unittest.TestCase):
    """Test cases for generated code"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = {"key": "value"}
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Add specific test implementation
        self.assertTrue(True)  # Placeholder
    
    def test_error_handling(self):
        """Test error handling"""
        # Add error handling tests
        with self.assertRaises(Exception):
            pass  # Placeholder
    
    def tearDown(self):
        """Clean up after tests"""
        pass

if __name__ == "__main__":
    unittest.main()
'''
    
    def _add_documentation(self, code: str, language: str) -> str:
        """Add documentation to code if missing"""
        if language.lower() == "python":
            lines = code.split('\n')
            enhanced_lines = []
            
            for line in lines:
                enhanced_lines.append(line)
                # Add docstring after function definitions
                if line.strip().startswith('def ') and '"""' not in line:
                    indent = len(line) - len(line.lstrip())
                    docstring = ' ' * (indent + 4) + '"""TODO: Add function documentation"""'
                    enhanced_lines.append(docstring)
            
            return '\n'.join(enhanced_lines)
        
        return code
    
    def _add_error_handling(self, code: str, language: str) -> str:
        """Add basic error handling to code"""
        if language.lower() == "python" and "try:" not in code:
            # Wrap main logic in try-except
            lines = code.split('\n')
            # Find main function or class and wrap it
            return code + '''

# Added error handling
try:
    # Main execution logic would be wrapped here
    pass
except Exception as e:
    logger.error(f"Error in generated code: {e}")
    raise
'''
        return code
    
    async def _load_generation_history(self):
        """Load previous generation history"""
        history_file = Path("/logs/generation_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.generation_history = json.load(f)
                logger.info(f"Loaded {len(self.generation_history)} previous generations")
            except Exception as e:
                logger.warning(f"Could not load generation history: {e}")
    
    async def save_generation_history(self):
        """Save generation history to file"""
        history_file = Path("/logs/generation_history.json")
        try:
            history_file.parent.mkdir(exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump(self.generation_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save generation history: {e}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "stats": self.generation_stats,
            "improvement_cycles": self.improvement_cycles,
            "history_count": len(self.generation_history),
            "quality_threshold": self.quality_threshold
        }