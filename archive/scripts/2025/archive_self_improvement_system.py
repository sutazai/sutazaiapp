#!/usr/bin/env python3
"""
Self-Improvement AI Code Generation System for SutazAI
Enables the system to analyze and improve its own code autonomously
"""

import os
import ast
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import aiofiles
import git
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes code quality and identifies improvement opportunities"""
    
    def __init__(self):
        self.metrics = {}
        self.suggestions = []
        
    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file for improvement opportunities"""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                
            tree = ast.parse(content)
            
            analysis = {
                "file": file_path,
                "metrics": {
                    "lines": len(content.splitlines()),
                    "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                    "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                    "imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                },
                "complexity": self._calculate_complexity(tree),
                "suggestions": []
            }
            
            # Check for common improvements
            if analysis["metrics"]["lines"] > 500:
                analysis["suggestions"].append({
                    "type": "refactor",
                    "message": "File is too long, consider splitting into modules"
                })
                
            if analysis["complexity"] > 10:
                analysis["suggestions"].append({
                    "type": "simplify",
                    "message": "High complexity detected, consider refactoring"
                })
                
            # Check for missing docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        analysis["suggestions"].append({
                            "type": "documentation",
                            "message": f"Missing docstring for {node.name}"
                        })
                        
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {"file": file_path, "error": str(e)}
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

class CodeGenerator:
    """Generates improved code based on analysis"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load code generation templates"""
        return {
            "function": '''def {name}({params}) -> {return_type}:
    """{docstring}"""
    {body}
''',
            "class": '''class {name}({bases}):
    """{docstring}"""
    
    def __init__(self{params}):
        {init_body}
    
    {methods}
''',
            "module": '''#!/usr/bin/env python3
"""
{module_docstring}
"""

{imports}

{constants}

{classes}

{functions}

if __name__ == "__main__":
    {main_block}
'''
        }
    
    async def generate_improvement(self, analysis: Dict[str, Any], suggestion: Dict[str, Any]) -> Optional[str]:
        """Generate improved code based on analysis and suggestion"""
        
        if suggestion["type"] == "refactor":
            return await self._generate_refactored_code(analysis)
        elif suggestion["type"] == "documentation":
            return await self._generate_documentation(analysis, suggestion)
        elif suggestion["type"] == "optimization":
            return await self._generate_optimized_code(analysis)
        elif suggestion["type"] == "test":
            return await self._generate_tests(analysis)
        else:
            return None
    
    async def _generate_refactored_code(self, analysis: Dict[str, Any]) -> str:
        """Generate refactored version of code"""
        prompt = f"""
        Refactor the following code metrics into a better structure:
        File: {analysis['file']}
        Lines: {analysis['metrics']['lines']}
        Functions: {analysis['metrics']['functions']}
        Complexity: {analysis['complexity']}
        
        Generate a refactored version that:
        1. Reduces complexity
        2. Improves readability
        3. Follows best practices
        4. Maintains functionality
        """
        
        response = await self.llm_client.generate(prompt)
        return response
    
    async def _generate_documentation(self, analysis: Dict[str, Any], suggestion: Dict[str, Any]) -> str:
        """Generate documentation for code"""
        prompt = f"""
        Generate comprehensive documentation for:
        {suggestion['message']}
        
        Include:
        1. Clear docstring
        2. Parameter descriptions
        3. Return value description
        4. Usage examples
        5. Any exceptions raised
        """
        
        response = await self.llm_client.generate(prompt)
        return response
    
    async def _generate_optimized_code(self, analysis: Dict[str, Any]) -> str:
        """Generate performance-optimized version of code"""
        prompt = f"""
        Optimize the code with these metrics:
        Complexity: {analysis['complexity']}
        
        Focus on:
        1. Algorithm efficiency
        2. Memory usage
        3. Async/await patterns
        4. Caching strategies
        5. Parallel processing
        """
        
        response = await self.llm_client.generate(prompt)
        return response
    
    async def _generate_tests(self, analysis: Dict[str, Any]) -> str:
        """Generate test cases for code"""
        prompt = f"""
        Generate comprehensive test cases for:
        Functions: {analysis['metrics']['functions']}
        Classes: {analysis['metrics']['classes']}
        
        Include:
        1. Unit tests
        2. Integration tests
        3. Edge cases
        4. Error handling tests
        5. Performance tests
        """
        
        response = await self.llm_client.generate(prompt)
        return response

class SelfImprovementSystem:
    """Main self-improvement system that coordinates analysis and generation"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.analyzer = CodeAnalyzer()
        self.generator = None  # Will be initialized with LLM client
        self.improvement_history = []
        self.repo = git.Repo(project_root)
        
    async def initialize(self, llm_client):
        """Initialize the self-improvement system"""
        self.generator = CodeGenerator(llm_client)
        await self.load_improvement_history()
        
    async def load_improvement_history(self):
        """Load history of improvements"""
        history_file = self.project_root / ".improvement_history.json"
        if history_file.exists():
            async with aiofiles.open(history_file, 'r') as f:
                content = await f.read()
                self.improvement_history = json.loads(content)
    
    async def save_improvement_history(self):
        """Save improvement history"""
        history_file = self.project_root / ".improvement_history.json"
        async with aiofiles.open(history_file, 'w') as f:
            await f.write(json.dumps(self.improvement_history, indent=2))
    
    async def analyze_project(self) -> Dict[str, Any]:
        """Analyze entire project for improvements"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": 0,
            "total_suggestions": 0,
            "by_type": {},
            "files": []
        }
        
        # Find all Python files
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            # Skip virtual environments and cache
            if any(part in str(file_path) for part in ["venv", "__pycache__", ".git"]):
                continue
                
            analysis = await self.analyzer.analyze_file(str(file_path))
            if "error" not in analysis:
                results["files_analyzed"] += 1
                results["total_suggestions"] += len(analysis.get("suggestions", []))
                
                for suggestion in analysis.get("suggestions", []):
                    stype = suggestion["type"]
                    results["by_type"][stype] = results["by_type"].get(stype, 0) + 1
                
                if analysis.get("suggestions"):
                    results["files"].append(analysis)
        
        return results
    
    async def generate_improvements(self, analysis_results: Dict[str, Any], 
                                  max_improvements: int = 5) -> List[Dict[str, Any]]:
        """Generate improvements based on analysis"""
        improvements = []
        improvement_count = 0
        
        for file_analysis in analysis_results.get("files", []):
            if improvement_count >= max_improvements:
                break
                
            for suggestion in file_analysis.get("suggestions", []):
                if improvement_count >= max_improvements:
                    break
                    
                # Generate improvement
                improved_code = await self.generator.generate_improvement(
                    file_analysis, suggestion
                )
                
                if improved_code:
                    improvement = {
                        "file": file_analysis["file"],
                        "type": suggestion["type"],
                        "description": suggestion["message"],
                        "improved_code": improved_code,
                        "timestamp": datetime.now().isoformat()
                    }
                    improvements.append(improvement)
                    improvement_count += 1
        
        return improvements
    
    async def apply_improvement(self, improvement: Dict[str, Any], 
                              require_approval: bool = True) -> Dict[str, Any]:
        """Apply an improvement to the codebase"""
        
        if require_approval:
            # In a real system, this would wait for user approval
            logger.info(f"Improvement requires approval: {improvement['description']}")
            return {
                "status": "pending_approval",
                "improvement": improvement
            }
        
        try:
            # Create a new branch for the improvement
            branch_name = f"improvement/{improvement['type']}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.repo.create_head(branch_name)
            self.repo.head.reference = self.repo.heads[branch_name]
            
            # Apply the improvement
            file_path = improvement["file"]
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(improvement["improved_code"])
            
            # Commit the change
            self.repo.index.add([file_path])
            commit_message = f"AI Improvement: {improvement['description']}"
            self.repo.index.commit(commit_message)
            
            # Record in history
            self.improvement_history.append({
                "timestamp": improvement["timestamp"],
                "file": improvement["file"],
                "type": improvement["type"],
                "description": improvement["description"],
                "branch": branch_name,
                "commit": str(self.repo.head.commit)
            })
            
            await self.save_improvement_history()
            
            return {
                "status": "success",
                "branch": branch_name,
                "commit": str(self.repo.head.commit)
            }
            
        except Exception as e:
            logger.error(f"Error applying improvement: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def continuous_improvement_loop(self, interval_hours: int = 24):
        """Run continuous improvement loop"""
        while True:
            try:
                logger.info("Starting self-improvement analysis...")
                
                # Analyze project
                analysis = await self.analyze_project()
                logger.info(f"Analysis complete: {analysis['files_analyzed']} files, "
                          f"{analysis['total_suggestions']} suggestions")
                
                # Generate improvements
                if analysis["total_suggestions"] > 0:
                    improvements = await self.generate_improvements(analysis, max_improvements=3)
                    logger.info(f"Generated {len(improvements)} improvements")
                    
                    # Apply improvements (with approval in production)
                    for improvement in improvements:
                        result = await self.apply_improvement(improvement, require_approval=True)
                        logger.info(f"Improvement result: {result['status']}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in improvement loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get statistics about improvements"""
        stats = {
            "total_improvements": len(self.improvement_history),
            "by_type": {},
            "by_file": {},
            "recent_improvements": self.improvement_history[-10:]
        }
        
        for improvement in self.improvement_history:
            # By type
            itype = improvement["type"]
            stats["by_type"][itype] = stats["by_type"].get(itype, 0) + 1
            
            # By file
            file = improvement["file"]
            stats["by_file"][file] = stats["by_file"].get(file, 0) + 1
        
        return stats

# Singleton instance
_self_improvement_system = None

def get_self_improvement_system() -> SelfImprovementSystem:
    """Get the singleton self-improvement system instance"""
    global _self_improvement_system
    if _self_improvement_system is None:
        _self_improvement_system = SelfImprovementSystem()
    return _self_improvement_system