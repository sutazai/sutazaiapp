"""
Code Enhancement Engine
Advanced automated code analysis, improvement, and suggestion system
"""

import ast
import asyncio
import logging
import time
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import re
import uuid

logger = logging.getLogger(__name__)

@dataclass
class CodeMetrics:
    """Code quality metrics"""
    lines_of_code: int
    cyclomatic_complexity: int
    maintainability_index: float
    technical_debt_ratio: float
    test_coverage: float
    security_score: float
    performance_score: float

@dataclass
class Enhancement:
    """Code enhancement suggestion"""
    id: str
    file_path: str
    line_number: int
    enhancement_type: str
    priority: str  # high, medium, low
    title: str
    description: str
    original_code: str
    enhanced_code: str
    reasoning: str
    confidence: float
    estimated_impact: str
    implementation_difficulty: str
    automated_fix: bool
    test_required: bool
    created_at: float
    status: str = "pending"  # pending, approved, applied, rejected

class CodeEnhancementEngine:
    """
    Advanced Code Enhancement Engine
    Analyzes code and provides intelligent improvement suggestions
    """
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "code_enhancement"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhancement tracking
        self.enhancements = {}
        self.metrics_history = []
        self.performance_baselines = {}
        
        # Pattern libraries
        self.patterns = self._load_enhancement_patterns()
        self.security_patterns = self._load_security_patterns()
        self.performance_patterns = self._load_performance_patterns()
        
        # Initialize
        self._load_existing_data()
    
    def _load_enhancement_patterns(self) -> Dict[str, Any]:
        """Load code enhancement patterns"""
        return {
            "type_hints": {
                "pattern": r"def\s+\w+\([^)]*\)(?!\s*->)",
                "suggestion": "Add type hints for better code clarity and IDE support",
                "priority": "medium",
                "automated": True
            },
            "error_handling": {
                "patterns": [
                    r"open\s*\(",
                    r"requests\.",
                    r"json\.loads",
                    r"int\s*\(",
                    r"float\s*\("
                ],
                "suggestion": "Add proper error handling",
                "priority": "high",
                "automated": True
            },
            "docstrings": {
                "pattern": r"def\s+\w+\([^)]*\):\s*\n(?!\s*\"\"\")",
                "suggestion": "Add docstring for better documentation",
                "priority": "medium",
                "automated": True
            },
            "magic_numbers": {
                "pattern": r"\b\d{2,}\b",
                "suggestion": "Replace magic numbers with named constants",
                "priority": "medium",
                "automated": False
            },
            "long_functions": {
                "threshold": 50,
                "suggestion": "Consider breaking down long functions",
                "priority": "medium",
                "automated": False
            }
        }
    
    def _load_security_patterns(self) -> Dict[str, Any]:
        """Load security analysis patterns"""
        return {
            "sql_injection": {
                "patterns": [
                    r"execute\s*\(\s*[\"'].*%.*[\"']\s*%",
                    r"\.format\s*\(.*sql",
                    r"f[\"'].*{.*}.*[\"'].*execute"
                ],
                "severity": "critical",
                "suggestion": "Use parameterized queries to prevent SQL injection"
            },
            "hardcoded_secrets": {
                "patterns": [
                    r"password\s*=\s*[\"'][^\"']+[\"']",
                    r"api_key\s*=\s*[\"'][^\"']+[\"']",
                    r"secret\s*=\s*[\"'][^\"']+[\"']",
                    r"token\s*=\s*[\"'][^\"']+[\"']"
                ],
                "severity": "high",
                "suggestion": "Use environment variables or secure vaults for secrets"
            },
            "unsafe_functions": {
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"subprocess.*shell\s*=\s*True",
                    r"os\.system\s*\("
                ],
                "severity": "high",
                "suggestion": "Replace with safer alternatives"
            },
            "weak_crypto": {
                "patterns": [
                    r"md5\s*\(",
                    r"sha1\s*\(",
                    r"DES\s*\(",
                    r"RC4\s*\("
                ],
                "severity": "medium",
                "suggestion": "Use stronger cryptographic algorithms"
            }
        }
    
    def _load_performance_patterns(self) -> Dict[str, Any]:
        """Load performance optimization patterns"""
        return {
            "inefficient_loops": {
                "patterns": [
                    r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(",
                    r"\w+\s*\+=\s*\w+\s*\+\s*\w+"
                ],
                "suggestion": "Optimize loop patterns for better performance",
                "impact": "medium"
            },
            "repeated_calculations": {
                "patterns": [
                    r"len\s*\(\s*\w+\s*\).*len\s*\(\s*\w+\s*\)",
                    r"\w+\.\w+\(\).*\w+\.\w+\(\)"
                ],
                "suggestion": "Cache repeated calculations",
                "impact": "low"
            },
            "memory_inefficient": {
                "patterns": [
                    r"\[\s*.*\s*for\s+.*\s+in\s+.*\s*if\s+.*\]",
                    r"\.join\s*\(\s*\[.*\]\s*\)"
                ],
                "suggestion": "Consider using generators for memory efficiency",
                "impact": "medium"
            },
            "blocking_io": {
                "patterns": [
                    r"requests\.get\s*\(",
                    r"open\s*\(",
                    r"time\.sleep\s*\("
                ],
                "suggestion": "Consider async/await for I/O operations",
                "impact": "high"
            }
        }
    
    async def analyze_file(self, file_path: str, generate_fixes: bool = True) -> Dict[str, Any]:
        """Comprehensively analyze a Python file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"error": "File not found"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for deep analysis
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {
                    "error": "Syntax error",
                    "details": str(e),
                    "line": e.lineno if hasattr(e, 'lineno') else None
                }
            
            analysis_result = {
                "file_path": str(file_path),
                "timestamp": time.time(),
                "metrics": await self._calculate_metrics(content, tree),
                "enhancements": await self._generate_enhancements(file_path, content, tree) if generate_fixes else [],
                "security_issues": await self._analyze_security(content),
                "performance_issues": await self._analyze_performance(content),
                "code_smells": await self._detect_code_smells(content, tree),
                "test_recommendations": await self._suggest_tests(content, tree),
                "documentation_score": self._calculate_documentation_score(content, tree)
            }
            
            # Store analysis for learning
            await self._store_analysis_result(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"File analysis failed for {file_path}: {e}")
            return {"error": str(e)}
    
    async def _calculate_metrics(self, content: str, tree: ast.AST) -> CodeMetrics:
        """Calculate comprehensive code metrics"""
        lines = content.splitlines()
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(tree)
        
        # Maintainability index (simplified)
        maintainability = max(0, 171 - 5.2 * complexity - 0.23 * len(code_lines))
        
        # Technical debt ratio (based on various factors)
        debt_factors = [
            len([line for line in lines if 'TODO' in line or 'FIXME' in line]) * 5,
            len([line for line in lines if len(line) > 100]) * 2,
            complexity * 3
        ]
        debt_ratio = min(100, sum(debt_factors) / max(len(code_lines), 1) * 10)
        
        return CodeMetrics(
            lines_of_code=len(code_lines),
            cyclomatic_complexity=complexity,
            maintainability_index=maintainability,
            technical_debt_ratio=debt_ratio,
            test_coverage=0.0,  # Would integrate with coverage tools
            security_score=self._calculate_security_score(content),
            performance_score=self._calculate_performance_score(content)
        )
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _calculate_security_score(self, content: str) -> float:
        """Calculate security score (0-100)"""
        score = 100
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns.get("patterns", []):
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    severity_penalty = {
                        "critical": 30,
                        "high": 20,
                        "medium": 10,
                        "low": 5
                    }
                    score -= matches * severity_penalty.get(patterns.get("severity", "low"), 5)
        
        return max(0, score)
    
    def _calculate_performance_score(self, content: str) -> float:
        """Calculate performance score (0-100)"""
        score = 100
        
        for category, patterns in self.performance_patterns.items():
            for pattern in patterns.get("patterns", []):
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    impact_penalty = {
                        "high": 15,
                        "medium": 10,
                        "low": 5
                    }
                    score -= matches * impact_penalty.get(patterns.get("impact", "low"), 5)
        
        return max(0, score)
    
    async def _generate_enhancements(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Generate intelligent code enhancements"""
        enhancements = []
        lines = content.splitlines()
        
        # Type hints enhancement
        enhancements.extend(await self._suggest_type_hints(file_path, content, tree))
        
        # Error handling enhancements
        enhancements.extend(await self._suggest_error_handling(file_path, content, lines))
        
        # Documentation enhancements
        enhancements.extend(await self._suggest_documentation(file_path, content, tree))
        
        # Performance enhancements
        enhancements.extend(await self._suggest_performance_improvements(file_path, content, lines))
        
        # Code structure enhancements
        enhancements.extend(await self._suggest_structural_improvements(file_path, content, tree))
        
        return enhancements[:20]  # Limit to top 20 suggestions
    
    async def _suggest_type_hints(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Suggest type hints for functions"""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.returns:  # No return type annotation
                    enhancement = Enhancement(
                        id=str(uuid.uuid4()),
                        file_path=str(file_path),
                        line_number=node.lineno,
                        enhancement_type="type_hints",
                        priority="medium",
                        title=f"Add type hints to function '{node.name}'",
                        description="Adding type hints improves code readability and enables better IDE support",
                        original_code=self._get_code_at_line(content, node.lineno),
                        enhanced_code=self._generate_type_hint_example(node),
                        reasoning="Type hints provide better code documentation and catch type-related errors early",
                        confidence=0.8,
                        estimated_impact="medium",
                        implementation_difficulty="easy",
                        automated_fix=True,
                        test_required=False,
                        created_at=time.time()
                    )
                    suggestions.append(asdict(enhancement))
        
        return suggestions
    
    def _generate_type_hint_example(self, node: ast.FunctionDef) -> str:
        """Generate example with type hints"""
        args = [arg.arg for arg in node.args.args]
        args_with_types = [f"{arg}: Any" for arg in args]
        return f"def {node.name}({', '.join(args_with_types)}) -> Any:"
    
    async def _suggest_error_handling(self, file_path: Path, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Suggest error handling improvements"""
        suggestions = []
        
        for i, line in enumerate(lines):
            # Check for file operations without error handling
            if "open(" in line and not any("try:" in lines[max(0, i-3):i+1]):
                enhancement = Enhancement(
                    id=str(uuid.uuid4()),
                    file_path=str(file_path),
                    line_number=i + 1,
                    enhancement_type="error_handling",
                    priority="high",
                    title="Add error handling for file operation",
                    description="File operations should be wrapped in try-except blocks",
                    original_code=line.strip(),
                    enhanced_code=self._generate_error_handling_example(line.strip()),
                    reasoning="File operations can fail due to permissions, missing files, etc.",
                    confidence=0.9,
                    estimated_impact="high",
                    implementation_difficulty="easy",
                    automated_fix=True,
                    test_required=True,
                    created_at=time.time()
                )
                suggestions.append(asdict(enhancement))
        
        return suggestions[:5]  # Limit suggestions
    
    def _generate_error_handling_example(self, original_line: str) -> str:
        """Generate error handling example"""
        return f"""try:
    {original_line}
except FileNotFoundError:
    logger.error("File not found")
    # Handle appropriately
except Exception as e:
    logger.error(f"File operation failed: {{e}}")"""
    
    async def _suggest_documentation(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Suggest documentation improvements"""
        suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    enhancement = Enhancement(
                        id=str(uuid.uuid4()),
                        file_path=str(file_path),
                        line_number=node.lineno,
                        enhancement_type="documentation",
                        priority="medium",
                        title=f"Add docstring to {type(node).__name__.lower()} '{node.name}'",
                        description="Docstrings improve code maintainability and enable automatic documentation generation",
                        original_code=self._get_code_at_line(content, node.lineno),
                        enhanced_code=self._generate_docstring_example(node),
                        reasoning="Good documentation makes code more maintainable and easier to understand",
                        confidence=0.7,
                        estimated_impact="medium",
                        implementation_difficulty="easy",
                        automated_fix=True,
                        test_required=False,
                        created_at=time.time()
                    )
                    suggestions.append(asdict(enhancement))
        
        return suggestions[:5]
    
    def _generate_docstring_example(self, node: ast.AST) -> str:
        """Generate docstring example"""
        if isinstance(node, ast.FunctionDef):
            return f'''def {node.name}():
    """
    Brief description of what this function does.
    
    Args:
        param1: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        Exception: Description of when this exception is raised
    """'''
        elif isinstance(node, ast.ClassDef):
            return f'''class {node.name}:
    """
    Brief description of what this class does.
    
    Attributes:
        attribute1: Description of attribute
        
    Methods:
        method1: Description of method
    """'''
        return ""
    
    async def _suggest_performance_improvements(self, file_path: Path, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Suggest performance improvements"""
        suggestions = []
        
        for i, line in enumerate(lines):
            # Check for inefficient loop patterns
            if re.search(r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(", line):
                enhancement = Enhancement(
                    id=str(uuid.uuid4()),
                    file_path=str(file_path),
                    line_number=i + 1,
                    enhancement_type="performance",
                    priority="medium",
                    title="Optimize loop pattern",
                    description="Use enumerate() instead of range(len()) for better readability and slight performance improvement",
                    original_code=line.strip(),
                    enhanced_code=self._generate_enumerate_example(line.strip()),
                    reasoning="enumerate() is more pythonic and slightly more efficient",
                    confidence=0.8,
                    estimated_impact="low",
                    implementation_difficulty="easy",
                    automated_fix=True,
                    test_required=False,
                    created_at=time.time()
                )
                suggestions.append(asdict(enhancement))
        
        return suggestions[:3]
    
    def _generate_enumerate_example(self, original_line: str) -> str:
        """Generate enumerate example"""
        # This is a simplified example - would need more sophisticated parsing
        return original_line.replace("for i in range(len(", "for i, item in enumerate(")
    
    async def _suggest_structural_improvements(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Suggest structural improvements"""
        suggestions = []
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines_in_function = self._count_function_lines(content, node)
                if lines_in_function > 50:
                    enhancement = Enhancement(
                        id=str(uuid.uuid4()),
                        file_path=str(file_path),
                        line_number=node.lineno,
                        enhancement_type="structure",
                        priority="medium",
                        title=f"Consider refactoring long function '{node.name}'",
                        description=f"Function has {lines_in_function} lines. Consider breaking it into smaller functions",
                        original_code=f"def {node.name}(...): # {lines_in_function} lines",
                        enhanced_code="# Break into smaller, focused functions",
                        reasoning="Smaller functions are easier to test, understand, and maintain",
                        confidence=0.6,
                        estimated_impact="medium",
                        implementation_difficulty="medium",
                        automated_fix=False,
                        test_required=True,
                        created_at=time.time()
                    )
                    suggestions.append(asdict(enhancement))
        
        return suggestions[:2]
    
    def _count_function_lines(self, content: str, node: ast.FunctionDef) -> int:
        """Count lines in a function"""
        lines = content.splitlines()
        start_line = node.lineno - 1
        
        # Find end of function (simplified)
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = start_line + 1
        
        while end_line < len(lines):
            line = lines[end_line]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent_level:
                break
            end_line += 1
        
        return end_line - start_line
    
    async def _analyze_security(self, content: str) -> List[Dict[str, Any]]:
        """Analyze security vulnerabilities"""
        security_issues = []
        
        for category, config in self.security_patterns.items():
            for pattern in config.get("patterns", []):
                matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    security_issues.append({
                        "category": category,
                        "severity": config.get("severity", "medium"),
                        "line_number": line_num,
                        "description": config.get("suggestion", "Security issue detected"),
                        "code_snippet": match.group(0),
                        "recommendation": config.get("suggestion", "Review and fix this security issue")
                    })
        
        return security_issues
    
    async def _analyze_performance(self, content: str) -> List[Dict[str, Any]]:
        """Analyze performance issues"""
        performance_issues = []
        
        for category, config in self.performance_patterns.items():
            for pattern in config.get("patterns", []):
                matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    performance_issues.append({
                        "category": category,
                        "impact": config.get("impact", "medium"),
                        "line_number": line_num,
                        "description": config.get("suggestion", "Performance issue detected"),
                        "code_snippet": match.group(0),
                        "recommendation": config.get("suggestion", "Consider optimizing this code")
                    })
        
        return performance_issues
    
    async def _detect_code_smells(self, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect code smells"""
        smells = []
        
        # Long parameter lists
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    smells.append({
                        "type": "long_parameter_list",
                        "line_number": node.lineno,
                        "description": f"Function '{node.name}' has {len(node.args.args)} parameters",
                        "suggestion": "Consider using a configuration object or breaking down the function"
                    })
        
        # Deep nesting
        nesting_depth = self._calculate_max_nesting_depth(tree)
        if nesting_depth > 4:
            smells.append({
                "type": "deep_nesting",
                "description": f"Maximum nesting depth is {nesting_depth}",
                "suggestion": "Consider extracting nested logic into separate functions"
            })
        
        return smells
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                current_depth += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_depth(child, current_depth)
        
        calculate_depth(tree)
        return max_depth
    
    async def _suggest_tests(self, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Suggest test cases"""
        test_suggestions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                test_suggestions.append({
                    "function_name": node.name,
                    "line_number": node.lineno,
                    "suggested_tests": [
                        "Test with valid inputs",
                        "Test with edge cases",
                        "Test error conditions"
                    ],
                    "test_file_suggestion": f"test_{Path(self.project_root).name}_{node.name}.py"
                })
        
        return test_suggestions[:5]
    
    def _calculate_documentation_score(self, content: str, tree: ast.AST) -> float:
        """Calculate documentation score"""
        total_items = 0
        documented_items = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total_items += 1
                if ast.get_docstring(node):
                    documented_items += 1
        
        return (documented_items / total_items * 100) if total_items > 0 else 100
    
    def _get_code_at_line(self, content: str, line_number: int) -> str:
        """Get code at specific line"""
        lines = content.splitlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1].strip()
        return ""
    
    async def _store_analysis_result(self, result: Dict[str, Any]):
        """Store analysis result for learning"""
        try:
            analysis_file = self.data_dir / f"analysis_{int(time.time())}.json"
            with open(analysis_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Keep only last 100 analysis files
            analysis_files = sorted(self.data_dir.glob("analysis_*.json"))
            if len(analysis_files) > 100:
                for old_file in analysis_files[:-100]:
                    old_file.unlink()
                    
        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")
    
    def _load_existing_data(self):
        """Load existing enhancement data"""
        try:
            data_file = self.data_dir / "enhancements.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.enhancements = data.get("enhancements", {})
                    self.metrics_history = data.get("metrics_history", [])
                    
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
    
    async def apply_enhancement(self, enhancement_id: str, user_email: str) -> bool:
        """Apply an approved enhancement"""
        try:
            enhancement = self.enhancements.get(enhancement_id)
            if not enhancement or enhancement.get("status") != "approved":
                return False
            
            # Apply the enhancement (simplified implementation)
            if enhancement.get("automated_fix"):
                # This would implement the actual code modification
                logger.info(f"Applied enhancement {enhancement_id}")
                enhancement["status"] = "applied"
                enhancement["applied_by"] = user_email
                enhancement["applied_at"] = time.time()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply enhancement {enhancement_id}: {e}")
            return False
    
    async def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get summary of all enhancements"""
        total = len(self.enhancements)
        by_status = {}
        by_priority = {}
        
        for enhancement in self.enhancements.values():
            status = enhancement.get("status", "pending")
            priority = enhancement.get("priority", "medium")
            
            by_status[status] = by_status.get(status, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        return {
            "total_enhancements": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "automated_fixes_available": len([e for e in self.enhancements.values() if e.get("automated_fix")]),
            "estimated_impact_high": len([e for e in self.enhancements.values() if e.get("estimated_impact") == "high"])
        }

# Global instance
code_enhancement_engine = CodeEnhancementEngine()

# Convenience functions
async def analyze_file(file_path: str) -> Dict[str, Any]:
    """Analyze a file"""
    return await code_enhancement_engine.analyze_file(file_path)

async def apply_enhancement(enhancement_id: str, user_email: str) -> bool:
    """Apply an enhancement"""
    return await code_enhancement_engine.apply_enhancement(enhancement_id, user_email)

async def get_enhancement_summary() -> Dict[str, Any]:
    """Get enhancement summary"""
    return await code_enhancement_engine.get_enhancement_summary()