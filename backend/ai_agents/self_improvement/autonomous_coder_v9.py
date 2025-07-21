#!/usr/bin/env python3
"""
SutazAI v9 Autonomous Self-Improvement System
AI system that analyzes, improves, and enhances its own codebase
"""

import asyncio
import aiofiles
import aiohttp
import ast
import black
import isort
import subprocess
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import difflib
import tempfile
import shutil
import git
from concurrent.futures import ThreadPoolExecutor
import pylint.lint
import mypy.api
import bandit.core.test_properties as bandit_test
from bandit.core import config
from bandit.core import manager

# Import our services
from ..services.container_communicator_v9 import ContainerCommunicator
from ..services.batch_processor_v9 import BatchProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovementType(Enum):
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    STYLE = "style"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"

@dataclass
class CodeAnalysis:
    """Results of code analysis"""
    file_path: str
    language: str
    complexity_score: float
    quality_score: float
    security_score: float
    test_coverage: float
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[str]

@dataclass
class ImprovementProposal:
    """A proposed code improvement"""
    id: str
    improvement_type: ImprovementType
    target_files: List[str]
    description: str
    rationale: str
    code_changes: Dict[str, str]  # file_path -> new_content
    impact_assessment: Dict[str, Any]
    risk_level: str  # low, medium, high
    expected_benefits: List[str]
    tests_affected: List[str]
    approval_status: ApprovalStatus
    created_at: float
    approved_at: Optional[float] = None
    implemented_at: Optional[float] = None

@dataclass
class ImprovementResults:
    """Results of implementing improvements"""
    proposal_id: str
    success: bool
    files_modified: List[str]
    tests_passed: bool
    performance_impact: Dict[str, Any]
    quality_improvement: Dict[str, Any]
    error_message: Optional[str] = None

class CodeAnalyzer:
    """Analyzes code quality, security, and performance"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def analyze_file(self, file_path: str) -> CodeAnalysis:
        """Analyze a single file"""
        try:
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Determine language
            language = self._detect_language(file_path)
            
            if language == "python":
                return await self._analyze_python_file(file_path, content)
            elif language in ["javascript", "typescript"]:
                return await self._analyze_js_file(file_path, content)
            else:
                return await self._analyze_generic_file(file_path, content)
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return self._create_error_analysis(file_path, str(e))

    async def _analyze_python_file(self, file_path: str, content: str) -> CodeAnalysis:
        """Analyze Python file"""
        issues = []
        metrics = {}
        suggestions = []
        
        try:
            # Parse AST for complexity analysis
            tree = ast.parse(content)
            complexity_score = self._calculate_complexity(tree)
            
            # Run pylint analysis
            pylint_issues = await self._run_pylint(file_path)
            issues.extend(pylint_issues)
            
            # Run mypy type checking
            mypy_issues = await self._run_mypy(file_path)
            issues.extend(mypy_issues)
            
            # Run Bandit security analysis
            security_issues = await self._run_bandit(file_path)
            issues.extend(security_issues)
            
            # Calculate metrics
            metrics = {
                "lines_of_code": len(content.split('\n')),
                "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                "imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                "complexity": complexity_score
            }
            
            # Generate suggestions
            suggestions = self._generate_python_suggestions(tree, issues)
            
            # Calculate scores
            quality_score = self._calculate_quality_score(issues, metrics)
            security_score = self._calculate_security_score(security_issues)
            
            return CodeAnalysis(
                file_path=file_path,
                language="python",
                complexity_score=complexity_score,
                quality_score=quality_score,
                security_score=security_score,
                test_coverage=0.0,  # TODO: Implement coverage analysis
                issues=issues,
                metrics=metrics,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Python analysis failed for {file_path}: {e}")
            return self._create_error_analysis(file_path, str(e))

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity

    async def _run_pylint(self, file_path: str) -> List[Dict[str, Any]]:
        """Run pylint analysis"""
        try:
            loop = asyncio.get_event_loop()
            
            def run_pylint():
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    report_file = f.name
                
                try:
                    pylint.lint.Run([
                        file_path,
                        f'--output-format=json',
                        f'--reports=n',
                        f'--output={report_file}'
                    ], exit=False)
                    
                    with open(report_file, 'r') as f:
                        result = f.read()
                    
                    Path(report_file).unlink()
                    
                    if result.strip():
                        return json.loads(result)
                    return []
                    
                except Exception as e:
                    logger.warning(f"Pylint analysis failed: {e}")
                    return []
            
            pylint_output = await loop.run_in_executor(self.executor, run_pylint)
            
            issues = []
            for issue in pylint_output:
                issues.append({
                    "tool": "pylint",
                    "type": issue.get("type", "unknown"),
                    "message": issue.get("message", ""),
                    "line": issue.get("line", 0),
                    "column": issue.get("column", 0),
                    "severity": self._map_pylint_severity(issue.get("type", ""))
                })
            
            return issues
            
        except Exception as e:
            logger.warning(f"Pylint analysis failed for {file_path}: {e}")
            return []

    async def _run_mypy(self, file_path: str) -> List[Dict[str, Any]]:
        """Run mypy type checking"""
        try:
            loop = asyncio.get_event_loop()
            
            def run_mypy():
                stdout, stderr, exit_status = mypy.api.run([
                    file_path,
                    '--show-error-codes',
                    '--no-error-summary'
                ])
                return stdout, stderr, exit_status
            
            stdout, stderr, exit_status = await loop.run_in_executor(self.executor, run_mypy)
            
            issues = []
            if stdout:
                for line in stdout.strip().split('\n'):
                    if ':' in line and 'error:' in line:
                        parts = line.split(':')
                        if len(parts) >= 4:
                            issues.append({
                                "tool": "mypy",
                                "type": "type_error",
                                "message": ':'.join(parts[3:]).strip(),
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2]) if parts[2].isdigit() else 0,
                                "severity": "error"
                            })
            
            return issues
            
        except Exception as e:
            logger.warning(f"Mypy analysis failed for {file_path}: {e}")
            return []

    async def _run_bandit(self, file_path: str) -> List[Dict[str, Any]]:
        """Run Bandit security analysis"""
        try:
            loop = asyncio.get_event_loop()
            
            def run_bandit():
                try:
                    conf = config.BanditConfig()
                    b_mgr = manager.BanditManager(conf, 'file')
                    b_mgr.discover_files([file_path])
                    b_mgr.run_tests()
                    return b_mgr.get_issue_list()
                except Exception as e:
                    logger.warning(f"Bandit analysis failed: {e}")
                    return []
            
            bandit_issues = await loop.run_in_executor(self.executor, run_bandit)
            
            issues = []
            for issue in bandit_issues:
                issues.append({
                    "tool": "bandit",
                    "type": "security",
                    "message": issue.text,
                    "line": issue.lineno,
                    "column": 0,
                    "severity": issue.severity.lower(),
                    "confidence": issue.confidence.lower(),
                    "test_id": issue.test_id
                })
            
            return issues
            
        except Exception as e:
            logger.warning(f"Bandit analysis failed for {file_path}: {e}")
            return []

    def _generate_python_suggestions(self, tree: ast.AST, issues: List[Dict]) -> List[str]:
        """Generate improvement suggestions for Python code"""
        suggestions = []
        
        # Analyze function complexity
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = self._calculate_complexity(node)
                if func_complexity > 10:
                    suggestions.append(f"Function '{node.name}' has high complexity ({func_complexity}). Consider breaking it into smaller functions.")
        
        # Check for missing docstrings
        functions_without_docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    functions_without_docstrings.append(node.name)
        
        if functions_without_docstrings:
            suggestions.append(f"Add docstrings to: {', '.join(functions_without_docstrings[:3])}")
        
        # Analyze issues for patterns
        error_types = [issue.get("type", "") for issue in issues]
        if error_types.count("convention") > 5:
            suggestions.append("Multiple style issues detected. Consider running black and isort.")
        
        if any("unused" in issue.get("message", "").lower() for issue in issues):
            suggestions.append("Remove unused imports and variables to improve code clarity.")
        
        return suggestions

    def _calculate_quality_score(self, issues: List[Dict], metrics: Dict) -> float:
        """Calculate overall code quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            severity = issue.get("severity", "info")
            if severity == "error":
                base_score -= 10
            elif severity == "warning":
                base_score -= 5
            elif severity == "convention":
                base_score -= 2
        
        # Adjust for complexity
        complexity = metrics.get("complexity", 1)
        if complexity > 15:
            base_score -= (complexity - 15) * 2
        
        return max(0.0, min(100.0, base_score))

    def _calculate_security_score(self, security_issues: List[Dict]) -> float:
        """Calculate security score (0-100)"""
        base_score = 100.0
        
        for issue in security_issues:
            severity = issue.get("severity", "low")
            if severity == "high":
                base_score -= 20
            elif severity == "medium":
                base_score -= 10
            elif severity == "low":
                base_score -= 5
        
        return max(0.0, min(100.0, base_score))

    def _map_pylint_severity(self, pylint_type: str) -> str:
        """Map pylint message type to severity"""
        mapping = {
            "error": "error",
            "warning": "warning",
            "refactor": "info",
            "convention": "convention",
            "info": "info"
        }
        return mapping.get(pylint_type, "info")

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        suffix = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust"
        }
        return language_map.get(suffix, "unknown")

    async def _analyze_js_file(self, file_path: str, content: str) -> CodeAnalysis:
        """Analyze JavaScript/TypeScript file"""
        # Placeholder implementation - would use tools like ESLint
        return CodeAnalysis(
            file_path=file_path,
            language="javascript",
            complexity_score=5.0,
            quality_score=80.0,
            security_score=90.0,
            test_coverage=0.0,
            issues=[],
            metrics={"lines_of_code": len(content.split('\n'))},
            suggestions=["Consider adding TypeScript types"]
        )

    async def _analyze_generic_file(self, file_path: str, content: str) -> CodeAnalysis:
        """Analyze generic file"""
        return CodeAnalysis(
            file_path=file_path,
            language="unknown",
            complexity_score=1.0,
            quality_score=100.0,
            security_score=100.0,
            test_coverage=0.0,
            issues=[],
            metrics={"lines_of_code": len(content.split('\n'))},
            suggestions=[]
        )

    def _create_error_analysis(self, file_path: str, error: str) -> CodeAnalysis:
        """Create analysis result for error case"""
        return CodeAnalysis(
            file_path=file_path,
            language="unknown",
            complexity_score=0.0,
            quality_score=0.0,
            security_score=0.0,
            test_coverage=0.0,
            issues=[{"tool": "analyzer", "type": "error", "message": error, "line": 0, "column": 0, "severity": "error"}],
            metrics={},
            suggestions=[]
        )

class ImprovementGenerator:
    """Generates improvement proposals using AI models"""
    
    def __init__(self, communicator: ContainerCommunicator):
        self.communicator = communicator
        self.models = {
            "code_generation": "deepseek-r1:8b",
            "analysis": "qwen3:8b",
            "refactoring": "deepseek-coder:33b"
        }

    async def generate_improvements(self, analysis: CodeAnalysis) -> List[ImprovementProposal]:
        """Generate improvement proposals based on analysis"""
        proposals = []
        
        try:
            # Generate different types of improvements
            if analysis.quality_score < 80:
                quality_proposal = await self._generate_quality_improvement(analysis)
                if quality_proposal:
                    proposals.append(quality_proposal)
            
            if analysis.security_score < 90:
                security_proposal = await self._generate_security_improvement(analysis)
                if security_proposal:
                    proposals.append(security_proposal)
            
            if analysis.complexity_score > 10:
                complexity_proposal = await self._generate_complexity_improvement(analysis)
                if complexity_proposal:
                    proposals.append(complexity_proposal)
            
            # Generate documentation improvements
            if "docstring" in str(analysis.suggestions).lower():
                doc_proposal = await self._generate_documentation_improvement(analysis)
                if doc_proposal:
                    proposals.append(doc_proposal)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Error generating improvements for {analysis.file_path}: {e}")
            return []

    async def _generate_quality_improvement(self, analysis: CodeAnalysis) -> Optional[ImprovementProposal]:
        """Generate code quality improvement proposal"""
        try:
            # Read current file content
            async with aiofiles.open(analysis.file_path, 'r', encoding='utf-8') as f:
                current_content = await f.read()
            
            # Create improvement prompt
            prompt = f"""
            Analyze and improve the following Python code for better quality:
            
            File: {analysis.file_path}
            Current Quality Score: {analysis.quality_score}/100
            
            Issues found:
            {self._format_issues(analysis.issues)}
            
            Code:
            ```python
            {current_content}
            ```
            
            Please provide:
            1. Improved version of the code
            2. Explanation of changes made
            3. Expected benefits
            
            Focus on:
            - Code style and formatting
            - Variable naming
            - Function structure
            - Error handling
            """
            
            # Call AI model for improvement
            result = await self.communicator.call_service(
                service_name="deepseek-server",
                method="generate",
                params={
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.3
                }
            )
            
            improved_code = self._extract_code_from_response(result.get("response", ""))
            
            if improved_code and improved_code != current_content:
                return ImprovementProposal(
                    id=self._generate_proposal_id("quality", analysis.file_path),
                    improvement_type=ImprovementType.CODE_QUALITY,
                    target_files=[analysis.file_path],
                    description="Improve code quality and style",
                    rationale=f"Current quality score is {analysis.quality_score}/100. Issues include: {', '.join([i['message'][:50] for i in analysis.issues[:3]])}",
                    code_changes={analysis.file_path: improved_code},
                    impact_assessment={
                        "quality_improvement": "medium",
                        "risk_level": "low",
                        "breaking_changes": False
                    },
                    risk_level="low",
                    expected_benefits=[
                        "Improved code readability",
                        "Better maintainability",
                        "Reduced technical debt"
                    ],
                    tests_affected=[],
                    approval_status=ApprovalStatus.PENDING,
                    created_at=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating quality improvement: {e}")
            return None

    async def _generate_security_improvement(self, analysis: CodeAnalysis) -> Optional[ImprovementProposal]:
        """Generate security improvement proposal"""
        security_issues = [i for i in analysis.issues if i.get("tool") == "bandit"]
        
        if not security_issues:
            return None
        
        try:
            # Read current file content
            async with aiofiles.open(analysis.file_path, 'r', encoding='utf-8') as f:
                current_content = await f.read()
            
            # Create security improvement prompt
            prompt = f"""
            Fix the following security issues in this Python code:
            
            File: {analysis.file_path}
            Security Score: {analysis.security_score}/100
            
            Security Issues:
            {self._format_security_issues(security_issues)}
            
            Code:
            ```python
            {current_content}
            ```
            
            Please provide secure version of the code that addresses all security issues.
            Explain what security improvements were made.
            """
            
            result = await self.communicator.call_service(
                service_name="deepseek-server",
                method="generate",
                params={
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.2
                }
            )
            
            improved_code = self._extract_code_from_response(result.get("response", ""))
            
            if improved_code and improved_code != current_content:
                return ImprovementProposal(
                    id=self._generate_proposal_id("security", analysis.file_path),
                    improvement_type=ImprovementType.SECURITY,
                    target_files=[analysis.file_path],
                    description="Fix security vulnerabilities",
                    rationale=f"Security issues detected: {', '.join([i['message'][:50] for i in security_issues])}",
                    code_changes={analysis.file_path: improved_code},
                    impact_assessment={
                        "security_improvement": "high",
                        "risk_level": "medium",
                        "breaking_changes": False
                    },
                    risk_level="medium",
                    expected_benefits=[
                        "Improved security posture",
                        "Reduced vulnerability risk",
                        "Better compliance"
                    ],
                    tests_affected=[],
                    approval_status=ApprovalStatus.PENDING,
                    created_at=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating security improvement: {e}")
            return None

    async def _generate_complexity_improvement(self, analysis: CodeAnalysis) -> Optional[ImprovementProposal]:
        """Generate complexity reduction proposal"""
        try:
            # Read current file content
            async with aiofiles.open(analysis.file_path, 'r', encoding='utf-8') as f:
                current_content = await f.read()
            
            prompt = f"""
            Reduce the complexity of this Python code by refactoring:
            
            File: {analysis.file_path}
            Current Complexity Score: {analysis.complexity_score}
            
            Code:
            ```python
            {current_content}
            ```
            
            Please refactor the code to:
            1. Reduce cyclomatic complexity
            2. Break down large functions
            3. Improve readability
            4. Maintain the same functionality
            
            Provide the refactored code and explain the improvements.
            """
            
            result = await self.communicator.call_service(
                service_name="deepseek-server",
                method="generate",
                params={
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.3
                }
            )
            
            improved_code = self._extract_code_from_response(result.get("response", ""))
            
            if improved_code and improved_code != current_content:
                return ImprovementProposal(
                    id=self._generate_proposal_id("complexity", analysis.file_path),
                    improvement_type=ImprovementType.ARCHITECTURE,
                    target_files=[analysis.file_path],
                    description="Reduce code complexity through refactoring",
                    rationale=f"High complexity score of {analysis.complexity_score} detected",
                    code_changes={analysis.file_path: improved_code},
                    impact_assessment={
                        "complexity_reduction": "high",
                        "risk_level": "medium",
                        "breaking_changes": False
                    },
                    risk_level="medium",
                    expected_benefits=[
                        "Improved code maintainability",
                        "Easier testing",
                        "Better readability"
                    ],
                    tests_affected=[],
                    approval_status=ApprovalStatus.PENDING,
                    created_at=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating complexity improvement: {e}")
            return None

    async def _generate_documentation_improvement(self, analysis: CodeAnalysis) -> Optional[ImprovementProposal]:
        """Generate documentation improvement proposal"""
        try:
            # Read current file content
            async with aiofiles.open(analysis.file_path, 'r', encoding='utf-8') as f:
                current_content = await f.read()
            
            prompt = f"""
            Add comprehensive documentation to this Python code:
            
            File: {analysis.file_path}
            
            Code:
            ```python
            {current_content}
            ```
            
            Please add:
            1. Module docstring
            2. Function docstrings with parameters, returns, and examples
            3. Class docstrings
            4. Inline comments for complex logic
            
            Follow Google or NumPy docstring style.
            """
            
            result = await self.communicator.call_service(
                service_name="qwen-server",
                method="generate",
                params={
                    "prompt": prompt,
                    "max_tokens": 2048,
                    "temperature": 0.3
                }
            )
            
            improved_code = self._extract_code_from_response(result.get("response", ""))
            
            if improved_code and improved_code != current_content:
                return ImprovementProposal(
                    id=self._generate_proposal_id("documentation", analysis.file_path),
                    improvement_type=ImprovementType.DOCUMENTATION,
                    target_files=[analysis.file_path],
                    description="Add comprehensive documentation",
                    rationale="Missing or insufficient documentation detected",
                    code_changes={analysis.file_path: improved_code},
                    impact_assessment={
                        "documentation_improvement": "high",
                        "risk_level": "low",
                        "breaking_changes": False
                    },
                    risk_level="low",
                    expected_benefits=[
                        "Better code understanding",
                        "Improved maintainability",
                        "Easier onboarding"
                    ],
                    tests_affected=[],
                    approval_status=ApprovalStatus.PENDING,
                    created_at=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating documentation improvement: {e}")
            return None

    def _format_issues(self, issues: List[Dict]) -> str:
        """Format issues for AI prompt"""
        formatted = []
        for issue in issues[:5]:  # Limit to top 5 issues
            formatted.append(f"- Line {issue.get('line', 0)}: {issue.get('message', 'Unknown issue')} ({issue.get('severity', 'info')})")
        return '\n'.join(formatted)

    def _format_security_issues(self, issues: List[Dict]) -> str:
        """Format security issues for AI prompt"""
        formatted = []
        for issue in issues:
            formatted.append(f"- Line {issue.get('line', 0)}: {issue.get('message', 'Security issue')} (Severity: {issue.get('severity', 'low')})")
        return '\n'.join(formatted)

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code from AI response"""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        return None

    def _generate_proposal_id(self, improvement_type: str, file_path: str) -> str:
        """Generate unique proposal ID"""
        content = f"{improvement_type}_{file_path}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

class AutonomousCoder:
    """Main autonomous coding system"""
    
    def __init__(self, 
                 project_root: str = "/opt/sutazaiapp",
                 approval_required: bool = True,
                 auto_approve_low_risk: bool = False):
        
        self.project_root = Path(project_root)
        self.approval_required = approval_required
        self.auto_approve_low_risk = auto_approve_low_risk
        
        # Components
        self.communicator = None
        self.analyzer = CodeAnalyzer()
        self.improvement_generator = None
        self.batch_processor = None
        
        # State
        self.improvement_history = []
        self.pending_proposals = {}
        self.running = False
        
        # Configuration
        self.scan_interval = 3600  # 1 hour
        self.max_proposals_per_cycle = 5
        self.excluded_paths = {
            ".git", "__pycache__", ".pytest_cache", "node_modules",
            ".mypy_cache", ".coverage", "*.pyc", "*.pyo"
        }

    async def initialize(self):
        """Initialize the autonomous coder"""
        try:
            # Initialize communicator
            self.communicator = ContainerCommunicator(
                service_name="autonomous-coder",
                service_type="ai_agent",
                capabilities=["code_analysis", "code_improvement", "self_improvement"]
            )
            
            await self.communicator.initialize()
            
            # Initialize components
            self.improvement_generator = ImprovementGenerator(self.communicator)
            self.batch_processor = BatchProcessor(enable_monitoring=True)
            
            # Register handlers
            await self.communicator.register_handler("analyze_code", self.analyze_code_handler)
            await self.communicator.register_handler("get_proposals", self.get_proposals_handler)
            await self.communicator.register_handler("approve_proposal", self.approve_proposal_handler)
            await self.communicator.register_handler("get_improvement_history", self.get_history_handler)
            
            logger.info("AutonomousCoder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutonomousCoder: {e}")
            raise

    async def start_autonomous_improvement(self):
        """Start the autonomous improvement loop"""
        self.running = True
        logger.info("Starting autonomous improvement loop")
        
        while self.running:
            try:
                await self.improvement_cycle()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Improvement cycle error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def improvement_cycle(self):
        """Single improvement cycle"""
        logger.info("Starting improvement cycle")
        
        try:
            # 1. Discover Python files to analyze
            python_files = await self.discover_python_files()
            logger.info(f"Found {len(python_files)} Python files to analyze")
            
            # 2. Analyze files in batches
            analyses = await self.analyze_files_batch(python_files)
            
            # 3. Generate improvement proposals
            all_proposals = []
            for analysis in analyses:
                if analysis.quality_score < 90 or analysis.security_score < 95:
                    proposals = await self.improvement_generator.generate_improvements(analysis)
                    all_proposals.extend(proposals)
            
            # 4. Sort and limit proposals
            sorted_proposals = self.prioritize_proposals(all_proposals)
            limited_proposals = sorted_proposals[:self.max_proposals_per_cycle]
            
            # 5. Auto-approve low-risk proposals if enabled
            if self.auto_approve_low_risk:
                for proposal in limited_proposals:
                    if proposal.risk_level == "low":
                        proposal.approval_status = ApprovalStatus.AUTO_APPROVED
                        await self.implement_proposal(proposal)
            
            # 6. Store pending proposals
            for proposal in limited_proposals:
                if proposal.approval_status == ApprovalStatus.PENDING:
                    self.pending_proposals[proposal.id] = proposal
            
            # 7. Broadcast improvement event
            await self.communicator.broadcast_event(
                "improvement_cycle_completed",
                {
                    "files_analyzed": len(analyses),
                    "proposals_generated": len(all_proposals),
                    "proposals_pending": len([p for p in limited_proposals if p.approval_status == ApprovalStatus.PENDING]),
                    "proposals_auto_approved": len([p for p in limited_proposals if p.approval_status == ApprovalStatus.AUTO_APPROVED])
                }
            )
            
            logger.info(f"Improvement cycle completed: {len(limited_proposals)} proposals generated")
            
        except Exception as e:
            logger.error(f"Improvement cycle failed: {e}")

    async def discover_python_files(self) -> List[str]:
        """Discover Python files in the project"""
        python_files = []
        
        for file_path in self.project_root.rglob("*.py"):
            # Skip excluded paths
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
            
            # Skip test files for now (could be analyzed separately)
            if "test_" in file_path.name or "_test.py" in file_path.name:
                continue
            
            python_files.append(str(file_path))
        
        return python_files

    async def analyze_files_batch(self, files: List[str]) -> List[CodeAnalysis]:
        """Analyze multiple files in batch"""
        try:
            # Use batch processor for efficient analysis
            results = await self.batch_processor.process_files_batch(
                files=files,
                operation="analyze_code",
                batch_size=20
            )
            
            analyses = []
            for file_path, result in results.results.items():
                if isinstance(result, dict) and result.get("success"):
                    # Convert result to CodeAnalysis if needed
                    analysis = await self.analyzer.analyze_file(file_path)
                    analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return []

    def prioritize_proposals(self, proposals: List[ImprovementProposal]) -> List[ImprovementProposal]:
        """Prioritize improvement proposals"""
        def priority_score(proposal: ImprovementProposal) -> float:
            score = 0.0
            
            # Security improvements get highest priority
            if proposal.improvement_type == ImprovementType.SECURITY:
                score += 100
            
            # Performance improvements
            elif proposal.improvement_type == ImprovementType.PERFORMANCE:
                score += 80
            
            # Code quality improvements
            elif proposal.improvement_type == ImprovementType.CODE_QUALITY:
                score += 60
            
            # Architecture improvements
            elif proposal.improvement_type == ImprovementType.ARCHITECTURE:
                score += 70
            
            # Documentation improvements
            elif proposal.improvement_type == ImprovementType.DOCUMENTATION:
                score += 40
            
            # Adjust for risk level
            if proposal.risk_level == "low":
                score += 20
            elif proposal.risk_level == "high":
                score -= 30
            
            return score
        
        return sorted(proposals, key=priority_score, reverse=True)

    async def implement_proposal(self, proposal: ImprovementProposal) -> ImprovementResults:
        """Implement an approved improvement proposal"""
        try:
            logger.info(f"Implementing proposal {proposal.id}: {proposal.description}")
            
            # Create backup branch
            repo = git.Repo(self.project_root)
            backup_branch = f"backup-{proposal.id}"
            repo.git.checkout("-b", backup_branch)
            
            files_modified = []
            
            try:
                # Apply code changes
                for file_path, new_content in proposal.code_changes.items():
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(new_content)
                    files_modified.append(file_path)
                
                # Format code
                await self.format_code(files_modified)
                
                # Run tests
                tests_passed = await self.run_tests(proposal.tests_affected)
                
                if tests_passed:
                    # Commit changes
                    repo.git.add(files_modified)
                    repo.git.commit("-m", f"Auto-improvement: {proposal.description}")
                    
                    # Switch back to main branch and merge
                    repo.git.checkout("main")
                    repo.git.merge(backup_branch)
                    
                    # Update proposal
                    proposal.approval_status = ApprovalStatus.APPROVED
                    proposal.implemented_at = time.time()
                    
                    # Record improvement
                    result = ImprovementResults(
                        proposal_id=proposal.id,
                        success=True,
                        files_modified=files_modified,
                        tests_passed=tests_passed,
                        performance_impact={},
                        quality_improvement={}
                    )
                    
                    self.improvement_history.append(result)
                    
                    logger.info(f"Successfully implemented proposal {proposal.id}")
                    return result
                
                else:
                    # Tests failed, rollback
                    repo.git.checkout("main")
                    repo.git.branch("-D", backup_branch)
                    
                    return ImprovementResults(
                        proposal_id=proposal.id,
                        success=False,
                        files_modified=[],
                        tests_passed=False,
                        performance_impact={},
                        quality_improvement={},
                        error_message="Tests failed after implementation"
                    )
                    
            except Exception as e:
                # Error during implementation, rollback
                repo.git.checkout("main")
                repo.git.branch("-D", backup_branch)
                raise e
                
        except Exception as e:
            logger.error(f"Failed to implement proposal {proposal.id}: {e}")
            return ImprovementResults(
                proposal_id=proposal.id,
                success=False,
                files_modified=[],
                tests_passed=False,
                performance_impact={},
                quality_improvement={},
                error_message=str(e)
            )

    async def format_code(self, files: List[str]):
        """Format code using black and isort"""
        for file_path in files:
            if file_path.endswith('.py'):
                try:
                    # Format with black
                    subprocess.run([
                        "black", "--line-length", "88", file_path
                    ], check=True, capture_output=True)
                    
                    # Sort imports with isort
                    subprocess.run([
                        "isort", file_path
                    ], check=True, capture_output=True)
                    
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Code formatting failed for {file_path}: {e}")

    async def run_tests(self, test_files: List[str]) -> bool:
        """Run tests to validate changes"""
        try:
            if not test_files:
                # Run all tests if no specific tests specified
                result = subprocess.run([
                    "python", "-m", "pytest", 
                    str(self.project_root / "tests"),
                    "-v", "--tb=short"
                ], capture_output=True, text=True, cwd=self.project_root)
            else:
                # Run specific tests
                result = subprocess.run([
                    "python", "-m", "pytest"
                ] + test_files + ["-v", "--tb=short"], 
                capture_output=True, text=True, cwd=self.project_root)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False

    # Service handlers
    async def analyze_code_handler(self, file_path: str) -> Dict[str, Any]:
        """Handler for code analysis requests"""
        try:
            analysis = await self.analyzer.analyze_file(file_path)
            return asdict(analysis)
        except Exception as e:
            logger.error(f"Analysis handler error: {e}")
            return {"error": str(e)}

    async def get_proposals_handler(self) -> Dict[str, Any]:
        """Handler for getting pending proposals"""
        try:
            proposals = []
            for proposal in self.pending_proposals.values():
                proposals.append(asdict(proposal))
            
            return {"proposals": proposals}
        except Exception as e:
            logger.error(f"Get proposals handler error: {e}")
            return {"error": str(e)}

    async def approve_proposal_handler(self, proposal_id: str, approved: bool) -> Dict[str, Any]:
        """Handler for proposal approval"""
        try:
            if proposal_id not in self.pending_proposals:
                return {"error": f"Proposal {proposal_id} not found"}
            
            proposal = self.pending_proposals[proposal_id]
            
            if approved:
                proposal.approval_status = ApprovalStatus.APPROVED
                proposal.approved_at = time.time()
                
                # Implement the proposal
                result = await self.implement_proposal(proposal)
                
                # Remove from pending
                del self.pending_proposals[proposal_id]
                
                return {"success": True, "implemented": result.success}
            else:
                proposal.approval_status = ApprovalStatus.REJECTED
                del self.pending_proposals[proposal_id]
                
                return {"success": True, "implemented": False}
                
        except Exception as e:
            logger.error(f"Approval handler error: {e}")
            return {"error": str(e)}

    async def get_history_handler(self) -> Dict[str, Any]:
        """Handler for getting improvement history"""
        try:
            return {"history": [asdict(result) for result in self.improvement_history[-20:]]}
        except Exception as e:
            logger.error(f"History handler error: {e}")
            return {"error": str(e)}

    async def stop(self):
        """Stop the autonomous coder"""
        self.running = False
        if self.communicator:
            await self.communicator.shutdown()
        if self.batch_processor:
            self.batch_processor.cleanup()

# Main entry point
async def main():
    """Main function to run autonomous coder"""
    coder = AutonomousCoder(
        project_root="/opt/sutazaiapp",
        approval_required=True,
        auto_approve_low_risk=True
    )
    
    try:
        await coder.initialize()
        await coder.start_autonomous_improvement()
    except KeyboardInterrupt:
        logger.info("Shutting down autonomous coder...")
    finally:
        await coder.stop()

if __name__ == "__main__":
    asyncio.run(main())