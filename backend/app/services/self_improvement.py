"""
AI-Driven Self-Improvement Feedback Loop Service
Enables the system to analyze its performance and autonomously improve
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, asdict
import aiofiles
import ast
import re

from app.core.config import settings
from app.services.model_manager import ModelManager
from app.services.vector_db_manager import VectorDBManager
from app.core.agi_brain import AGIBrain, ReasoningType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    timestamp: datetime
    metric_type: str
    value: float
    context: Dict[str, Any]


@dataclass
class ImprovementSuggestion:
    """Improvement suggestion from analysis"""
    category: str
    priority: str  # high, medium, low
    description: str
    implementation: Dict[str, Any]
    expected_impact: float
    confidence: float


@dataclass
class CodeImprovement:
    """Code improvement task"""
    file_path: str
    line_start: int
    line_end: int
    issue_type: str
    current_code: str
    suggested_code: str
    explanation: str


class SelfImprovementService:
    """
    Autonomous self-improvement service that:
    1. Monitors system performance
    2. Analyzes patterns and issues
    3. Generates improvement suggestions
    4. Implements changes autonomously
    5. Validates improvements
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.vector_db = VectorDBManager()
        self.agi_brain = AGIBrain()
        
        # Performance tracking
        self.metrics_buffer: List[PerformanceMetric] = []
        self.improvement_history: List[Dict[str, Any]] = []
        
        # Batch processing settings
        self.batch_size = getattr(settings, "SELF_IMPROVEMENT_BATCH_SIZE", 50)
        self.min_confidence_threshold = 0.7
        self.max_concurrent_improvements = 5
        
        # File patterns to analyze
        self.code_patterns = {
            "python": r"\.py$",
            "javascript": r"\.(js|ts|jsx|tsx)$",
            "config": r"\.(json|yml|yaml|env)$"
        }
        
        # Issue detection patterns
        self.issue_patterns = {
            "performance": [
                r"time\.sleep\s*\(\s*\d+\s*\)",  # Blocking sleep
                r"for\s+\w+\s+in\s+.*:\s*\n\s*for\s+\w+\s+in",  # Nested loops
                r"\.append\s*\(.*\)\s*in\s+.*for",  # List comprehension opportunity
            ],
            "error_handling": [
                r"except\s*:",  # Bare except
                r"except\s+Exception\s*:",  # Too broad exception
                r"pass\s*$",  # Empty except block
            ],
            "code_quality": [
                r"TODO|FIXME|HACK",  # Technical debt markers
                r"print\s*\(",  # Debug prints
                r"import\s+\*",  # Star imports
            ],
            "security": [
                r"eval\s*\(",  # Eval usage
                r"exec\s*\(",  # Exec usage
                r"pickle\.loads",  # Unsafe deserialization
            ]
        }
        
        # Start background monitoring
        self._start_monitoring = False
    
    async def start_monitoring(self):
        """Start the self-improvement monitoring loop"""
        self._start_monitoring = True
        logger.info("Starting self-improvement monitoring")
        
        # Run multiple monitoring tasks concurrently
        await asyncio.gather(
            self._monitor_performance(),
            self._analyze_codebase(),
            self._process_improvements(),
            return_exceptions=True
        )
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._start_monitoring = False
        logger.info("Stopping self-improvement monitoring")
    
    async def _monitor_performance(self):
        """Monitor system performance metrics"""
        while self._start_monitoring:
            try:
                # Collect various metrics
                metrics = await self._collect_metrics()
                
                for metric in metrics:
                    self.metrics_buffer.append(metric)
                
                # Analyze if buffer is full
                if len(self.metrics_buffer) >= 100:
                    await self._analyze_metrics()
                    self.metrics_buffer = self.metrics_buffer[-50:]  # Keep recent metrics
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self) -> List[PerformanceMetric]:
        """Collect current system metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # API response times
            if hasattr(self, '_api_latencies'):
                avg_latency = np.mean(self._api_latencies) if self._api_latencies else 0
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_type="api_latency",
                    value=avg_latency,
                    context={"endpoint": "aggregate"}
                ))
            
            # Model performance
            # TODO: Implement get_model_stats in ModelManager
            # model_stats = await self.model_manager.get_model_stats()
            # for model, stats in model_stats.items():
            #     metrics.append(PerformanceMetric(
            #         timestamp=timestamp,
            #         metric_type="model_performance",
            #         value=stats.get("avg_inference_time", 0),
            #         context={"model": model, "requests": stats.get("total_requests", 0)}
            #     ))
            
            # Memory usage
            import psutil
            process = psutil.Process()
            memory_percent = process.memory_percent()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_type="memory_usage",
                value=memory_percent,
                context={"rss": process.memory_info().rss}
            ))
            
            # Error rates
            if hasattr(self, '_error_counts'):
                error_rate = sum(self._error_counts.values()) / max(len(self._api_latencies), 1)
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_type="error_rate",
                    value=error_rate,
                    context={"errors": dict(self._error_counts)}
                ))
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def _analyze_metrics(self):
        """Analyze collected metrics for patterns and issues"""
        try:
            # Group metrics by type
            metrics_by_type = defaultdict(list)
            for metric in self.metrics_buffer:
                metrics_by_type[metric.metric_type].append(metric)
            
            suggestions = []
            
            # Analyze each metric type
            for metric_type, metrics in metrics_by_type.items():
                values = [m.value for m in metrics]
                
                if metric_type == "api_latency":
                    avg_latency = np.mean(values)
                    if avg_latency > 1.0:  # > 1 second average
                        suggestions.append(ImprovementSuggestion(
                            category="performance",
                            priority="high",
                            description="High API latency detected",
                            implementation={
                                "action": "optimize_api_endpoints",
                                "targets": ["caching", "query_optimization", "async_processing"]
                            },
                            expected_impact=0.5,
                            confidence=0.85
                        ))
                
                elif metric_type == "memory_usage":
                    if max(values) > 80:  # > 80% memory
                        suggestions.append(ImprovementSuggestion(
                            category="resource",
                            priority="high",
                            description="High memory usage detected",
                            implementation={
                                "action": "memory_optimization",
                                "targets": ["garbage_collection", "cache_cleanup", "object_pooling"]
                            },
                            expected_impact=0.3,
                            confidence=0.9
                        ))
                
                elif metric_type == "error_rate":
                    if np.mean(values) > 0.05:  # > 5% error rate
                        suggestions.append(ImprovementSuggestion(
                            category="reliability",
                            priority="high",
                            description="High error rate detected",
                            implementation={
                                "action": "error_analysis",
                                "targets": ["exception_handling", "input_validation", "retry_logic"]
                            },
                            expected_impact=0.7,
                            confidence=0.8
                        ))
            
            # Process suggestions
            for suggestion in suggestions:
                await self._queue_improvement(suggestion)
            
        except Exception as e:
            logger.error(f"Error analyzing metrics: {e}")
    
    async def _analyze_codebase(self):
        """Continuously analyze codebase for improvements"""
        while self._start_monitoring:
            try:
                # Find Python files to analyze
                code_files = await self._find_code_files()
                
                # Analyze in batches
                for i in range(0, len(code_files), self.batch_size):
                    batch = code_files[i:i + self.batch_size]
                    improvements = await self._analyze_code_batch(batch)
                    
                    # Queue improvements
                    for improvement in improvements:
                        await self._queue_code_improvement(improvement)
                    
                    await asyncio.sleep(5)  # Pause between batches
                
                # Wait before next full scan
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in codebase analysis: {e}")
                await asyncio.sleep(3600)
    
    async def _find_code_files(self) -> List[Path]:
        """Find all code files to analyze"""
        code_files = []
        
        try:
            base_path = Path("/app")  # Assuming app is mounted here
            
            for pattern_type, pattern in self.code_patterns.items():
                for file_path in base_path.rglob("*"):
                    if file_path.is_file() and re.match(pattern, str(file_path)):
                        # Skip test files and migrations
                        if not any(skip in str(file_path) for skip in ["test_", "_test.", "migrations", "__pycache__"]):
                            code_files.append(file_path)
            
        except Exception as e:
            logger.error(f"Error finding code files: {e}")
        
        return code_files[:self.batch_size * 10]  # Limit total files
    
    async def _analyze_code_batch(self, files: List[Path]) -> List[CodeImprovement]:
        """Analyze a batch of code files"""
        improvements = []
        
        for file_path in files:
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                
                # Check for various issues
                file_improvements = await self._analyze_file_content(file_path, content)
                improvements.extend(file_improvements)
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
        
        return improvements
    
    async def _analyze_file_content(self, file_path: Path, content: str) -> List[CodeImprovement]:
        """Analyze file content for improvements"""
        improvements = []
        lines = content.split('\n')
        
        try:
            # Check for pattern-based issues
            for issue_type, patterns in self.issue_patterns.items():
                for pattern in patterns:
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            # Use AI to suggest improvement
                            suggestion = await self._generate_improvement_suggestion(
                                file_path, i, line, issue_type
                            )
                            if suggestion:
                                improvements.append(suggestion)
            
            # Use AI for deeper analysis
            if file_path.suffix == '.py':
                # Analyze Python AST
                try:
                    tree = ast.parse(content)
                    ast_improvements = await self._analyze_python_ast(file_path, tree, lines)
                    improvements.extend(ast_improvements)
                except SyntaxError:
                    logger.warning(f"Syntax error in {file_path}")
            
        except Exception as e:
            logger.error(f"Error in file content analysis: {e}")
        
        return improvements[:10]  # Limit improvements per file
    
    async def _analyze_python_ast(self, file_path: Path, tree: ast.AST, lines: List[str]) -> List[CodeImprovement]:
        """Analyze Python AST for improvements"""
        improvements = []
        
        class ImprovementVisitor(ast.NodeVisitor):
            def __init__(self):
                self.suggestions = []
            
            def visit_FunctionDef(self, node):
                # Check for long functions
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        self.suggestions.append({
                            'line': node.lineno,
                            'issue': 'long_function',
                            'name': node.name,
                            'length': func_length
                        })
                
                # Check for too many arguments
                if len(node.args.args) > 5:
                    self.suggestions.append({
                        'line': node.lineno,
                        'issue': 'too_many_args',
                        'name': node.name,
                        'arg_count': len(node.args.args)
                    })
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                # Check for empty except blocks
                for handler in node.handlers:
                    if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                        self.suggestions.append({
                            'line': handler.lineno,
                            'issue': 'empty_except'
                        })
                
                self.generic_visit(node)
        
        visitor = ImprovementVisitor()
        visitor.visit(tree)
        
        # Convert visitor suggestions to improvements
        for suggestion in visitor.suggestions:
            improvement = await self._create_ast_improvement(
                file_path, suggestion, lines
            )
            if improvement:
                improvements.append(improvement)
        
        return improvements
    
    async def _generate_improvement_suggestion(
        self, file_path: Path, line_num: int, line: str, issue_type: str
    ) -> Optional[CodeImprovement]:
        """Generate improvement suggestion using AI"""
        try:
            # Use AGI brain to analyze and suggest improvement
            result = await self.agi_brain.think(
                input_data={
                    "task": "code_improvement",
                    "file": str(file_path),
                    "line": line,
                    "line_number": line_num,
                    "issue_type": issue_type
                },
                reasoning_type=ReasoningType.DEDUCTIVE
            )
            
            if result.get("confidence", 0) > self.min_confidence_threshold:
                return CodeImprovement(
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=line_num,
                    issue_type=issue_type,
                    current_code=line,
                    suggested_code=result.get("suggested_code", line),
                    explanation=result.get("explanation", "")
                )
        
        except Exception as e:
            logger.error(f"Error generating improvement suggestion: {e}")
        
        return None
    
    async def _create_ast_improvement(
        self, file_path: Path, suggestion: Dict, lines: List[str]
    ) -> Optional[CodeImprovement]:
        """Create improvement from AST analysis"""
        try:
            line_num = suggestion['line'] - 1  # 0-indexed
            
            if suggestion['issue'] == 'long_function':
                return CodeImprovement(
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=line_num + suggestion['length'],
                    issue_type="refactoring",
                    current_code=f"Function '{suggestion['name']}' is {suggestion['length']} lines long",
                    suggested_code="Consider breaking this function into smaller, more focused functions",
                    explanation=f"Functions longer than 50 lines are harder to maintain and test. Consider extracting logical sections into separate functions."
                )
            
            elif suggestion['issue'] == 'too_many_args':
                return CodeImprovement(
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=line_num,
                    issue_type="refactoring",
                    current_code=f"Function '{suggestion['name']}' has {suggestion['arg_count']} arguments",
                    suggested_code="Consider using a configuration object or builder pattern",
                    explanation="Functions with more than 5 arguments are difficult to use and maintain. Consider grouping related parameters."
                )
            
            elif suggestion['issue'] == 'empty_except':
                return CodeImprovement(
                    file_path=str(file_path),
                    line_start=line_num,
                    line_end=line_num,
                    issue_type="error_handling",
                    current_code=lines[line_num] if line_num < len(lines) else "",
                    suggested_code="except SpecificException as e:\n    logger.error(f'Error occurred: {e}')",
                    explanation="Empty except blocks hide errors. Always log or handle exceptions appropriately."
                )
        
        except Exception as e:
            logger.error(f"Error creating AST improvement: {e}")
        
        return None
    
    async def _queue_improvement(self, suggestion: ImprovementSuggestion):
        """Queue an improvement suggestion for processing"""
        try:
            # Store in vector DB for retrieval
            await self.vector_db.add_documents(
                collection_name="improvements",
                documents=[{
                    "id": f"imp_{datetime.now().timestamp()}",
                    "content": json.dumps(asdict(suggestion)),
                    "metadata": {
                        "category": suggestion.category,
                        "priority": suggestion.priority,
                        "timestamp": datetime.now().isoformat()
                    }
                }]
            )
            
            # If high priority, process immediately
            if suggestion.priority == "high" and suggestion.confidence > 0.8:
                await self._process_improvement(suggestion)
            
        except Exception as e:
            logger.error(f"Error queuing improvement: {e}")
    
    async def _queue_code_improvement(self, improvement: CodeImprovement):
        """Queue a code improvement for processing"""
        try:
            # Store for batch processing
            await self.vector_db.add_documents(
                collection_name="code_improvements",
                documents=[{
                    "id": f"code_imp_{datetime.now().timestamp()}",
                    "content": json.dumps(asdict(improvement)),
                    "metadata": {
                        "file": improvement.file_path,
                        "issue_type": improvement.issue_type,
                        "timestamp": datetime.now().isoformat()
                    }
                }]
            )
        
        except Exception as e:
            logger.error(f"Error queuing code improvement: {e}")
    
    async def _process_improvements(self):
        """Process queued improvements"""
        while self._start_monitoring:
            try:
                # Get pending improvements
                # TODO: Fix VectorDB search - needs query_embedding not query string
                # improvements = await self.vector_db.search(
                #     collection="improvements",
                #     query_embedding=await self._get_embedding("pending improvements"),
                #     limit=self.max_concurrent_improvements
                # )
                improvements = []  # Temporarily disabled
                
                # Process each improvement
                tasks = []
                for imp_doc in improvements:
                    imp_data = json.loads(imp_doc['content'])
                    suggestion = ImprovementSuggestion(**imp_data)
                    tasks.append(self._process_improvement(suggestion))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Also process code improvements
                await self._process_code_improvements()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing improvements: {e}")
                await asyncio.sleep(600)
    
    async def _process_improvement(self, suggestion: ImprovementSuggestion):
        """Process a single improvement suggestion"""
        try:
            logger.info(f"Processing improvement: {suggestion.description}")
            
            # Implementation based on category
            if suggestion.category == "performance":
                await self._implement_performance_improvement(suggestion)
            elif suggestion.category == "resource":
                await self._implement_resource_improvement(suggestion)
            elif suggestion.category == "reliability":
                await self._implement_reliability_improvement(suggestion)
            
            # Record improvement
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "suggestion": asdict(suggestion),
                "status": "completed"
            })
            
            # Save history
            await self._save_improvement_history()
            
        except Exception as e:
            logger.error(f"Error processing improvement: {e}")
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "suggestion": asdict(suggestion),
                "status": "failed",
                "error": str(e)
            })
    
    async def _process_code_improvements(self):
        """Process batched code improvements"""
        try:
            # Get code improvements by file
            # TODO: Fix VectorDB search - needs query_embedding not query string
            # improvements = await self.vector_db.search(
            #     collection="code_improvements",
            #     query_embedding=await self._get_embedding("pending code improvements"),
            #     limit=self.batch_size
            # )
            improvements = []  # Temporarily disabled
            
            # Group by file
            improvements_by_file = defaultdict(list)
            for imp_doc in improvements:
                imp_data = json.loads(imp_doc['content'])
                improvement = CodeImprovement(**imp_data)
                improvements_by_file[improvement.file_path].append(improvement)
            
            # Apply improvements file by file
            for file_path, file_improvements in improvements_by_file.items():
                await self._apply_code_improvements(file_path, file_improvements)
            
        except Exception as e:
            logger.error(f"Error processing code improvements: {e}")
    
    async def _apply_code_improvements(self, file_path: str, improvements: List[CodeImprovement]):
        """Apply improvements to a single file"""
        try:
            # Sort improvements by line number (reverse to avoid offset issues)
            improvements.sort(key=lambda x: x.line_start, reverse=True)
            
            # Read file
            async with aiofiles.open(file_path, 'r') as f:
                lines = await f.readlines()
            
            # Apply each improvement
            for improvement in improvements:
                if improvement.line_start < len(lines):
                    # Log the change
                    logger.info(f"Applying improvement to {file_path}:{improvement.line_start}")
                    
                    # Simple replacement for now
                    # In production, use more sophisticated code modification
                    if improvement.suggested_code and improvement.suggested_code != improvement.current_code:
                        lines[improvement.line_start] = improvement.suggested_code + '\n'
            
            # Write back
            async with aiofiles.open(file_path, 'w') as f:
                await f.writelines(lines)
            
            logger.info(f"Applied {len(improvements)} improvements to {file_path}")
            
        except Exception as e:
            logger.error(f"Error applying code improvements to {file_path}: {e}")
    
    async def _implement_performance_improvement(self, suggestion: ImprovementSuggestion):
        """Implement performance-related improvements"""
        implementation = suggestion.implementation
        action = implementation.get("action")
        
        if action == "optimize_api_endpoints":
            # Enable caching for slow endpoints
            logger.info("Implementing API endpoint optimization")
            # This would actually modify API configurations
            
        elif action == "database_optimization":
            # Add indexes, optimize queries
            logger.info("Implementing database optimization")
            # This would run database optimization scripts
    
    async def _implement_resource_improvement(self, suggestion: ImprovementSuggestion):
        """Implement resource-related improvements"""
        implementation = suggestion.implementation
        action = implementation.get("action")
        
        if action == "memory_optimization":
            # Implement garbage collection, clear caches
            logger.info("Implementing memory optimization")
            import gc
            gc.collect()
            
            # Clear model caches if needed
            if hasattr(self.model_manager, 'clear_cache'):
                await self.model_manager.clear_cache()
    
    async def _implement_reliability_improvement(self, suggestion: ImprovementSuggestion):
        """Implement reliability-related improvements"""
        implementation = suggestion.implementation
        action = implementation.get("action")
        
        if action == "error_analysis":
            # Analyze error patterns and add better handling
            logger.info("Implementing error handling improvements")
            # This would analyze logs and add error handling
    
    async def _save_improvement_history(self):
        """Save improvement history to file"""
        try:
            history_file = Path("/app/data/improvement_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(history_file, 'w') as f:
                await f.write(json.dumps(self.improvement_history, indent=2))
        
        except Exception as e:
            logger.error(f"Error saving improvement history: {e}")
    
    async def get_improvement_report(self) -> Dict[str, Any]:
        """Get a report of improvements made"""
        try:
            total_improvements = len(self.improvement_history)
            successful = sum(1 for imp in self.improvement_history if imp['status'] == 'completed')
            failed = sum(1 for imp in self.improvement_history if imp['status'] == 'failed')
            
            # Group by category
            by_category = defaultdict(int)
            for imp in self.improvement_history:
                category = imp['suggestion']['category']
                by_category[category] += 1
            
            return {
                "total_improvements": total_improvements,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_improvements if total_improvements > 0 else 0,
                "by_category": dict(by_category),
                "recent_improvements": self.improvement_history[-10:],
                "metrics_analyzed": len(self.metrics_buffer),
                "active_monitoring": self._start_monitoring
            }
        
        except Exception as e:
            logger.error(f"Error generating improvement report: {e}")
            return {}


# Singleton instance
self_improvement_service = SelfImprovementService()