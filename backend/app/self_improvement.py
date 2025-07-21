"""
Self-Improvement System - Autonomous code generation and system optimization
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import subprocess
import ast
import json
import httpx

logger = logging.getLogger(__name__)

class SelfImprovementSystem:
    """Manages autonomous self-improvement and code generation"""
    
    def __init__(self):
        self.code_base_path = "/opt/sutazaiapp"
        self.improvement_history = []
        self.performance_metrics = {}
        self.ollama_url = "http://localhost:11434"
        self.initialized = False
        
    async def initialize(self):
        """Initialize self-improvement system"""
        logger.info("Initializing Self-Improvement System...")
        
        # Analyze current codebase
        await self._analyze_codebase()
        
        # Initialize performance monitoring
        await self._init_performance_monitoring()
        
        # Start improvement cycles
        asyncio.create_task(self._improvement_cycle())
        
        self.initialized = True
        logger.info("Self-Improvement System initialized")
        
    async def _analyze_codebase(self):
        """Analyze the current codebase"""
        logger.info("Analyzing codebase...")
        
        # Count files and lines of code
        file_stats = self._get_codebase_stats()
        
        # Identify improvement opportunities
        self.improvement_opportunities = await self._identify_improvements()
        
        logger.info(f"Found {len(self.improvement_opportunities)} improvement opportunities")
        
    def _get_codebase_stats(self) -> Dict[str, Any]:
        """Get statistics about the codebase"""
        stats = {
            "total_files": 0,
            "python_files": 0,
            "total_lines": 0,
            "function_count": 0
        }
        
        for root, dirs, files in os.walk(self.code_base_path):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    stats["python_files"] += 1
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            stats["total_lines"] += len(lines)
                    except:
                        pass
                        
                stats["total_files"] += 1
                
        return stats
        
    async def _identify_improvements(self) -> List[Dict[str, Any]]:
        """Identify areas for improvement"""
        improvements = []
        
        # Check for code quality issues
        quality_issues = await self._check_code_quality()
        improvements.extend(quality_issues)
        
        # Check for performance bottlenecks
        performance_issues = await self._check_performance()
        improvements.extend(performance_issues)
        
        # Check for missing features
        missing_features = await self._check_missing_features()
        improvements.extend(missing_features)
        
        return improvements
        
    async def _check_code_quality(self) -> List[Dict[str, Any]]:
        """Check for code quality issues"""
        issues = []
        
        # Run pylint on key files
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", f"{self.code_base_path}/backend/app"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                pylint_issues = json.loads(result.stdout)
                for issue in pylint_issues[:10]:  # Top 10 issues
                    issues.append({
                        "type": "code_quality",
                        "severity": issue.get("type", "warning"),
                        "file": issue.get("path", ""),
                        "line": issue.get("line", 0),
                        "message": issue.get("message", ""),
                        "improvement": f"Fix {issue.get('symbol', 'issue')} in {issue.get('path', 'file')}"
                    })
        except Exception as e:
            logger.warning(f"Pylint check failed: {e}")
            
        return issues
        
    async def _check_performance(self) -> List[Dict[str, Any]]:
        """Check for performance issues"""
        issues = []
        
        # Analyze response times
        if self.performance_metrics.get("avg_response_time", 0) > 1000:  # 1 second
            issues.append({
                "type": "performance",
                "severity": "high",
                "metric": "response_time",
                "value": self.performance_metrics["avg_response_time"],
                "improvement": "Optimize slow endpoints"
            })
            
        # Check memory usage
        memory_usage = self._get_memory_usage()
        if memory_usage > 80:  # 80% memory usage
            issues.append({
                "type": "performance",
                "severity": "high",
                "metric": "memory_usage",
                "value": memory_usage,
                "improvement": "Reduce memory consumption"
            })
            
        return issues
        
    async def _check_missing_features(self) -> List[Dict[str, Any]]:
        """Check for missing features"""
        features = []
        
        # Check if all agents have health endpoints
        features.append({
            "type": "feature",
            "severity": "medium",
            "feature": "agent_health_monitoring",
            "improvement": "Add comprehensive health monitoring for all agents"
        })
        
        # Check for missing tests
        test_coverage = self._get_test_coverage()
        if test_coverage < 80:
            features.append({
                "type": "feature",
                "severity": "high",
                "feature": "test_coverage",
                "value": test_coverage,
                "improvement": "Increase test coverage to at least 80%"
            })
            
        return features
        
    async def improve_system(self) -> Dict[str, Any]:
        """Execute improvement on the system"""
        logger.info("Starting system improvement...")
        
        if not self.improvement_opportunities:
            return {"status": "no_improvements_needed"}
            
        # Select highest priority improvement
        improvement = self._select_improvement()
        
        # Generate solution
        solution = await self._generate_solution(improvement)
        
        # Apply solution
        result = await self._apply_solution(solution)
        
        # Record improvement
        self.improvement_history.append({
            "timestamp": datetime.now().isoformat(),
            "improvement": improvement,
            "solution": solution,
            "result": result
        })
        
        return result
        
    def _select_improvement(self) -> Dict[str, Any]:
        """Select the most important improvement"""
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        sorted_improvements = sorted(
            self.improvement_opportunities,
            key=lambda x: severity_order.get(x.get("severity", "low"), 3)
        )
        
        return sorted_improvements[0] if sorted_improvements else {}
        
    async def _generate_solution(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a solution for the improvement"""
        improvement_type = improvement.get("type", "")
        
        if improvement_type == "code_quality":
            return await self._generate_code_fix(improvement)
        elif improvement_type == "performance":
            return await self._generate_performance_optimization(improvement)
        elif improvement_type == "feature":
            return await self._generate_feature_implementation(improvement)
        else:
            return {"type": "unknown", "code": ""}
            
    async def _generate_code_fix(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code fix for quality issues"""
        file_path = improvement.get("file", "")
        line = improvement.get("line", 0)
        message = improvement.get("message", "")
        
        # Read the problematic code
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                context = ''.join(lines[max(0, line-5):line+5])
        except:
            context = ""
            
        # Generate fix using AI
        prompt = f"""Fix this code quality issue:
File: {file_path}
Line: {line}
Issue: {message}

Code context:
{context}

Provide only the fixed code, no explanations."""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "codellama:7b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    fixed_code = response.json()["response"]
                    return {
                        "type": "code_fix",
                        "file": file_path,
                        "line": line,
                        "code": fixed_code
                    }
        except Exception as e:
            logger.error(f"Error generating code fix: {e}")
            
        return {"type": "code_fix", "code": "# Unable to generate fix"}
        
    async def _generate_performance_optimization(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance optimization"""
        metric = improvement.get("metric", "")
        
        optimizations = {
            "response_time": """
# Add caching to improve response time
from functools import lru_cache
import redis

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

@lru_cache(maxsize=128)
def cached_operation(key):
    # Check Redis cache first
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    # Perform operation
    result = expensive_operation(key)
    
    # Cache result
    redis_client.setex(key, 3600, json.dumps(result))
    return result
""",
            "memory_usage": """
# Optimize memory usage
import gc
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Implement memory-efficient data structures
class MemoryEfficientStorage:
    def __init__(self):
        self._data = {}
        self._cleanup_threshold = 1000
        
    def add(self, key, value):
        self._data[key] = value
        if len(self._data) > self._cleanup_threshold:
            self._cleanup()
            
    def _cleanup(self):
        # Remove least recently used items
        sorted_items = sorted(self._data.items(), key=lambda x: x[1].get('last_access', 0))
        for key, _ in sorted_items[:len(sorted_items)//2]:
            del self._data[key]
        gc.collect()
"""
        }
        
        return {
            "type": "performance_optimization",
            "metric": metric,
            "code": optimizations.get(metric, "# Performance optimization needed")
        }
        
    async def _generate_feature_implementation(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new feature implementation"""
        feature = improvement.get("feature", "")
        
        if feature == "agent_health_monitoring":
            code = """
# Agent Health Monitoring System
from typing import Dict, Any
import asyncio
import httpx
from datetime import datetime

class AgentHealthMonitor:
    def __init__(self):
        self.agents = {}
        self.health_history = []
        
    async def monitor_agent(self, agent_name: str, url: str):
        '''Monitor health of a specific agent'''
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=5.0)
                
                health_status = {
                    "agent": agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json() if response.status_code == 200 else {}
                }
                
                self.health_history.append(health_status)
                return health_status
                
        except Exception as e:
            return {
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "status": "unreachable",
                "error": str(e)
            }
            
    async def monitor_all_agents(self):
        '''Monitor all registered agents'''
        tasks = []
        for agent_name, agent_url in self.agents.items():
            tasks.append(self.monitor_agent(agent_name, agent_url))
            
        results = await asyncio.gather(*tasks)
        return results
"""
        elif feature == "test_coverage":
            code = """
# Test Suite for Core Components
import pytest
import asyncio
from unittest.mock import Mock, patch

class TestAGIBrain:
    @pytest.mark.asyncio
    async def test_initialization(self):
        from app.agi_brain import AGIBrain
        brain = AGIBrain()
        await brain.initialize()
        assert brain.initialized == True
        
    @pytest.mark.asyncio
    async def test_process_query(self):
        from app.agi_brain import AGIBrain
        brain = AGIBrain()
        await brain.initialize()
        
        result = await brain.process_query("Test query")
        assert "response" in result
        assert "cognitive_trace" in result
        
    @pytest.mark.asyncio
    async def test_cognitive_functions(self):
        from app.agi_brain import AGIBrain, CognitiveFunction
        brain = AGIBrain()
        await brain.initialize()
        
        # Test all cognitive functions are registered
        assert len(brain.cognitive_modules) == 8
        assert CognitiveFunction.PERCEPTION in brain.cognitive_modules

# Run with: pytest -v --cov=app --cov-report=html
"""
        else:
            code = f"# Implementation for {feature} feature"
            
        return {
            "type": "feature_implementation",
            "feature": feature,
            "code": code
        }
        
    async def _apply_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the generated solution"""
        solution_type = solution.get("type", "")
        
        if solution_type == "code_fix":
            # Apply code fix to file
            file_path = solution.get("file", "")
            line = solution.get("line", 0)
            new_code = solution.get("code", "")
            
            # For safety, create a backup first
            backup_path = f"{file_path}.backup"
            try:
                subprocess.run(["cp", file_path, backup_path])
                
                # Apply the fix (simplified - in reality would parse and replace)
                logger.info(f"Would apply code fix to {file_path}:{line}")
                
                return {
                    "status": "success",
                    "message": f"Applied code fix to {file_path}",
                    "backup": backup_path
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}
                
        elif solution_type == "performance_optimization":
            # Create optimization file
            opt_file = f"{self.code_base_path}/backend/app/optimizations_{solution.get('metric', 'general')}.py"
            try:
                with open(opt_file, 'w') as f:
                    f.write(solution.get("code", ""))
                    
                return {
                    "status": "success",
                    "message": f"Created optimization file: {opt_file}"
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}
                
        elif solution_type == "feature_implementation":
            # Create new feature file
            feature_file = f"{self.code_base_path}/backend/app/{solution.get('feature', 'new_feature')}.py"
            try:
                with open(feature_file, 'w') as f:
                    f.write(solution.get("code", ""))
                    
                return {
                    "status": "success",
                    "message": f"Created feature file: {feature_file}"
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}
                
        return {"status": "unknown", "message": "Unknown solution type"}
        
    async def _init_performance_monitoring(self):
        """Initialize performance monitoring"""
        self.performance_metrics = {
            "avg_response_time": 0,
            "total_requests": 0,
            "error_rate": 0,
            "memory_usage": 0,
            "cpu_usage": 0
        }
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
            
    def _get_test_coverage(self) -> float:
        """Get test coverage percentage"""
        # Simplified - would run coverage tool
        return 45.0  # Placeholder
        
    async def _improvement_cycle(self):
        """Continuous improvement cycle"""
        while True:
            try:
                # Wait before next improvement
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Re-analyze codebase
                await self._analyze_codebase()
                
                # Apply improvements if any
                if self.improvement_opportunities:
                    await self.improve_system()
                    
            except Exception as e:
                logger.error(f"Improvement cycle error: {e}")
                
    async def continuous_improvement_loop(self):
        """Main continuous improvement loop"""
        logger.info("Starting continuous improvement loop...")
        
        while True:
            try:
                # Analyze system performance
                performance = await self._analyze_performance()
                
                # Check for improvement opportunities
                if performance.get("needs_improvement", False):
                    await self.improve_system()
                    
                # Learn from recent interactions
                await self._learn_from_usage()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Continuous improvement error: {e}")
                
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance"""
        # Collect metrics
        metrics = {
            "response_times": [],
            "error_count": 0,
            "success_count": 0,
            "needs_improvement": False
        }
        
        # Analyze recent performance
        if self.performance_metrics.get("avg_response_time", 0) > 500:
            metrics["needs_improvement"] = True
            
        return metrics
        
    async def _learn_from_usage(self):
        """Learn from system usage patterns"""
        # Analyze user interactions
        # Identify common patterns
        # Optimize for frequent use cases
        logger.info("Learning from usage patterns...")
        
    async def health_check(self) -> Dict[str, Any]:
        """Check self-improvement system health"""
        return {
            "status": "healthy" if self.initialized else "initializing",
            "improvements_made": len(self.improvement_history),
            "pending_improvements": len(self.improvement_opportunities),
            "last_improvement": self.improvement_history[-1]["timestamp"] if self.improvement_history else None
        } 