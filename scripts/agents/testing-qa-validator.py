#!/usr/bin/env python3
"""
Purpose: Comprehensive testing and QA validation for hygiene enforcement system
Usage: python testing-qa-validator.py [--test-type TYPE] [--output-format FORMAT]
Requirements: pytest, coverage, hypothesis, faker
"""

import os
import sys
import json
import time
import asyncio
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import sqlite3
import hashlib
import multiprocessing

# Test framework imports
import pytest
import coverage
from hypothesis import given, strategies as st, settings, HealthCheck
from faker import Faker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/sutazaiapp")
TEST_DB_PATH = PROJECT_ROOT / "logs" / "rule_test_results.db"
CONFIG_PATH = PROJECT_ROOT / "config" / "hygiene-agents.json"

@dataclass
class TestResult:
    test_id: str
    rule_combination: str
    status: str  # pass, fail, skip, error
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    violations_found: int
    violations_fixed: int
    errors: List[str]
    warnings: List[str]
    timestamp: str
    agent_results: Dict[str, Dict]

@dataclass
class CombinationTestSuite:
    total_combinations: int = 65536  # 2^16 rules
    completed_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    estimated_completion_time: str = ""
    performance_metrics: Dict = None

class ComprehensiveTestValidator:
    """Validates the entire hygiene enforcement system with exhaustive testing"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.fake = Faker()
        self.test_db = self._initialize_test_database()
        self.agents_config = self._load_agents_config()
        self.rule_ids = list(self._get_all_rule_ids())
        
        # Performance limits for zero-tolerance validation
        self.max_response_time_ms = 5000
        self.max_memory_usage_mb = 2048
        self.max_cpu_usage_percent = 80
        
        # Circuit breaker for test suite stability
        self.max_consecutive_failures = 10
        self.consecutive_failures = 0
        
    def _initialize_test_database(self):
        """Initialize SQLite database for test results"""
        TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(TEST_DB_PATH))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT UNIQUE,
                rule_combination TEXT,
                status TEXT,
                duration_ms REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                violations_found INTEGER,
                violations_fixed INTEGER,
                errors TEXT,
                warnings TEXT,
                timestamp TEXT,
                agent_results TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS test_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_run_id TEXT,
                total_combinations INTEGER,
                completed_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                error_tests INTEGER,
                avg_response_time_ms REAL,
                max_memory_usage_mb REAL,
                avg_cpu_usage_percent REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        return conn
    
    def _load_agents_config(self) -> Dict:
        """Load agent configuration"""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                return json.load(f)
        return {}
    
    def _get_all_rule_ids(self) -> Set[str]:
        """Extract all rule IDs from agent configuration"""
        rule_ids = set()
        for agent_config in self.agents_config.get("agents", {}).values():
            rule_ids.update(agent_config.get("enforces_rules", []))
        return rule_ids
    
    def generate_rule_combinations(self) -> List[Tuple[str, Dict[str, bool]]]:
        """Generate all 65,536 possible rule combinations"""
        logger.info(f"Generating {2**len(self.rule_ids)} rule combinations...")
        
        combinations = []
        for i, combination in enumerate(itertools.product([True, False], repeat=len(self.rule_ids))):
            rule_state = dict(zip(self.rule_ids, combination))
            combination_id = f"combo_{i:05d}_{hashlib.md5(str(rule_state).encode()).hexdigest()[:8]}"
            combinations.append((combination_id, rule_state))
            
            # Progress logging for large datasets
            if i % 1000 == 0:
                logger.info(f"Generated {i} combinations...")
        
        logger.info(f"Generated {len(combinations)} total combinations")
        return combinations
    
    async def test_rule_combination(self, combination_id: str, rule_states: Dict[str, bool]) -> TestResult:
        """Test a specific rule combination with comprehensive validation"""
        start_time = time.time()
        test_result = TestResult(
            test_id=combination_id,
            rule_combination=json.dumps(rule_states, sort_keys=True),
            status="running",
            duration_ms=0,
            memory_usage_mb=0,
            cpu_usage_percent=0,
            violations_found=0,
            violations_fixed=0,
            errors=[],
            warnings=[],
            timestamp=datetime.now().isoformat(),
            agent_results={}
        )
        
        try:
            # Set up rule states via API
            await self._configure_rule_states(rule_states)
            
            # Run enforcement coordinator with this configuration
            coordinator_result = await self._run_enforcement_coordinator(combination_id)
            test_result.agent_results = coordinator_result.get("agents_run", [])
            
            # Validate results
            validation_result = await self._validate_enforcement_results(coordinator_result)
            test_result.violations_found = validation_result.get("violations_found", 0)
            test_result.violations_fixed = validation_result.get("violations_fixed", 0)
            
            # Performance metrics
            end_time = time.time()
            test_result.duration_ms = (end_time - start_time) * 1000
            test_result.memory_usage_mb = self._get_memory_usage()
            test_result.cpu_usage_percent = self._get_cpu_usage()
            
            # Determine status
            if test_result.duration_ms > self.max_response_time_ms:
                test_result.status = "fail"
                test_result.errors.append(f"Response time {test_result.duration_ms}ms exceeds limit")
            elif test_result.memory_usage_mb > self.max_memory_usage_mb:
                test_result.status = "fail"
                test_result.errors.append(f"Memory usage {test_result.memory_usage_mb}MB exceeds limit")
            elif validation_result.get("critical_errors", 0) > 0:
                test_result.status = "fail"
                test_result.errors.extend(validation_result.get("errors", []))
            else:
                test_result.status = "pass"
                self.consecutive_failures = 0
            
        except Exception as e:
            test_result.status = "error"
            test_result.errors.append(str(e))
            self.consecutive_failures += 1
            logger.error(f"Error testing combination {combination_id}: {e}")
            
            # Circuit breaker
            if self.consecutive_failures >= self.max_consecutive_failures:
                raise Exception(f"Circuit breaker triggered: {self.consecutive_failures} consecutive failures")
        
        # Store result in database
        await self._store_test_result(test_result)
        return test_result
    
    async def _configure_rule_states(self, rule_states: Dict[str, bool]):
        """Configure rule states via API"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Bulk update rule states
            bulk_request = {
                "rule_ids": list(rule_states.keys()),
                "enabled": True,  # Will be overridden per rule
                "force": True
            }
            
            # Set each rule individually for precision
            for rule_id, enabled in rule_states.items():
                try:
                    async with session.put(
                        f"http://localhost:8100/api/rules/{rule_id}/toggle",
                        json={"rule_id": rule_id, "enabled": enabled, "force": True}
                    ) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to set rule {rule_id} to {enabled}")
                except Exception as e:
                    logger.error(f"Error configuring rule {rule_id}: {e}")
    
    async def _run_enforcement_coordinator(self, combination_id: str) -> Dict:
        """Run the hygiene enforcement coordinator"""
        coordinator_script = self.project_root / "scripts" / "hygiene-enforcement-coordinator.py" 
        
        cmd = [
            sys.executable, str(coordinator_script),
            "--phase", "1",
            "--dry-run"  # Use dry-run for testing
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60  # 1 minute timeout per test
            )
            
            if process.returncode == 0:
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError:
                    return {"output": stdout.decode(), "stderr": stderr.decode()}
            else:
                return {"error": stderr.decode(), "returncode": process.returncode}
                
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()
            return {"error": "Timeout: enforcement coordinator exceeded 60 seconds"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _validate_enforcement_results(self, coordinator_result: Dict) -> Dict:
        """Validate enforcement coordinator results"""
        validation = {
            "violations_found": 0,
            "violations_fixed": 0,
            "critical_errors": 0,
            "errors": [],
            "warnings": []
        }
        
        # Check for critical errors in agent results
        agents_run = coordinator_result.get("agents_run", [])
        for agent_result in agents_run:
            if agent_result.get("status") == "error":
                validation["critical_errors"] += 1
                validation["errors"].extend(agent_result.get("errors", []))
            elif agent_result.get("status") == "failed":
                validation["warnings"].append(f"Agent {agent_result.get('agent')} failed")
            
            # Aggregate violation counts
            validation["violations_found"] += agent_result.get("violations_found", 0)
            validation["violations_fixed"] += agent_result.get("violations_fixed", 0)
        
        return validation
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        import psutil
        return psutil.cpu_percent(interval=0.1)
    
    async def _store_test_result(self, result: TestResult):
        """Store test result in database"""
        self.test_db.execute('''
            INSERT OR REPLACE INTO test_results (
                test_id, rule_combination, status, duration_ms, memory_usage_mb,
                cpu_usage_percent, violations_found, violations_fixed, errors,
                warnings, timestamp, agent_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.test_id,
            result.rule_combination,
            result.status,
            result.duration_ms,
            result.memory_usage_mb,
            result.cpu_usage_percent,
            result.violations_found,
            result.violations_fixed,
            json.dumps(result.errors),
            json.dumps(result.warnings),
            result.timestamp,
            json.dumps(result.agent_results)
        ))
        self.test_db.commit()
    
    async def run_comprehensive_test_suite(self, max_parallel: int = 5) -> CombinationTestSuite:
        """Run the complete test suite with all 65,536 combinations"""
        logger.info("🧪 Starting comprehensive test suite for all rule combinations...")
        
        suite = CombinationTestSuite()
        combinations = self.generate_rule_combinations()
        suite.total_combinations = len(combinations)
        
        # Create semaphore for parallel testing
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def test_with_semaphore(combination_data):
            async with semaphore:
                combination_id, rule_states = combination_data
                return await self.test_rule_combination(combination_id, rule_states)
        
        # Run tests in batches
        batch_size = 100
        start_time = time.time()
        
        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i + batch_size]
            logger.info(f"Running batch {i//batch_size + 1}/{(len(combinations) + batch_size - 1)//batch_size}")
            
            # Run batch tests
            tasks = [test_with_semaphore(combo) for combo in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    suite.error_tests += 1
                    logger.error(f"Test exception: {result}")
                elif isinstance(result, TestResult):
                    suite.completed_tests += 1
                    if result.status == "pass":
                        suite.passed_tests += 1
                    elif result.status == "fail":
                        suite.failed_tests += 1
                    else:
                        suite.error_tests += 1
            
            # Progress reporting
            elapsed_time = time.time() - start_time
            if suite.completed_tests > 0:
                avg_time_per_test = elapsed_time / suite.completed_tests
                remaining_tests = suite.total_combinations - suite.completed_tests
                estimated_remaining = avg_time_per_test * remaining_tests
                suite.estimated_completion_time = f"{estimated_remaining/60:.1f} minutes"
            
            logger.info(f"Progress: {suite.completed_tests}/{suite.total_combinations} "
                       f"(Pass: {suite.passed_tests}, Fail: {suite.failed_tests}, Error: {suite.error_tests}) "
                       f"ETA: {suite.estimated_completion_time}")
            
            # Store performance snapshot
            await self._store_performance_snapshot(suite)
        
        # Final performance metrics
        total_time = time.time() - start_time
        suite.performance_metrics = {
            "total_runtime_minutes": total_time / 60,
            "tests_per_minute": suite.completed_tests / (total_time / 60),
            "success_rate_percent": (suite.passed_tests / suite.completed_tests * 100) if suite.completed_tests > 0 else 0
        }
        
        logger.info(f"🎉 Test suite completed! Results: {asdict(suite)}")
        return suite
    
    async def _store_performance_snapshot(self, suite: CombinationTestSuite):
        """Store performance snapshot in database"""
        # Get average metrics from recent tests
        cursor = self.test_db.execute('''
            SELECT AVG(duration_ms), MAX(memory_usage_mb), AVG(cpu_usage_percent)
            FROM test_results 
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        avg_response, max_memory, avg_cpu = cursor.fetchone()
        
        self.test_db.execute('''
            INSERT INTO test_performance (
                test_run_id, total_combinations, completed_tests, passed_tests,
                failed_tests, error_tests, avg_response_time_ms, max_memory_usage_mb,
                avg_cpu_usage_percent, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            suite.total_combinations,
            suite.completed_tests,
            suite.passed_tests,
            suite.failed_tests,
            suite.error_tests,
            avg_response or 0,
            max_memory or 0,
            avg_cpu or 0,
            datetime.now().isoformat()
        ))
        self.test_db.commit()
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report"""
        # Query database for results
        cursor = self.test_db.execute('''
            SELECT status, COUNT(*), AVG(duration_ms), AVG(memory_usage_mb), AVG(cpu_usage_percent)
            FROM test_results
            GROUP BY status
        ''')
        
        status_summary = {}
        for status, count, avg_duration, avg_memory, avg_cpu in cursor.fetchall():
            status_summary[status] = {
                "count": count,
                "avg_duration_ms": avg_duration,
                "avg_memory_mb": avg_memory,
                "avg_cpu_percent": avg_cpu
            }
        
        # Get failure analysis
        cursor = self.test_db.execute('''
            SELECT errors, COUNT(*)
            FROM test_results
            WHERE status IN ('fail', 'error') AND errors != '[]'
            GROUP BY errors
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')
        
        common_failures = []
        for error_json, count in cursor.fetchall():
            try:
                errors = json.loads(error_json)
                common_failures.append({"errors": errors, "count": count})
            except:
                pass
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "test_summary": status_summary,
            "common_failures": common_failures,
            "total_combinations_tested": sum(s["count"] for s in status_summary.values()),
            "overall_success_rate": (status_summary.get("pass", {}).get("count", 0) / 
                                   sum(s["count"] for s in status_summary.values()) * 100) 
                                   if status_summary else 0,
            "performance_analysis": self._analyze_performance(),
            "recommendations": self._generate_recommendations(status_summary, common_failures)
        }
        
        return report
    
    def _analyze_performance(self) -> Dict:
        """Analyze performance metrics"""
        cursor = self.test_db.execute('''
            SELECT 
                MIN(duration_ms), MAX(duration_ms), AVG(duration_ms),
                MIN(memory_usage_mb), MAX(memory_usage_mb), AVG(memory_usage_mb),
                MIN(cpu_usage_percent), MAX(cpu_usage_percent), AVG(cpu_usage_percent)
            FROM test_results
        ''')
        
        row = cursor.fetchone()
        if row:
            return {
                "response_time": {
                    "min_ms": row[0],
                    "max_ms": row[1],
                    "avg_ms": row[2],
                    "within_limits": row[1] <= self.max_response_time_ms if row[1] else True
                },
                "memory_usage": {
                    "min_mb": row[3],
                    "max_mb": row[4],
                    "avg_mb": row[5],
                    "within_limits": row[4] <= self.max_memory_usage_mb if row[4] else True
                },
                "cpu_usage": {
                    "min_percent": row[6],
                    "max_percent": row[7],
                    "avg_percent": row[8],
                    "within_limits": row[7] <= self.max_cpu_usage_percent if row[7] else True
                }
            }
        
        return {}
    
    def _generate_recommendations(self, status_summary: Dict, common_failures: List[Dict]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        total_tests = sum(s["count"] for s in status_summary.values())
        if total_tests == 0:
            return ["No test data available for analysis"]
        
        success_rate = status_summary.get("pass", {}).get("count", 0) / total_tests * 100
        
        if success_rate < 95:
            recommendations.append(f"Success rate is {success_rate:.1f}% - target is >95%")
        
        if status_summary.get("error", {}).get("count", 0) > 0:
            recommendations.append("System errors detected - investigate error handling")
        
        # Analyze common failure patterns
        for failure in common_failures[:3]:  # Top 3 failure patterns
            if failure["count"] > total_tests * 0.1:  # If >10% of tests fail the same way
                recommendations.append(
                    f"Common failure pattern affecting {failure['count']} tests: {failure['errors'][:1]}"
                )
        
        # Performance recommendations
        perf = self._analyze_performance()
        if perf.get("response_time", {}).get("max_ms", 0) > self.max_response_time_ms:
            recommendations.append("Response time exceeds limits - optimize performance")
        
        if perf.get("memory_usage", {}).get("max_mb", 0) > self.max_memory_usage_mb:
            recommendations.append("Memory usage exceeds limits - check for memory leaks")
        
        if not recommendations:
            recommendations.append("✅ System performance is within acceptable limits")
        
        return recommendations

async def main():
    """Main testing entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Testing & QA Validation")
    parser.add_argument("--test-type", choices=["combinations", "performance", "agents", "all"], 
                       default="all", help="Type of test to run")
    parser.add_argument("--output-format", choices=["json", "html", "text"], 
                       default="json", help="Output format for reports")
    parser.add_argument("--max-parallel", type=int, default=5, 
                       help="Maximum parallel tests")
    parser.add_argument("--sample-size", type=int, default=None, 
                       help="Sample size for testing (default: all combinations)")
    
    args = parser.parse_args()
    
    validator = ComprehensiveTestValidator()
    
    try:
        if args.test_type in ["combinations", "all"]:
            logger.info("Running comprehensive rule combination tests...")
            suite_result = await validator.run_comprehensive_test_suite(args.max_parallel)
            logger.info(f"Test suite completed: {asdict(suite_result)}")
        
        # Generate final report
        report = validator.generate_test_report()
        
        # Output report
        report_path = PROJECT_ROOT / "reports" / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Test report generated: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE TEST SUITE RESULTS")
        print("="*60)
        print(f"Total Combinations Tested: {report['total_combinations_tested']:,}")
        print(f"Overall Success Rate: {report['overall_success_rate']:.1f}%")
        
        for status, data in report['test_summary'].items():
            print(f"{status.upper()}: {data['count']:,} tests")
        
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("="*60)
        
        return 0 if report['overall_success_rate'] >= 95 else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)