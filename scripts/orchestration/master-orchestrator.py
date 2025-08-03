#!/usr/bin/env python3
"""
Purpose: Master orchestrator for comprehensive codebase hygiene enforcement
Usage: python master-orchestrator.py [--mode=async|sync] [--rules=1,2,3] [--dry-run]
Requirements: Redis, Celery, asyncio, specialized enforcement agents
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

import redis
from celery import Celery, group, chord
from celery.result import AsyncResult

# Configuration
PROJECT_ROOT = Path("/opt/sutazaiapp")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ORCHESTRATION_CONFIG = PROJECT_ROOT / "config" / "orchestration.json"

# Initialize Celery
app = Celery('hygiene_orchestrator', broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Initialize Redis client for state management
redis_client = redis.from_url(REDIS_URL)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "master-orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RulePriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class EnforcementStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RuleEnforcementTask:
    rule_id: int
    priority: RulePriority
    agent_name: str
    module_path: str
    async_execution: bool = True
    dependencies: List[int] = field(default_factory=list)
    status: EnforcementStatus = EnforcementStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class HygieneOrchestrationMaster:
    """Master orchestrator for comprehensive hygiene enforcement"""
    
    def __init__(self, config_path: Path = ORCHESTRATION_CONFIG):
        self.config = self._load_config(config_path)
        self.enforcement_tasks: Dict[int, RuleEnforcementTask] = {}
        self.active_tasks: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_rule_tasks()
        
    def _load_config(self, config_path: Path) -> Dict:
        """Load orchestration configuration"""
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        else:
            # Load from architecture JSON
            arch_path = PROJECT_ROOT / "HYGIENE_ORCHESTRATION_ARCHITECTURE.json"
            if arch_path.exists():
                with open(arch_path) as f:
                    arch = json.load(f)
                    return arch.get("core_components", {})
            return {}
    
    def _initialize_rule_tasks(self):
        """Initialize enforcement tasks for all rules"""
        rule_configs = {
            1: ("mega-code-auditor", "rule1_reality_enforcer.py", RulePriority.HIGH),
            2: ("senior-backend-developer", "rule2_stability_guardian.py", RulePriority.CRITICAL),
            3: ("mega-code-auditor", "rule3_comprehensive_analyzer.py", RulePriority.HIGH),
            4: ("system-optimizer-reorganizer", "rule4_reuse_enforcer.py", RulePriority.MEDIUM),
            5: ("mega-code-auditor", "rule5_professionalism_checker.py", RulePriority.MEDIUM),
            6: ("document-knowledge-manager", "rule6_doc_standardizer.py", RulePriority.MEDIUM),
            7: ("system-optimizer-reorganizer", "rule7_script_organizer.py", RulePriority.HIGH),
            8: ("senior-backend-developer", "rule8_python_enforcer.py", RulePriority.HIGH),
            9: ("multi-agent-coordinator", "rule9_version_controller.py", RulePriority.CRITICAL),
            10: ("mega-code-auditor", "rule10_safety_validator.py", RulePriority.CRITICAL),
            11: ("container-orchestrator-k3s", "rule11_docker_organizer.py", RulePriority.HIGH),
            12: ("deploy-automation-master", "rule12_deploy_unifier.py", RulePriority.CRITICAL),
            13: ("garbage-collector", "rule13_garbage_collector.py", RulePriority.CRITICAL),
            14: ("multi-agent-coordinator", "rule14_agent_selector.py", RulePriority.LOW),
            15: ("document-knowledge-manager", "rule15_doc_deduplicator.py", RulePriority.MEDIUM),
            16: ("system-optimizer-reorganizer", "rule16_llm_enforcer.py", RulePriority.LOW),
        }
        
        # Define rule dependencies
        dependencies = {
            2: [10],   # No breaking requires safety validation first
            4: [3],    # Reuse requires full analysis first
            7: [13],   # Script organization after garbage cleanup
            9: [13],   # Version control after garbage cleanup
            11: [13],  # Docker organization after cleanup
            12: [3, 7], # Deploy unification needs analysis and script org
        }
        
        for rule_id, (agent, module, priority) in rule_configs.items():
            self.enforcement_tasks[rule_id] = RuleEnforcementTask(
                rule_id=rule_id,
                priority=priority,
                agent_name=agent,
                module_path=f"/opt/sutazaiapp/scripts/enforcement/{module}",
                dependencies=dependencies.get(rule_id, [])
            )
    
    async def _check_system_resources(self) -> Tuple[bool, Dict]:
        """Check if system has enough resources for enforcement"""
        try:
            # Check CPU usage
            cpu_cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1"
            cpu_result = await asyncio.create_subprocess_shell(
                cpu_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            cpu_output, _ = await cpu_result.communicate()
            cpu_usage = float(cpu_output.decode().strip() or 0)
            
            # Check memory
            mem_cmd = "free -m | awk 'NR==2{printf \"%.2f\", $3*100/$2 }'"
            mem_result = await asyncio.create_subprocess_shell(
                mem_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            mem_output, _ = await mem_result.communicate()
            mem_usage = float(mem_output.decode().strip() or 0)
            
            # Check active tasks
            active_count = len(self.active_tasks)
            
            resources = {
                "cpu_usage": cpu_usage,
                "memory_usage": mem_usage,
                "active_tasks": active_count,
                "available": cpu_usage < 80 and mem_usage < 80 and active_count < 3
            }
            
            return resources["available"], resources
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True, {"error": str(e)}
    
    def _get_executable_rules(self, requested_rules: List[int]) -> List[int]:
        """Get rules that can be executed based on dependencies"""
        executable = []
        completed = {rule_id for rule_id, task in self.enforcement_tasks.items() 
                    if task.status == EnforcementStatus.COMPLETED}
        
        for rule_id in requested_rules:
            task = self.enforcement_tasks.get(rule_id)
            if not task:
                continue
                
            # Check if all dependencies are completed
            if all(dep in completed for dep in task.dependencies):
                executable.append(rule_id)
            else:
                logger.info(f"Rule {rule_id} waiting for dependencies: {task.dependencies}")
        
        return executable
    
    @app.task(name='enforce_rule_async')
    def enforce_rule_async(self, rule_id: int, dry_run: bool = False) -> Dict:
        """Celery task for async rule enforcement"""
        task = self.enforcement_tasks.get(rule_id)
        if not task:
            return {"status": "error", "message": f"Unknown rule: {rule_id}"}
        
        try:
            # Update task status
            task.status = EnforcementStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Execute enforcement module
            result = self._execute_enforcement_module(task, dry_run)
            
            # Update task with result
            task.status = EnforcementStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            
            return {
                "rule_id": rule_id,
                "status": "completed",
                "result": result,
                "duration": (task.completed_at - task.started_at).total_seconds()
            }
            
        except Exception as e:
            task.status = EnforcementStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            return {
                "rule_id": rule_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _execute_enforcement_module(self, task: RuleEnforcementTask, dry_run: bool) -> Dict:
        """Execute the enforcement module for a rule"""
        # For now, simulate execution
        # In production, this would actually run the enforcement module
        logger.info(f"Executing enforcement for rule {task.rule_id} using {task.agent_name}")
        
        # Simulate some work
        import time
        time.sleep(2)
        
        return {
            "violations_found": 10,
            "violations_fixed": 8 if not dry_run else 0,
            "agent": task.agent_name,
            "dry_run": dry_run
        }
    
    async def orchestrate_enforcement(
        self,
        rules: Optional[List[int]] = None,
        phase: Optional[int] = None,
        async_mode: bool = True,
        dry_run: bool = False
    ) -> Dict:
        """Main orchestration method"""
        logger.info("Starting hygiene enforcement orchestration")
        
        # Determine which rules to enforce
        if rules:
            target_rules = rules
        elif phase:
            phase_mapping = {
                1: [13, 12, 9, 2, 10],  # Critical
                2: [11, 8, 1, 3, 7],    # Structural
                3: [6, 15, 4, 5, 14, 16]  # Organizational
            }
            target_rules = phase_mapping.get(phase, [])
        else:
            target_rules = list(range(1, 17))  # All rules
        
        # Check system resources
        can_proceed, resources = await self._check_system_resources()
        if not can_proceed:
            return {
                "status": "deferred",
                "reason": "Insufficient system resources",
                "resources": resources
            }
        
        # Start enforcement
        start_time = datetime.utcnow()
        results = {
            "start_time": start_time.isoformat(),
            "mode": "async" if async_mode else "sync",
            "dry_run": dry_run,
            "target_rules": target_rules,
            "rule_results": {}
        }
        
        if async_mode:
            # Execute rules asynchronously with Celery
            results["rule_results"] = await self._orchestrate_async(target_rules, dry_run)
        else:
            # Execute rules synchronously
            results["rule_results"] = await self._orchestrate_sync(target_rules, dry_run)
        
        # Calculate summary
        end_time = datetime.utcnow()
        results["end_time"] = end_time.isoformat()
        results["duration"] = (end_time - start_time).total_seconds()
        results["summary"] = self._calculate_summary(results["rule_results"])
        
        # Save orchestration report
        self._save_report(results)
        
        return results
    
    async def _orchestrate_async(self, rules: List[int], dry_run: bool) -> Dict:
        """Orchestrate rules asynchronously using Celery"""
        results = {}
        pending_rules = rules.copy()
        completed_rules = set()
        
        while pending_rules:
            # Get rules that can be executed now
            executable = self._get_executable_rules(pending_rules)
            if not executable:
                # No rules can be executed, might be circular dependency
                logger.error(f"Cannot execute remaining rules: {pending_rules}")
                break
            
            # Create Celery group for parallel execution
            job_group = group(
                self.enforce_rule_async.s(rule_id, dry_run)
                for rule_id in executable
            )
            
            # Execute and wait for results
            group_result = job_group.apply_async()
            
            # Monitor execution
            while not group_result.ready():
                await asyncio.sleep(1)
            
            # Collect results
            for rule_id, result in zip(executable, group_result.get()):
                results[f"rule_{rule_id}"] = result
                completed_rules.add(rule_id)
                pending_rules.remove(rule_id)
                
                # Update task status
                task = self.enforcement_tasks[rule_id]
                task.status = EnforcementStatus.COMPLETED
        
        return results
    
    async def _orchestrate_sync(self, rules: List[int], dry_run: bool) -> Dict:
        """Orchestrate rules synchronously"""
        results = {}
        
        for rule_id in self._get_execution_order(rules):
            task = self.enforcement_tasks.get(rule_id)
            if not task:
                continue
            
            # Check dependencies
            deps_met = all(
                self.enforcement_tasks[dep].status == EnforcementStatus.COMPLETED
                for dep in task.dependencies
            )
            
            if not deps_met:
                results[f"rule_{rule_id}"] = {
                    "status": "skipped",
                    "reason": "Dependencies not met"
                }
                continue
            
            # Execute enforcement
            try:
                task.status = EnforcementStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                result = self._execute_enforcement_module(task, dry_run)
                
                task.status = EnforcementStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = result
                
                results[f"rule_{rule_id}"] = {
                    "status": "completed",
                    "result": result
                }
                
            except Exception as e:
                task.status = EnforcementStatus.FAILED
                task.error = str(e)
                
                results[f"rule_{rule_id}"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results
    
    def _get_execution_order(self, rules: List[int]) -> List[int]:
        """Get topologically sorted execution order based on dependencies"""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(rule_id):
            if rule_id in visited:
                return
            visited.add(rule_id)
            
            task = self.enforcement_tasks.get(rule_id)
            if task:
                for dep in task.dependencies:
                    if dep in rules:
                        visit(dep)
            
            order.append(rule_id)
        
        for rule_id in rules:
            visit(rule_id)
        
        return order
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate enforcement summary statistics"""
        summary = {
            "total_rules": len(results),
            "completed": sum(1 for r in results.values() if r.get("status") == "completed"),
            "failed": sum(1 for r in results.values() if r.get("status") == "failed"),
            "skipped": sum(1 for r in results.values() if r.get("status") == "skipped"),
            "total_violations": 0,
            "total_fixed": 0
        }
        
        for result in results.values():
            if result.get("status") == "completed" and "result" in result:
                summary["total_violations"] += result["result"].get("violations_found", 0)
                summary["total_fixed"] += result["result"].get("violations_fixed", 0)
        
        return summary
    
    def _save_report(self, results: Dict):
        """Save orchestration report"""
        report_dir = PROJECT_ROOT / "reports" / "orchestration"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"orchestration_report_{timestamp}.json"
        
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Orchestration report saved to {report_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Master hygiene orchestrator")
    parser.add_argument("--rules", type=str, help="Comma-separated rule IDs")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Enforcement phase")
    parser.add_argument("--mode", choices=["async", "sync"], default="async", help="Execution mode")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    # Parse rules if provided
    rules = None
    if args.rules:
        rules = [int(r) for r in args.rules.split(",")]
    
    # Create orchestrator
    orchestrator = HygieneOrchestrationMaster()
    
    # Run orchestration
    results = await orchestrator.orchestrate_enforcement(
        rules=rules,
        phase=args.phase,
        async_mode=(args.mode == "async"),
        dry_run=args.dry_run
    )
    
    # Print summary
    print(json.dumps(results["summary"], indent=2))
    
    return 0 if results["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))