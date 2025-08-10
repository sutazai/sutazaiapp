#!/usr/bin/env python3
"""
Purpose: Coordinates systematic codebase hygiene enforcement across all 16 rules
Usage: python hygiene-enforcement-coordinator.py [--phase=1|2|3] [--dry-run]
Requirements: Python 3.8+, git, specialized AI agents available
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
import asyncio
import aiohttp
import psutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ResourceUsage:
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    
class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300, reset_timeout: int = 600):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    def call(self, func, *args, **kwargs):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker opened: {e}")
            
            raise

class ResourceMonitor:
    """Monitor system resources to prevent overload"""
    
    def __init__(self, cpu_limit: float = 80.0, memory_limit_mb: float = 8192):
        self.cpu_limit = cpu_limit
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process()
        
    def get_usage(self) -> ResourceUsage:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get disk I/O if available
        try:
            io_counters = self.process.io_counters()
            disk_io_mb = (io_counters.read_bytes + io_counters.write_bytes) / 1024 / 1024
        except:
            disk_io_mb = 0
            
        return ResourceUsage(cpu_percent, memory_mb, disk_io_mb)
    
    def check_limits(self) -> bool:
        """Check if resource usage is within limits"""
        usage = self.get_usage()
        return usage.cpu_percent < self.cpu_limit and usage.memory_mb < self.memory_limit_mb

class HygieneEnforcementCoordinator:
    """Enhanced coordinator with rule control integration and resource management"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.archive_root = self.project_root / "archive"
        self.enforcement_log = self.project_root / "logs" / "hygiene-enforcement.log"
        self.dry_run = False
        
        # Rule control API endpoint
        self.rule_control_api = "http://localhost:8100/api"
        
        # Load agent configuration
        self.config_file = self.project_root / "config" / "hygiene-agents.json"
        self.agents = {}
        self.load_configuration()
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        self.circuit_breakers = {}
        
        # Rule state cache
        self.rule_states = {}
        self.last_rule_fetch = None
        self.rule_cache_ttl = 60  # seconds
        
    def load_configuration(self):
        """Load agent configuration from JSON file"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                config = json.load(f)
                self.agents = config.get("agents", {})
                self.global_settings = config.get("global_settings", {})
        else:
            logger.warning(f"Configuration file not found: {self.config_file}")
            
    async def fetch_rule_states(self) -> Dict[str, bool]:
        """Fetch current rule states from Rule Control API"""
        # Check cache
        if self.last_rule_fetch and (time.time() - self.last_rule_fetch) < self.rule_cache_ttl:
            return self.rule_states
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.rule_control_api}/rules") as response:
                    if response.status == 200:
                        data = await response.json()
                        rules = data.get("rules", [])
                        self.rule_states = {rule["id"]: rule["enabled"] for rule in rules}
                        self.last_rule_fetch = time.time()
                        logger.info(f"Fetched rule states: {len(self.rule_states)} rules")
                    else:
                        logger.error(f"Failed to fetch rule states: {response.status}")
        except Exception as e:
            logger.error(f"Error connecting to Rule Control API: {e}")
            # Use cached states or default to all enabled
            if not self.rule_states:
                self.rule_states = {agent_data.get("enforces_rules", [])[0]: True 
                                  for agent_data in self.agents.values() 
                                  if agent_data.get("enforces_rules")}
        
        return self.rule_states
    
    def should_run_agent(self, agent_name: str) -> bool:
        """Check if an agent should run based on rule states"""
        agent_config = self.agents.get(agent_name, {})
        enforced_rules = agent_config.get("enforces_rules", [])
        
        # Agent runs if ANY of its rules are enabled
        for rule_id in enforced_rules:
            if self.rule_states.get(rule_id, True):  # Default to enabled if unknown
                return True
        
        return False
    
    def get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent"""
        if agent_name not in self.circuit_breakers:
            cb_config = self.global_settings.get("circuit_breaker", {})
            self.circuit_breakers[agent_name] = CircuitBreaker(
                failure_threshold=cb_config.get("failure_threshold", 3),
                timeout=cb_config.get("timeout_seconds", 300),
                reset_timeout=cb_config.get("reset_timeout_seconds", 600)
            )
        return self.circuit_breakers[agent_name]
    
    def log_action(self, message: str, level: str = "INFO"):
        """Log enforcement actions with timestamp"""
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Ensure log directory exists
        self.enforcement_log.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.enforcement_log, "a") as f:
            f.write(log_entry + "\n")
        
        # Use logger for consistency
        getattr(logger, level.lower(), logger.info)(message)
    
    def create_archive_directory(self, rule_name: str) -> Path:
        """Create dated archive directory for safe file removal"""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        archive_dir = self.archive_root / f"{date_str}-{rule_name}-cleanup"
        
        if not self.dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            
        return archive_dir
    
    def find_violations(self, rule_name: str) -> List[Path]:
        """Find all files violating a specific rule"""
        violations = []
        rule_config = self.rule_violations.get(rule_name, {})
        patterns = rule_config.get("patterns", [])
        
        for pattern in patterns:
            # Use find command for better performance with large repos
            try:
                result = subprocess.run(
                    ["find", str(self.project_root), "-name", pattern, "-type", "f"],
                    capture_output=True, text=True, check=True
                )
                
                file_paths = [Path(p.strip()) for p in result.stdout.split("\n") if p.strip()]
                violations.extend(file_paths)
                
            except subprocess.CalledProcessError as e:
                self.log_action(f"Error finding {pattern}: {e}", "ERROR")
        
        return list(set(violations))  # Remove duplicates
    
    def verify_file_safety(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Verify file can be safely removed (Rule 10 compliance)"""
        references = []
        safe_to_remove = True
        
        # Search for references in codebase
        try:
            result = subprocess.run(
                ["grep", "-r", file_path.name, str(self.project_root)],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                references = result.stdout.strip().split("\n")
                # Filter out self-references
                references = [ref for ref in references if str(file_path) not in ref]
                
                if references:
                    safe_to_remove = False
                    
        except subprocess.CalledProcessError:
            # No references found (grep returns 1 when no matches)
            pass
        
        return safe_to_remove, references
    
    def archive_file(self, file_path: Path, archive_dir: Path) -> bool:
        """Safely archive a file before removal"""
        try:
            # Preserve directory structure in archive
            relative_path = file_path.relative_to(self.project_root)
            archive_path = archive_dir / relative_path
            
            # Create parent directories in archive
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.dry_run:
                # Copy file to archive
                subprocess.run(["cp", str(file_path), str(archive_path)], check=True)
                self.log_action(f"Archived: {file_path} -> {archive_path}")
                
            return True
            
        except Exception as e:
            self.log_action(f"Failed to archive {file_path}: {e}", "ERROR")
            return False
    
    def enforce_rule_13(self) -> Dict[str, int]:
        """Enforce Rule 13: No garbage, no rot"""
        self.log_action("=== ENFORCING RULE 13: NO GARBAGE, NO ROT ===")
        
        violations = self.find_violations("rule_13")
        archive_dir = self.create_archive_directory("rule-13-junk-files")
        
        stats = {"found": len(violations), "archived": 0, "removed": 0, "skipped": 0}
        
        for file_path in violations:
            # Skip if file doesn't exist (might be symlink or already removed)
            if not file_path.exists():
                continue
                
            # Verify safety before removal
            safe, references = self.verify_file_safety(file_path)
            
            if safe:
                # Archive then remove
                if self.archive_file(file_path, archive_dir):
                    if not self.dry_run:
                        file_path.unlink()
                        stats["removed"] += 1
                        self.log_action(f"Removed junk file: {file_path}")
                    else:
                        self.log_action(f"[DRY RUN] Would remove: {file_path}")
                    stats["archived"] += 1
                else:
                    stats["skipped"] += 1
            else:
                self.log_action(f"Skipped {file_path} (has references): {references[:3]}", "WARN")
                stats["skipped"] += 1
        
        self.log_action(f"Rule 13 enforcement complete: {stats}")
        return stats
    
    def enforce_rule_12(self) -> Dict[str, int]:
        """Enforce Rule 12: One canonical deployment script"""
        self.log_action("=== ENFORCING RULE 12: SINGLE DEPLOYMENT SCRIPT ===")
        
        # Find all deployment-related scripts
        deploy_scripts = []
        for pattern in ["deploy*.sh", "*deploy*.py", "validate*deploy*"]:
            result = subprocess.run(
                ["find", str(self.project_root), "-name", pattern, "-type", "f"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                scripts = [Path(p.strip()) for p in result.stdout.split("\n") if p.strip()]
                deploy_scripts.extend(scripts)
        
        stats = {"found": len(deploy_scripts), "consolidated": 0, "archived": 0}
        
        if len(deploy_scripts) <= 1:
            self.log_action("Rule 12 already compliant: ≤1 deployment script found")
            return stats
        
        # Create archive for backup
        archive_dir = self.create_archive_directory("rule-12-deploy-consolidation")
        
        # Identify primary deployment script (largest or most recent)
        primary_script = max(deploy_scripts, key=lambda p: p.stat().st_size)
        self.log_action(f"Primary deployment script identified: {primary_script}")
        
        # Archive other scripts
        for script in deploy_scripts:
            if script != primary_script:
                if self.archive_file(script, archive_dir):
                    stats["archived"] += 1
                    self.log_action(f"Archived redundant deploy script: {script}")
        
        self.log_action("NEXT: Coordinate with deploy-automation-master agent for consolidation")
        
        return stats
    
    async def run_agent(self, agent_name: str, agent_config: Dict) -> Dict:
        """Run a specific hygiene enforcement agent with resource management"""
        start_time = time.time()
        result = {
            "agent": agent_name,
            "status": "pending",
            "started": datetime.datetime.now().isoformat(),
            "enforced_rules": agent_config.get("enforces_rules", []),
            "violations_found": 0,
            "violations_fixed": 0,
            "errors": []
        }
        
        # Check if agent should run based on rule states
        if not self.should_run_agent(agent_name):
            result["status"] = "skipped"
            result["reason"] = "All enforced rules are disabled"
            return result
        
        # Check resource limits
        if not self.resource_monitor.check_limits():
            result["status"] = "deferred"
            result["reason"] = "Resource limits exceeded"
            return result
        
        # Get circuit breaker
        circuit_breaker = self.get_circuit_breaker(agent_name)
        
        try:
            # Run agent through circuit breaker
            agent_path = self.project_root / "scripts" / "agents" / f"{agent_name}.py"
            if not agent_path.exists():
                result["status"] = "error"
                result["errors"].append(f"Agent script not found: {agent_path}")
                return result
            
            # Execute agent with resource limits
            resource_limits = agent_config.get("resource_limits", {})
            timeout = resource_limits.get("timeout_seconds", 300)
            
            cmd = [sys.executable, str(agent_path)]
            if self.dry_run:
                cmd.append("--dry-run")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                if process.returncode == 0:
                    result["status"] = "success"
                    # Parse agent output (assuming JSON format)
                    try:
                        output_data = json.loads(stdout.decode())
                        result.update(output_data)
                    except:
                        result["output"] = stdout.decode()
                else:
                    result["status"] = "failed"
                    result["errors"].append(stderr.decode())
                    
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                result["status"] = "timeout"
                result["errors"].append(f"Agent exceeded timeout of {timeout} seconds")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Error running agent {agent_name}: {e}")
        
        result["completed"] = datetime.datetime.now().isoformat()
        result["duration_seconds"] = time.time() - start_time
        
        return result
    
    async def run_phase(self, phase: int) -> Dict[str, Dict]:
        """Run specific enforcement phase with async agent execution"""
        # Fetch current rule states
        await self.fetch_rule_states()
        
        results = {"phase": phase, "agents_run": []}
        
        # Define phase agent groups
        phase_agents = {
            1: [  # Critical violations
                "garbage-collection-enforcer",
                "deployment-script-enforcer", 
                "functionality-preservation-validator",
                "safe-cleanup-validator"
            ],
            2: [  # Structural violations  
                "docker-structure-validator",
                "script-consolidation-enforcer",
                "version-control-enforcer",
                "documentation-structure-enforcer"
            ],
            3: [  # Organizational violations
                "professional-standards-validator",
                "ai-agent-router",
                "ollama-enforcement-agent",
                "script-reuse-enforcer"
            ]
        }
        
        agents_to_run = phase_agents.get(phase, [])
        
        # Run agents with concurrency limit
        max_concurrent = self.global_settings.get("max_concurrent_agents", 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(agent_name):
            async with semaphore:
                agent_config = self.agents.get(agent_name, {})
                return await self.run_agent(agent_name, agent_config)
        
        # Execute agents concurrently
        tasks = [run_with_semaphore(agent) for agent in agents_to_run]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(agent_results):
            if isinstance(result, Exception):
                results["agents_run"].append({
                    "agent": agents_to_run[i],
                    "status": "error",
                    "error": str(result)
                })
            else:
                results["agents_run"].append(result)
        
        # Summary statistics
        results["summary"] = {
            "total_agents": len(agents_to_run),
            "successful": sum(1 for r in results["agents_run"] if r.get("status") == "success"),
            "failed": sum(1 for r in results["agents_run"] if r.get("status") == "failed"),
            "skipped": sum(1 for r in results["agents_run"] if r.get("status") == "skipped"),
            "errors": sum(1 for r in results["agents_run"] if r.get("status") == "error")
        }
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate enforcement report"""
        report_path = self.project_root / "logs" / f"hygiene-report-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "dry_run": self.dry_run,
            "results": results,
            "next_steps": [
                "Coordinate with specialized agents for remaining rules",
                "Implement automated prevention mechanisms", 
                "Schedule regular compliance audits"
            ]
        }
        
        if not self.dry_run:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
        
        return str(report_path)

def main():
    parser = argparse.ArgumentParser(description="Coordinate codebase hygiene enforcement")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
                       help="Enforcement phase to run (1=critical, 2=structural, 3=organizational)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    parser.add_argument("--all-phases", action="store_true",
                       help="Run all phases sequentially")
    
    args = parser.parse_args()
    
    coordinator = HygieneEnforcementCoordinator(args.project_root)
    coordinator.dry_run = args.dry_run
    
    async def run_async():
        try:
            if args.all_phases:
                all_results = {}
                for phase in [1, 2, 3]:
                    coordinator.log_action(f"=== RUNNING PHASE {phase} ===")
                    results = await coordinator.run_phase(phase)
                    all_results[f"phase_{phase}"] = results
                    
                    # Add delay between phases to prevent resource exhaustion
                    if phase < 3:
                        await asyncio.sleep(5)
                
                report_path = coordinator.generate_report(all_results)
                coordinator.log_action("All phases complete")
            else:
                results = await coordinator.run_phase(args.phase)
                report_path = coordinator.generate_report(results)
                coordinator.log_action(f"Enforcement phase {args.phase} complete")
            
            coordinator.log_action(f"Report generated: {report_path}")
            return 0
            
        except Exception as e:
            coordinator.log_action(f"Enforcement failed: {e}", "ERROR")
            logger.exception("Enforcement failed")
            return 1
    
    # Run async main
    return asyncio.run(run_async())

if __name__ == "__main__":
    sys.exit(main())