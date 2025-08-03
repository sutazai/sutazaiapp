#!/usr/bin/env python3
"""
Purpose: Orchestrates specialized AI agents for systematic hygiene enforcement
Usage: python hygiene-agent-orchestrator.py --rule=<rule_number> [--dry-run]
Requirements: Specialized agents available, agent coordination system
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Optional

class AgentOrchestrator:
    """Orchestrates specialized agents for rule enforcement"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.agent_registry = self._load_agent_registry()
        self.execution_log = self.project_root / "logs" / "agent-orchestration.log"
        
    def _load_agent_registry(self) -> Dict:
        """Load available agents and their capabilities"""
        return {
            "garbage-collector": {
                "rules": [13],
                "capabilities": ["file_cleanup", "junk_removal", "archive_management"],
                "command": "python .claude/agents/garbage-collector.py",
                "config_file": "config/agents/garbage-collector.yaml"
            },
            "deploy-automation-master": {
                "rules": [12],
                "capabilities": ["script_consolidation", "deployment_automation", "ci_cd_integration"],
                "command": "python .claude/agents/deploy-automation-master.py", 
                "config_file": "config/agents/deploy-automation-master.yaml"
            },
            "multi-agent-coordinator": {
                "rules": [9, 14],
                "capabilities": ["directory_consolidation", "agent_management", "coordination"],
                "command": "python .claude/agents/multi-agent-coordinator.py",
                "config_file": "config/agents/multi-agent-coordinator.yaml"
            },
            "container-orchestrator-k3s": {
                "rules": [11],
                "capabilities": ["docker_organization", "container_optimization", "k8s_deployment"],
                "command": "python .claude/agents/container-orchestrator-k3s.py",
                "config_file": "config/agents/container-orchestrator-k3s.yaml"
            },
            "senior-backend-developer": {
                "rules": [8, 2],
                "capabilities": ["python_documentation", "code_quality", "backend_architecture"],
                "command": "python .claude/agents/senior-backend-developer.py",
                "config_file": "config/agents/senior-backend-developer.yaml"
            },
            "mega-code-auditor": {
                "rules": [1, 2, 3],
                "capabilities": ["code_analysis", "quality_gates", "compliance_checking"],
                "command": "python .claude/agents/mega-code-auditor.py",
                "config_file": "config/agents/mega-code-auditor.yaml"
            },
            "system-optimizer-reorganizer": {
                "rules": [7, 10],
                "capabilities": ["structure_optimization", "performance_improvement", "reorganization"],
                "command": "python .claude/agents/system-optimizer-reorganizer.py",
                "config_file": "config/agents/system-optimizer-reorganizer.yaml"
            },
            "document-knowledge-manager": {
                "rules": [6, 15],
                "capabilities": ["documentation_standardization", "knowledge_organization"],
                "command": "python .claude/agents/document-knowledge-manager.py",
                "config_file": "config/agents/document-knowledge-manager.yaml"
            }
        }
    
    def log_action(self, message: str, level: str = "INFO"):
        """Log orchestration actions"""
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        self.execution_log.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.execution_log, "a") as f:
            f.write(log_entry + "\n")
        
        print(log_entry)
    
    def get_agents_for_rule(self, rule_number: int) -> List[str]:
        """Get agents capable of enforcing a specific rule"""
        capable_agents = []
        
        for agent_name, agent_config in self.agent_registry.items():
            if rule_number in agent_config.get("rules", []):
                capable_agents.append(agent_name)
        
        return capable_agents
    
    def create_agent_task(self, agent_name: str, rule_number: int, task_params: Dict) -> Dict:
        """Create standardized task for agent execution"""
        task = {
            "agent": agent_name,
            "rule": rule_number,
            "task_id": f"{agent_name}_rule_{rule_number}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "parameters": task_params,
            "project_root": str(self.project_root),
            "timestamp": datetime.datetime.now().isoformat(),
            "safety_mode": True,  # Always enforce Rule 10 (no breaking changes)
            "dry_run": task_params.get("dry_run", False)
        }
        
        return task
    
    def execute_agent_task(self, task: Dict) -> Dict:
        """Execute task with specified agent"""
        agent_name = task["agent"]
        agent_config = self.agent_registry.get(agent_name)
        
        if not agent_config:
            return {
                "status": "error",
                "message": f"Agent {agent_name} not found in registry",
                "task_id": task["task_id"]
            }
        
        # Create task file for agent
        task_file = self.project_root / "tmp" / f"{task['task_id']}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(task_file, "w") as f:
            json.dump(task, f, indent=2)
        
        self.log_action(f"Executing task {task['task_id']} with agent {agent_name}")
        
        try:
            # Execute agent command with task file
            cmd = agent_config["command"].split() + ["--task-file", str(task_file)]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            execution_result = {
                "status": "success" if result.returncode == 0 else "error",
                "task_id": task["task_id"],
                "agent": agent_name,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": datetime.datetime.now().isoformat()
            }
            
            # Clean up task file
            if task_file.exists():
                task_file.unlink()
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "task_id": task["task_id"],
                "message": "Agent execution timed out after 1 hour"
            }
        except Exception as e:
            return {
                "status": "error", 
                "task_id": task["task_id"],
                "message": f"Agent execution failed: {str(e)}"
            }
    
    def coordinate_rule_enforcement(self, rule_number: int, dry_run: bool = False) -> Dict:
        """Coordinate enforcement of a specific rule"""
        self.log_action(f"=== COORDINATING RULE {rule_number} ENFORCEMENT ===")
        
        capable_agents = self.get_agents_for_rule(rule_number)
        
        if not capable_agents:
            return {
                "status": "error",
                "message": f"No agents available for rule {rule_number}",
                "rule": rule_number
            }
        
        results = []
        
        # Define rule-specific task parameters
        rule_tasks = {
            13: {"action": "cleanup_junk_files", "patterns": ["*.backup*", "*.tmp", "*.bak"]},
            12: {"action": "consolidate_deployment_scripts", "target": "/deploy.sh"},
            9: {"action": "consolidate_directories", "target_structure": "standard"},
            11: {"action": "organize_docker_structure", "target_dir": "/docker"},
            8: {"action": "add_python_documentation", "header_format": "standard"},
            1: {"action": "validate_real_implementations", "check_fantasy_elements": True},
            2: {"action": "verify_no_breaking_changes", "test_mode": "comprehensive"},
            3: {"action": "analyze_full_system", "scope": "complete"}
        }
        
        task_params = rule_tasks.get(rule_number, {"action": "general_compliance_check"})
        task_params["dry_run"] = dry_run
        
        # Execute with each capable agent
        for agent_name in capable_agents:
            task = self.create_agent_task(agent_name, rule_number, task_params)
            result = self.execute_agent_task(task)
            results.append(result)
            
            self.log_action(f"Agent {agent_name} result: {result['status']}")
            
            if result["status"] != "success":
                self.log_action(f"Agent {agent_name} failed: {result.get('message', 'Unknown error')}", "ERROR")
        
        return {
            "rule": rule_number,
            "agents_used": capable_agents,
            "results": results,
            "overall_status": "success" if all(r["status"] == "success" for r in results) else "partial_failure"
        }
    
    def orchestrate_full_enforcement(self, priority_rules: List[int], dry_run: bool = False) -> Dict:
        """Orchestrate enforcement across multiple rules in priority order"""
        self.log_action("=== STARTING FULL HYGIENE ENFORCEMENT ORCHESTRATION ===")
        
        enforcement_results = {}
        overall_success = True
        
        for rule_number in priority_rules:
            try:
                result = self.coordinate_rule_enforcement(rule_number, dry_run)
                enforcement_results[f"rule_{rule_number}"] = result
                
                if result.get("overall_status") != "success":
                    overall_success = False
                    self.log_action(f"Rule {rule_number} enforcement had issues", "WARN")
                
            except Exception as e:
                self.log_action(f"Failed to enforce rule {rule_number}: {e}", "ERROR")
                enforcement_results[f"rule_{rule_number}"] = {
                    "status": "error",
                    "message": str(e)
                }
                overall_success = False
        
        final_report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_status": "success" if overall_success else "partial_failure",
            "rules_processed": priority_rules,
            "results": enforcement_results,
            "summary": {
                "total_rules": len(priority_rules),
                "successful_rules": sum(1 for r in enforcement_results.values() 
                                      if r.get("overall_status") == "success"),
                "failed_rules": sum(1 for r in enforcement_results.values() 
                                  if r.get("overall_status") != "success")
            }
        }
        
        # Save orchestration report
        report_path = self.project_root / "logs" / f"orchestration-report-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)
        
        self.log_action(f"Orchestration complete. Report: {report_path}")
        
        return final_report

def main():
    parser = argparse.ArgumentParser(description="Orchestrate AI agents for hygiene enforcement")
    parser.add_argument("--rule", type=int, choices=range(1, 17),
                       help="Specific rule to enforce (1-16)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                       help="Enforcement phase (1=critical, 2=structural, 3=organizational)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    orchestrator = AgentOrchestrator(args.project_root)
    
    try:
        if args.rule:
            # Enforce specific rule
            result = orchestrator.coordinate_rule_enforcement(args.rule, args.dry_run)
            print(json.dumps(result, indent=2))
            return 0 if result.get("overall_status") == "success" else 1
            
        elif args.phase:
            # Enforce by phase priority
            phase_rules = {
                1: [13, 12, 9],      # Critical violations
                2: [11, 8, 1, 2, 3], # Structural violations  
                3: [7, 6, 15, 4, 5, 10, 14, 16]  # Organizational violations
            }
            
            priority_rules = phase_rules.get(args.phase, [])
            result = orchestrator.orchestrate_full_enforcement(priority_rules, args.dry_run)
            print(json.dumps(result, indent=2))
            return 0 if result.get("overall_status") == "success" else 1
        
        else:
            # Enforce all rules in priority order
            all_rules = [13, 12, 9, 11, 8, 1, 2, 3, 7, 6, 15, 4, 5, 10, 14, 16]
            result = orchestrator.orchestrate_full_enforcement(all_rules, args.dry_run)
            print(json.dumps(result, indent=2))
            return 0 if result.get("overall_status") == "success" else 1
            
    except Exception as e:
        orchestrator.log_action(f"Orchestration failed: {e}", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main())