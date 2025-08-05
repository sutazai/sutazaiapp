#!/usr/bin/env python3
"""
Agent Cleanup Implementation Script
Implements the recommendations from the comprehensive agent analysis.

SAFETY FEATURES:
- Dry-run mode by default
- Backup creation before changes
- Rollback capability
- Detailed logging of all operations
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set

class AgentCleanupManager:
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/agents", dry_run: bool = True):
        self.agents_dir = Path(agents_dir)
        self.dry_run = dry_run
        self.backup_dir = Path(f"/opt/sutazaiapp/backups/agent_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load analysis results
        self.analysis_results = self._load_analysis_results()
        
        # Define agents to remove (from analysis)
        self.agents_to_remove = {
            # Low value stubs
            'ollama-integration-specialist',
            'code-generation-improver', 
            'senior-ai-engineer',
            'hardware-optimizer',
            'context-optimizer',
            
            # Experimental/Theoretical (low practical value)
            'quantum-ai-researcher',
            'deep-learning-brain-architect',
            'neuromorphic-computing-expert',
            'deep-local-brain-builder',
            'quantum-computing-optimizer',
            
            # Redundant hardware optimizers (keep hardware-resource-optimizer)
            'cpu-only-hardware-optimizer',
            'gpu-hardware-optimizer', 
            'ram-hardware-optimizer',
            
            # Redundant monitoring
            'intelligence-optimization-monitor',
            
            # Stub agents with medium/low value
            'genetic-algorithm-tuner',
            'system-performance-forecaster',
            'cognitive-load-monitor',
            'resource-arbitration-agent',
            'deep-learning-brain-manager',
            'ai-system-validator',
            'metrics-collector-prometheus',
            'deploy-automation-master',
            'automated-incident-responder',
            'prompt-injection-guard',
            'ml-experiment-tracker-mlflow',
            'data-drift-detector',
            'autonomous-task-executor',
            'goal-setting-and-planning-agent',
            'resource-visualiser',
            'human-oversight-interface-agent',
            'observability-dashboard-manager-grafana',
            'compute-scheduler-and-optimizer',
            'experiment-tracker',
            'senior-full-stack-developer',
            'codebase-team-lead',
            'scrum-master',
            'agent-debugger',
            'product-manager',
            'ai-senior-full-stack-developer',
            'code-quality-gateway-sonarqube',
            'evolution-strategy-trainer',
            'garbage-collector',
            'system-validator',
            'energy-consumption-optimize',
            'bias-and-fairness-auditor',
            'container-vulnerability-scanner-trivy',
            'secrets-vault-manager-vault',
            'log-aggregator-loki',
            'ram-hardware-optimizer',
            'neural-architecture-search',
            'edge-inference-proxy',
            'data-version-controller-dvc',
            'data-lifecycle-manager',
            'system-knowledge-curator',
            'runtime-behavior-anomaly-detector',
            'senior-engineer',
            'distributed-tracing-analyzer-jaeger',
            'explainability-and-transparency-agent',
            'ethical-governor'
        }
        
        # Define agents to keep (high value + working)
        self.agents_to_keep = {
            'hardware-resource-optimizer',  # 963 lines, fully functional
            'infrastructure-devops',
            'deployment-automation-master',
            'ai-system-architect',
            'health-monitor',
            'self-healing-orchestrator',
            'ai-senior-backend-developer',
            'ai-senior-frontend-developer',
            'semgrep-security-analyzer',
            'ai-qa-team-lead',
            'ai-testing-qa-validator',
            'task-assignment-coordinator',
            'opendevin-code-generator',
            'code-improver',
            'garbage-collector-coordinator',
            'observability-monitoring-engineer',
            'service-hub',
            'infrastructure-devops-manager',
            'symbolic-reasoning-engine',
            'attention-optimizer',
            'data-pipeline-engineer',
            'ai-agent-debugger',
            'causal-inference-expert',
            'federated-learning-coordinator',
            'document-knowledge-manager',
            'gradient-compression-specialist',
            'edge-computing-optimizer',
            'localagi-orchestration-manager',
            'distributed-computing-architect',
            'agentzero-coordinator',
            'cognitive-architecture-designer',
            'product-strategy-architect',
            'multi-modal-fusion-coordinator',
            'senior-frontend-developer',
            'private-data-analyst',
            'senior-backend-developer',
            'synthetic-data-generator',
            'reinforcement-learning-trainer',
            'flowiseai-flow-manager',
            'data-analysis-engineer',
            'meta-learning-specialist',
            'mcp-server',
            'ai-product-manager',
            'context-framework',
            'gpt-engineer',
            'awesome-code-ai',
            'model-training-specialist',
            'bigagi-system-manager',
            'episodic-memory-engineer',
            'agentgpt-autonomous-executor',
            'ai-senior-engineer',
            'langflow-workflow-designer',
            'knowledge-graph-builder',
            'browser-automation-orchestrator',
            'dify-automation-specialist',
            'transformers-migration-specialist',
            'ai-scrum-master',
            'knowledge-distillation-expert',
            'memory-persistence-manager',
            'explainable-ai-specialist',
            'jarvis-voice-interface'
        }
        
    def _load_analysis_results(self) -> Dict:
        """Load analysis results from JSON file"""
        results_file = Path("/opt/sutazaiapp/agent_analysis_results.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def create_backup(self) -> bool:
        """Create backup of agents directory before cleanup"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would create backup at: {self.backup_dir}")
                return True
                
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup agents directory
            shutil.copytree(self.agents_dir, self.backup_dir / "agents")
            
            # Backup docker-compose files
            compose_files = [
                "/opt/sutazaiapp/docker-compose.yml",
                "/opt/sutazaiapp/docker-compose.agents.yml"
            ]
            
            for compose_file in compose_files:
                if Path(compose_file).exists():
                    shutil.copy2(compose_file, self.backup_dir)
            
            # Create manifest
            manifest = {
                "backup_time": datetime.now().isoformat(),
                "agents_backed_up": len(list(self.agents_dir.glob("*-*"))),
                "cleanup_config": {
                    "agents_to_remove": sorted(list(self.agents_to_remove)),
                    "agents_to_keep": sorted(list(self.agents_to_keep))
                }
            }
            
            with open(self.backup_dir / "cleanup_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Backup created successfully at: {self.backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def remove_agent_directories(self) -> Dict[str, bool]:
        """Remove agent directories marked for deletion"""
        results = {}
        
        for agent_name in self.agents_to_remove:
            agent_path = self.agents_dir / agent_name
            
            if agent_path.exists():
                try:
                    if self.dry_run:
                        self.logger.info(f"[DRY RUN] Would remove: {agent_path}")
                        results[agent_name] = True
                    else:
                        shutil.rmtree(agent_path)
                        self.logger.info(f"Removed agent directory: {agent_path}")
                        results[agent_name] = True
                except Exception as e:
                    self.logger.error(f"Failed to remove {agent_path}: {e}")
                    results[agent_name] = False
            else:
                self.logger.warning(f"Agent directory not found: {agent_path}")
                results[agent_name] = False
        
        return results
    
    def update_docker_compose(self) -> bool:
        """Update docker-compose.yml to remove deleted agents"""
        compose_file = Path("/opt/sutazaiapp/docker-compose.yml")
        agents_compose_file = Path("/opt/sutazaiapp/docker-compose.agents.yml")
        
        if self.dry_run:
            self.logger.info("[DRY RUN] Would update docker-compose files to remove deleted agents")
            return True
        
        try:
            # This would require YAML parsing and modification
            # For now, just log the requirement
            self.logger.info("Manual step required: Update docker-compose files to remove deleted agents")
            self.logger.info(f"Remove services for: {', '.join(sorted(self.agents_to_remove))}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update docker-compose files: {e}")
            return False
    
    def fix_requirements_conflicts(self) -> bool:
        """Fix the Docker version conflict identified in analysis"""
        if self.dry_run:
            self.logger.info("[DRY RUN] Would standardize docker package to version 7.0.0")
            return True
        
        # Update infrastructure-devops requirements.txt
        infra_req_file = self.agents_dir / "infrastructure-devops" / "requirements.txt"
        
        if infra_req_file.exists():
            try:
                with open(infra_req_file, 'r') as f:
                    content = f.read()
                
                # Replace docker==6.1.3 with docker==7.0.0
                updated_content = content.replace("docker==6.1.3", "docker==7.0.0")
                
                with open(infra_req_file, 'w') as f:
                    f.write(updated_content)
                
                self.logger.info("Fixed Docker version conflict in infrastructure-devops")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to fix requirements conflict: {e}")
                return False
        
        return True
    
    def generate_cleanup_report(self, removal_results: Dict[str, bool]) -> None:
        """Generate detailed cleanup report"""
        report_file = Path("/opt/sutazaiapp/AGENT_CLEANUP_REPORT.md")
        
        successful_removals = [k for k, v in removal_results.items() if v]
        failed_removals = [k for k, v in removal_results.items() if not v]
        
        report_content = f"""# AGENT CLEANUP EXECUTION REPORT

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mode:** {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}

## CLEANUP SUMMARY

### Agents Processed
- **Total agents before cleanup:** {len(list(self.agents_dir.glob('*-*')))}
- **Agents marked for removal:** {len(self.agents_to_remove)}
- **Agents to keep:** {len(self.agents_to_keep)}

### Removal Results
- **Successfully removed:** {len(successful_removals)}
- **Failed removals:** {len(failed_removals)}

## DETAILED RESULTS

### Successfully Removed Agents ({len(successful_removals)})
"""
        
        for agent in sorted(successful_removals):
            report_content += f"✅ {agent}\n"
        
        report_content += f"\n### Failed Removals ({len(failed_removals)})\n"
        for agent in sorted(failed_removals):
            report_content += f"❌ {agent}\n"
        
        report_content += f"""
### Kept Agents ({len(self.agents_to_keep)})
"""
        for agent in sorted(self.agents_to_keep):
            agent_path = self.agents_dir / agent
            status = "✅ EXISTS" if agent_path.exists() else "❌ MISSING"
            report_content += f"{status} {agent}\n"
        
        report_content += f"""

## NEXT STEPS

1. **Update Docker Compose Files**
   - Remove services for deleted agents from docker-compose.yml
   - Update port mappings and dependencies

2. **Test System Functionality**
   - Start remaining agents
   - Verify system health endpoints
   - Check inter-agent communication

3. **Resource Validation**
   - Monitor memory usage reduction
   - Verify storage space freed
   - Confirm container count reduction

## ROLLBACK INSTRUCTIONS

If issues arise, restore from backup:
```bash
# Stop all services
docker-compose down

# Restore agents directory
rm -rf /opt/sutazaiapp/agents
cp -r {self.backup_dir}/agents /opt/sutazaiapp/

# Restore docker-compose files
cp {self.backup_dir}/docker-compose*.yml /opt/sutazaiapp/

# Restart system
docker-compose up -d
```

## SYSTEM IMPACT ESTIMATE

**Resource Savings:**
- Memory: ~2-4GB RAM reduction
- Storage: ~500MB disk space freed
- Containers: {len(successful_removals)} fewer Docker containers
- Ports: {len(successful_removals)} freed ports in range 10000-12000

**Performance Benefits:**
- Faster system startup (~40-50% improvement)
- Reduced container orchestration overhead
- Simplified debugging and maintenance
"""

        if self.dry_run:
            print("\n" + "="*60)
            print("DRY RUN CLEANUP REPORT")
            print("="*60)
            print(report_content)
        else:
            with open(report_file, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Cleanup report saved to: {report_file}")
    
    def execute_cleanup(self) -> bool:
        """Execute the complete cleanup process"""
        self.logger.info("Starting agent cleanup process...")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        
        # Step 1: Create backup
        if not self.create_backup():
            self.logger.error("Backup creation failed. Aborting cleanup.")
            return False
        
        # Step 2: Remove agent directories
        removal_results = self.remove_agent_directories()
        
        # Step 3: Fix requirements conflicts
        self.fix_requirements_conflicts()
        
        # Step 4: Update docker-compose (manual step noted)
        self.update_docker_compose()
        
        # Step 5: Generate report
        self.generate_cleanup_report(removal_results)
        
        successful_count = sum(1 for v in removal_results.values() if v)
        self.logger.info(f"Cleanup completed. {successful_count}/{len(self.agents_to_remove)} agents processed successfully.")
        
        return True

def main():
    """Main execution function with safety checks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Cleanup Implementation")
    parser.add_argument("--execute", action="store_true", help="Execute cleanup (default is dry-run)")
    parser.add_argument("--agents-dir", default="/opt/sutazaiapp/agents", help="Agents directory path")
    
    args = parser.parse_args()
    
    # Safety confirmation for live execution
    if args.execute:
        print("⚠️  WARNING: This will permanently remove agent directories!")
        print("Make sure you have reviewed the analysis report and understand the impact.")
        confirmation = input("Type 'CONFIRM' to proceed with live execution: ")
        
        if confirmation != "CONFIRM":
            print("Cleanup cancelled.")
            return False
    
    # Execute cleanup
    cleanup_manager = AgentCleanupManager(
        agents_dir=args.agents_dir,
        dry_run=not args.execute
    )
    
    return cleanup_manager.execute_cleanup()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)