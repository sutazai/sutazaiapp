#!/usr/bin/env python3
"""
Purpose: Simple agent activator using existing Docker Compose infrastructure
Usage: python simple-agent-activator.py --phase=<1|2|3> [--agents=agent1,agent2]
Requirements: Docker, docker-compose
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

class SimpleAgentActivator:
    """Simple agent activator using existing compose infrastructure"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        
        # Phase 1 Critical Agents (available in main compose)
        self.critical_agents = [
            "ai-system-validator",
            "ai-testing-qa-validator", 
            "hardware-resource-optimizer",
            "ollama-integration-specialist",
            "semgrep-security-analyzer",
            "mega-code-auditor",
            "document-knowledge-manager",
            "testing-qa-validator"
        ]
    
    def log_action(self, message: str):
        """Simple logging"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def activate_agents(self, agents: list, dry_run: bool = False) -> dict:
        """Activate specified agents using docker-compose"""
        self.log_action(f"{'DRY RUN: ' if dry_run else ''}Activating agents: {agents}")
        
        if dry_run:
            return {"status": "dry_run", "agents": agents}
        
        results = {"activated": [], "failed": []}
        
        for agent in agents:
            try:
                self.log_action(f"Starting agent: {agent}")
                
                # Try to start the service
                cmd = ["docker-compose", "up", "-d", agent]
                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.log_action(f"✅ Successfully started: {agent}")
                    results["activated"].append(agent)
                else:
                    self.log_action(f"❌ Failed to start {agent}: {result.stderr}")
                    results["failed"].append(agent)
                
                # Brief pause between starts
                time.sleep(5)
                
            except subprocess.TimeoutExpired:
                self.log_action(f"⏰ Timeout starting {agent}")
                results["failed"].append(agent)
            except Exception as e:
                self.log_action(f"❌ Error starting {agent}: {e}")
                results["failed"].append(agent)
        
        return results
    
    def check_agent_status(self, agent: str) -> str:
        """Check if an agent container is running"""
        try:
            cmd = ["docker", "ps", "--filter", f"name=sutazai-{agent}", "--format", "{{.Status}}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                if "Up" in result.stdout:
                    return "running"
                else:
                    return "stopped"
            else:
                return "not_found"
        except:
            return "unknown"
    
    def get_available_agents(self) -> list:
        """Get list of agents available in docker-compose"""
        try:
            cmd = ["docker-compose", "config", "--services"]
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return [service.strip() for service in result.stdout.split('\n') if service.strip()]
            else:
                return []
        except:
            return []
    
    def run_phase_activation(self, phase: int, dry_run: bool = False) -> dict:
        """Run activation for a specific phase"""
        self.log_action(f"=== PHASE {phase} ACTIVATION ===")
        
        if phase == 1:
            target_agents = self.critical_agents
        else:
            self.log_action(f"Phase {phase} not implemented in simple activator")
            return {"status": "not_implemented", "phase": phase}
        
        # Filter to only agents that exist in compose
        available_agents = self.get_available_agents()
        valid_agents = [agent for agent in target_agents if agent in available_agents]
        missing_agents = [agent for agent in target_agents if agent not in available_agents]
        
        if missing_agents:
            self.log_action(f"⚠️  Agents not found in compose: {missing_agents}")
        
        if not valid_agents:
            self.log_action("❌ No valid agents found to activate")
            return {"status": "no_agents", "phase": phase}
        
        self.log_action(f"Found {len(valid_agents)} valid agents to activate")
        
        # Check current status
        running_agents = []
        stopped_agents = []
        
        for agent in valid_agents:
            status = self.check_agent_status(agent)
            if status == "running":
                running_agents.append(agent)
            else:
                stopped_agents.append(agent)
        
        self.log_action(f"Currently running: {len(running_agents)}, Stopped: {len(stopped_agents)}")
        
        if not stopped_agents:
            self.log_action("✅ All agents are already running!")
            return {"status": "already_running", "phase": phase, "running": running_agents}
        
        # Activate stopped agents
        results = self.activate_agents(stopped_agents, dry_run)
        
        return {
            "status": "completed",
            "phase": phase,
            "already_running": running_agents,
            "activation_results": results
        }

def main():
    parser = argparse.ArgumentParser(description="Simple agent activator")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
                       help="Phase to activate (currently only phase 1 implemented)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--agents", type=str,
                       help="Comma-separated list of specific agents to activate")
    parser.add_argument("--list-available", action="store_true",
                       help="List available agents in docker-compose")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    activator = SimpleAgentActivator(args.project_root)
    
    try:
        if args.list_available:
            # List available agents
            agents = activator.get_available_agents()
            print(f"Available agents in docker-compose ({len(agents)}):")
            for agent in sorted(agents):
                status = activator.check_agent_status(agent)
                status_icon = "🟢" if status == "running" else "🔴" if status == "stopped" else "⚪"
                print(f"  {status_icon} {agent} ({status})")
            return 0
        
        elif args.agents:
            # Activate specific agents
            agent_list = [agent.strip() for agent in args.agents.split(',')]
            results = activator.activate_agents(agent_list, args.dry_run)
            print(f"Results: {results}")
            return 0 if not results.get("failed") else 1
        
        else:
            # Activate phase
            results = activator.run_phase_activation(args.phase, args.dry_run)
            print(f"Phase {args.phase} activation results:")
            print(f"Status: {results['status']}")
            
            if results['status'] == 'completed':
                activation = results['activation_results']
                print(f"✅ Activated: {len(activation['activated'])} agents")
                print(f"❌ Failed: {len(activation['failed'])} agents")
                print(f"🟢 Already running: {len(results['already_running'])} agents")
                
                if activation['failed']:
                    print("Failed agents:", activation['failed'])
                    return 1
            
            return 0
            
    except Exception as e:
        print(f"❌ Activation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())