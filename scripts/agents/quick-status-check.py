#!/usr/bin/env python3
"""
Quick Status Check - Get current agent system status
"""

import docker
import json
from datetime import datetime
from typing import Dict, List

def get_system_status():
    """Get quick system status"""
    docker_client = docker.from_env()
    
    # Get all containers
    containers = docker_client.containers.list(all=True, filters={"name": "sutazai-"})
    
    # Categorize containers
    status_summary = {
        "running": [],
        "restarting": [],
        "exited": [],
        "created": [],
        "unhealthy": [],
        "other": []
    }
    
    core_services = ["postgres", "redis", "ollama", "neo4j", "qdrant", "chromadb", "backend"]
    
    for container in containers:
        name = container.name.replace("sutazai-", "")
        status = container.status
        
        # Check health if available
        health = container.attrs.get("State", {}).get("Health", {})
        health_status = health.get("Status", "unknown") if health else "no_health_check"
        
        container_info = {
            "name": name,
            "status": status,
            "health": health_status,
            "is_core": name in core_services,
            "restart_count": container.attrs.get("RestartCount", 0)
        }
        
        if status == "running":
            if health_status == "unhealthy":
                status_summary["unhealthy"].append(container_info)
            else:
                status_summary["running"].append(container_info)
        elif status == "restarting":
            status_summary["restarting"].append(container_info)
        elif status == "exited":
            status_summary["exited"].append(container_info)
        elif status == "created":
            status_summary["created"].append(container_info)
        else:
            status_summary["other"].append(container_info)
    
    return status_summary

def print_status_report(status):
    """Print formatted status report"""
    print("🤖 SutazAI Agent System Status")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Count totals
    total_containers = sum(len(containers) for containers in status.values())
    running_count = len(status["running"])
    
    print(f"📊 SUMMARY")
    print(f"Total Containers: {total_containers}")
    print(f"Running: {running_count}")
    print(f"Restarting: {len(status['restarting'])}")
    print(f"Exited: {len(status['exited'])}")
    print(f"Created (not started): {len(status['created'])}")
    print(f"Unhealthy: {len(status['unhealthy'])}")
    print()
    
    # Core services status
    print("🔧 CORE SERVICES")
    core_running = [c for c in status["running"] if c["is_core"]]
    core_issues = []
    for category in ["restarting", "exited", "unhealthy"]:
        core_issues.extend([c for c in status[category] if c["is_core"]])
    
    for service in core_running:
        print(f"  ✅ {service['name']} - running")
    
    for service in core_issues:
        print(f"  ❌ {service['name']} - {service['status']} (health: {service['health']})")
    print()
    
    # Agent status
    print("🤖 AGENT STATUS")
    
    if status["running"]:
        agent_running = [c for c in status["running"] if not c["is_core"]]
        if agent_running:
            print(f"  ✅ Running ({len(agent_running)}):")
            for agent in sorted(agent_running, key=lambda x: x["name"])[:10]:  # Show first 10
                health_indicator = "🟢" if agent["health"] in ["healthy", "no_health_check"] else "🟡"
                print(f"    {health_indicator} {agent['name']}")
            if len(agent_running) > 10:
                print(f"    ... and {len(agent_running) - 10} more")
            print()
    
    if status["restarting"]:
        print(f"  🔄 Restarting ({len(status['restarting'])}):")
        for agent in sorted(status["restarting"], key=lambda x: x["restart_count"], reverse=True)[:10]:
            print(f"    🔄 {agent['name']} (restarts: {agent['restart_count']})")
        if len(status["restarting"]) > 10:
            print(f"    ... and {len(status['restarting']) - 10} more")
        print()
    
    if status["unhealthy"]:
        print(f"  🔴 Unhealthy ({len(status['unhealthy'])}):")
        for agent in status["unhealthy"]:
            print(f"    🔴 {agent['name']} - {agent['health']}")
        print()
    
    if status["exited"]:
        print(f"  ⏹️ Exited ({len(status['exited'])}):")
        for agent in status["exited"][:5]:  # Show first 5
            print(f"    ⏹️ {agent['name']}")
        if len(status["exited"]) > 5:
            print(f"    ... and {len(status['exited']) - 5} more")
        print()
    
    # Health assessment
    healthy_agents = len([c for c in status["running"] if not c["is_core"]])
    total_agents = total_containers - len([c for c in status["running"] + status["restarting"] + status["exited"] + status["unhealthy"] if c["is_core"]])
    
    if total_agents > 0:
        health_percentage = (healthy_agents / total_agents) * 100
        print(f"🏥 SYSTEM HEALTH")
        print(f"Agent Health Rate: {health_percentage:.1f}% ({healthy_agents}/{total_agents})")
        
        if health_percentage >= 80:
            print("Status: 🟢 EXCELLENT")
        elif health_percentage >= 60:
            print("Status: 🟡 GOOD")
        elif health_percentage >= 40:
            print("Status: 🟠 FAIR")
        else:
            print("Status: 🔴 POOR")

def main():
    """Main status check"""
    try:
        status = get_system_status()
        print_status_report(status)
        
        # Save to file for monitoring
        from pathlib import Path
        status_file = Path("/opt/sutazaiapp/logs/current-agent-status.json")
        status_file.parent.mkdir(exist_ok=True)
        
        with open(status_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "status": status
            }, f, indent=2)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())