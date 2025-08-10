#!/usr/bin/env python3
"""
Update restart policies for containers based on their phase
"""
import subprocess

# Restart policies by phase
RESTART_POLICIES = {
    "critical": {
        "Name": "unless-stopped",
        "MaximumRetryCount": 0
    },
    "performance": {
        "Name": "on-failure",
        "MaximumRetryCount": 5
    },
    "specialized": {
        "Name": "on-failure", 
        "MaximumRetryCount": 3
    }
}

def get_container_phase(container_name):
    """Determine container phase based on name or port"""
    # Critical agents
    critical_keywords = [
        "agentzero-coordinator", "agent-orchestrator", 
        "task-assignment-coordinator", "autonomous-system-controller",
        "bigagi-system-manager"
    ]
    
    for keyword in critical_keywords:
        if keyword in container_name.lower():
            return "critical"
    
    # Performance agents
    performance_keywords = [
        "optimizer", "analyzer", "processor", "builder"
    ]
    
    for keyword in performance_keywords:
        if keyword in container_name.lower():
            return "performance"
    
    # Default to specialized
    return "specialized"

def update_container_restart_policy(container_name, policy):
    """Update container restart policy"""
    try:
        # Stop container first
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        
        # Update restart policy
        cmd = ["docker", "update"]
        
        if policy["Name"] == "on-failure":
            cmd.extend(["--restart", f"on-failure:{policy['MaximumRetryCount']}"])
        else:
            cmd.extend(["--restart", policy["Name"]])
        
        cmd.append(container_name)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Updated {container_name} restart policy to {policy['Name']}")
        else:
            print(f"✗ Failed to update {container_name}: {result.stderr}")
            
        # Start container again
        subprocess.run(["docker", "start", container_name], capture_output=True)
        
    except Exception as e:
        print(f"✗ Error updating {container_name}: {e}")

def main():
    print("Updating container restart policies...")
    print("=" * 50)
    
    # Get all containers
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    
    containers = result.stdout.strip().split('\n')
    
    for container in containers:
        if container and container.startswith("sutazai-"):
            phase = get_container_phase(container)
            policy = RESTART_POLICIES[phase]
            update_container_restart_policy(container, policy)
    
    print("\n✓ Restart policy update completed")

if __name__ == "__main__":
    main()
