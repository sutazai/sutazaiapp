#!/usr/bin/env python3
"""
Deploy MCP Services to DinD Orchestrator
Uses the MCP Manager API to deploy all MCP services from manifests
"""
import requests
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any

MANAGER_URL = "http://localhost:18081"
MANIFEST_DIR = Path("/opt/sutazaiapp/docker/dind/orchestrator/mcp-manifests")

def load_manifest(manifest_file: Path) -> Dict[str, Any]:
    """Load and parse a YAML manifest file"""
    with open(manifest_file, 'r') as f:
        return yaml.safe_load(f)

def convert_manifest_to_api_format(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Convert manifest format to API format"""
    spec = manifest['spec']
    
    # Build Docker image if needed
    image = spec.get('image', 'node:18-alpine')
    
    # Build API payload - only include fields the Manager expects
    api_config = {
        'name': spec['container_name'],
        'image': image,
        'environment': spec.get('environment', {}),
        'ports': spec.get('ports', {}),
        'restart_policy': spec.get('restart_policy', 'unless-stopped')
    }
    
    # Convert volumes to proper format
    volumes = spec.get('volumes', {})
    # Ensure volumes is a dict with string:string format
    if isinstance(volumes, dict):
        # Convert any non-string values to strings
        api_config['volumes'] = {str(k): str(v) for k, v in volumes.items()}
    else:
        api_config['volumes'] = {}
    
    return api_config

def deploy_container(api_config: Dict[str, Any]) -> bool:
    """Deploy a single container via Manager API"""
    name = api_config['name']
    
    try:
        print(f"🚀 Deploying {name}...")
        
        # Make API request
        response = requests.post(
            f"{MANAGER_URL}/containers",
            json=api_config,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'unknown')
            
            if status == 'deployed':
                print(f"✅ Successfully deployed {name}")
                return True
            elif status == 'already_running':
                print(f"⚠️  {name} is already running")
                return True
            else:
                print(f"⚠️  {name} deployment status: {status}")
                return False
        else:
            print(f"❌ Failed to deploy {name}: HTTP {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error deploying {name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error deploying {name}: {e}")
        return False

def check_manager_health() -> bool:
    """Check if MCP Manager is healthy"""
    try:
        response = requests.get(f"{MANAGER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ MCP Manager is healthy")
            return True
    except:
        pass
    
    print("❌ MCP Manager is not healthy")
    return False

def list_deployed_containers() -> List[Dict[str, Any]]:
    """List currently deployed containers"""
    try:
        response = requests.get(f"{MANAGER_URL}/containers", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def main():
    """Main deployment function"""
    print("=" * 60)
    print("MCP Service Deployment to DinD Orchestrator")
    print("=" * 60)
    
    # Check manager health
    if not check_manager_health():
        print("Cannot proceed without healthy Manager")
        return 1
    
    # List existing containers
    existing = list_deployed_containers()
    print(f"\n📦 Currently deployed: {len(existing)} containers")
    if existing:
        for container in existing:
            print(f"  - {container.get('name', 'unknown')}: {container.get('status', 'unknown')}")
    
    # Load all manifests
    manifest_files = sorted(MANIFEST_DIR.glob("*.yml"))
    print(f"\n📋 Found {len(manifest_files)} manifests to deploy")
    
    # Deploy each service
    deployed = 0
    failed = 0
    skipped = 0
    
    for manifest_file in manifest_files:
        service_name = manifest_file.stem
        
        # Skip postgres as it was noted as problematic
        if 'postgres' in service_name:
            print(f"⏭️  Skipping {service_name} (known issues)")
            skipped += 1
            continue
        
        try:
            # Load manifest
            manifest = load_manifest(manifest_file)
            
            # Convert to API format
            api_config = convert_manifest_to_api_format(manifest)
            
            # Deploy container
            if deploy_container(api_config):
                deployed += 1
            else:
                failed += 1
                
            # Small delay between deployments
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error processing {service_name}: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("Deployment Summary")
    print("=" * 60)
    print(f"✅ Deployed: {deployed}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"📊 Total: {deployed + failed + skipped}")
    
    # List final state
    print("\n📦 Final container state:")
    final_containers = list_deployed_containers()
    for container in final_containers:
        print(f"  - {container.get('name', 'unknown')}: {container.get('status', 'unknown')}")
    
    print("\n✨ Deployment complete!")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    exit(main())