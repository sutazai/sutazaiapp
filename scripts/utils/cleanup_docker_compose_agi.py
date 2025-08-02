#!/usr/bin/env python3
"""
Comprehensive cleanup script to remove ALL AGI/ASI/fantasy references from docker-compose files
"""

import os
import re
import yaml
from pathlib import Path
import shutil

# AGI/ASI service mappings
SERVICE_REPLACEMENTS = {
    'backend-agi': 'backend',
    'localagi': None,  # Remove entirely
    'bigagi': None,    # Remove entirely
    'jarvis-agi': 'jarvis',
    'agi-coordinator': 'task-coordinator',
    'neural-engine': 'processing-engine',
    'brain-service': 'coordinator-service'
}

# Container name replacements
CONTAINER_REPLACEMENTS = {
    'sutazai-backend-agi': 'sutazai-backend',
    'sutazai-localagi': None,  # Remove
    'sutazai-bigagi': None,    # Remove
    'sutazai-jarvis-agi': 'sutazai-jarvis',
    'agi-coordinator': 'task-coordinator',
    'neural-container': 'processing-container',
    'brain-container': 'coordinator-container'
}

# Environment variable replacements
ENV_REPLACEMENTS = {
    'AGI_MODE': 'AUTOMATION_MODE',
    'ASI_ENABLED': 'ADVANCED_AUTOMATION_ENABLED',
    'NEURAL_NETWORK': 'PROCESSING_NETWORK',
    'BRAIN_ENDPOINT': 'COORDINATOR_ENDPOINT',
    'state management_LEVEL': 'SYSTEM_STATE_LEVEL',
    'QUANTUM_ENABLED': 'OPTIMIZATION_ENABLED'
}

def clean_docker_compose_file(filepath: Path) -> bool:
    """Clean a single docker-compose file"""
    print(f"\n🔧 Processing: {filepath}")
    
    try:
        # Backup original
        backup_path = filepath.with_suffix('.yml.agi_backup')
        shutil.copy2(filepath, backup_path)
        print(f"  📋 Created backup: {backup_path}")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or 'services' not in data:
            print(f"  ⚠️  No services found in {filepath}")
            return False
        
        changes_made = False
        services_to_remove = []
        services_to_rename = {}
        
        # Process services
        for service_name, service_config in data['services'].items():
            # Check if service should be removed or renamed
            if service_name in SERVICE_REPLACEMENTS:
                new_name = SERVICE_REPLACEMENTS[service_name]
                if new_name is None:
                    services_to_remove.append(service_name)
                    print(f"  🗑️  Removing service: {service_name}")
                    changes_made = True
                else:
                    services_to_rename[service_name] = new_name
                    print(f"  ✏️  Renaming service: {service_name} → {new_name}")
                    changes_made = True
            
            # Update container names
            if 'container_name' in service_config:
                old_container = service_config['container_name']
                for old, new in CONTAINER_REPLACEMENTS.items():
                    if old in old_container:
                        if new is None:
                            # This service will be removed
                            break
                        new_container = old_container.replace(old, new)
                        service_config['container_name'] = new_container
                        print(f"  📦 Container: {old_container} → {new_container}")
                        changes_made = True
                        break
            
            # Update image names
            if 'image' in service_config:
                old_image = service_config['image']
                if 'agi' in old_image or 'ASI' in old_image:
                    new_image = old_image.replace('-agi', '').replace('_agi', '')
                    new_image = new_image.replace('ASI', 'automation')
                    service_config['image'] = new_image
                    print(f"  🖼️  Image: {old_image} → {new_image}")
                    changes_made = True
            
            # Update build context
            if 'build' in service_config:
                if isinstance(service_config['build'], dict) and 'context' in service_config['build']:
                    old_context = service_config['build']['context']
                    if 'agi' in old_context:
                        new_context = old_context.replace('-agi', '').replace('_agi', '')
                        service_config['build']['context'] = new_context
                        print(f"  🏗️  Build context: {old_context} → {new_context}")
                        changes_made = True
            
            # Update environment variables
            if 'environment' in service_config:
                for env_key, env_value in list(service_config['environment'].items()):
                    # Replace environment variable names
                    if env_key in ENV_REPLACEMENTS:
                        new_key = ENV_REPLACEMENTS[env_key]
                        service_config['environment'][new_key] = service_config['environment'].pop(env_key)
                        print(f"  🔧 Env var: {env_key} → {new_key}")
                        changes_made = True
                    
                    # Replace values containing AGI/ASI references
                    if isinstance(env_value, str):
                        new_value = env_value
                        for old, new in SERVICE_REPLACEMENTS.items():
                            if old in env_value and new is not None:
                                new_value = new_value.replace(old, new)
                        
                        if 'agi' in new_value:
                            new_value = new_value.replace('-agi', '').replace('_agi', '')
                        
                        if new_value != env_value:
                            service_config['environment'][env_key] = new_value
                            print(f"  📝 Env value: {env_key}={env_value} → {new_value}")
                            changes_made = True
            
            # Update depends_on
            if 'depends_on' in service_config:
                new_depends = []
                for dep in service_config['depends_on']:
                    if dep in SERVICE_REPLACEMENTS:
                        new_dep = SERVICE_REPLACEMENTS[dep]
                        if new_dep is not None:
                            new_depends.append(new_dep)
                            print(f"  🔗 Dependency: {dep} → {new_dep}")
                            changes_made = True
                    else:
                        new_depends.append(dep)
                service_config['depends_on'] = new_depends
            
            # Update volumes
            if 'volumes' in service_config:
                new_volumes = []
                for volume in service_config['volumes']:
                    if isinstance(volume, str) and 'agi' in volume:
                        new_volume = volume.replace('-agi', '').replace('_agi', '')
                        new_volumes.append(new_volume)
                        print(f"  💾 Volume: {volume} → {new_volume}")
                        changes_made = True
                    else:
                        new_volumes.append(volume)
                service_config['volumes'] = new_volumes
            
            # Update networks
            if 'networks' in service_config:
                new_networks = []
                for network in service_config['networks']:
                    if isinstance(network, str) and 'agi' in network:
                        new_network = network.replace('-agi', '').replace('_agi', '')
                        new_networks.append(new_network)
                        print(f"  🌐 Network: {network} → {new_network}")
                        changes_made = True
                    else:
                        new_networks.append(network)
                service_config['networks'] = new_networks
        
        # Remove services marked for deletion
        for service in services_to_remove:
            del data['services'][service]
        
        # Rename services
        for old_name, new_name in services_to_rename.items():
            data['services'][new_name] = data['services'].pop(old_name)
        
        # Update volume definitions
        if 'volumes' in data:
            new_volumes = {}
            for vol_name, vol_config in data['volumes'].items():
                if 'agi' in vol_name:
                    new_vol_name = vol_name.replace('-agi', '').replace('_agi', '')
                    new_volumes[new_vol_name] = vol_config
                    print(f"  💿 Volume definition: {vol_name} → {new_vol_name}")
                    changes_made = True
                else:
                    new_volumes[vol_name] = vol_config
            data['volumes'] = new_volumes
        
        # Update network definitions
        if 'networks' in data:
            new_networks = {}
            for net_name, net_config in data['networks'].items():
                if 'agi' in net_name:
                    new_net_name = net_name.replace('-agi', '').replace('_agi', '')
                    new_networks[new_net_name] = net_config
                    print(f"  🌐 Network definition: {net_name} → {new_net_name}")
                    changes_made = True
                else:
                    new_networks[net_name] = net_config
            data['networks'] = new_networks
        
        if changes_made:
            # Write cleaned file
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"  ✅ Successfully cleaned {filepath}")
            return True
        else:
            print(f"  ℹ️  No changes needed for {filepath}")
            # Remove backup if no changes
            backup_path.unlink()
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")
        return False

def clean_dockerfile_references(root_path: Path):
    """Clean AGI references from Dockerfiles"""
    print("\n🐳 Cleaning Dockerfiles...")
    
    dockerfiles = list(root_path.glob('**/Dockerfile*'))
    cleaned_count = 0
    
    for dockerfile in dockerfiles:
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            original = content
            
            # Replace backend-agi references
            content = re.sub(r'backend-agi', 'backend', content)
            content = re.sub(r'BACKEND_AGI', 'BACKEND', content)
            content = re.sub(r'backend_agi', 'backend', content)
            
            # Replace AGI/ASI references in comments
            content = re.sub(r'AGI system', 'automation system', content)
            content = re.sub(r'ASI capabilities', 'advanced automation capabilities', content)
            
            if content != original:
                # Backup
                backup_path = dockerfile.with_suffix('.agi_backup')
                shutil.copy2(dockerfile, backup_path)
                
                # Write cleaned content
                with open(dockerfile, 'w') as f:
                    f.write(content)
                
                print(f"  ✅ Cleaned: {dockerfile}")
                cleaned_count += 1
                
        except Exception as e:
            print(f"  ❌ Error cleaning {dockerfile}: {e}")
    
    print(f"  📊 Cleaned {cleaned_count} Dockerfiles")

def main():
    """Main cleanup function"""
    print("🚀 Docker Compose AGI/ASI Cleanup Script")
    print("=" * 60)
    
    root_path = Path('/opt/sutazaiapp')
    
    # Find all docker-compose files
    compose_files = list(root_path.glob('docker-compose*.yml'))
    compose_files.extend(root_path.glob('docker-compose*.yaml'))
    compose_files.extend(root_path.glob('**/docker-compose*.yml'))
    compose_files.extend(root_path.glob('**/docker-compose*.yaml'))
    
    # Remove duplicates
    compose_files = list(set(compose_files))
    
    print(f"\n📋 Found {len(compose_files)} docker-compose files to process")
    
    cleaned_count = 0
    for compose_file in compose_files:
        if clean_docker_compose_file(compose_file):
            cleaned_count += 1
    
    # Also clean Dockerfiles
    clean_dockerfile_references(root_path)
    
    print("\n" + "=" * 60)
    print(f"✅ Cleanup Complete!")
    print(f"📊 Cleaned {cleaned_count} docker-compose files")
    print("\n🔍 Next steps:")
    print("1. Review changes: git diff")
    print("2. Test services: docker-compose config")
    print("3. Restart services: docker-compose down && docker-compose up -d")
    
    # Final verification
    print("\n🔍 Verification scan...")
    remaining = []
    for compose_file in compose_files:
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
            if 'agi' in content.lower() or 'asi' in content.lower():
                if 'backup' not in str(compose_file):
                    remaining.append(compose_file)
        except:
            pass
    
    if remaining:
        print(f"⚠️  Found {len(remaining)} files with possible remaining AGI/ASI references:")
        for f in remaining:
            print(f"  - {f}")
    else:
        print("✅ All docker-compose files are clean!")

if __name__ == "__main__":
    main()