#!/usr/bin/env python3
"""
SutazAI Agent State Backup System
Backs up all agent states, configurations, and runtime data
"""

import os
import sys
import json
import logging
import datetime
import shutil
import tarfile
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/agent-state-backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentStateBackupSystem:
    """Agent state backup system for SutazAI"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.agent_backup_dir = self.backup_root / 'agents'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure backup directory exists
        self.agent_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Could not initialize Docker client: {e}")
            self.docker_client = None
        
        # Agent-related paths to backup
        self.agent_paths = {
            'agent_configs': '/opt/sutazaiapp/agents/configs/',
            'agent_registry': '/opt/sutazaiapp/agents/agent_registry.json',
            'agent_status': '/opt/sutazaiapp/agents/agent_status.json',
            'collective_intelligence': '/opt/sutazaiapp/data/collective_intelligence/',
            'agent_data': '/opt/sutazaiapp/data/',
            'workflow_reports': '/opt/sutazaiapp/data/workflow_reports/'
        }
        
        # Agent service ports for state extraction
        self.agent_service_ports = self.discover_agent_services()
    
    def discover_agent_services(self) -> Dict[str, int]:
        """Discover running agent services and their ports"""
        services = {}
        
        if not self.docker_client:
            return services
        
        try:
            # Get running containers with agent-related names
            containers = self.docker_client.containers.list()
            
            for container in containers:
                container_name = container.name
                if any(keyword in container_name.lower() for keyword in 
                      ['agent', 'ai-', 'sutazai']):
                    
                    # Get port mappings
                    port_mappings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                    
                    for container_port, host_ports in port_mappings.items():
                        if host_ports:
                            host_port = host_ports[0]['HostPort']
                            services[container_name] = int(host_port)
                            break
            
            logger.info(f"Discovered {len(services)} agent services")
            
        except Exception as e:
            logger.error(f"Error discovering agent services: {e}")
        
        return services
    
    def backup_agent_state_files(self) -> List[Dict]:
        """Backup agent state files and configurations"""
        results = []
        
        for category, path in self.agent_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Agent path not found: {path}")
                continue
            
            try:
                # Create category backup directory
                category_dir = self.agent_backup_dir / category / self.timestamp
                category_dir.mkdir(parents=True, exist_ok=True)
                
                source_path = Path(path)
                
                if source_path.is_file():
                    # Backup single file
                    dest_file = category_dir / source_path.name
                    shutil.copy2(source_path, dest_file)
                    
                    checksum = self.calculate_checksum(dest_file)
                    
                    results.append({
                        'category': category,
                        'source': path,
                        'destination': str(dest_file),
                        'type': 'file',
                        'size': dest_file.stat().st_size,
                        'checksum': checksum,
                        'status': 'success'
                    })
                    
                elif source_path.is_dir():
                    # Backup directory
                    dest_dir = category_dir / source_path.name
                    shutil.copytree(source_path, dest_dir, dirs_exist_ok=True)
                    
                    # Calculate directory stats
                    total_size = 0
                    file_count = 0
                    checksums = []
                    
                    for file_path in dest_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
                            checksums.append(self.calculate_checksum(file_path))
                    
                    combined_checksum = hashlib.sha256(''.join(sorted(checksums)).encode()).hexdigest()
                    
                    results.append({
                        'category': category,
                        'source': path,
                        'destination': str(dest_dir),
                        'type': 'directory',
                        'size': total_size,
                        'file_count': file_count,
                        'checksum': combined_checksum,
                        'status': 'success'
                    })
                
                logger.info(f"Successfully backed up {category}: {path}")
                
            except Exception as e:
                logger.error(f"Error backing up {category} from {path}: {e}")
                results.append({
                    'category': category,
                    'source': path,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def backup_agent_runtime_states(self) -> List[Dict]:
        """Backup runtime states from running agents"""
        results = []
        
        for service_name, port in self.agent_service_ports.items():
            try:
                # Try to get agent health/status
                health_data = self.get_agent_health(port)
                
                if health_data:
                    # Save agent runtime state
                    state_file = self.agent_backup_dir / 'runtime_states' / self.timestamp / f"{service_name}_state.json"
                    state_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(state_file, 'w') as f:
                        json.dump(health_data, f, indent=2)
                    
                    checksum = self.calculate_checksum(state_file)
                    
                    results.append({
                        'service': service_name,
                        'port': port,
                        'state_file': str(state_file),
                        'size': state_file.stat().st_size,
                        'checksum': checksum,
                        'status': 'success'
                    })
                    
                    logger.info(f"Backed up runtime state for {service_name}")
                
            except Exception as e:
                logger.error(f"Error backing up runtime state for {service_name}: {e}")
                results.append({
                    'service': service_name,
                    'port': port,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def get_agent_health(self, port: int, timeout: int = 5) -> Optional[Dict]:
        """Get agent health/status from HTTP endpoint"""
        health_endpoints = ['/health', '/status', '/api/health', '/api/status']
        
        for endpoint in health_endpoints:
            try:
                url = f"http://localhost:{port}{endpoint}"
                response = requests.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    return {
                        'timestamp': self.timestamp,
                        'endpoint': endpoint,
                        'status_code': response.status_code,
                        'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                        'headers': dict(response.headers)
                    }
                    
            except Exception as e:
                continue
        
        return None
    
    def backup_agent_logs(self) -> List[Dict]:
        """Backup agent-specific log files"""
        results = []
        
        log_patterns = [
            '/opt/sutazaiapp/logs/agent-*.log',
            '/opt/sutazaiapp/logs/*agent*.log',
            '/opt/sutazaiapp/logs/ai-*.log'
        ]
        
        for pattern in log_patterns:
            matching_files = list(Path('/').glob(pattern.lstrip('/')))
            
            for log_file in matching_files:
                try:
                    # Copy log file to backup
                    dest_file = self.agent_backup_dir / 'logs' / self.timestamp / log_file.name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(log_file, dest_file)
                    
                    checksum = self.calculate_checksum(dest_file)
                    
                    results.append({
                        'source': str(log_file),
                        'destination': str(dest_file),
                        'size': dest_file.stat().st_size,
                        'checksum': checksum,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    logger.error(f"Error backing up log file {log_file}: {e}")
                    results.append({
                        'source': str(log_file),
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results
    
    def backup_docker_agent_volumes(self) -> List[Dict]:
        """Backup Docker volumes used by agent containers"""
        results = []
        
        if not self.docker_client:
            return results
        
        try:
            volumes = self.docker_client.volumes.list()
            
            for volume in volumes:
                volume_name = volume.name
                
                # Only backup agent-related volumes
                if any(keyword in volume_name.lower() for keyword in 
                      ['agent', 'ai-', 'sutazai', 'ollama']):
                    
                    try:
                        # Create temporary container to backup volume
                        volume_backup_file = self.agent_backup_dir / 'volumes' / self.timestamp / f"{volume_name}.tar.gz"
                        volume_backup_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Use tar to backup the volume
                        container = self.docker_client.containers.run(
                            'alpine:latest',
                            f'tar czf /backup.tar.gz -C /data .',
                            volumes={volume_name: {'bind': '/data', 'mode': 'ro'}},
                            detach=True,
                            remove=True
                        )
                        
                        # Wait for container to complete
                        container.wait()
                        
                        # Copy the backup file from container
                        bits, _ = container.get_archive('/backup.tar.gz')
                        with open(volume_backup_file, 'wb') as f:
                            for chunk in bits:
                                f.write(chunk)
                        
                        checksum = self.calculate_checksum(volume_backup_file)
                        
                        results.append({
                            'volume': volume_name,
                            'backup_file': str(volume_backup_file),
                            'size': volume_backup_file.stat().st_size,
                            'checksum': checksum,
                            'status': 'success'
                        })
                        
                        logger.info(f"Backed up Docker volume: {volume_name}")
                        
                    except Exception as e:
                        logger.error(f"Error backing up volume {volume_name}: {e}")
                        results.append({
                            'volume': volume_name,
                            'status': 'failed',
                            'error': str(e)
                        })
        
        except Exception as e:
            logger.error(f"Error listing Docker volumes: {e}")
        
        return results
    
    def create_agent_state_snapshot(self) -> Dict:
        """Create comprehensive agent state snapshot"""
        try:
            snapshot = {
                'timestamp': self.timestamp,
                'backup_date': datetime.datetime.now().isoformat(),
                'discovered_services': self.agent_service_ports,
                'system_info': {
                    'hostname': os.uname().nodename,
                    'python_version': sys.version,
                    'docker_available': self.docker_client is not None
                }
            }
            
            # Add Docker container info
            if self.docker_client:
                containers_info = []
                for container in self.docker_client.containers.list():
                    if any(keyword in container.name.lower() for keyword in 
                          ['agent', 'ai-', 'sutazai']):
                        containers_info.append({
                            'name': container.name,
                            'image': container.image.tags[0] if container.image.tags else 'unknown',
                            'status': container.status,
                            'created': container.attrs['Created'],
                            'ports': container.attrs.get('NetworkSettings', {}).get('Ports', {})
                        })
                
                snapshot['docker_containers'] = containers_info
            
            # Save snapshot
            snapshot_file = self.agent_backup_dir / f"agent_snapshot_{self.timestamp}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
            
            checksum = self.calculate_checksum(snapshot_file)
            
            return {
                'snapshot_file': str(snapshot_file),
                'size': snapshot_file.stat().st_size,
                'checksum': checksum,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error creating agent state snapshot: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    # calculate_checksum centralized in scripts.lib.file_utils
    
    def run_agent_state_backup(self) -> Dict:
        """Run complete agent state backup"""
        start_time = time.time()
        logger.info(f"Starting agent state backup - {self.timestamp}")
        
        all_results = []
        
        # Backup agent state files
        file_results = self.backup_agent_state_files()
        all_results.extend(file_results)
        
        # Backup runtime states
        runtime_results = self.backup_agent_runtime_states()
        all_results.extend(runtime_results)
        
        # Backup agent logs
        log_results = self.backup_agent_logs()
        all_results.extend(log_results)
        
        # Backup Docker volumes
        volume_results = self.backup_docker_agent_volumes()
        all_results.extend(volume_results)
        
        # Create state snapshot
        snapshot_result = self.create_agent_state_snapshot()
        all_results.append(snapshot_result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Create manifest
        manifest = {
            'timestamp': self.timestamp,
            'backup_date': datetime.datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_backups': len(all_results),
            'successful_backups': len([r for r in all_results if r.get('status') == 'success']),
            'failed_backups': len([r for r in all_results if r.get('status') == 'failed']),
            'discovered_services': len(self.agent_service_ports),
            'backup_results': all_results
        }
        
        manifest_file = self.agent_backup_dir / f"agent_backup_manifest_{self.timestamp}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Agent state backup completed in {duration:.2f} seconds")
        logger.info(f"Successful: {manifest['successful_backups']}, Failed: {manifest['failed_backups']}")
        
        return manifest

def main():
    """Main entry point"""
    try:
        agent_backup = AgentStateBackupSystem()
        result = agent_backup.run_agent_state_backup()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/agent_state_backup_summary_{agent_backup.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code
        if result['failed_backups'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Agent state backup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
