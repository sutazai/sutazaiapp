#!/usr/bin/env python3
"""
SutazAI Ollama Model Backup System
Backs up Ollama models, configurations, and model registry
"""

import os
import sys
import json
import logging
import datetime
import shutil
import tarfile
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/ollama-model-backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OllamaModelBackupSystem:
    """Ollama model backup system for SutazAI"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.model_backup_dir = self.backup_root / 'models' / 'ollama'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure backup directory exists
        self.model_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Ollama configuration
        self.ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_data_dir = os.environ.get('OLLAMA_MODELS', '/root/.ollama')
        
        # Model storage paths
        self.model_paths = {
            'ollama_models': f"{self.ollama_data_dir}/models",
            'ollama_config': f"{self.ollama_data_dir}",
            'model_configs': "/opt/sutazaiapp/agents/configs",
            'ollama_scripts': "/opt/sutazaiapp/scripts/models/ollama"
        }
    
    def get_ollama_models_list(self) -> List[Dict]:
        """Get list of installed Ollama models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                return models_data.get('models', [])
            else:
                logger.error(f"Failed to get models list: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting Ollama models list: {e}")
            
            # Fallback: try command line
            try:
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    models = []
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 3:
                                models.append({
                                    'name': parts[0],
                                    'id': parts[1] if len(parts) > 1 else '',
                                    'size': parts[2] if len(parts) > 2 else '',
                                    'modified': ' '.join(parts[3:]) if len(parts) > 3 else ''
                                })
                    
                    return models
                else:
                    logger.error(f"ollama list command failed: {result.stderr}")
                    
            except Exception as cmd_e:
                logger.error(f"Error running ollama list command: {cmd_e}")
            
            return []
    
    def backup_model_data_directory(self) -> Dict:
        """Backup the entire Ollama models directory"""
        try:
            models_dir = Path(self.model_paths['ollama_models'])
            
            if not models_dir.exists():
                logger.warning(f"Ollama models directory not found: {models_dir}")
                return {
                    'type': 'ollama_models_directory',
                    'status': 'skipped',
                    'reason': 'directory_not_found'
                }
            
            # Create backup archive
            backup_file = self.model_backup_dir / f"ollama_models_{self.timestamp}.tar.gz"
            
            logger.info(f"Creating backup of Ollama models directory: {models_dir}")
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(models_dir, arcname='ollama_models')
            
            # Calculate checksum
            checksum = self.calculate_checksum(backup_file)
            
            return {
                'type': 'ollama_models_directory',
                'source': str(models_dir),
                'backup_file': str(backup_file),
                'size': backup_file.stat().st_size,
                'checksum': checksum,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error backing up Ollama models directory: {e}")
            return {
                'type': 'ollama_models_directory',
                'status': 'failed',
                'error': str(e)
            }
    
    def backup_individual_models(self, models_list: List[Dict]) -> List[Dict]:
        """Backup individual models using ollama export"""
        results = []
        
        for model in models_list:
            model_name = model.get('name', '')
            if not model_name:
                continue
            
            try:
                # Create individual model backup
                model_backup_file = self.model_backup_dir / 'individual' / f"{model_name.replace(':', '_')}_{self.timestamp}.gguf"
                model_backup_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Exporting model: {model_name}")
                
                # Try to export model
                result = subprocess.run([
                    'ollama', 'save', model_name, str(model_backup_file)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    checksum = self.calculate_checksum(model_backup_file)
                    
                    results.append({
                        'type': 'individual_model',
                        'model_name': model_name,
                        'model_info': model,
                        'backup_file': str(model_backup_file),
                        'size': model_backup_file.stat().st_size,
                        'checksum': checksum,
                        'status': 'success'
                    })
                    
                    logger.info(f"Successfully exported model: {model_name}")
                else:
                    logger.error(f"Failed to export model {model_name}: {result.stderr}")
                    results.append({
                        'type': 'individual_model',
                        'model_name': model_name,
                        'status': 'failed',
                        'error': result.stderr
                    })
                    
            except Exception as e:
                logger.error(f"Error backing up model {model_name}: {e}")
                results.append({
                    'type': 'individual_model',
                    'model_name': model_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def backup_model_configurations(self) -> List[Dict]:
        """Backup model configuration files"""
        results = []
        
        config_patterns = [
            '/opt/sutazaiapp/agents/configs/*.modelfile',
            '/opt/sutazaiapp/config/ollama*.yaml',
            '/opt/sutazaiapp/config/ollama*.json'
        ]
        
        for pattern in config_patterns:
            matching_files = list(Path('/').glob(pattern.lstrip('/')))
            
            for config_file in matching_files:
                try:
                    dest_file = self.model_backup_dir / 'configs' / self.timestamp / config_file.name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(config_file, dest_file)
                    
                    checksum = self.calculate_checksum(dest_file)
                    
                    results.append({
                        'type': 'model_config',
                        'source': str(config_file),
                        'destination': str(dest_file),
                        'size': dest_file.stat().st_size,
                        'checksum': checksum,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    logger.error(f"Error backing up config file {config_file}: {e}")
                    results.append({
                        'type': 'model_config',
                        'source': str(config_file),
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results
    
    def backup_ollama_system_info(self) -> Dict:
        """Backup Ollama system information and version"""
        try:
            system_info = {
                'timestamp': self.timestamp,
                'ollama_host': self.ollama_host,
                'ollama_data_dir': self.ollama_data_dir
            }
            
            # Get Ollama version
            try:
                result = subprocess.run(['ollama', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    system_info['ollama_version'] = result.stdout.strip()
            except Exception:
                pass
            
            # Get system info from API
            try:
                response = requests.get(f"{self.ollama_host}/api/version", timeout=5)
                if response.status_code == 200:
                    system_info['api_version'] = response.json()
            except Exception:
                pass
            
            # Get running models
            try:
                response = requests.get(f"{self.ollama_host}/api/ps", timeout=5)
                if response.status_code == 200:
                    system_info['running_models'] = response.json()
            except Exception:
                pass
            
            # Save system info
            info_file = self.model_backup_dir / f"ollama_system_info_{self.timestamp}.json"
            with open(info_file, 'w') as f:
                json.dump(system_info, f, indent=2)
            
            checksum = self.calculate_checksum(info_file)
            
            return {
                'type': 'system_info',
                'info_file': str(info_file),
                'size': info_file.stat().st_size,
                'checksum': checksum,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error backing up Ollama system info: {e}")
            return {
                'type': 'system_info',
                'status': 'failed',
                'error': str(e)
            }
    
    def create_model_registry_backup(self, models_list: List[Dict]) -> Dict:
        """Create a comprehensive model registry backup"""
        try:
            registry = {
                'timestamp': self.timestamp,
                'backup_date': datetime.datetime.now().isoformat(),
                'total_models': len(models_list),
                'models': models_list,
                'ollama_host': self.ollama_host,
                'data_directory': self.ollama_data_dir
            }
            
            # Add model details
            for model in models_list:
                model_name = model.get('name', '')
                if model_name:
                    try:
                        # Get model info
                        response = requests.post(
                            f"{self.ollama_host}/api/show",
                            json={'name': model_name},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            model_info = response.json()
                            model['detailed_info'] = model_info
                            
                    except Exception:
                        pass
            
            registry_file = self.model_backup_dir / f"model_registry_{self.timestamp}.json"
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            
            checksum = self.calculate_checksum(registry_file)
            
            return {
                'type': 'model_registry',
                'registry_file': str(registry_file),
                'size': registry_file.stat().st_size,
                'checksum': checksum,
                'models_count': len(models_list),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error creating model registry backup: {e}")
            return {
                'type': 'model_registry',
                'status': 'failed',
                'error': str(e)
            }
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def run_ollama_model_backup(self) -> Dict:
        """Run complete Ollama model backup"""
        start_time = time.time()
        logger.info(f"Starting Ollama model backup - {self.timestamp}")
        
        all_results = []
        
        # Get list of installed models
        models_list = self.get_ollama_models_list()
        logger.info(f"Found {len(models_list)} Ollama models")
        
        # Backup entire models directory
        directory_result = self.backup_model_data_directory()
        all_results.append(directory_result)
        
        # Backup individual models (optional, for easier restore)
        if models_list:
            individual_results = self.backup_individual_models(models_list)
            all_results.extend(individual_results)
        
        # Backup model configurations
        config_results = self.backup_model_configurations()
        all_results.extend(config_results)
        
        # Backup system info
        system_info_result = self.backup_ollama_system_info()
        all_results.append(system_info_result)
        
        # Create model registry
        registry_result = self.create_model_registry_backup(models_list)
        all_results.append(registry_result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate total backup size
        total_size = sum(r.get('size', 0) for r in all_results if r.get('size'))
        
        # Create manifest
        manifest = {
            'timestamp': self.timestamp,
            'backup_date': datetime.datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_models': len(models_list),
            'total_backups': len(all_results),
            'successful_backups': len([r for r in all_results if r.get('status') == 'success']),
            'failed_backups': len([r for r in all_results if r.get('status') == 'failed']),
            'total_backup_size': total_size,
            'models_list': models_list,
            'backup_results': all_results
        }
        
        manifest_file = self.model_backup_dir / f"ollama_backup_manifest_{self.timestamp}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Ollama model backup completed in {duration:.2f} seconds")
        logger.info(f"Total size: {total_size / (1024**3):.2f} GB")
        logger.info(f"Successful: {manifest['successful_backups']}, Failed: {manifest['failed_backups']}")
        
        return manifest

def main():
    """Main entry point"""
    try:
        model_backup = OllamaModelBackupSystem()
        result = model_backup.run_ollama_model_backup()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/ollama_model_backup_summary_{model_backup.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code
        if result['failed_backups'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Ollama model backup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()