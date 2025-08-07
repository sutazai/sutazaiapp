#!/usr/bin/env python3
"""
SutazAI Master Backup Orchestrator
Central orchestration system for all backup operations implementing 3-2-1 strategy
"""

import os
import sys
import json
import logging
import datetime
import time
import subprocess
import argparse
import threading
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/backup-orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SutazAIBackupOrchestrator:
    """Master backup orchestration system"""
    
    def __init__(self):
        self.backup_root = Path('/opt/sutazaiapp/data/backups')
        self.scripts_root = Path('/opt/sutazaiapp/scripts/backup-automation')
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directories exist
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Load orchestration configuration
        self.config = self.load_orchestration_config()
        
        # Backup systems and their scripts
        self.backup_systems = {
            'database': {
                'script': self.scripts_root / 'core' / 'database-backup-system.py',
                'description': 'Database backup (PostgreSQL, SQLite, Loki)',
                'priority': 1,
                'timeout_minutes': 60
            },
            'configuration': {
                'script': self.scripts_root / 'config' / 'config-backup-system.py',
                'description': 'Configuration files backup',
                'priority': 2,
                'timeout_minutes': 30
            },
            'agent_state': {
                'script': self.scripts_root / 'agents' / 'agent-state-backup-system.py',
                'description': 'Agent state and runtime data backup',
                'priority': 3,
                'timeout_minutes': 45
            },
            'models': {
                'script': self.scripts_root / 'models' / 'ollama-model-backup-system.py',
                'description': 'AI model backup (Ollama)',
                'priority': 4,
                'timeout_minutes': 120
            },
            'monitoring': {
                'script': self.scripts_root / 'monitoring' / 'monitoring-data-retention-system.py',
                'description': 'Monitoring data retention',
                'priority': 5,
                'timeout_minutes': 30
            },
            'logs': {
                'script': self.scripts_root / 'logs' / 'log-archival-system.py',
                'description': 'Log archival and compression',
                'priority': 6,
                'timeout_minutes': 30
            }
        }
        
        # Post-backup operations
        self.post_backup_operations = {
            'verification': {
                'script': self.scripts_root / 'verification' / 'backup-verification-system.py',
                'description': 'Backup integrity verification',
                'timeout_minutes': 45
            },
            'offsite_replication': {
                'script': self.scripts_root / 'offsite' / 'offsite-backup-replication-system.py',
                'description': 'Offsite backup replication (3-2-1 strategy)',
                'timeout_minutes': 180
            },
            'restore_testing': {
                'script': self.scripts_root / 'restore' / 'restore-testing-system.py',
                'description': 'Restore testing validation',
                'timeout_minutes': 60
            },
            'monitoring': {
                'script': self.scripts_root / 'alerts' / 'backup-monitoring-alerting-system.py',
                'description': 'Backup monitoring and alerting',
                'timeout_minutes': 15
            }
        }
        
        # Execution state
        self.execution_state = {
            'start_time': None,
            'end_time': None,
            'current_operation': None,
            'completed_operations': [],
            'failed_operations': [],
            'total_operations': 0,
            'interruption_requested': False
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def load_orchestration_config(self) -> Dict:
        """Load orchestration configuration"""
        config_file = Path('/opt/sutazaiapp/config/backup-orchestration-config.json')
        
        default_config = {
            'orchestration': {
                'enabled': True,
                'parallel_execution': True,
                'max_concurrent_operations': 3,
                'stop_on_critical_failure': True,
                'retry_failed_operations': True,
                'max_retries': 2
            },
            'schedule': {
                'daily_backup_time': '02:00',
                'weekly_backup_day': 'Sunday',
                'weekly_backup_time': '03:00',
                'monthly_backup_day': 1,
                'monthly_backup_time': '04:00'
            },
            'operations': {
                'database': {'enabled': True, 'daily': True, 'weekly': True, 'monthly': True},
                'configuration': {'enabled': True, 'daily': True, 'weekly': True, 'monthly': True},
                'agent_state': {'enabled': True, 'daily': True, 'weekly': False, 'monthly': False},
                'models': {'enabled': True, 'daily': False, 'weekly': True, 'monthly': True},
                'monitoring': {'enabled': True, 'daily': True, 'weekly': False, 'monthly': False},
                'logs': {'enabled': True, 'daily': True, 'weekly': False, 'monthly': False}
            },
            'post_operations': {
                'verification': {'enabled': True, 'run_after': ['database', 'configuration']},
                'offsite_replication': {'enabled': True, 'run_after': ['verification']},
                'restore_testing': {'enabled': True, 'frequency': 'weekly'},
                'monitoring': {'enabled': True, 'run_always': True}
            },
            '3_2_1_validation': {
                'enabled': True,
                'local_copies_required': 2,
                'offsite_copies_required': 1,
                'verify_different_media': True
            }
        }
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                return {**default_config, **loaded_config}
            else:
                # Create default config
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Error loading orchestration config: {e}")
            return default_config
    
    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals for graceful shutdown"""
        logger.warning(f"Received signal {signum}, requesting graceful shutdown...")
        self.execution_state['interruption_requested'] = True
    
    def execute_backup_operation(self, operation_name: str, operation_config: Dict) -> Dict:
        """Execute a single backup operation"""
        start_time = time.time()
        
        try:
            script_path = operation_config['script']
            timeout_seconds = operation_config.get('timeout_minutes', 30) * 60
            
            logger.info(f"Starting {operation_name}: {operation_config['description']}")
            
            # Check if script exists
            if not script_path.exists():
                return {
                    'operation': operation_name,
                    'status': 'failed',
                    'error': f'Script not found: {script_path}',
                    'duration': 0
                }
            
            # Execute the backup script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(script_path.parent)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"Completed {operation_name} successfully in {duration:.2f} seconds")
                return {
                    'operation': operation_name,
                    'status': 'success',
                    'duration': duration,
                    'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
            else:
                logger.error(f"Failed {operation_name} after {duration:.2f} seconds: {result.stderr}")
                return {
                    'operation': operation_name,
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr,
                    'stdout': result.stdout[-1000:] if result.stdout else ''
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout {operation_name} after {timeout_seconds} seconds")
            return {
                'operation': operation_name,
                'status': 'timeout',
                'duration': timeout_seconds,
                'error': f'Operation timed out after {timeout_seconds} seconds'
            }
        except Exception as e:
            logger.error(f"Error executing {operation_name}: {e}")
            return {
                'operation': operation_name,
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def run_parallel_operations(self, operations: Dict[str, Dict]) -> List[Dict]:
        """Run backup operations in parallel"""
        results = []
        max_workers = self.config['orchestration'].get('max_concurrent_operations', 3)
        
        # Sort operations by priority
        sorted_operations = sorted(
            operations.items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all operations
            future_to_operation = {
                executor.submit(self.execute_backup_operation, name, config): name
                for name, config in sorted_operations
            }
            
            # Process completed operations
            for future in as_completed(future_to_operation):
                if self.execution_state['interruption_requested']:
                    logger.warning("Interruption requested, cancelling remaining operations")
                    break
                
                operation_name = future_to_operation[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    self.execution_state['current_operation'] = operation_name
                    
                    if result['status'] == 'success':
                        self.execution_state['completed_operations'].append(operation_name)
                    else:
                        self.execution_state['failed_operations'].append(operation_name)
                        
                        # Check if we should stop on critical failure
                        if (self.config['orchestration'].get('stop_on_critical_failure', True) and
                            operation_name in ['database', 'configuration']):
                            logger.critical(f"Critical operation {operation_name} failed, stopping execution")
                            break
                            
                except Exception as e:
                    logger.error(f"Error getting result for {operation_name}: {e}")
                    results.append({
                        'operation': operation_name,
                        'status': 'error',
                        'error': str(e),
                        'duration': 0
                    })
        
        return results
    
    def run_sequential_operations(self, operations: Dict[str, Dict]) -> List[Dict]:
        """Run backup operations sequentially"""
        results = []
        
        # Sort operations by priority
        sorted_operations = sorted(
            operations.items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        for operation_name, operation_config in sorted_operations:
            if self.execution_state['interruption_requested']:
                logger.warning("Interruption requested, stopping sequential execution")
                break
            
            self.execution_state['current_operation'] = operation_name
            
            result = self.execute_backup_operation(operation_name, operation_config)
            results.append(result)
            
            if result['status'] == 'success':
                self.execution_state['completed_operations'].append(operation_name)
            else:
                self.execution_state['failed_operations'].append(operation_name)
                
                # Check if we should stop on critical failure
                if (self.config['orchestration'].get('stop_on_critical_failure', True) and
                    operation_name in ['database', 'configuration']):
                    logger.critical(f"Critical operation {operation_name} failed, stopping execution")
                    break
        
        return results
    
    def run_post_backup_operations(self, backup_results: List[Dict]) -> List[Dict]:
        """Run post-backup operations (verification, offsite replication, etc.)"""
        post_results = []
        
        # Determine which post-operations to run
        completed_backups = [r['operation'] for r in backup_results if r['status'] == 'success']
        
        for post_op_name, post_op_config in self.post_backup_operations.items():
            post_config = self.config['post_operations'].get(post_op_name, {})
            
            if not post_config.get('enabled', True):
                continue
            
            # Check if this post-operation should run
            should_run = False
            
            if post_config.get('run_always', False):
                should_run = True
            elif 'run_after' in post_config:
                # Check if any of the required operations completed successfully
                required_ops = post_config['run_after']
                if any(op in completed_backups for op in required_ops):
                    should_run = True
            else:
                # Default: run if any backup completed
                should_run = len(completed_backups) > 0
            
            if should_run:
                logger.info(f"Running post-backup operation: {post_op_name}")
                result = self.execute_backup_operation(post_op_name, post_op_config)
                post_results.append(result)
        
        return post_results
    
    def validate_3_2_1_strategy(self) -> Dict:
        """Validate that 3-2-1 backup strategy is being followed"""
        validation_config = self.config.get('3_2_1_validation', {})
        
        if not validation_config.get('enabled', True):
            return {'status': 'disabled', 'message': '3-2-1 validation disabled'}
        
        try:
            local_copies_required = validation_config.get('local_copies_required', 2)
            offsite_copies_required = validation_config.get('offsite_copies_required', 1)
            
            validation_results = {
                'local_copies': 0,
                'offsite_copies': 0,
                'different_media': [],
                'validation_status': 'unknown'
            }
            
            # Count local copies
            local_backup_dirs = ['daily', 'weekly', 'monthly']
            for backup_dir in local_backup_dirs:
                backup_path = self.backup_root / backup_dir
                if backup_path.exists() and any(backup_path.iterdir()):
                    validation_results['local_copies'] += 1
                    validation_results['different_media'].append('local_disk')
            
            # Check offsite copies
            offsite_path = self.backup_root / 'offsite'
            if offsite_path.exists() and any(offsite_path.iterdir()):
                validation_results['offsite_copies'] += 1
                validation_results['different_media'].append('offsite')
            
            # Validate requirements
            local_ok = validation_results['local_copies'] >= local_copies_required
            offsite_ok = validation_results['offsite_copies'] >= offsite_copies_required
            media_ok = len(set(validation_results['different_media'])) >= 2
            
            if local_ok and offsite_ok and media_ok:
                validation_results['validation_status'] = 'compliant'
            else:
                validation_results['validation_status'] = 'non_compliant'
                validation_results['issues'] = []
                
                if not local_ok:
                    validation_results['issues'].append(f"Insufficient local copies: {validation_results['local_copies']}/{local_copies_required}")
                if not offsite_ok:
                    validation_results['issues'].append(f"Insufficient offsite copies: {validation_results['offsite_copies']}/{offsite_copies_required}")
                if not media_ok:
                    validation_results['issues'].append("Not using different media types")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating 3-2-1 strategy: {e}")
            return {
                'validation_status': 'error',
                'error': str(e)
            }
    
    def run_full_backup_cycle(self, backup_type: str = 'daily') -> Dict:
        """Run a complete backup cycle"""
        self.execution_state['start_time'] = time.time()
        logger.info(f"Starting {backup_type} backup cycle - {self.timestamp}")
        
        # Filter operations based on backup type and configuration
        enabled_operations = {}
        for op_name, op_config in self.backup_systems.items():
            op_settings = self.config['operations'].get(op_name, {})
            
            if (op_settings.get('enabled', True) and 
                op_settings.get(backup_type, False)):
                enabled_operations[op_name] = op_config
        
        self.execution_state['total_operations'] = len(enabled_operations)
        
        if not enabled_operations:
            logger.warning(f"No operations enabled for {backup_type} backup")
            return {
                'timestamp': self.timestamp,
                'backup_type': backup_type,
                'status': 'no_operations',
                'message': f'No backup operations configured for {backup_type}'
            }
        
        logger.info(f"Running {len(enabled_operations)} backup operations: {list(enabled_operations.keys())}")
        
        # Execute backup operations
        if self.config['orchestration'].get('parallel_execution', True):
            backup_results = self.run_parallel_operations(enabled_operations)
        else:
            backup_results = self.run_sequential_operations(enabled_operations)
        
        # Run post-backup operations
        post_backup_results = self.run_post_backup_operations(backup_results)
        
        # Validate 3-2-1 strategy
        strategy_validation = self.validate_3_2_1_strategy()
        
        self.execution_state['end_time'] = time.time()
        total_duration = self.execution_state['end_time'] - self.execution_state['start_time']
        
        # Calculate overall status
        successful_backups = len([r for r in backup_results if r['status'] == 'success'])
        failed_backups = len([r for r in backup_results if r['status'] in ['failed', 'error', 'timeout']])
        
        if self.execution_state['interruption_requested']:
            overall_status = 'interrupted'
        elif failed_backups == 0:
            overall_status = 'success'
        elif successful_backups > 0:
            overall_status = 'partial_success'
        else:
            overall_status = 'failed'
        
        # Create comprehensive report
        execution_report = {
            'timestamp': self.timestamp,
            'backup_cycle_date': datetime.datetime.now().isoformat(),
            'backup_type': backup_type,
            'overall_status': overall_status,
            'duration_minutes': total_duration / 60,
            'execution_summary': {
                'total_operations': len(enabled_operations),
                'successful_operations': successful_backups,
                'failed_operations': failed_backups,
                'interrupted': self.execution_state['interruption_requested']
            },
            'backup_results': backup_results,
            'post_backup_results': post_backup_results,
            'strategy_validation': strategy_validation,
            'execution_state': self.execution_state,
            'configuration': {
                'parallel_execution': self.config['orchestration']['parallel_execution'],
                'max_concurrent_operations': self.config['orchestration']['max_concurrent_operations'],
                'stop_on_critical_failure': self.config['orchestration']['stop_on_critical_failure']
            }
        }
        
        # Save execution report
        report_file = self.backup_root / f"backup_cycle_report_{backup_type}_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(execution_report, f, indent=2)
        
        # Log summary
        logger.info(f"{backup_type.title()} backup cycle completed in {total_duration/60:.2f} minutes")
        logger.info(f"Status: {overall_status}, Success: {successful_backups}/{len(enabled_operations)}")
        
        if strategy_validation['validation_status'] == 'compliant':
            logger.info("3-2-1 backup strategy: COMPLIANT")
        elif strategy_validation['validation_status'] == 'non_compliant':
            logger.warning(f"3-2-1 backup strategy: NON-COMPLIANT - {strategy_validation.get('issues', [])}")
        
        return execution_report
    
    def setup_scheduled_backups(self):
        """Set up scheduled backup execution"""
        schedule_config = self.config.get('schedule', {})
        
        # Daily backups
        daily_time = schedule_config.get('daily_backup_time', '02:00')
        schedule.every().day.at(daily_time).do(self.run_full_backup_cycle, 'daily')
        logger.info(f"Scheduled daily backups at {daily_time}")
        
        # Weekly backups
        weekly_day = schedule_config.get('weekly_backup_day', 'Sunday')
        weekly_time = schedule_config.get('weekly_backup_time', '03:00')
        getattr(schedule.every(), weekly_day.lower()).at(weekly_time).do(self.run_full_backup_cycle, 'weekly')
        logger.info(f"Scheduled weekly backups on {weekly_day} at {weekly_time}")
        
        # Monthly backups
        monthly_day = schedule_config.get('monthly_backup_day', 1)
        monthly_time = schedule_config.get('monthly_backup_time', '04:00')
        # Note: schedule library doesn't support monthly directly, would need custom logic
        
        logger.info("Backup scheduling configured")
    
    def run_scheduler(self):
        """Run the backup scheduler"""
        logger.info("Starting backup scheduler...")
        self.setup_scheduled_backups()
        
        try:
            while not self.execution_state['interruption_requested']:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        
        logger.info("Backup scheduler stopped")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='SutazAI Backup Orchestrator')
    parser.add_argument('command', choices=['run', 'daily', 'weekly', 'monthly', 'schedule', 'status'], 
                       help='Command to execute')
    parser.add_argument('--parallel', action='store_true', help='Run operations in parallel')
    parser.add_argument('--sequential', action='store_true', help='Run operations sequentially')
    parser.add_argument('--operations', nargs='+', help='Specific operations to run')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        orchestrator = SutazAIBackupOrchestrator()
        
        # Override configuration based on command line arguments
        if args.parallel:
            orchestrator.config['orchestration']['parallel_execution'] = True
        elif args.sequential:
            orchestrator.config['orchestration']['parallel_execution'] = False
        
        if args.command == 'run' or args.command == 'daily':
            result = orchestrator.run_full_backup_cycle('daily')
        elif args.command == 'weekly':
            result = orchestrator.run_full_backup_cycle('weekly')
        elif args.command == 'monthly':
            result = orchestrator.run_full_backup_cycle('monthly')
        elif args.command == 'schedule':
            orchestrator.run_scheduler()
            sys.exit(0)
        elif args.command == 'status':
            validation = orchestrator.validate_3_2_1_strategy()
            print(json.dumps(validation, indent=2))
            sys.exit(0 if validation['validation_status'] == 'compliant' else 1)
        
        # Exit with appropriate code
        if result['overall_status'] in ['success']:
            sys.exit(0)
        elif result['overall_status'] in ['partial_success', 'interrupted']:
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Backup orchestration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()