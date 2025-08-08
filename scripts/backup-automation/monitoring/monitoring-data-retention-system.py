#!/usr/bin/env python3
"""
SutazAI Monitoring Data Retention System
Manages retention and archival of monitoring data (Prometheus, Grafana, Loki)
"""

import os
import sys
import json
import logging
import datetime
import shutil
import tarfile
import hashlib
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/monitoring-retention.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonitoringDataRetentionSystem:
    """Monitoring data retention and archival system"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.monitoring_backup_dir = self.backup_root / 'monitoring'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure backup directory exists
        self.monitoring_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring data paths
        self.monitoring_paths = {
            'prometheus_data': '/var/lib/prometheus',
            'grafana_data': '/var/lib/grafana',
            'loki_data': '/opt/sutazaiapp/data/loki',
            'monitoring_configs': '/opt/sutazaiapp/monitoring',
            'health_reports': '/opt/sutazaiapp/reports',
            'compliance_reports': '/opt/sutazaiapp/compliance-reports',
            'logs_db': '/opt/sutazaiapp/monitoring/logs.db',
            'health_db': '/opt/sutazaiapp/monitoring/hygiene.db'
        }
        
        # Retention policies (days)
        self.retention_policies = {
            'prometheus_metrics': 90,
            'grafana_dashboards': 365,
            'loki_logs': 30,
            'health_reports': 180,
            'compliance_reports': 365,
            'performance_metrics': 90,
            'alert_history': 180
        }
        
        # Service endpoints
        self.service_endpoints = {
            'prometheus': 'http://localhost:9090',
            'grafana': 'http://localhost:3000',
            'loki': 'http://localhost:3100'
        }
    
    def backup_prometheus_data(self) -> Dict:
        """Backup Prometheus metrics data"""
        try:
            prometheus_path = Path(self.monitoring_paths['prometheus_data'])
            
            if not prometheus_path.exists():
                logger.warning(f"Prometheus data directory not found: {prometheus_path}")
                return {
                    'type': 'prometheus_data',
                    'status': 'skipped',
                    'reason': 'directory_not_found'
                }
            
            # Create snapshot using Prometheus API
            snapshot_result = self.create_prometheus_snapshot()
            
            if snapshot_result.get('status') == 'success':
                # Backup the snapshot
                backup_file = self.monitoring_backup_dir / f"prometheus_snapshot_{self.timestamp}.tar.gz"
                
                with tarfile.open(backup_file, 'w:gz') as tar:
                    snapshot_dir = prometheus_path / 'snapshots' / snapshot_result['snapshot_name']
                    tar.add(snapshot_dir, arcname='prometheus_snapshot')
                
                # Clean up snapshot
                shutil.rmtree(snapshot_dir, ignore_errors=True)
                
                checksum = self.calculate_checksum(backup_file)
                
                return {
                    'type': 'prometheus_data',
                    'backup_file': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'checksum': checksum,
                    'snapshot_name': snapshot_result['snapshot_name'],
                    'status': 'success'
                }
            else:
                # Fallback: direct directory backup
                backup_file = self.monitoring_backup_dir / f"prometheus_data_{self.timestamp}.tar.gz"
                
                with tarfile.open(backup_file, 'w:gz') as tar:
                    tar.add(prometheus_path, arcname='prometheus_data')
                
                checksum = self.calculate_checksum(backup_file)
                
                return {
                    'type': 'prometheus_data',
                    'backup_file': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'checksum': checksum,
                    'method': 'direct_backup',
                    'status': 'success'
                }
                
        except Exception as e:
            logger.error(f"Error backing up Prometheus data: {e}")
            return {
                'type': 'prometheus_data',
                'status': 'failed',
                'error': str(e)
            }
    
    def create_prometheus_snapshot(self) -> Dict:
        """Create Prometheus snapshot via API"""
        try:
            response = requests.post(
                f"{self.service_endpoints['prometheus']}/api/v1/admin/tsdb/snapshot",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return {
                        'status': 'success',
                        'snapshot_name': data['data']['name']
                    }
            
            logger.warning(f"Prometheus snapshot creation failed: {response.text}")
            return {'status': 'failed', 'error': response.text}
            
        except Exception as e:
            logger.warning(f"Could not create Prometheus snapshot via API: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def backup_grafana_data(self) -> Dict:
        """Backup Grafana dashboards and data"""
        try:
            grafana_path = Path(self.monitoring_paths['grafana_data'])
            
            if not grafana_path.exists():
                logger.warning(f"Grafana data directory not found: {grafana_path}")
                return {
                    'type': 'grafana_data',
                    'status': 'skipped',
                    'reason': 'directory_not_found'
                }
            
            # Export dashboards via API if possible
            dashboards = self.export_grafana_dashboards()
            
            # Create backup
            backup_file = self.monitoring_backup_dir / f"grafana_data_{self.timestamp}.tar.gz"
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(grafana_path, arcname='grafana_data')
                
                # Add exported dashboards if available
                if dashboards:
                    dashboards_file = self.monitoring_backup_dir / f"grafana_dashboards_{self.timestamp}.json"
                    with open(dashboards_file, 'w') as f:
                        json.dump(dashboards, f, indent=2)
                    tar.add(dashboards_file, arcname='grafana_dashboards.json')
                    dashboards_file.unlink()  # Remove temp file
            
            checksum = self.calculate_checksum(backup_file)
            
            return {
                'type': 'grafana_data',
                'backup_file': str(backup_file),
                'size': backup_file.stat().st_size,
                'checksum': checksum,
                'dashboards_exported': len(dashboards) if dashboards else 0,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error backing up Grafana data: {e}")
            return {
                'type': 'grafana_data',
                'status': 'failed',
                'error': str(e)
            }
    
    def export_grafana_dashboards(self) -> Optional[List[Dict]]:
        """Export Grafana dashboards via API"""
        try:
            # Get list of dashboards
            response = requests.get(
                f"{self.service_endpoints['grafana']}/api/search",
                timeout=10
            )
            
            if response.status_code != 200:
                return None
            
            dashboards_list = response.json()
            exported_dashboards = []
            
            for dashboard_info in dashboards_list:
                if dashboard_info.get('type') == 'dash-db':
                    dashboard_uid = dashboard_info.get('uid')
                    if dashboard_uid:
                        # Get dashboard details
                        dash_response = requests.get(
                            f"{self.service_endpoints['grafana']}/api/dashboards/uid/{dashboard_uid}",
                            timeout=10
                        )
                        
                        if dash_response.status_code == 200:
                            exported_dashboards.append(dash_response.json())
            
            return exported_dashboards
            
        except Exception as e:
            logger.warning(f"Could not export Grafana dashboards: {e}")
            return None
    
    def backup_loki_data(self) -> Dict:
        """Backup Loki log data"""
        try:
            loki_path = Path(self.monitoring_paths['loki_data'])
            
            if not loki_path.exists():
                logger.warning(f"Loki data directory not found: {loki_path}")
                return {
                    'type': 'loki_data',
                    'status': 'skipped',
                    'reason': 'directory_not_found'
                }
            
            # Create incremental backup based on date
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.retention_policies['loki_logs'])
            
            backup_file = self.monitoring_backup_dir / f"loki_data_{self.timestamp}.tar.gz"
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                # Only backup recent data for efficiency
                for item in loki_path.rglob('*'):
                    if item.is_file():
                        file_time = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                        if file_time >= cutoff_date:
                            arcname = item.relative_to(loki_path.parent)
                            tar.add(item, arcname=arcname)
            
            checksum = self.calculate_checksum(backup_file)
            
            return {
                'type': 'loki_data',
                'backup_file': str(backup_file),
                'size': backup_file.stat().st_size,
                'checksum': checksum,
                'retention_days': self.retention_policies['loki_logs'],
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error backing up Loki data: {e}")
            return {
                'type': 'loki_data',
                'status': 'failed',
                'error': str(e)
            }
    
    def backup_health_reports(self) -> Dict:
        """Backup health and compliance reports"""
        try:
            results = []
            
            # Backup reports directories
            for report_type in ['health_reports', 'compliance_reports']:
                reports_path = Path(self.monitoring_paths[report_type])
                
                if not reports_path.exists():
                    continue
                
                backup_file = self.monitoring_backup_dir / f"{report_type}_{self.timestamp}.tar.gz"
                
                with tarfile.open(backup_file, 'w:gz') as tar:
                    tar.add(reports_path, arcname=report_type)
                
                checksum = self.calculate_checksum(backup_file)
                
                results.append({
                    'type': report_type,
                    'backup_file': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'checksum': checksum
                })
            
            return {
                'type': 'health_reports',
                'reports': results,
                'total_files': len(results),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error backing up health reports: {e}")
            return {
                'type': 'health_reports',
                'status': 'failed',
                'error': str(e)
            }
    
    def backup_monitoring_databases(self) -> List[Dict]:
        """Backup monitoring SQLite databases"""
        results = []
        
        db_files = [
            self.monitoring_paths['logs_db'],
            self.monitoring_paths['health_db']
        ]
        
        for db_path in db_files:
            if not os.path.exists(db_path):
                continue
            
            try:
                db_name = os.path.basename(db_path)
                backup_file = self.monitoring_backup_dir / f"{db_name}_{self.timestamp}"
                
                # Use SQLite backup
                source_conn = sqlite3.connect(db_path)
                backup_conn = sqlite3.connect(str(backup_file))
                source_conn.backup(backup_conn)
                backup_conn.close()
                source_conn.close()
                
                # Compress
                compressed_file = f"{backup_file}.gz"
                subprocess.run(['gzip', str(backup_file)], check=True)
                
                checksum = self.calculate_checksum(Path(compressed_file))
                
                results.append({
                    'type': 'monitoring_database',
                    'database': db_path,
                    'backup_file': compressed_file,
                    'size': os.path.getsize(compressed_file),
                    'checksum': checksum,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Error backing up monitoring database {db_path}: {e}")
                results.append({
                    'type': 'monitoring_database',
                    'database': db_path,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def cleanup_old_monitoring_data(self) -> Dict:
        """Clean up old monitoring data based on retention policies"""
        cleanup_results = {}
        
        for data_type, retention_days in self.retention_policies.items():
            try:
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
                
                if data_type == 'loki_logs':
                    cleaned = self.cleanup_loki_old_data(cutoff_date)
                    cleanup_results[data_type] = cleaned
                elif data_type == 'health_reports':
                    cleaned = self.cleanup_old_reports(cutoff_date)
                    cleanup_results[data_type] = cleaned
                
            except Exception as e:
                logger.error(f"Error cleaning up {data_type}: {e}")
                cleanup_results[data_type] = {'status': 'failed', 'error': str(e)}
        
        return cleanup_results
    
    def cleanup_loki_old_data(self, cutoff_date: datetime.datetime) -> Dict:
        """Clean up old Loki data"""
        try:
            loki_path = Path(self.monitoring_paths['loki_data'])
            if not loki_path.exists():
                return {'status': 'skipped', 'reason': 'directory_not_found'}
            
            cleaned_files = 0
            cleaned_size = 0
            
            for item in loki_path.rglob('*'):
                if item.is_file():
                    file_time = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_size = item.stat().st_size
                        item.unlink()
                        cleaned_files += 1
                        cleaned_size += file_size
            
            return {
                'status': 'success',
                'files_cleaned': cleaned_files,
                'size_cleaned': cleaned_size
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def cleanup_old_reports(self, cutoff_date: datetime.datetime) -> Dict:
        """Clean up old health and compliance reports"""
        try:
            cleaned_files = 0
            cleaned_size = 0
            
            for report_type in ['health_reports', 'compliance_reports']:
                reports_path = Path(self.monitoring_paths[report_type])
                if not reports_path.exists():
                    continue
                
                for item in reports_path.rglob('*'):
                    if item.is_file():
                        file_time = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_size = item.stat().st_size
                            item.unlink()
                            cleaned_files += 1
                            cleaned_size += file_size
            
            return {
                'status': 'success',
                'files_cleaned': cleaned_files,
                'size_cleaned': cleaned_size
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    # calculate_checksum centralized in scripts.lib.file_utils
    
    def run_monitoring_retention(self) -> Dict:
        """Run complete monitoring data retention process"""
        start_time = time.time()
        logger.info(f"Starting monitoring data retention - {self.timestamp}")
        
        all_results = []
        
        # Backup Prometheus data
        prometheus_result = self.backup_prometheus_data()
        all_results.append(prometheus_result)
        
        # Backup Grafana data
        grafana_result = self.backup_grafana_data()
        all_results.append(grafana_result)
        
        # Backup Loki data
        loki_result = self.backup_loki_data()
        all_results.append(loki_result)
        
        # Backup health reports
        reports_result = self.backup_health_reports()
        all_results.append(reports_result)
        
        # Backup monitoring databases
        db_results = self.backup_monitoring_databases()
        all_results.extend(db_results)
        
        # Clean up old data
        cleanup_results = self.cleanup_old_monitoring_data()
        
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
            'retention_policies': self.retention_policies,
            'cleanup_results': cleanup_results,
            'backup_results': all_results
        }
        
        manifest_file = self.monitoring_backup_dir / f"monitoring_retention_manifest_{self.timestamp}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Monitoring data retention completed in {duration:.2f} seconds")
        logger.info(f"Successful: {manifest['successful_backups']}, Failed: {manifest['failed_backups']}")
        
        return manifest

def main():
    """Main entry point"""
    try:
        retention_system = MonitoringDataRetentionSystem()
        result = retention_system.run_monitoring_retention()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/monitoring_retention_summary_{retention_system.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code
        if result['failed_backups'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Monitoring data retention failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
