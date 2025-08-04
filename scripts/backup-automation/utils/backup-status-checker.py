#!/usr/bin/env python3
"""
SutazAI Backup Status Checker
Quick status check for backup system health
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List

def check_backup_status() -> Dict:
    """Check overall backup system status"""
    backup_root = Path('/opt/sutazaiapp/data/backups')
    
    status = {
        'timestamp': datetime.datetime.now().isoformat(),
        'overall_health': 'unknown',
        'categories': {},
        'recent_backups': [],
        'issues': [],
        'storage_usage': {}
    }
    
    # Check each backup category
    categories = ['daily', 'weekly', 'monthly', 'postgres', 'sqlite', 'config', 'agents', 'models']
    
    for category in categories:
        category_path = backup_root / category
        category_status = {
            'exists': category_path.exists(),
            'file_count': 0,
            'latest_backup': None,
            'size_mb': 0
        }
        
        if category_path.exists():
            files = list(category_path.rglob('*'))
            backup_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
            
            category_status['file_count'] = len(backup_files)
            
            if backup_files:
                # Find latest backup
                latest = max(backup_files, key=lambda x: x.stat().st_mtime)
                age_hours = (datetime.datetime.now().timestamp() - latest.stat().st_mtime) / 3600
                
                category_status['latest_backup'] = {
                    'file': latest.name,
                    'age_hours': round(age_hours, 1),
                    'size_mb': round(latest.stat().st_size / (1024*1024), 2)
                }
                
                # Add to recent backups if less than 48 hours old
                if age_hours < 48:
                    status['recent_backups'].append({
                        'category': category,
                        'file': latest.name,
                        'age_hours': round(age_hours, 1)
                    })
                
                # Check for old backups
                if age_hours > 26:  # Alert if older than 26 hours
                    status['issues'].append(f"{category} backup is {age_hours:.1f} hours old")
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in backup_files)
                category_status['size_mb'] = round(total_size / (1024*1024), 2)
        
        status['categories'][category] = category_status
    
    # Check storage usage
    try:
        statvfs = os.statvfs(str(backup_root))
        total_bytes = statvfs.f_frsize * statvfs.f_blocks
        available_bytes = statvfs.f_frsize * statvfs.f_bavail
        used_bytes = total_bytes - available_bytes
        usage_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
        
        status['storage_usage'] = {
            'total_gb': round(total_bytes / (1024**3), 2),
            'used_gb': round(used_bytes / (1024**3), 2),
            'available_gb': round(available_bytes / (1024**3), 2),
            'usage_percent': round(usage_percent, 1)
        }
        
        if usage_percent > 90:
            status['issues'].append(f"Storage {usage_percent:.1f}% full")
            
    except Exception as e:
        status['storage_usage']['error'] = str(e)
    
    # Determine overall health
    if not status['issues']:
        if len(status['recent_backups']) >= 3:
            status['overall_health'] = 'healthy'
        elif len(status['recent_backups']) >= 1:
            status['overall_health'] = 'warning'
        else:
            status['overall_health'] = 'critical'
    else:
        if len(status['issues']) <= 2:
            status['overall_health'] = 'warning'
        else:
            status['overall_health'] = 'critical'
    
    return status

def print_status_summary(status: Dict):
    """Print a human-readable status summary"""
    print("=" * 60)
    print("SutazAI Backup System Status")
    print("=" * 60)
    print(f"Overall Health: {status['overall_health'].upper()}")
    print(f"Timestamp: {status['timestamp']}")
    print()
    
    # Storage info
    storage = status.get('storage_usage', {})
    if 'total_gb' in storage:
        print(f"Storage: {storage['used_gb']} GB / {storage['total_gb']} GB ({storage['usage_percent']}% used)")
        print()
    
    # Recent backups
    print("Recent Backups (last 48 hours):")
    if status['recent_backups']:
        for backup in sorted(status['recent_backups'], key=lambda x: x['age_hours']):
            print(f"  {backup['category']}: {backup['file']} ({backup['age_hours']}h ago)")
    else:
        print("  No recent backups found!")
    print()
    
    # Issues
    if status['issues']:
        print("Issues:")
        for issue in status['issues']:
            print(f"  ⚠️  {issue}")
        print()
    
    # Category details
    print("Backup Categories:")
    for category, info in status['categories'].items():
        if info['exists'] and info['file_count'] > 0:
            latest = info.get('latest_backup', {})
            print(f"  {category}: {info['file_count']} files, {info['size_mb']} MB")
            if latest:
                print(f"    Latest: {latest['file']} ({latest['age_hours']}h ago, {latest['size_mb']} MB)")
        else:
            print(f"  {category}: No backups found")

def main():
    """Main entry point"""
    try:
        status = check_backup_status()
        
        # Check if JSON output is requested
        if len(sys.argv) > 1 and sys.argv[1] == '--json':
            print(json.dumps(status, indent=2))
        else:
            print_status_summary(status)
        
        # Exit with appropriate code
        if status['overall_health'] == 'healthy':
            sys.exit(0)
        elif status['overall_health'] == 'warning':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"Error checking backup status: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()