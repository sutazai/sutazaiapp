#!/usr/bin/env python3
"""
Garbage Collection Monitoring Integration
Purpose: Integrates with existing hygiene and compliance systems
Usage: python garbage-collection-monitor.py
Requirements: Python 3.8+
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import logging

BASE_DIR = Path("/opt/sutazaiapp")
DB_PATH = BASE_DIR / "data" / "garbage-collection.db"
COMPLIANCE_REPORT = BASE_DIR / "compliance-reports" / "latest.json"

class GarbageCollectionMonitor:
    """Monitor and report garbage collection status to compliance system"""
    
    def __init__(self):
        self.logger = logging.getLogger("GCMonitor")
        self.logger.setLevel(logging.INFO)
        
        # Setup logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def get_gc_health(self):
        """Get garbage collection health status"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Check if GC has run recently
                last_run = conn.execute("""
                    SELECT run_time, errors FROM collection_runs 
                    ORDER BY run_time DESC LIMIT 1
                """).fetchone()
                
                if not last_run:
                    return {
                        'status': 'error',
                        'message': 'No garbage collection runs found',
                        'health_score': 0
                    }
                
                last_run_time = datetime.fromisoformat(last_run[0].replace(' ', 'T'))
                hours_since_run = (datetime.now() - last_run_time).total_seconds() / 3600
                
                # Get disk usage
                disk_usage = conn.execute("""
                    SELECT percent_used FROM disk_usage 
                    ORDER BY timestamp DESC LIMIT 1
                """).fetchone()
                
                # Calculate health score
                health_score = 100
                status_messages = []
                
                # Check run frequency
                if hours_since_run > 2:
                    health_score -= 20
                    status_messages.append(f"Last run {hours_since_run:.1f} hours ago")
                
                # Check errors
                if last_run[1] > 0:
                    health_score -= 30
                    status_messages.append(f"{last_run[1]} errors in last run")
                
                # Check disk usage
                if disk_usage and disk_usage[0] > 80:
                    health_score -= 30
                    status_messages.append(f"Disk usage at {disk_usage[0]:.1f}%")
                elif disk_usage and disk_usage[0] > 60:
                    health_score -= 10
                    status_messages.append(f"Disk usage at {disk_usage[0]:.1f}%")
                
                return {
                    'status': 'healthy' if health_score >= 70 else 'warning' if health_score >= 40 else 'error',
                    'health_score': health_score,
                    'messages': status_messages,
                    'last_run': last_run_time.isoformat(),
                    'disk_usage': disk_usage[0] if disk_usage else 0
                }
                
        except Exception as e:
            self.logger.error(f"Error checking GC health: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'health_score': 0
            }
    
    def update_compliance_report(self):
        """Update compliance report with garbage collection status"""
        gc_health = self.get_gc_health()
        
        # Read existing compliance report
        if COMPLIANCE_REPORT.exists():
            with open(COMPLIANCE_REPORT) as f:
                compliance_data = json.load(f)
        else:
            compliance_data = {
                'timestamp': datetime.now().isoformat(),
                'services': {}
            }
        
        # Add garbage collection status
        compliance_data['services']['garbage_collection'] = {
            'status': gc_health['status'],
            'health_score': gc_health['health_score'],
            'details': gc_health
        }
        
        # Write back
        with open(COMPLIANCE_REPORT, 'w') as f:
            json.dump(compliance_data, f, indent=2)
        
        self.logger.info(f"Updated compliance report with GC status: {gc_health['status']}")
        
        return gc_health
    
    def check_and_fix_issues(self):
        """Check for issues and attempt to fix them"""
        gc_health = self.get_gc_health()
        
        if gc_health['health_score'] < 70:
            self.logger.warning(f"GC health score low: {gc_health['health_score']}")
            
            # Check if service is running
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', 'sutazai-garbage-collection'],
                    capture_output=True, text=True
                )
                
                if result.stdout.strip() != 'active':
                    self.logger.error("GC service not running, attempting to start...")
                    subprocess.run(['systemctl', 'start', 'sutazai-garbage-collection'])
                    
            except Exception as e:
                self.logger.error(f"Error checking service status: {e}")
            
            # If disk usage is high, run immediate collection
            if 'disk_usage' in gc_health and gc_health['disk_usage'] > 80:
                self.logger.warning("High disk usage detected, running immediate collection...")
                try:
                    subprocess.run([
                        'python3', 
                        str(BASE_DIR / 'scripts' / 'garbage-collection-system.py')
                    ])
                except Exception as e:
                    self.logger.error(f"Error running immediate collection: {e}")
    
    def generate_summary(self):
        """Generate summary report"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Get statistics for last 7 days
                week_ago = datetime.now() - timedelta(days=7)
                weekly_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as runs,
                        SUM(files_processed) as total_processed,
                        SUM(files_archived) as total_archived,
                        SUM(files_deleted) as total_deleted,
                        SUM(space_freed_mb) as total_freed_mb,
                        AVG(duration_seconds) as avg_duration
                    FROM collection_runs
                    WHERE run_time > ?
                """, (week_ago,)).fetchone()
                
                return {
                    'period': 'last_7_days',
                    'total_runs': weekly_stats[0],
                    'files_processed': weekly_stats[1] or 0,
                    'files_archived': weekly_stats[2] or 0,
                    'files_deleted': weekly_stats[3] or 0,
                    'space_freed_gb': (weekly_stats[4] or 0) / 1024,
                    'avg_duration_minutes': (weekly_stats[5] or 0) / 60
                }
                
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {}

def main():
    """Main monitoring function"""
    monitor = GarbageCollectionMonitor()
    
    # Check health
    health = monitor.get_gc_health()
    print(f"Garbage Collection Health: {health['status']} (Score: {health['health_score']})")
    
    # Update compliance report
    monitor.update_compliance_report()
    
    # Check and fix issues
    monitor.check_and_fix_issues()
    
    # Generate summary
    summary = monitor.generate_summary()
    print("\nWeekly Summary:")
    print(f"  Total runs: {summary.get('total_runs', 0)}")
    print(f"  Files processed: {summary.get('files_processed', 0)}")
    print(f"  Space freed: {summary.get('space_freed_gb', 0):.2f} GB")
    
    # Integration with compliance monitoring
    print("\n📊 Integration with Compliance System:")
    print(f"  ✓ Status updated in: {COMPLIANCE_REPORT}")
    print(f"  ✓ Health score: {health['health_score']}/100")
    print(f"  ✓ Disk usage: {health.get('disk_usage', 0):.1f}%")

if __name__ == "__main__":
    main()