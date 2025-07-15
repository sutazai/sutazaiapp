#!/usr/bin/env python3
"""
SutazAI System Dashboard
Real-time system monitoring and management interface
"""

import asyncio
import json
import time
import psutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemDashboard:
    """Real-time system dashboard for SutazAI"""
    
    def __init__(self):
        self.root_dir = Path("/opt/sutazaiapp")
        self.running = False
        
    def get_system_overview(self) -> Dict[str, Any]:
        """Get complete system overview"""
        return {
            "system_info": self._get_system_info(),
            "resource_usage": self._get_resource_usage(),
            "service_status": self._get_service_status(),
            "ai_components": self._get_ai_components_status(),
            "database_status": self._get_database_status(),
            "security_status": self._get_security_status(),
            "performance_metrics": self._get_performance_metrics(),
            "recent_activity": self._get_recent_activity()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "hostname": os.uname().nodename,
            "platform": f"{os.uname().sysname} {os.uname().release}",
            "python_version": sys.version.split()[0],
            "sutazai_version": "1.0.0",
            "installation_path": str(self.root_dir),
            "uptime": self._get_uptime(),
            "current_time": datetime.now().isoformat(),
            "timezone": time.tzname[0]
        }
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.root_dir))
        
        return {
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "core_count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2)
            },
            "network": self._get_network_stats()
        }
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except:
            return {"status": "unavailable"}
    
    def _get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        services = {}
        
        # Check critical files to determine service status
        critical_services = {
            "database": "data/sutazai.db",
            "neural_network": "data/neural_network_state.json",
            "knowledge_graph": "data/knowledge_graph_state.json",
            "model_registry": "data/model_registry.json",
            "api_server": "main.py",
            "security_system": "sutazai/core/acm.py"
        }
        
        for service, file_path in critical_services.items():
            file_exists = (self.root_dir / file_path).exists()
            services[service] = {
                "status": "active" if file_exists else "inactive",
                "last_check": datetime.now().isoformat()
            }
        
        return services
    
    def _get_ai_components_status(self) -> Dict[str, Any]:
        """Get AI components status"""
        components = {}
        
        # Check AI component files
        ai_components = {
            "code_generation_module": "sutazai/core/cgm.py",
            "knowledge_graph": "sutazai/core/kg.py",
            "authorization_control": "sutazai/core/acm.py",
            "neural_link_networks": "sutazai/nln/neural_node.py",
            "secure_storage": "sutazai/core/secure_storage.py"
        }
        
        for component, file_path in ai_components.items():
            file_exists = (self.root_dir / file_path).exists()
            components[component] = {
                "status": "operational" if file_exists else "offline",
                "file_path": file_path,
                "last_modified": self._get_file_modified_time(file_path)
            }
        
        return components
    
    def _get_database_status(self) -> Dict[str, Any]:
        """Get database status"""
        db_path = self.root_dir / "data/sutazai.db"
        
        if not db_path.exists():
            return {"status": "offline", "reason": "Database file not found"}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Get database size
            db_size = db_path.stat().st_size
            
            # Get record counts
            record_counts = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    count = cursor.fetchone()[0]
                    record_counts[table[0]] = count
                except:
                    record_counts[table[0]] = "N/A"
            
            conn.close()
            
            return {
                "status": "online",
                "database_size_mb": round(db_size / (1024**2), 2),
                "table_count": len(tables),
                "tables": [table[0] for table in tables],
                "record_counts": record_counts,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _get_security_status(self) -> Dict[str, Any]:
        """Get security status"""
        security_files = {
            "authorization_control": "sutazai/core/acm.py",
            "secure_storage": "sutazai/core/secure_storage.py",
            "security_fixes": "security_fix.py",
            "environment_config": ".env"
        }
        
        security_status = {}
        for component, file_path in security_files.items():
            file_exists = (self.root_dir / file_path).exists()
            security_status[component] = {
                "status": "active" if file_exists else "missing",
                "file_path": file_path
            }
        
        # Check for security reports
        security_reports = list(self.root_dir.glob("*SECURITY*.json"))
        
        return {
            "components": security_status,
            "security_reports": len(security_reports),
            "last_security_scan": self._get_latest_security_scan(),
            "overall_status": "secured" if all(s["status"] == "active" for s in security_status.values()) else "needs_attention"
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}
        
        # Check for performance reports
        perf_reports = [
            "PERFORMANCE_OPTIMIZATION_REPORT.json",
            "STORAGE_OPTIMIZATION_REPORT.json",
            "FINAL_VALIDATION_REPORT.json"
        ]
        
        for report_file in perf_reports:
            report_path = self.root_dir / report_file
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        report_data = json.load(f)
                        metrics[report_file.replace('.json', '')] = {
                            "status": "available",
                            "timestamp": report_data.get("timestamp", "unknown"),
                            "last_modified": self._get_file_modified_time(report_file)
                        }
                except:
                    metrics[report_file.replace('.json', '')] = {"status": "corrupted"}
            else:
                metrics[report_file.replace('.json', '')] = {"status": "missing"}
        
        return metrics
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent system activity"""
        activities = []
        
        # Check log files for recent activity
        log_files = ["logs/sutazai.log", "logs/error.log", "logs/security.log"]
        
        for log_file in log_files:
            log_path = self.root_dir / log_file
            if log_path.exists():
                try:
                    # Get file modification time
                    mod_time = datetime.fromtimestamp(log_path.stat().st_mtime)
                    activities.append({
                        "type": "log_update",
                        "description": f"Log file updated: {log_file}",
                        "timestamp": mod_time.isoformat(),
                        "source": log_file
                    })
                except:
                    pass
        
        # Check for recent reports
        json_files = list(self.root_dir.glob("*.json"))
        for json_file in json_files[-5:]:  # Last 5 JSON files
            try:
                mod_time = datetime.fromtimestamp(json_file.stat().st_mtime)
                activities.append({
                    "type": "report_generated",
                    "description": f"Report generated: {json_file.name}",
                    "timestamp": mod_time.isoformat(),
                    "source": json_file.name
                })
            except:
                pass
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return activities[:10]  # Return last 10 activities
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))
            return uptime_str
        except:
            return "unknown"
    
    def _get_file_modified_time(self, file_path: str) -> str:
        """Get file modification time"""
        try:
            full_path = self.root_dir / file_path
            if full_path.exists():
                mod_time = datetime.fromtimestamp(full_path.stat().st_mtime)
                return mod_time.isoformat()
            return "file_not_found"
        except:
            return "error"
    
    def _get_latest_security_scan(self) -> str:
        """Get latest security scan time"""
        security_files = ["security_fix.py", "security_hardening.py"]
        latest_time = None
        
        for file_path in security_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                mod_time = datetime.fromtimestamp(full_path.stat().st_mtime)
                if latest_time is None or mod_time > latest_time:
                    latest_time = mod_time
        
        return latest_time.isoformat() if latest_time else "never"
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        overview = self.get_system_overview()
        
        # Calculate health scores
        health_scores = self._calculate_health_scores(overview)
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "system_health": health_scores,
            "overview": overview,
            "recommendations": self._generate_recommendations(overview, health_scores),
            "alerts": self._generate_alerts(overview)
        }
        
        return report
    
    def _calculate_health_scores(self, overview: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system health scores"""
        scores = {}
        
        # Resource health (0-100)
        cpu_score = max(0, 100 - overview["resource_usage"]["cpu"]["usage_percent"])
        memory_score = max(0, 100 - overview["resource_usage"]["memory"]["usage_percent"])
        disk_score = max(0, 100 - overview["resource_usage"]["disk"]["usage_percent"])
        
        scores["resource_health"] = round((cpu_score + memory_score + disk_score) / 3, 1)
        
        # Service health
        active_services = sum(1 for s in overview["service_status"].values() if s["status"] == "active")
        total_services = len(overview["service_status"])
        scores["service_health"] = round((active_services / total_services) * 100, 1)
        
        # AI components health
        operational_components = sum(1 for c in overview["ai_components"].values() if c["status"] == "operational")
        total_components = len(overview["ai_components"])
        scores["ai_health"] = round((operational_components / total_components) * 100, 1)
        
        # Database health
        db_status = overview["database_status"]["status"]
        scores["database_health"] = 100 if db_status == "online" else 0
        
        # Security health
        security_status = overview["security_status"]["overall_status"]
        scores["security_health"] = 100 if security_status == "secured" else 50
        
        # Overall health
        all_scores = [scores["resource_health"], scores["service_health"], 
                     scores["ai_health"], scores["database_health"], scores["security_health"]]
        scores["overall_health"] = round(sum(all_scores) / len(all_scores), 1)
        
        return scores
    
    def _generate_recommendations(self, overview: Dict[str, Any], health_scores: Dict[str, Any]) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Resource recommendations
        if overview["resource_usage"]["cpu"]["usage_percent"] > 80:
            recommendations.append("High CPU usage detected - consider optimizing workloads")
        
        if overview["resource_usage"]["memory"]["usage_percent"] > 80:
            recommendations.append("High memory usage detected - consider increasing RAM or optimizing memory usage")
        
        if overview["resource_usage"]["disk"]["usage_percent"] > 80:
            recommendations.append("High disk usage detected - consider cleanup or additional storage")
        
        # Service recommendations
        inactive_services = [name for name, status in overview["service_status"].items() if status["status"] != "active"]
        if inactive_services:
            recommendations.append(f"Inactive services detected: {', '.join(inactive_services)}")
        
        # AI component recommendations
        offline_components = [name for name, status in overview["ai_components"].items() if status["status"] != "operational"]
        if offline_components:
            recommendations.append(f"Offline AI components: {', '.join(offline_components)}")
        
        # Database recommendations
        if overview["database_status"]["status"] != "online":
            recommendations.append("Database is not online - check database connectivity")
        
        # Security recommendations
        if overview["security_status"]["overall_status"] != "secured":
            recommendations.append("Security system needs attention - review security components")
        
        # General recommendations
        if health_scores["overall_health"] < 70:
            recommendations.append("Overall system health is below optimal - review all components")
        
        return recommendations
    
    def _generate_alerts(self, overview: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system alerts"""
        alerts = []
        
        # Critical alerts
        if overview["resource_usage"]["cpu"]["usage_percent"] > 90:
            alerts.append({
                "level": "critical",
                "message": "CPU usage is critically high",
                "value": f"{overview['resource_usage']['cpu']['usage_percent']}%"
            })
        
        if overview["resource_usage"]["memory"]["usage_percent"] > 90:
            alerts.append({
                "level": "critical",
                "message": "Memory usage is critically high",
                "value": f"{overview['resource_usage']['memory']['usage_percent']}%"
            })
        
        if overview["resource_usage"]["disk"]["usage_percent"] > 90:
            alerts.append({
                "level": "critical",
                "message": "Disk usage is critically high",
                "value": f"{overview['resource_usage']['disk']['usage_percent']}%"
            })
        
        # Warning alerts
        if overview["database_status"]["status"] != "online":
            alerts.append({
                "level": "warning",
                "message": "Database is not online",
                "value": overview["database_status"]["status"]
            })
        
        if overview["security_status"]["overall_status"] != "secured":
            alerts.append({
                "level": "warning",
                "message": "Security system needs attention",
                "value": overview["security_status"]["overall_status"]
            })
        
        return alerts
    
    def display_dashboard(self):
        """Display interactive dashboard"""
        os.system('clear')
        
        print("=" * 80)
        print("🚀 SutazAI System Dashboard")
        print("=" * 80)
        
        overview = self.get_system_overview()
        health_scores = self._calculate_health_scores(overview)
        
        # System Info
        print(f"\n📊 System Information:")
        print(f"   Hostname: {overview['system_info']['hostname']}")
        print(f"   Platform: {overview['system_info']['platform']}")
        print(f"   Python: {overview['system_info']['python_version']}")
        print(f"   SutazAI: {overview['system_info']['sutazai_version']}")
        print(f"   Uptime: {overview['system_info']['uptime']}")
        
        # Health Scores
        print(f"\n💊 Health Scores:")
        for metric, score in health_scores.items():
            color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            print(f"   {color} {metric.replace('_', ' ').title()}: {score}%")
        
        # Resource Usage
        print(f"\n⚡ Resource Usage:")
        cpu = overview['resource_usage']['cpu']
        memory = overview['resource_usage']['memory']
        disk = overview['resource_usage']['disk']
        
        print(f"   CPU: {cpu['usage_percent']}% ({cpu['core_count']} cores)")
        print(f"   Memory: {memory['usage_percent']}% ({memory['used_gb']:.1f}GB / {memory['total_gb']:.1f}GB)")
        print(f"   Disk: {disk['usage_percent']}% ({disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB)")
        
        # Service Status
        print(f"\n🔧 Service Status:")
        for service, status in overview['service_status'].items():
            status_icon = "🟢" if status['status'] == 'active' else "🔴"
            print(f"   {status_icon} {service.replace('_', ' ').title()}: {status['status']}")
        
        # AI Components
        print(f"\n🤖 AI Components:")
        for component, status in overview['ai_components'].items():
            status_icon = "🟢" if status['status'] == 'operational' else "🔴"
            print(f"   {status_icon} {component.replace('_', ' ').title()}: {status['status']}")
        
        # Database Status
        print(f"\n🗄️ Database Status:")
        db_status = overview['database_status']
        db_icon = "🟢" if db_status['status'] == 'online' else "🔴"
        print(f"   {db_icon} Status: {db_status['status']}")
        if db_status['status'] == 'online':
            print(f"   📊 Tables: {db_status['table_count']}")
            print(f"   💾 Size: {db_status['database_size_mb']}MB")
        
        # Recent Activity
        print(f"\n📋 Recent Activity:")
        for activity in overview['recent_activity'][:5]:
            print(f"   • {activity['description']}")
        
        print(f"\n🔄 Dashboard refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

def main():
    """Main dashboard function"""
    dashboard = SystemDashboard()
    
    try:
        while True:
            dashboard.display_dashboard()
            time.sleep(5)  # Refresh every 5 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed.")
        return True
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)