#!/usr/bin/env python3
"""
MCP Server Performance Monitor

Continuous monitoring script for MCP server performance.
Can be run as a cron job or monitoring service.
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/mcp-manager/benchmark_results/monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPPerformanceMonitor:
    def __init__(self):
        self.results_dir = Path("/opt/sutazaiapp/mcp-manager/benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_mcp_config(self):
        """Load MCP server configurations"""
        servers = {}
        config_path = Path("/opt/sutazaiapp/.mcp.json")
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                for name, server_config in config.get("mcpServers", {}).items():
                    servers[name] = {
                        "command": server_config["command"],
                        "args": server_config.get("args", []),
                        "type": server_config.get("type", "stdio")
                    }
        
        return servers
    
    def check_server_health(self, name, config):
        """Quick health check for a server"""
        if not config["command"].endswith('.sh'):
            return {"status": "N/A", "message": "Not a wrapper script"}
        
        try:
            result = subprocess.run(
                ['/bin/bash', config["command"], '--selfcheck'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "message": result.stdout.strip() if result.returncode == 0 else result.stderr.strip(),
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "message": "Health check timed out"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def test_server_startup(self, name, config):
        """Test server startup time"""
        try:
            if config["command"].endswith('.sh'):
                cmd = ['/bin/bash', config["command"]] + config["args"]
            else:
                cmd = [config["command"]] + config["args"]
            
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it time to start
            time.sleep(1)
            startup_time = time.time() - start_time
            
            # Check if running
            if process.poll() is None:
                # Clean shutdown
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                return {
                    "startup_success": True,
                    "startup_time_ms": startup_time * 1000
                }
            else:
                return {
                    "startup_success": False,
                    "startup_time_ms": 0,
                    "error": "Process exited immediately"
                }
                
        except Exception as e:
            return {
                "startup_success": False,
                "startup_time_ms": 0,
                "error": str(e)
            }
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        logger.info("🔄 Starting MCP performance monitoring cycle")
        
        servers = self.load_mcp_config()
        cycle_results = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "servers_monitored": len(servers),
            "results": []
        }
        
        working_servers = 0
        total_startup_time = 0
        health_pass = 0
        
        for name, config in servers.items():
            logger.info(f"📊 Monitoring {name}")
            
            # Health check
            health_result = self.check_server_health(name, config)
            
            # Startup test
            startup_result = self.test_server_startup(name, config)
            
            # Combine results
            server_result = {
                "name": name,
                "health": health_result,
                "startup": startup_result,
                "overall_status": "WORKING" if (
                    startup_result.get("startup_success", False) and 
                    health_result.get("status") in ["PASS", "N/A"]
                ) else "FAILED"
            }
            
            cycle_results["results"].append(server_result)
            
            # Update counters
            if server_result["overall_status"] == "WORKING":
                working_servers += 1
                total_startup_time += startup_result.get("startup_time_ms", 0)
            
            if health_result.get("status") == "PASS":
                health_pass += 1
        
        # Calculate summary metrics
        avg_startup = total_startup_time / working_servers if working_servers > 0 else 0
        
        cycle_results["summary"] = {
            "working_servers": working_servers,
            "total_servers": len(servers),
            "availability_percent": (working_servers / len(servers)) * 100,
            "health_pass_percent": (health_pass / len(servers)) * 100,
            "average_startup_ms": avg_startup
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"monitor_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(cycle_results, f, indent=2)
        
        # Log summary
        summary = cycle_results["summary"]
        logger.info(f"✅ Monitoring complete: {summary['working_servers']}/{summary['total_servers']} servers working")
        logger.info(f"📊 Availability: {summary['availability_percent']:.1f}%, Avg startup: {summary['average_startup_ms']:.0f}ms")
        
        # Check for alerts
        self.check_alerts(cycle_results)
        
        return cycle_results
    
    def check_alerts(self, results):
        """Check for performance alerts"""
        summary = results["summary"]
        
        # Alert thresholds
        MIN_AVAILABILITY = 80.0  # 80%
        MAX_STARTUP_TIME = 2000  # 2 seconds
        MIN_HEALTH_PASS = 70.0   # 70%
        
        alerts = []
        
        if summary["availability_percent"] < MIN_AVAILABILITY:
            alerts.append(f"🚨 LOW AVAILABILITY: {summary['availability_percent']:.1f}% (threshold: {MIN_AVAILABILITY}%)")
        
        if summary["average_startup_ms"] > MAX_STARTUP_TIME:
            alerts.append(f"⏰ SLOW STARTUP: {summary['average_startup_ms']:.0f}ms (threshold: {MAX_STARTUP_TIME}ms)")
        
        if summary["health_pass_percent"] < MIN_HEALTH_PASS:
            alerts.append(f"🏥 HEALTH ISSUES: {summary['health_pass_percent']:.1f}% passing (threshold: {MIN_HEALTH_PASS}%)")
        
        # Log alerts
        for alert in alerts:
            logger.warning(alert)
        
        # Save alerts to file if any
        if alerts:
            alert_file = self.results_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(alert_file, 'a') as f:
                f.write(f"\n{datetime.now().isoformat()}\n")
                for alert in alerts:
                    f.write(f"{alert}\n")
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        today = datetime.now().strftime('%Y%m%d')
        monitor_files = list(self.results_dir.glob(f"monitor_{today}_*.json"))
        
        if not monitor_files:
            logger.info("No monitoring data for today")
            return
        
        # Load all monitoring data for today
        daily_data = []
        for file in monitor_files:
            with open(file) as f:
                daily_data.append(json.load(f))
        
        # Calculate daily statistics
        availability_values = [d["summary"]["availability_percent"] for d in daily_data]
        startup_values = [d["summary"]["average_startup_ms"] for d in daily_data if d["summary"]["average_startup_ms"] > 0]
        
        daily_stats = {
            "date": today,
            "monitoring_cycles": len(daily_data),
            "avg_availability": sum(availability_values) / len(availability_values) if availability_values else 0,
            "min_availability": min(availability_values) if availability_values else 0,
            "avg_startup_time": sum(startup_values) / len(startup_values) if startup_values else 0,
            "max_startup_time": max(startup_values) if startup_values else 0
        }
        
        # Save daily report
        report_file = self.results_dir / f"daily_report_{today}.json"
        with open(report_file, 'w') as f:
            json.dump(daily_stats, f, indent=2)
        
        logger.info(f"📅 Daily report saved: {daily_stats['avg_availability']:.1f}% avg availability")

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Performance Monitor")
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=300, help="Monitoring interval in seconds (default: 300)")
    parser.add_argument("--daily-report", action="store_true", help="Generate daily report")
    
    args = parser.parse_args()
    
    monitor = MCPPerformanceMonitor()
    
    if args.daily_report:
        monitor.generate_daily_report()
        return
    
    if args.continuous:
        logger.info(f"🔄 Starting continuous monitoring (interval: {args.interval}s)")
        try:
            while True:
                monitor.run_monitoring_cycle()
                logger.info(f"😴 Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("🛑 Continuous monitoring stopped")
    else:
        # Single monitoring cycle
        monitor.run_monitoring_cycle()

if __name__ == "__main__":
    main()