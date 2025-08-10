#!/usr/bin/env python3
"""
SutazAI Startup Validation and Performance Testing
Validates startup time reduction and system stability
"""

import asyncio
import json
import logging
import time
import subprocess
import statistics
from pathlib import Path
import requests
import docker
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupValidator:
    """Validates startup optimization performance and stability"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.baseline_times = {}
        self.optimized_times = {}
        
    def record_baseline_startup(self, runs: int = 3) -> Dict[str, float]:
        """Record baseline startup times using traditional sequential startup"""
        logger.info(f"Recording baseline startup times over {runs} runs")
        
        all_times = []
        
        for run in range(runs):
            logger.info(f"Baseline run {run + 1}/{runs}")
            
            # Stop all services first
            self.stop_all_services()
            time.sleep(10)  # Wait for clean shutdown
            
            # Start services sequentially (traditional method)
            start_time = time.time()
            
            # Critical services first
            critical_services = ["postgres", "redis", "neo4j"]
            for service in critical_services:
                self.start_single_service(service)
                self.wait_for_service_health(service, timeout=60)
            
            # Infrastructure services
            infra_services = ["chromadb", "qdrant", "ollama"]
            for service in infra_services:
                self.start_single_service(service)
                self.wait_for_service_health(service, timeout=60)
            
            # Core application
            core_services = ["backend", "frontend"]
            for service in core_services:
                self.start_single_service(service)
                self.wait_for_service_health(service, timeout=60)
            
            # Sample of AI agents (not all 69 for baseline - too long)
            sample_agents = [
                "letta", "autogpt", "crewai", "aider", "langflow",
                "gpt-engineer", "privategpt", "shellgpt"
            ]
            for service in sample_agents:
                self.start_single_service(service)
                # Don't wait for health check on AI agents to speed up baseline
            
            total_time = time.time() - start_time
            all_times.append(total_time)
            
            logger.info(f"Baseline run {run + 1} completed in {total_time:.2f}s")
        
        average_time = statistics.mean(all_times)
        self.baseline_times = {
            "individual_runs": all_times,
            "average": average_time,
            "min": min(all_times),
            "max": max(all_times),
            "std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0
        }
        
        logger.info(f"Baseline average: {average_time:.2f}s (±{self.baseline_times['std_dev']:.2f}s)")
        return self.baseline_times
    
    def test_optimized_startup(self, runs: int = 3) -> Dict[str, float]:
        """Test optimized startup times using fast_start.sh"""
        logger.info(f"Testing optimized startup times over {runs} runs")
        
        all_times = []
        
        for run in range(runs):
            logger.info(f"Optimized run {run + 1}/{runs}")
            
            # Stop all services first
            self.stop_all_services()
            time.sleep(10)  # Wait for clean shutdown
            
            # Use optimized startup script
            start_time = time.time()
            
            result = subprocess.run([
                str(self.project_root / "scripts" / "fast_start.sh"),
                "full"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            total_time = time.time() - start_time
            all_times.append(total_time)
            
            if result.returncode == 0:
                logger.info(f"Optimized run {run + 1} completed in {total_time:.2f}s")
            else:
                logger.error(f"Optimized run {run + 1} failed: {result.stderr}")
        
        average_time = statistics.mean(all_times)
        self.optimized_times = {
            "individual_runs": all_times,
            "average": average_time,
            "min": min(all_times),
            "max": max(all_times),
            "std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0
        }
        
        logger.info(f"Optimized average: {average_time:.2f}s (±{self.optimized_times['std_dev']:.2f}s)")
        return self.optimized_times
    
    def start_single_service(self, service_name: str) -> bool:
        """Start a single service using docker-compose"""
        try:
            result = subprocess.run([
                "docker", "compose",
                "-f", str(self.project_root / "docker-compose.yml"),
                "up", "-d", service_name
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
            return False
    
    def wait_for_service_health(self, service_name: str, timeout: int = 30) -> bool:
        """Wait for service to become healthy"""
        container_name = f"sutazai-{service_name}"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container = self.docker_client.containers.get(container_name)
                
                if container.status == 'running':
                    # Check health if available
                    health = container.attrs.get('State', {}).get('Health', {})
                    if health:
                        if health.get('Status') == 'healthy':
                            return True
                    else:
                        # No health check, assume healthy if running
                        return True
                
                time.sleep(2)
                
            except docker.errors.NotFound:
                time.sleep(1)
            except Exception as e:
                logger.debug(f"Health check error for {service_name}: {e}")
                time.sleep(1)
        
        return False
    
    def stop_all_services(self):
        """Stop all SutazAI services"""
        try:
            subprocess.run([
                "docker", "compose",
                "-f", str(self.project_root / "docker-compose.yml"),
                "down", "--remove-orphans"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Wait for containers to stop
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
    
    def validate_system_stability(self, duration_minutes: int = 10) -> Dict[str, any]:
        """Validate system stability after optimized startup"""
        logger.info(f"Validating system stability for {duration_minutes} minutes")
        
        stability_results = {
            "test_duration_minutes": duration_minutes,
            "container_restarts": {},
            "resource_usage": [],
            "api_availability": [],
            "errors": []
        }
        
        # Start optimized system
        result = subprocess.run([
            str(self.project_root / "scripts" / "fast_start.sh"),
            "full"
        ], capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode != 0:
            stability_results["errors"].append("Failed to start optimized system")
            return stability_results
        
        # Monitor for specified duration
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        initial_containers = self.get_running_containers()
        
        while time.time() < end_time:
            # Check container stability
            current_containers = self.get_running_containers()
            
            for container_name in initial_containers:
                if container_name not in current_containers:
                    restart_count = stability_results["container_restarts"].get(container_name, 0)
                    stability_results["container_restarts"][container_name] = restart_count + 1
            
            # Monitor resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            stability_results["resource_usage"].append({
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            })
            
            # Test API availability
            api_available = self.test_api_endpoints()
            stability_results["api_availability"].append({
                "timestamp": time.time(),
                "backend_api": api_available["backend"],
                "ollama_api": api_available["ollama"],
                "frontend": api_available["frontend"]
            })
            
            time.sleep(30)  # Check every 30 seconds
        
        logger.info("Stability test completed")
        return stability_results
    
    def get_running_containers(self) -> List[str]:
        """Get list of running SutazAI containers"""
        try:
            containers = self.docker_client.containers.list(
                filters={"name": "sutazai-"}
            )
            return [c.name for c in containers if c.status == 'running']
        except Exception as e:
            logger.error(f"Error getting containers: {e}")
            return []
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test availability of key API endpoints"""
        endpoints = {
            "backend": "http://localhost:8000/health",
            "ollama": "http://localhost:10104/api/tags",
            "frontend": "http://localhost:8501"
        }
        
        results = {}
        
        for name, url in endpoints.items():
            try:
                response = requests.get(url, timeout=5)
                results[name] = response.status_code < 400
            except Exception:
                results[name] = False
        
        return results
    
    def calculate_optimization_metrics(self) -> Dict[str, any]:
        """Calculate optimization performance metrics"""
        if not self.baseline_times or not self.optimized_times:
            logger.error("Both baseline and optimized times must be recorded first")
            return {}
        
        baseline_avg = self.baseline_times["average"]
        optimized_avg = self.optimized_times["average"]
        
        improvement_seconds = baseline_avg - optimized_avg
        improvement_percentage = (improvement_seconds / baseline_avg) * 100
        
        metrics = {
            "baseline_average_s": baseline_avg,
            "optimized_average_s": optimized_avg,
            "improvement_seconds": improvement_seconds,
            "improvement_percentage": improvement_percentage,
            "target_achieved": improvement_percentage >= 50.0,
            "reliability_improvement": {
                "baseline_std_dev": self.baseline_times["std_dev"],
                "optimized_std_dev": self.optimized_times["std_dev"],
                "consistency_improved": self.optimized_times["std_dev"] < self.baseline_times["std_dev"]
            }
        }
        
        return metrics
    
    def generate_validation_report(self) -> Dict[str, any]:
        """Generate comprehensive validation report"""
        timestamp = int(time.time())
        report_file = self.project_root / "logs" / f"startup_validation_{timestamp}.json"
        
        # Calculate metrics
        optimization_metrics = self.calculate_optimization_metrics()
        
        # Run stability test
        stability_results = self.validate_system_stability(duration_minutes=5)
        
        # Compile full report
        report = {
            "validation_timestamp": timestamp,
            "validation_summary": {
                "target_achievement": optimization_metrics.get("target_achieved", False),
                "improvement_percentage": optimization_metrics.get("improvement_percentage", 0),
                "baseline_time_s": optimization_metrics.get("baseline_average_s", 0),
                "optimized_time_s": optimization_metrics.get("optimized_average_s", 0),
                "system_stable": len(stability_results.get("container_restarts", {})) == 0
            },
            "performance_metrics": optimization_metrics,
            "baseline_measurements": self.baseline_times,
            "optimized_measurements": self.optimized_times,
            "stability_test": stability_results,
            "system_info": {
                "cpu_cores": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total // (1024**3),
                "docker_version": self.get_docker_version()
            }
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved: {report_file}")
        
        # Print summary
        self.print_validation_summary(report)
        
        return report
    
    def get_docker_version(self) -> str:
        """Get Docker version"""
        try:
            return self.docker_client.version()["Version"]
        except Exception:
            return "unknown"
    
    def print_validation_summary(self, report: Dict[str, any]):
        """Print validation summary to console"""
        summary = report["validation_summary"]
        metrics = report["performance_metrics"]
        
        print("\n" + "="*60)
        print("🚀 SUTAZAI STARTUP OPTIMIZATION VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   Baseline startup time:     {metrics.get('baseline_average_s', 0):.2f}s")
        print(f"   Optimized startup time:    {metrics.get('optimized_average_s', 0):.2f}s")
        print(f"   Time improvement:          {metrics.get('improvement_seconds', 0):.2f}s")
        print(f"   Percentage improvement:    {metrics.get('improvement_percentage', 0):.1f}%")
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        if summary["target_achievement"]:
            print("   ✅ SUCCESS: Achieved 50%+ startup time reduction!")
        else:
            print(f"   ❌ MISSED: Only achieved {summary['improvement_percentage']:.1f}% reduction")
        
        print(f"\n🔧 SYSTEM STABILITY:")
        if summary["system_stable"]:
            print("   ✅ STABLE: No container restarts during testing")
        else:
            print("   ⚠️  UNSTABLE: Some containers restarted during testing")
        
        stability = report["stability_test"]
        api_tests = stability.get("api_availability", [])
        if api_tests:
            backend_success = sum(1 for test in api_tests if test["backend_api"]) / len(api_tests) * 100
            print(f"   API Availability: {backend_success:.1f}% uptime")
        
        print(f"\n📈 RECOMMENDATIONS:")
        if not summary["target_achievement"]:
            print("   • Consider increasing parallel startup limits")
            print("   • Review service dependencies for further optimization")
            print("   • Implement lazy loading for non-critical services")
        
        if not summary["system_stable"]:
            print("   • Review container resource limits")
            print("   • Add health check stabilization delays")
            print("   • Monitor resource contention during startup")
        
        print("="*60 + "\n")

async def main():
    """Main validation function"""
    validator = StartupValidator()
    
    try:
        logger.info("Starting SutazAI startup optimization validation")
        
        # Record baseline performance
        logger.info("Phase 1: Recording baseline startup times...")
        validator.record_baseline_startup(runs=3)
        
        # Test optimized performance  
        logger.info("Phase 2: Testing optimized startup times...")
        validator.test_optimized_startup(runs=3)
        
        # Generate comprehensive report
        logger.info("Phase 3: Generating validation report...")
        report = validator.generate_validation_report()
        
        # Return success/failure based on target achievement
        return report["validation_summary"]["target_achievement"]
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)