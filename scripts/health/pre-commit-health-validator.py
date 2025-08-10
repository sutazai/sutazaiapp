#!/usr/bin/env python3
"""
PRE-COMMIT HEALTH VALIDATOR
Quick system health validation before allowing commits

Consolidated from:
- validate_system_health.py (pre-commit)
- quick-system-check.py (pre-commit)

Purpose: Ensure system is healthy before code changes
Author: ULTRA SCRIPT CONSOLIDATION MASTER
"""

import sys
import time
import requests
from pathlib import Path

# Import the master health controller
sys.path.append(str(Path(__file__).parent))
try:
    from master_health_controller import HealthMaster
except ImportError:
    # Fallback if master controller is not available
    print("Warning: Master health controller not available, using basic checks")
    HealthMaster = None

class PreCommitHealthValidator:
    """Fast health validation for pre-commit hooks"""
    
    def __init__(self):
        self.critical_services = [
            ('Backend API', 'http://localhost:10010/health', 5),
            ('Ollama', 'http://localhost:10104/api/tags', 10),
            ('PostgreSQL', 'http://localhost:10000', 5),  # Assuming health endpoint
            ('Redis', 'http://localhost:10001', 3)        # Assuming health endpoint
        ]
        
    def quick_service_check(self, name: str, url: str, timeout: int) -> tuple:
        """Quick service health check with minimal overhead"""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, f"✅ {name}: OK ({response.status_code})"
            else:
                return False, f"❌ {name}: HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            return False, f"⏱️ {name}: Timeout ({timeout}s)"
        except requests.exceptions.ConnectionError:
            return False, f"🔴 {name}: Connection refused"
        except Exception as e:
            return False, f"💥 {name}: {str(e)[:50]}"
    
    def validate_critical_services(self) -> tuple:
        """Validate only critical services for speed"""
        print("🔍 Pre-commit health validation...")
        
        start_time = time.time()
        results = []
        all_healthy = True
        
        for name, url, timeout in self.critical_services:
            healthy, message = self.quick_service_check(name, url, timeout)
            results.append(message)
            if not healthy:
                all_healthy = False
        
        duration = time.time() - start_time
        
        # Print results
        for result in results:
            print(f"  {result}")
        
        print(f"⏱️ Validation completed in {duration:.2f}s")
        
        return all_healthy, results
    
    def check_docker_containers(self) -> tuple:
        """Quick Docker container status check"""
        try:
            import docker
            client = docker.from_env()
            
            # Get running SutazAI containers
            containers = client.containers.list(filters={'status': 'running'})
            sutazai_containers = [c for c in containers if 'sutazai' in c.name]
            
            if len(sutazai_containers) < 4:  # Minimum expected containers
                return False, f"⚠️ Only {len(sutazai_containers)} SutazAI containers running (expected 4+)"
            
            return True, f"✅ {len(sutazai_containers)} SutazAI containers running"
            
        except Exception as e:
            return False, f"❌ Docker check failed: {str(e)[:50]}"
    
    def validate_system(self) -> bool:
        """Main validation entry point"""
        print("=" * 60)
        print("SUTAZAI PRE-COMMIT HEALTH VALIDATION")
        print("=" * 60)
        
        # Quick container check
        container_ok, container_msg = self.check_docker_containers()
        print(f"  {container_msg}")
        
        # Critical service validation
        services_ok, service_results = self.validate_critical_services()
        
        # Overall result
        overall_ok = container_ok and services_ok
        
        print("=" * 60)
        if overall_ok:
            print("✅ SYSTEM HEALTHY - Commit allowed")
        else:
            print("❌ SYSTEM ISSUES DETECTED - Consider fixing before commit")
            print("\n💡 Run 'make health' for detailed analysis")
            
        print("=" * 60)
        
        return overall_ok


def main():
    """Main entry point for pre-commit validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-commit health validation')
    parser.add_argument('--strict', action='store_true',
                       help='Fail commit if any issues detected')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Create validator
    validator = PreCommitHealthValidator()
    
    # Run validation
    if args.quiet:
        # Suppress output for quiet mode
        import contextlib
        import io
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            is_healthy = validator.validate_system()
        
        if not is_healthy:
            print("❌ System health issues detected")
    else:
        is_healthy = validator.validate_system()
    
    # Exit based on result and strictness
    if args.strict and not is_healthy:
        print("\n🚫 Commit blocked due to system health issues")
        sys.exit(1)
    elif not is_healthy:
        print("\n⚠️ Commit allowed despite health issues (use --strict to block)")
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()