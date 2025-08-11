#!/usr/bin/env python3
"""
Manual Debug Testing Framework for Hardware Resource Optimizer

Purpose: Create real system stress and test actual optimization effects
Usage: python manual_debug_test.py [test_scenario]
Requirements: Agent running on port 8116, system permissions for testing
"""

import os
import sys
import time
import json
import requests
import subprocess
import tempfile
import threading
import random
import string
import psutil
import docker
from pathlib import Path
from typing import Dict, Any, List
from debug_tracer import debug_tracer

class ManualTestFramework:
    """Framework for manual testing with real system effects"""
    
    def __init__(self, agent_url="http://localhost:8116"):
        self.agent_url = agent_url
        self.test_data_dir = "/tmp/hardware_optimizer_test_data"
        self.cleanup_required = []
        
        # Ensure test directory exists
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Docker not available: {e}")
            self.docker_client = None
        
        print(f"Manual test framework initialized - agent: {agent_url}")
        print(f"Test data directory: {self.test_data_dir}")
    
    def verify_agent_running(self) -> bool:
        """Verify the agent is running and accessible"""
        try:
            response = requests.get(f"{self.agent_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"Agent healthy: {health_data}")
                return True
            else:
                print(f"Agent unhealthy - status: {response.status_code}")
                return False
        except Exception as e:
            print(f"Cannot connect to agent: {e}")
            return False
    
    def create_memory_pressure(self, mb_to_allocate: int = 100) -> List[bytearray]:
        """Create real memory pressure by allocating large blocks"""
        print(f"Creating memory pressure: {mb_to_allocate}MB")
        
        memory_blocks = []
        try:
            for i in range(mb_to_allocate):
                # Allocate 1MB blocks
                block = bytearray(1024 * 1024)  # 1MB
                # Fill with random data to prevent optimization
                for j in range(0, len(block), 1024):
                    block[j:j+1024] = os.urandom(1024)
                memory_blocks.append(block)
                
                if i % 10 == 0:
                    print(f"  Allocated {i+1}MB...")
                    
            print(f"Memory pressure created: {len(memory_blocks)}MB allocated")
            self.cleanup_required.append(('memory', memory_blocks))
            return memory_blocks
            
        except MemoryError as e:
            print(f"Memory allocation failed at {len(memory_blocks)}MB: {e}")
            return memory_blocks
    
    def create_disk_pressure(self, mb_to_create: int = 50) -> List[str]:
        """Create disk pressure with temporary files and duplicates"""
        print(f"Creating disk pressure: {mb_to_create}MB of test files")
        
        created_files = []
        
        # Create large files
        for i in range(5):
            file_path = os.path.join(self.test_data_dir, f"large_file_{i}.dat")
            with open(file_path, 'wb') as f:
                # Write 10MB of random data
                for chunk in range(10):
                    f.write(os.urandom(1024 * 1024))
            created_files.append(file_path)
            print(f"  Created large file: {file_path} (10MB)")
        
        # Create duplicate files
        original_data = os.urandom(1024 * 1024)  # 1MB of data
        for i in range(10):
            file_path = os.path.join(self.test_data_dir, f"duplicate_{i}.dat")
            with open(file_path, 'wb') as f:
                f.write(original_data)
            created_files.append(file_path)
        print(f"  Created 10 duplicate files (1MB each)")
        
        # Create old log files
        for i in range(5):
            log_path = os.path.join(self.test_data_dir, f"old_app_{i}.log")
            with open(log_path, 'w') as f:
                for line in range(10000):
                    f.write(f"Log entry {line} - {time.time()} - random data: {''.join(random.choices(string.ascii_letters, k=50))}\n")
            
            # Make files old
            old_time = time.time() - (40 * 24 * 3600)  # 40 days old
            os.utime(log_path, (old_time, old_time))
            created_files.append(log_path)
        print(f"  Created 5 old log files")
        
        # Create cache-like files
        cache_dir = os.path.join(self.test_data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        for i in range(20):
            cache_file = os.path.join(cache_dir, f"cache_item_{i}.tmp")
            with open(cache_file, 'wb') as f:
                f.write(os.urandom(500 * 1024))  # 500KB each
            
            # Make some cache files old
            if i % 3 == 0:
                old_time = time.time() - (10 * 24 * 3600)  # 10 days old
                os.utime(cache_file, (old_time, old_time))
            created_files.append(cache_file)
        print(f"  Created 20 cache files")
        
        self.cleanup_required.append(('disk', created_files))
        return created_files
    
    def create_cpu_pressure(self) -> List[threading.Thread]:
        """Create CPU pressure with background threads"""
        print("Creating CPU pressure with background threads")
        
        def cpu_intensive_task():
            """CPU intensive task for stress testing"""
            end_time = time.time() + 30  # Run for 30 seconds
            while time.time() < end_time:
                # CPU intensive operations
                [x**2 for x in range(1000)]
                sum(range(1000))
        
        threads = []
        cpu_count = psutil.cpu_count()
        
        for i in range(cpu_count):
            thread = threading.Thread(target=cpu_intensive_task)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        print(f"  Started {len(threads)} CPU intensive threads")
        self.cleanup_required.append(('cpu', threads))
        return threads
    
    def create_docker_pressure(self) -> List[str]:
        """Create Docker pressure with containers and images"""
        if not self.docker_client:
            print("Docker not available - skipping Docker pressure test")
            return []
        
        print("Creating Docker pressure")
        container_ids = []
        
        try:
            # Create some test containers
            for i in range(3):
                container = self.docker_client.containers.run(
                    "alpine:latest",
                    command="sleep 60",
                    detach=True,
                    name=f"test_container_{i}_{int(time.time())}"
                )
                container_ids.append(container.id)
                print(f"  Created container: {container.name}")
                time.sleep(1)
            
            # Stop containers to create "stopped" containers for cleanup
            for container_id in container_ids:
                container = self.docker_client.containers.get(container_id)
                container.stop()
                print(f"  Stopped container: {container.name}")
            
            self.cleanup_required.append(('docker', container_ids))
            return container_ids
            
        except Exception as e:
            print(f"Docker pressure creation failed: {e}")
            return container_ids
    
    def measure_system_before_after(self, test_name: str, test_func):
        """Measure system state before and after test execution"""
        print(f"\n{'='*60}")
        print(f"MANUAL TEST: {test_name}")
        print(f"{'='*60}")
        
        # Take before snapshot
        before_snapshot = debug_tracer.take_system_snapshot(f"before_{test_name}")
        
        # Execute test
        try:
            result = test_func()
            
            # Take after snapshot
            after_snapshot = debug_tracer.take_system_snapshot(f"after_{test_name}")
            
            # Calculate differences
            differences = debug_tracer._calculate_snapshot_differences(before_snapshot, after_snapshot)
            
            print(f"\nTEST RESULTS for {test_name}:")
            print(f"Memory change: {differences.get('memory_change_mb', 0):.2f}MB")
            print(f"Disk change: {differences.get('disk_freed_mb', 0):.2f}MB")
            print(f"Process change: {differences.get('process_count_change', 0)}")
            
            if isinstance(result, dict) and 'actions_taken' in result:
                print(f"Actions taken: {len(result['actions_taken'])}")
                for action in result['actions_taken'][:5]:  # Show first 5 actions
                    print(f"  - {action}")
            
            return {
                "test_name": test_name,
                "before": before_snapshot,
                "after": after_snapshot,
                "differences": differences,
                "result": result
            }
            
        except Exception as e:
            print(f"TEST FAILED: {test_name} - {e}")
            after_snapshot = debug_tracer.take_system_snapshot(f"error_{test_name}")
            return {
                "test_name": test_name,
                "before": before_snapshot,
                "after": after_snapshot,
                "error": str(e)
            }
    
    def test_memory_optimization_real_pressure(self):
        """Test memory optimization with real memory pressure"""
        def memory_test():
            # Create memory pressure
            memory_blocks = self.create_memory_pressure(50)  # 50MB
            
            print(f"Memory allocated, current usage: {psutil.virtual_memory().percent:.1f}%")
            
            # Call memory optimization
            response = requests.post(f"{self.agent_url}/optimize/memory")
            result = response.json()
            
            print(f"Memory optimization result: {result.get('status')}")
            
            # Keep memory blocks alive to test if optimization actually helps
            # In real scenario, we'd release them, but here we test the optimization under pressure
            time.sleep(2)
            
            return result
        
        return self.measure_system_before_after("memory_optimization_real_pressure", memory_test)
    
    def test_disk_optimization_real_files(self):
        """Test disk optimization with real files"""
        def disk_test():
            # Create disk pressure
            files_created = self.create_disk_pressure(100)  # 100MB of test files
            
            print(f"Files created, current disk usage: {psutil.disk_usage('/').percent:.1f}%")
            
            # Call comprehensive storage optimization
            response = requests.post(f"{self.agent_url}/optimize/storage?dry_run=false")
            result = response.json()
            
            print(f"Storage optimization result: {result.get('status')}")
            
            # Verify files were actually processed
            remaining_files = [f for f in files_created if os.path.exists(f)]
            print(f"Files remaining after optimization: {len(remaining_files)}/{len(files_created)}")
            
            return result
        
        return self.measure_system_before_after("disk_optimization_real_files", disk_test)
    
    def test_cpu_optimization_real_load(self):
        """Test CPU optimization with real CPU load"""
        def cpu_test():
            # Create CPU pressure
            cpu_threads = self.create_cpu_pressure()
            
            time.sleep(2)  # Let CPU load stabilize
            print(f"CPU load created, current usage: {psutil.cpu_percent(interval=1):.1f}%")
            
            # Call CPU optimization
            response = requests.post(f"{self.agent_url}/optimize/cpu")
            result = response.json()
            
            print(f"CPU optimization result: {result.get('status')}")
            
            # Check if process nice values were actually changed
            nice_changes = 0
            for proc in psutil.process_iter(['pid', 'name', 'nice']):
                try:
                    if proc.info['nice'] > 0:  # Processes with lower priority
                        nice_changes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            print(f"Processes with adjusted nice values: {nice_changes}")
            
            # Wait for threads to complete
            for thread in cpu_threads:
                thread.join(timeout=1)
            
            return result
        
        return self.measure_system_before_after("cpu_optimization_real_load", cpu_test)
    
    def test_docker_optimization_real_containers(self):
        """Test Docker optimization with real containers"""
        def docker_test():
            if not self.docker_client:
                return {"status": "skipped", "reason": "Docker not available"}
            
            # Create Docker pressure
            container_ids = self.create_docker_pressure()
            
            initial_containers = len(self.docker_client.containers.list(all=True))
            initial_images = len(self.docker_client.images.list())
            
            print(f"Docker pressure created - containers: {initial_containers}, images: {initial_images}")
            
            # Call Docker optimization
            response = requests.post(f"{self.agent_url}/optimize/docker")
            result = response.json()
            
            print(f"Docker optimization result: {result.get('status')}")
            
            # Verify actual cleanup
            final_containers = len(self.docker_client.containers.list(all=True))
            final_images = len(self.docker_client.images.list())
            
            print(f"Docker after cleanup - containers: {final_containers}, images: {final_images}")
            print(f"Actual containers removed: {initial_containers - final_containers}")
            print(f"Actual images removed: {initial_images - final_images}")
            
            return result
        
        return self.measure_system_before_after("docker_optimization_real_containers", docker_test)
    
    def test_duplicate_detection_real_files(self):
        """Test duplicate detection with real duplicate files"""
        def duplicate_test():
            # Create specific duplicate test files
            test_dir = os.path.join(self.test_data_dir, "duplicate_test")
            os.makedirs(test_dir, exist_ok=True)
            
            # Create files with identical content
            identical_content = os.urandom(1024 * 1024)  # 1MB
            duplicate_files = []
            
            for i in range(5):
                file_path = os.path.join(test_dir, f"identical_file_{i}.dat")
                with open(file_path, 'wb') as f:
                    f.write(identical_content)
                duplicate_files.append(file_path)
            
            print(f"Created {len(duplicate_files)} identical files")
            
            # Analyze duplicates
            response = requests.get(f"{self.agent_url}/analyze/storage/duplicates?path={test_dir}")
            analysis_result = response.json()
            
            print(f"Duplicate analysis: {analysis_result.get('duplicate_groups', 0)} groups found")
            print(f"Space wasted: {analysis_result.get('space_wasted_mb', 0):.2f}MB")
            
            # Optimize duplicates
            response = requests.post(f"{self.agent_url}/optimize/storage/duplicates?path={test_dir}&dry_run=false")
            optimization_result = response.json()
            
            # Check actual file removal
            remaining_files = [f for f in duplicate_files if os.path.exists(f)]
            print(f"Files remaining after deduplication: {len(remaining_files)}/{len(duplicate_files)}")
            
            # Should keep 1 file, remove 4 duplicates
            expected_remaining = 1
            actual_removed = len(duplicate_files) - len(remaining_files)
            
            print(f"Expected to remove: {len(duplicate_files) - expected_remaining}")
            print(f"Actually removed: {actual_removed}")
            
            self.cleanup_required.append(('disk', remaining_files))
            
            return {
                "analysis": analysis_result,
                "optimization": optimization_result,
                "verification": {
                    "files_created": len(duplicate_files),
                    "files_remaining": len(remaining_files),
                    "files_removed": actual_removed,
                    "removal_successful": actual_removed == (len(duplicate_files) - expected_remaining)
                }
            }
        
        return self.measure_system_before_after("duplicate_detection_real_files", duplicate_test)
    
    def test_cache_cleanup_real_caches(self):
        """Test cache cleanup with real cache files"""
        def cache_test():
            # Create real cache-like structure
            cache_dirs = [
                os.path.join(self.test_data_dir, ".cache", "test_app"),
                os.path.join(self.test_data_dir, ".cache", "pip"),
                os.path.join(self.test_data_dir, ".cache", "npm")
            ]
            
            cache_files = []
            for cache_dir in cache_dirs:
                os.makedirs(cache_dir, exist_ok=True)
                
                # Create cache files with different ages
                for i in range(10):
                    cache_file = os.path.join(cache_dir, f"cache_{i}.tmp")
                    with open(cache_file, 'wb') as f:
                        f.write(os.urandom(100 * 1024))  # 100KB each
                    
                    # Make some files old (8+ days)
                    if i % 2 == 0:
                        old_time = time.time() - (8 * 24 * 3600)
                        os.utime(cache_file, (old_time, old_time))
                    
                    cache_files.append(cache_file)
            
            print(f"Created {len(cache_files)} cache files in {len(cache_dirs)} directories")
            
            # Call cache optimization
            response = requests.post(f"{self.agent_url}/optimize/storage/cache")
            result = response.json()
            
            # Check what was actually cleaned
            remaining_files = [f for f in cache_files if os.path.exists(f)]
            removed_count = len(cache_files) - len(remaining_files)
            
            print(f"Cache files removed: {removed_count}/{len(cache_files)}")
            
            self.cleanup_required.append(('disk', remaining_files))
            
            return result
        
        return self.measure_system_before_after("cache_cleanup_real_caches", cache_test)
    
    def test_edge_case_permissions(self):
        """Test edge cases with permission denied scenarios"""
        def permission_test():
            # Create files with different permissions
            permission_test_dir = os.path.join(self.test_data_dir, "permission_test")
            os.makedirs(permission_test_dir, exist_ok=True)
            
            # Create file with restricted permissions
            restricted_file = os.path.join(permission_test_dir, "restricted.dat")
            with open(restricted_file, 'w') as f:
                f.write("This file has restricted permissions")
            
            os.chmod(restricted_file, 0o000)  # No permissions
            
            # Create directory with restricted permissions
            restricted_dir = os.path.join(permission_test_dir, "restricted_dir")
            os.makedirs(restricted_dir, exist_ok=True)
            os.chmod(restricted_dir, 0o000)
            
            print("Created files/directories with restricted permissions")
            
            # Try to analyze storage - should handle permission errors gracefully
            response = requests.get(f"{self.agent_url}/analyze/storage?path={permission_test_dir}")
            result = response.json()
            
            print(f"Storage analysis with permissions: {result.get('status')}")
            
            # Restore permissions for cleanup
            os.chmod(restricted_file, 0o644)
            os.chmod(restricted_dir, 0o755)
            
            self.cleanup_required.append(('disk', [restricted_file, restricted_dir]))
            
            return result
        
        return self.measure_system_before_after("edge_case_permissions", permission_test)
    
    def test_symbolic_links_and_special_files(self):
        """Test handling of symbolic links and special files"""
        def symlink_test():
            symlink_test_dir = os.path.join(self.test_data_dir, "symlink_test")
            os.makedirs(symlink_test_dir, exist_ok=True)
            
            # Create regular file
            regular_file = os.path.join(symlink_test_dir, "regular.txt")
            with open(regular_file, 'w') as f:
                f.write("Regular file content")
            
            # Create symbolic link
            symlink_file = os.path.join(symlink_test_dir, "symlink.txt")
            os.symlink(regular_file, symlink_file)
            
            # Create broken symbolic link
            broken_symlink = os.path.join(symlink_test_dir, "broken.txt")
            os.symlink("/nonexistent/file", broken_symlink)
            
            print("Created regular file, symlink, and broken symlink")
            
            # Analyze storage - should handle symlinks properly
            response = requests.get(f"{self.agent_url}/analyze/storage?path={symlink_test_dir}")
            result = response.json()
            
            print(f"Storage analysis with symlinks: {result.get('status')}")
            print(f"Files found: {result.get('total_files', 0)}")
            
            self.cleanup_required.append(('disk', [regular_file, symlink_file, broken_symlink]))
            
            return result
        
        return self.measure_system_before_after("symbolic_links_special_files", symlink_test)
    
    def test_extremely_large_directories(self):
        """Test with extremely large directories (many files)"""
        def large_dir_test():
            large_dir = os.path.join(self.test_data_dir, "large_directory")
            os.makedirs(large_dir, exist_ok=True)
            
            # Create many small files
            files_created = []
            print("Creating 1000 small files...")
            
            for i in range(1000):
                if i % 100 == 0:
                    print(f"  Created {i} files...")
                
                file_path = os.path.join(large_dir, f"small_file_{i:04d}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"Small file {i} content\n" * 10)
                files_created.append(file_path)
            
            print(f"Created {len(files_created)} small files")
            
            # Analyze this large directory
            start_time = time.time()
            response = requests.get(f"{self.agent_url}/analyze/storage?path={large_dir}")
            analysis_time = time.time() - start_time
            
            result = response.json()
            
            print(f"Analysis completed in {analysis_time:.2f} seconds")
            print(f"Files analyzed: {result.get('total_files', 0)}")
            
            self.cleanup_required.append(('disk', files_created))
            
            return {
                "analysis_result": result,
                "analysis_time": analysis_time,
                "files_created": len(files_created)
            }
        
        return self.measure_system_before_after("extremely_large_directories", large_dir_test)
    
    def run_comprehensive_manual_tests(self):
        """Run all manual tests and generate comprehensive report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MANUAL TESTING OF HARDWARE RESOURCE OPTIMIZER")
        print("="*80)
        
        if not self.verify_agent_running():
            print("ERROR: Agent is not running. Please start the agent first.")
            return
        
        # Run all tests
        test_results = []
        
        tests_to_run = [
            ("Memory Optimization with Real Pressure", self.test_memory_optimization_real_pressure),
            ("Disk Optimization with Real Files", self.test_disk_optimization_real_files),
            ("CPU Optimization with Real Load", self.test_cpu_optimization_real_load),
            ("Docker Optimization with Real Containers", self.test_docker_optimization_real_containers),
            ("Duplicate Detection with Real Files", self.test_duplicate_detection_real_files),
            ("Cache Cleanup with Real Caches", self.test_cache_cleanup_real_caches),
            ("Edge Case: Permission Denied", self.test_edge_case_permissions),
            ("Edge Case: Symbolic Links", self.test_symbolic_links_and_special_files),
            ("Stress Test: Large Directories", self.test_extremely_large_directories)
        ]
        
        for test_name, test_func in tests_to_run:
            try:
                print(f"\n{'='*60}")
                print(f"RUNNING: {test_name}")
                print(f"{'='*60}")
                
                result = test_func()
                test_results.append(result)
                
                print(f"✓ COMPLETED: {test_name}")
                
            except Exception as e:
                print(f"✗ FAILED: {test_name} - {e}")
                test_results.append({
                    "test_name": test_name,
                    "error": str(e),
                    "failed": True
                })
        
        # Generate final debug report
        debug_report = debug_tracer.generate_debug_report()
        
        # Create comprehensive test report
        comprehensive_report = {
            "timestamp": time.time(),
            "test_session_summary": {
                "total_tests": len(test_results),
                "successful_tests": len([t for t in test_results if not t.get('failed', False)]),
                "failed_tests": len([t for t in test_results if t.get('failed', False)])
            },
            "individual_test_results": test_results,
            "debug_trace_report": debug_report,
            "cleanup_required": self.cleanup_required
        }
        
        # Save comprehensive report
        report_file = f"/tmp/comprehensive_manual_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print("MANUAL TESTING COMPLETED")
        print(f"{'='*80}")
        print(f"Total tests: {comprehensive_report['test_session_summary']['total_tests']}")
        print(f"Successful: {comprehensive_report['test_session_summary']['successful_tests']}")
        print(f"Failed: {comprehensive_report['test_session_summary']['failed_tests']}")
        print(f"Comprehensive report saved: {report_file}")
        
        # Cleanup
        self.cleanup_test_artifacts()
        
        return comprehensive_report
    
    def cleanup_test_artifacts(self):
        """Clean up test artifacts"""
        print("\nCleaning up test artifacts...")
        
        for cleanup_type, items in self.cleanup_required:
            try:
                if cleanup_type == 'memory':
                    # Memory blocks will be garbage collected
                    del items
                    print("  Memory blocks released")
                    
                elif cleanup_type == 'disk':
                    for file_path in items:
                        if os.path.exists(file_path):
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                import shutil
                                shutil.rmtree(file_path)
                    print(f"  Removed {len(items)} disk artifacts")
                    
                elif cleanup_type == 'docker':
                    if self.docker_client:
                        for container_id in items:
                            try:
                                container = self.docker_client.containers.get(container_id)
                                container.remove(force=True)
                            except Exception:
                                pass
                        print(f"  Removed {len(items)} Docker containers")
                        
                elif cleanup_type == 'cpu':
                    # CPU threads should have completed
                    print("  CPU threads completed")
                    
            except Exception as e:
                print(f"  Cleanup error for {cleanup_type}: {e}")
        
        # Remove test data directory
        try:
            import shutil
            if os.path.exists(self.test_data_dir):
                shutil.rmtree(self.test_data_dir)
                print(f"  Removed test data directory: {self.test_data_dir}")
        except Exception as e:
            print(f"  Could not remove test data directory: {e}")

if __name__ == "__main__":
    test_framework = ManualTestFramework()
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if hasattr(test_framework, f"test_{test_name}"):
            test_func = getattr(test_framework, f"test_{test_name}")
            result = test_func()
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Test '{test_name}' not found")
    else:
        # Run comprehensive tests
        test_framework.run_comprehensive_manual_tests()