#!/usr/bin/env python3
"""
Debug-Enhanced Hardware Resource Optimizer Agent

Purpose: On-demand hardware resource optimization with detailed debug tracing
Usage: Runs optimizations when API endpoints are called with full tracing
Requirements: Docker, system access for optimization tasks, debug tracing
"""

import os
import sys
from pathlib import Path

# Add local shared directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the original agent
from app import HardwareResourceOptimizerAgent as OriginalAgent
from debug_tracer import debug_tracer

class DebugEnhancedHardwareResourceOptimizerAgent(OriginalAgent):
    """Debug-enhanced version with comprehensive tracing"""
    
    def __init__(self):
        super().__init__()
        self.debug_tracer = debug_tracer
        self.logger.info("Debug-enhanced hardware optimizer initialized with tracing")
    
    @debug_tracer.trace_function_execution("memory_optimization")
    def _optimize_memory(self):
        """Debug-enhanced memory optimization with tracing"""
        self.logger.info("STARTING memory optimization with debug tracing")
        
        # Call original method
        result = super()._optimize_memory()
        
        # Add debug information
        if isinstance(result, dict):
            result['debug_enhanced'] = True
            result['trace_id'] = len(self.debug_tracer.execution_traces)
        
        return result
    
    @debug_tracer.trace_function_execution("cpu_optimization")
    def _optimize_cpu(self):
        """Debug-enhanced CPU optimization with tracing"""
        self.logger.info("STARTING CPU optimization with debug tracing")
        
        # Take before snapshot of CPU processes
        before_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'nice']):
            try:
                info = proc.info
                if info['cpu_percent'] and info['cpu_percent'] > 1:
                    before_processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Call original method
        result = super()._optimize_cpu()
        
        # Take after snapshot and verify changes
        after_processes = []
        nice_changes_verified = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'nice']):
            try:
                info = proc.info
                if info['cpu_percent'] and info['cpu_percent'] > 1:
                    after_processes.append(info)
                    
                    # Find corresponding before process
                    before_proc = next((p for p in before_processes if p['pid'] == info['pid']), None)
                    if before_proc and before_proc['nice'] != info['nice']:
                        nice_changes_verified += 1
                        self.logger.info(f"VERIFIED: Process {info['name']} (PID {info['pid']}) nice changed from {before_proc['nice']} to {info['nice']}")
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Add verification results
        if isinstance(result, dict):
            result['debug_enhanced'] = True
            result['trace_id'] = len(self.debug_tracer.execution_traces)
            result['verified_nice_changes'] = nice_changes_verified
            result['before_high_cpu_processes'] = len(before_processes)
            result['after_high_cpu_processes'] = len(after_processes)
        
        return result
    
    @debug_tracer.trace_function_execution("disk_optimization")
    def _optimize_disk(self):
        """Debug-enhanced disk optimization with detailed file tracking"""
        self.logger.info("STARTING disk optimization with debug tracing")
        
        # Track files before optimization
        temp_dirs = ['/tmp', '/var/tmp']
        files_before = {}
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    files_before[temp_dir] = len(os.listdir(temp_dir))
                except PermissionError:
                    files_before[temp_dir] = 0
        
        # Call original method
        result = super()._optimize_disk()
        
        # Track files after optimization
        files_after = {}
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    files_after[temp_dir] = len(os.listdir(temp_dir))
                except PermissionError:
                    files_after[temp_dir] = 0
        
        # Calculate actual file removal
        actual_files_removed = 0
        for temp_dir in temp_dirs:
            if temp_dir in files_before and temp_dir in files_after:
                removed = files_before[temp_dir] - files_after[temp_dir]
                if removed > 0:
                    actual_files_removed += removed
                    self.logger.info(f"VERIFIED: Removed {removed} files from {temp_dir}")
        
        # Add verification results
        if isinstance(result, dict):
            result['debug_enhanced'] = True
            result['trace_id'] = len(self.debug_tracer.execution_traces)
            result['verified_files_removed'] = actual_files_removed
            result['files_before'] = files_before
            result['files_after'] = files_after
        
        return result
    
    @debug_tracer.trace_function_execution("docker_optimization")
    def _optimize_docker(self):
        """Debug-enhanced Docker optimization with container tracking"""
        self.logger.info("STARTING Docker optimization with debug tracing")
        
        if not self.docker_client:
            return super()._optimize_docker()
        
        # Track Docker resources before
        try:
            before_containers = self.docker_client.containers.list(all=True)
            before_images = self.docker_client.images.list()
            before_container_names = [c.name for c in before_containers]
            before_image_ids = [i.id for i in before_images]
            
            self.logger.info(f"BEFORE: {len(before_containers)} containers, {len(before_images)} images")
            
        except Exception as e:
            self.logger.error(f"Failed to get Docker state before optimization: {e}")
            return super()._optimize_docker()
        
        # Call original method
        result = super()._optimize_docker()
        
        # Track Docker resources after and verify
        try:
            after_containers = self.docker_client.containers.list(all=True)
            after_images = self.docker_client.images.list()
            after_container_names = [c.name for c in after_containers]
            after_image_ids = [i.id for i in after_images]
            
            # Identify what was actually removed
            removed_containers = [name for name in before_container_names if name not in after_container_names]
            removed_images = [img_id for img_id in before_image_ids if img_id not in after_image_ids]
            
            self.logger.info(f"AFTER: {len(after_containers)} containers, {len(after_images)} images")
            self.logger.info(f"VERIFIED: Removed containers: {removed_containers}")
            self.logger.info(f"VERIFIED: Removed images: {[img_id[:12] for img_id in removed_images]}")
            
            # Add verification results
            if isinstance(result, dict):
                result['debug_enhanced'] = True
                result['trace_id'] = len(self.debug_tracer.execution_traces)
                result['verified_containers_removed'] = len(removed_containers)
                result['verified_images_removed'] = len(removed_images)
                result['removed_container_names'] = removed_containers
                result['removed_image_ids'] = [img_id[:12] for img_id in removed_images]
            
        except Exception as e:
            self.logger.error(f"Failed to verify Docker optimization: {e}")
            if isinstance(result, dict):
                result['verification_error'] = str(e)
        
        return result
    
    @debug_tracer.trace_function_execution("storage_optimization_comprehensive")
    def _optimize_storage_comprehensive(self, dry_run: bool = False):
        """Debug-enhanced comprehensive storage optimization"""
        self.logger.info(f"STARTING comprehensive storage optimization (dry_run={dry_run}) with debug tracing")
        
        # Track initial state
        initial_disk = psutil.disk_usage('/')
        
        # Track temp directories
        temp_paths = ['/tmp', '/var/tmp']
        temp_files_before = {}
        
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    temp_files = []
                    for root, dirs, files in os.walk(temp_path):
                        temp_files.extend([os.path.join(root, f) for f in files])
                    temp_files_before[temp_path] = len(temp_files)
                except Exception as e:
                    temp_files_before[temp_path] = 0
                    self.logger.warning(f"Could not count files in {temp_path}: {e}")
        
        # Call original method
        result = super()._optimize_storage_comprehensive(dry_run)
        
        # Verify actual changes
        final_disk = psutil.disk_usage('/')
        actual_disk_freed = initial_disk.used - final_disk.used
        
        # Track temp directories after
        temp_files_after = {}
        actual_files_processed = 0
        
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    temp_files = []
                    for root, dirs, files in os.walk(temp_path):
                        temp_files.extend([os.path.join(root, f) for f in files])
                    temp_files_after[temp_path] = len(temp_files)
                    
                    # Calculate files processed
                    files_before = temp_files_before.get(temp_path, 0)
                    files_after = temp_files_after[temp_path]
                    if files_before > files_after:
                        processed = files_before - files_after
                        actual_files_processed += processed
                        self.logger.info(f"VERIFIED: Processed {processed} files in {temp_path}")
                        
                except Exception as e:
                    temp_files_after[temp_path] = 0
                    self.logger.warning(f"Could not count files in {temp_path}: {e}")
        
        # Add comprehensive verification
        if isinstance(result, dict):
            result['debug_enhanced'] = True
            result['trace_id'] = len(self.debug_tracer.execution_traces)
            result['actual_disk_freed_mb'] = actual_disk_freed / (1024 * 1024)
            result['verified_files_processed'] = actual_files_processed
            result['temp_files_before'] = temp_files_before
            result['temp_files_after'] = temp_files_after
            
            # Verify against expected results
            expected_space = result.get('estimated_space_freed_mb', 0)
            actual_space = actual_disk_freed / (1024 * 1024)
            
            result['space_verification'] = {
                'expected_mb': expected_space,
                'actual_mb': actual_space,
                'difference_mb': abs(expected_space - actual_space),
                'verified': abs(expected_space - actual_space) < max(expected_space * 0.5, 1.0)
            }
        
        return result
    
    def _safe_delete(self, filepath: str, dry_run: bool = False) -> bool:
        """Debug-enhanced safe delete with tracing"""
        self.debug_tracer.trace_file_operation("safe_delete", filepath, {"dry_run": dry_run})
        
        if dry_run:
            self.logger.debug(f"DRY RUN: Would delete {filepath}")
            return True
        
        file_size = 0
        if os.path.exists(filepath):
            try:
                file_size = os.path.getsize(filepath)
            except OSError:
                pass
        
        # Call original method
        result = super()._safe_delete(filepath, dry_run)
        
        if result:
            self.logger.info(f"VERIFIED DELETE: {filepath} ({file_size} bytes)")
        else:
            self.logger.warning(f"DELETE FAILED: {filepath}")
        
        return result
    
    # Add new debug endpoints
    def _setup_routes(self):
        """Setup FastAPI routes including debug endpoints"""
        # Call original setup
        super()._setup_routes()
        
        @self.app.get("/debug/trace-report")
        async def get_trace_report():
            """Get current debug trace report"""
            report = self.debug_tracer.generate_debug_report()
            return JSONResponse(content=report)
        
        @self.app.get("/debug/system-snapshot")
        async def take_system_snapshot(label: str = "manual"):
            """Take a system snapshot manually"""
            snapshot = self.debug_tracer.take_system_snapshot(label)
            return JSONResponse(content=snapshot)
        
        @self.app.post("/debug/verify-optimization")
        async def verify_optimization():
            """Verify the last optimization actually worked"""
            if len(self.debug_tracer.execution_traces) < 2:
                return JSONResponse(content={
                    "error": "Need at least 2 execution traces for verification"
                })
            
            # Get the last two traces for comparison
            last_trace = self.debug_tracer.execution_traces[-1]
            verification = self.debug_tracer.verify_actual_changes({
                "memory_freed_mb": last_trace.get('after_snapshot', {}).get('memory', {}).get('used', 0),
                "space_freed_mb": 10  # Example expected space freed
            })
            
            return JSONResponse(content=verification)

if __name__ == "__main__":
    # Use debug-enhanced agent
    agent = DebugEnhancedHardwareResourceOptimizerAgent()
    agent.start()