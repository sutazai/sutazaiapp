#!/usr/bin/env python3
"""
Monthly Deep Cleanup for SutazAI
Purpose: Performs comprehensive monthly cleanup and optimization
Usage: python monthly-cleanup.py [--force]
Requirements: Python 3.8+, full agent access
"""

import os
import sys
import json
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonthlyCleanup:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.archive_root = self.project_root / "archive"
        self.report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def cleanup_old_logs(self) -> int:
        """Remove logs older than 30 days"""
        cleaned = 0
        log_dirs = [
            self.project_root / "logs",
            self.project_root / "backend" / "logs",
            self.project_root / "frontend" / "logs"
        ]
        
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
                
            for log_file in log_dir.rglob("*.log*"):
                try:
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff_date:
                        log_file.unlink()
                        cleaned += 1
                        logger.info(f"Removed old log: {log_file}")
                except Exception as e:
                    logger.error(f"Failed to remove {log_file}: {e}")
                    
        return cleaned
    
    def cleanup_old_archives(self) -> int:
        """Remove archives older than 90 days"""
        cleaned = 0
        cutoff_date = datetime.now() - timedelta(days=90)
        
        if self.archive_root.exists():
            for archive in self.archive_root.iterdir():
                if archive.is_dir():
                    try:
                        # Parse date from directory name
                        date_str = archive.name.split('-')[0]
                        archive_date = datetime.strptime(date_str, "%Y%m%d")
                        
                        if archive_date < cutoff_date:
                            shutil.rmtree(archive)
                            cleaned += 1
                            logger.info(f"Removed old archive: {archive}")
                    except Exception as e:
                        # Suppressed exception (was bare except)
                        logger.debug(f"Suppressed exception: {e}")
                        pass
                        
        return cleaned
    
    def optimize_docker_images(self) -> Dict:
        """Clean up unused Docker images and containers"""
        results = {
            "containers_removed": 0,
            "images_removed": 0,
            "space_freed_mb": 0
        }
        
        try:
            # Remove stopped containers
            output = subprocess.run(
                ["docker", "container", "prune", "-f"],
                capture_output=True,
                text=True
            )
            if "Deleted Containers" in output.stdout:
                results["containers_removed"] = output.stdout.count('\n') - 1
                
            # Remove unused images
            output = subprocess.run(
                ["docker", "image", "prune", "-a", "-f"],
                capture_output=True,
                text=True
            )
            if "Total reclaimed space" in output.stdout:
                # Parse space freed
                space_line = [l for l in output.stdout.split('\n') if "Total reclaimed" in l][0]
                space_parts = space_line.split()
                if "GB" in space_line:
                    results["space_freed_mb"] = float(space_parts[-2]) * 1024
                elif "MB" in space_line:
                    results["space_freed_mb"] = float(space_parts[-2])
                    
            # Count removed images
            results["images_removed"] = output.stdout.count("Deleted Images")
            
        except Exception as e:
            logger.error(f"Docker cleanup failed: {e}")
            
        return results
    
    def consolidate_requirements(self) -> int:
        """Consolidate common requirements across Docker images"""
        consolidated = 0
        
        # Find all requirements.txt files
        requirements_files = list(self.project_root.rglob("requirements*.txt"))
        
        # Group by content hash
        content_map = {}
        for req_file in requirements_files:
            try:
                content = req_file.read_text().strip()
                content_hash = hash(content)
                if content_hash not in content_map:
                    content_map[content_hash] = []
                content_map[content_hash].append(req_file)
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
                
        # Find duplicates
        for content_hash, files in content_map.items():
            if len(files) > 1:
                # Keep the first, create symlinks for others
                base_file = files[0]
                for dup_file in files[1:]:
                    try:
                        relative_path = os.path.relpath(base_file, dup_file.parent)
                        dup_file.unlink()
                        dup_file.symlink_to(relative_path)
                        consolidated += 1
                        logger.info(f"Consolidated {dup_file} -> {base_file}")
                    except Exception as e:
                        # Suppressed exception (was bare except)
                        logger.debug(f"Suppressed exception: {e}")
                        pass
                        
        return consolidated
    
    def deep_agent_scan(self) -> Dict:
        """Run all hygiene agents in deep scan mode"""
        results = {}
        
        try:
            # Run comprehensive hygiene check
            output = subprocess.run(
                [sys.executable, str(self.project_root / "scripts" / "hygiene-enforcement-coordinator.py"), 
                 "--all-phases"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )
            
            if output.returncode == 0:
                results["status"] = "success"
                results["phases_completed"] = 3
            else:
                results["status"] = "partial"
                results["error"] = output.stderr
                
        except subprocess.TimeoutExpired:
            results["status"] = "timeout"
            results["error"] = "Deep scan exceeded 30 minute limit"
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def generate_monthly_report(self, results: Dict) -> Path:
        """Generate comprehensive monthly report"""
        report_path = self.project_root / "compliance-reports" / f"monthly-report-{self.report_date}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        report = {
            "type": "monthly_cleanup",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "next_scheduled": (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Update latest monthly report symlink
        latest_link = report_path.parent / "latest-monthly.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(report_path.name)
        
        return report_path
    
    def run(self) -> Dict:
        """Run complete monthly cleanup"""
        logger.info("Starting monthly deep cleanup...")
        
        results = {
            "started": datetime.now().isoformat(),
            "logs_cleaned": 0,
            "archives_cleaned": 0,
            "docker_cleanup": {},
            "requirements_consolidated": 0,
            "deep_scan": {},
            "errors": []
        }
        
        # 1. Clean old logs
        try:
            results["logs_cleaned"] = self.cleanup_old_logs()
        except Exception as e:
            results["errors"].append(f"Log cleanup failed: {e}")
            
        # 2. Clean old archives
        try:
            results["archives_cleaned"] = self.cleanup_old_archives()
        except Exception as e:
            results["errors"].append(f"Archive cleanup failed: {e}")
            
        # 3. Docker cleanup
        try:
            results["docker_cleanup"] = self.optimize_docker_images()
        except Exception as e:
            results["errors"].append(f"Docker cleanup failed: {e}")
            
        # 4. Requirements consolidation
        try:
            results["requirements_consolidated"] = self.consolidate_requirements()
        except Exception as e:
            results["errors"].append(f"Requirements consolidation failed: {e}")
            
        # 5. Deep agent scan
        try:
            results["deep_scan"] = self.deep_agent_scan()
        except Exception as e:
            results["errors"].append(f"Deep scan failed: {e}")
            
        results["completed"] = datetime.now().isoformat()
        
        # Generate report
        report_path = self.generate_monthly_report(results)
        logger.info(f"Monthly cleanup complete. Report: {report_path}")
        
        # Summary log
        logger.info(f"Cleanup summary:")
        logger.info(f"  - Logs cleaned: {results['logs_cleaned']}")
        logger.info(f"  - Archives cleaned: {results['archives_cleaned']}")
        logger.info(f"  - Docker space freed: {results['docker_cleanup'].get('space_freed_mb', 0):.1f} MB")
        logger.info(f"  - Requirements consolidated: {results['requirements_consolidated']}")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Monthly Deep Cleanup")
    parser.add_argument("--force", action="store_true", help="Force cleanup without prompts")
    parser.add_argument("--project-root", default="/opt/sutazaiapp", help="Project root")
    
    args = parser.parse_args()
    
    cleanup = MonthlyCleanup(args.project_root)
    results = cleanup.run()
    
    return 0 if not results.get("errors") else 1

if __name__ == "__main__":
    sys.exit(main())