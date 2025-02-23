#!/usr/bin/env python3
import logging
import os
import shutil
import subprocess
from typing import List, Tuple


class DiskCleanupManager:
    def __init__(self, log_file='/var/log/sutazai/disk_cleanup.log'):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Paths to investigate and potentially clean
        self.cleanup_paths = [
            '/tmp',
            '~/.cache',
            '~/Downloads',
            '/var/log',
            '/home/ai/.npm',
            '/home/ai/.docker',
            '/home/ai/.local/share/Trash'
        ]
        
        # File types to target for cleanup
        self.cleanup_extensions = [
            '.log', '.tmp', '.bak', '.old', 
            '.cache', '.backup', '.swp'
        ]

    def get_directory_size(self, path: str) -> float:
        """Calculate total size of a directory in GB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 ** 3)  # Convert to GB

    def find_large_files(self, start_path: str = '/', size_threshold_gb: float = 1.0) -> List[Tuple[str, float]]:
        """Find large files exceeding the size threshold."""
        large_files = []
        try:
            for dirpath, dirnames, filenames in os.walk(start_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        size = os.path.getsize(fp) / (1024 ** 3)  # Convert to GB
                        if size > size_threshold_gb:
                            large_files.append((fp, size))
                    except (OSError, FileNotFoundError):
                        pass
        except Exception as e:
            self.logger.error(f"Error finding large files: {e}")
        
        return sorted(large_files, key=lambda x: x[1], reverse=True)

    def cleanup_old_files(self, days: int = 30):
        """Remove files older than specified days."""
        for path in self.cleanup_paths:
            expanded_path = os.path.expanduser(path)
            if not os.path.exists(expanded_path):
                continue
            
            try:
                subprocess.run([
                    'find', expanded_path, 
                    '-type', 'f', 
                    '-mtime', f'+{days}', 
                    '-delete'
                ], check=True)
                self.logger.info(f"Cleaned up files older than {days} days in {path}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error cleaning {path}: {e}")

    def cleanup_by_extension(self):
        """Remove files with specific extensions."""
        for path in self.cleanup_paths:
            expanded_path = os.path.expanduser(path)
            if not os.path.exists(expanded_path):
                continue
            
            for ext in self.cleanup_extensions:
                try:
                    subprocess.run([
                        'find', expanded_path, 
                        '-type', 'f', 
                        '-name', f'*{ext}', 
                        '-delete'
                    ], check=True)
                    self.logger.info(f"Cleaned up {ext} files in {path}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error cleaning {ext} files in {path}: {e}")

    def docker_cleanup(self):
        """Clean up Docker resources."""
        try:
            # Remove dangling images
            subprocess.run(['docker', 'image', 'prune', '-f'], check=True)
            
            # Remove stopped containers
            subprocess.run(['docker', 'container', 'prune', '-f'], check=True)
            
            # Remove unused volumes
            subprocess.run(['docker', 'volume', 'prune', '-f'], check=True)
            
            self.logger.info("Docker resources cleaned up")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker cleanup error: {e}")

    def generate_cleanup_report(self):
        """Generate a report of cleanup actions."""
        report = {
            'large_files': self.find_large_files(),
            'cleanup_paths': {
                path: self.get_directory_size(os.path.expanduser(path)) 
                for path in self.cleanup_paths if os.path.exists(os.path.expanduser(path))
            }
        }
        
        print("\n🧹 Disk Cleanup Report 🧹")
        
        print("\nLarge Files (>1GB):")
        for file, size in report['large_files'][:10]:  # Top 10 large files
            print(f"  - {file}: {size:.2f} GB")
        
        print("\nDirectory Sizes:")
        for path, size in report['cleanup_paths'].items():
            print(f"  - {path}: {size:.2f} GB")

    def run_cleanup(self):
        """Execute comprehensive cleanup."""
        self.cleanup_old_files()
        self.cleanup_by_extension()
        self.docker_cleanup()
        self.generate_cleanup_report()

def main():
    cleanup_manager = DiskCleanupManager()
    cleanup_manager.run_cleanup()

if __name__ == '__main__':
    main() 