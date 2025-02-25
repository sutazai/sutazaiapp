"""
SutazAI Distribution Information Module
--------------------------------------
A simplified version of distribution information utilities for the SutazAI system.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union


class DistributionInfo:
    """Class for handling distribution information."""
    
    def __init__(self, name: str, version: str) -> None:
        """
        Initialize distribution information.
        
        Args:
            name: Package name
            version: Package version
        """
        self.name = name
        self.version = version
        self.metadata: Dict[str, str] = {}
        
    def get_dist_info_dir(self, base_dir: Union[str, Path]) -> Path:
        """
        Get the path to the .dist-info directory.
        
        Args:
            base_dir: Base directory
            
        Returns:
            Path to the .dist-info directory
        """
        base_path = Path(base_dir)
        return base_path / f"{self.name}-{self.version}.dist-info"
    
    def create_dist_info(self, base_dir: Union[str, Path]) -> Path:
        """
        Create a .dist-info directory.
        
        Args:
            base_dir: Base directory
            
        Returns:
            Path to the created .dist-info directory
        """
        dist_info_dir = self.get_dist_info_dir(base_dir)
        
        # Create the directory if it doesn't exist
        os.makedirs(dist_info_dir, exist_ok=True)
        
        # Create METADATA file
        with open(dist_info_dir / "METADATA", "w") as f:
            f.write(f"Metadata-Version: 2.1\n")
            f.write(f"Name: {self.name}\n")
            f.write(f"Version: {self.version}\n")
            
            for key, value in self.metadata.items():
                f.write(f"{key}: {value}\n")
        
        # Create WHEEL file
        with open(dist_info_dir / "WHEEL", "w") as f:
            f.write("Wheel-Version: 1.0\n")
            f.write("Generator: SutazAI\n")
            f.write("Root-Is-Purelib: true\n")
            f.write("Tag: py3-none-any\n")
        
        return dist_info_dir
    
    def add_metadata(self, key: str, value: str) -> None:
        """
        Add metadata to the distribution information.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    @staticmethod
    def backup_directory(dir_path: Union[str, Path]) -> Path:
        """
        Create a backup of a directory.
        
        Args:
            dir_path: Directory to backup
            
        Returns:
            Path to the backup directory
        """
        dir_path = Path(dir_path)
        backup_path = dir_path.with_suffix(".bak")
        
        if backup_path.exists():
            shutil.rmtree(backup_path)
        
        shutil.copytree(dir_path, backup_path)
        return backup_path
    
    @staticmethod
    def restore_from_backup(backup_path: Union[str, Path], target_path: Union[str, Path]) -> None:
        """
        Restore a directory from a backup.
        
        Args:
            backup_path: Path to the backup directory
            target_path: Path to restore to
        """
        backup_path = Path(backup_path)
        target_path = Path(target_path)
        
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.copytree(backup_path, target_path)


def get_distribution_info(name: str, version: str) -> DistributionInfo:
    """
    Get distribution information.
    
    Args:
        name: Package name
        version: Package version
        
    Returns:
        Distribution information
    """
    return DistributionInfo(name, version)
