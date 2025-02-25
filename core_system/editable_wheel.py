"""
SutazAI Editable Wheel Module
-----------------------------
A simplified version of editable wheel utilities for the SutazAI system.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional


class EditableMode(Enum):
    """
    Possible editable installation modes:
    `lenient` (new files automatically added to the package - DEFAULT);
    `strict` (requires a new installation when files are added/removed); or
    `compat` (attempts to emulate `python setup.py develop` - DEPRECATED).
    """
    STRICT = "strict"
    LENIENT = "lenient"
    COMPAT = "compat"  # Deprecated


class EditableWheel:
    """
    Build 'editable' wheel for development.
    This is a simplified version of the setuptools editable_wheel command.
    """
    
    def __init__(
        self, 
        name: str, 
        version: str,
        dist_dir: Optional[str] = None,
        mode: str = "lenient"
    ) -> None:
        """
        Initialize the editable wheel builder.
        
        Args:
            name: Package name
            version: Package version
            dist_dir: Directory to put the wheel in
            mode: Installation mode (lenient, strict, or compat)
        """
        self.name = name
        self.version = version
        self.project_dir = os.curdir
        self.dist_dir = Path(dist_dir or os.path.join(self.project_dir, "dist"))
        self.mode = self._convert_mode(mode)
        self.logger = logging.getLogger(__name__)
    
    def _convert_mode(self, mode: Optional[str]) -> EditableMode:
        """Convert string mode to enum value."""
        if not mode:
            return EditableMode.LENIENT  # default

        try:
            return EditableMode[mode.upper()]
        except KeyError:
            raise ValueError(f"Invalid editable mode: {mode!r}. Try: 'strict'.")
    
    def build(self) -> Path:
        """
        Build an editable wheel.
        
        Returns:
            Path to the created wheel file
        """
        self.dist_dir.mkdir(exist_ok=True)
        
        # Create wheel filename
        tag = "py3-none-any"  # Universal wheel
        build_tag = "0.editable"  # According to PEP 427 needs to start with digit
        archive_name = f"{self.name}-{build_tag}-{tag}.whl"
        wheel_path = Path(self.dist_dir, archive_name)
        
        # Create temporary directories for building
        temp_dir = Path(self.project_dir, "build", f"__editable__.{self.name}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .dist-info directory
        dist_info_dir = Path(temp_dir, f"{self.name}-{self.version}.dist-info")
        dist_info_dir.mkdir(exist_ok=True)
        
        # Create METADATA file
        with open(dist_info_dir / "METADATA", "w") as f:
            f.write("Metadata-Version: 2.1\n")
            f.write(f"Name: {self.name}\n")
            f.write(f"Version: {self.version}\n")
        
        # Create WHEEL file
        with open(dist_info_dir / "WHEEL", "w") as f:
            f.write("Wheel-Version: 1.0\n")
            f.write("Generator: SutazAI\n")
            f.write("Root-Is-Purelib: true\n")
            f.write("Tag: py3-none-any\n")
        
        # Create .pth file for editable install
        pth_file = Path(temp_dir, f"__editable__.{self.name}.pth")
        with open(pth_file, "w") as f:
            f.write(str(Path(self.project_dir).resolve()))
        
        # Create the wheel file
        self._create_wheel_file(wheel_path, temp_dir)
        
        return wheel_path
    
    def _create_wheel_file(self, wheel_path: Path, source_dir: Path) -> None:
        """
        Create a wheel file from the source directory.
        
        Args:
            wheel_path: Path to the wheel file to create
            source_dir: Directory containing the wheel contents
        """
        try:
            # If wheel package is available, use it
            from wheel.wheelfile import WheelFile
            
            with WheelFile(wheel_path, 'w') as wheel_obj:
                for path in source_dir.glob('**/*'):
                    if path.is_file():
                        rel_path = path.relative_to(source_dir)
                        wheel_obj.write(str(path), str(rel_path))
        except ImportError:
            # Fallback to simple zip file
            import zipfile
            
            with zipfile.ZipFile(wheel_path, 'w') as zip_obj:
                for path in source_dir.glob('**/*'):
                    if path.is_file():
                        rel_path = path.relative_to(source_dir)
                        zip_obj.write(str(path), str(rel_path))


def create_editable_wheel(
    name: str, 
    version: str,
    dist_dir: Optional[str] = None,
    mode: str = "lenient"
) -> Path:
    """
    Create an editable wheel.
    
    Args:
        name: Package name
        version: Package version
        dist_dir: Directory to put the wheel in
        mode: Installation mode (lenient, strict, or compat)
        
    Returns:
        Path to the created wheel file
    """
    wheel_builder = EditableWheel(name, version, dist_dir, mode)
    return wheel_builder.build()
