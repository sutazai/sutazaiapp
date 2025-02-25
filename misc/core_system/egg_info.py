"""
SutazAI Egg Info Module
-----------------------
A simplified version of egg_info utilities for the SutazAI system.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any


class EggInfo:
    """
    Class for creating and managing .egg-info directories.
    This is a simplified version that doesn't rely on setuptools internals.
    """

    def __init__(
        self, name: str, version: str, base_dir: Optional[str] = None
    ) -> None:
        """
        Initialize egg info.

        Args:
            name: Package name
            version: Package version
            base_dir: Base directory for the egg-info
        """
        self.name = name
        self.version = version
        self.base_dir = base_dir or os.curdir
        self.tag_build: Optional[str] = None
        self.tag_date: bool = False
        self.metadata: Dict[str, str] = {}

    def get_egg_info_dir(self) -> Path:
        """
        Get the path to the .egg-info directory.

        Returns:
            Path to the .egg-info directory
        """
        safe_name = self.name.replace("-", "_")
        return Path(self.base_dir) / f"{safe_name}.egg-info"

    def create_egg_info(self) -> Path:
        """
        Create an .egg-info directory.

        Returns:
            Path to the created .egg-info directory
        """
        egg_info_dir = self.get_egg_info_dir()

        # Create the directory if it doesn't exist
        os.makedirs(egg_info_dir, exist_ok=True)

        # Create PKG-INFO file
        self._write_pkg_info(egg_info_dir)

        # Create SOURCES.txt file
        self._write_sources_txt(egg_info_dir)

        # Create top_level.txt file
        self._write_top_level_txt(egg_info_dir)

        # Create dependency_links.txt file
        self._write_dependency_links_txt(egg_info_dir)

        # Create requires.txt file
        self._write_requires_txt(egg_info_dir)

        return egg_info_dir

    def _write_pkg_info(self, egg_info_dir: Path) -> None:
        """
        Write the PKG-INFO file.

        Args:
            egg_info_dir: Path to the .egg-info directory
        """
        pkg_info_path = egg_info_dir / "PKG-INFO"

        with open(pkg_info_path, "w") as f:
            f.write(f"Metadata-Version: 2.1\n")
            f.write(f"Name: {self.name}\n")
            f.write(f"Version: {self.tagged_version()}\n")

            for key, value in self.metadata.items():
                f.write(f"{key}: {value}\n")

    def _write_sources_txt(self, egg_info_dir: Path) -> None:
        """
        Write the SOURCES.txt file.

        Args:
            egg_info_dir: Path to the .egg-info directory
        """
        sources_path = egg_info_dir / "SOURCES.txt"

        with open(sources_path, "w") as f:
            # Add basic source files
            f.write("setup.py\n")
            f.write("README.md\n")

            # Add Python files
            for path in Path(self.base_dir).glob("**/*.py"):
                if not str(path).startswith(str(egg_info_dir)):
                    rel_path = path.relative_to(self.base_dir)
                    f.write(f"{rel_path}\n")

    def _write_top_level_txt(self, egg_info_dir: Path) -> None:
        """
        Write the top_level.txt file.

        Args:
            egg_info_dir: Path to the .egg-info directory
        """
        top_level_path = egg_info_dir / "top_level.txt"

        with open(top_level_path, "w") as f:
            # Assume the package name is the top-level module
            f.write(f"{self.name.replace('-', '_')}\n")

    def _write_dependency_links_txt(self, egg_info_dir: Path) -> None:
        """
        Write the dependency_links.txt file.

        Args:
            egg_info_dir: Path to the .egg-info directory
        """
        dependency_links_path = egg_info_dir / "dependency_links.txt"

        with open(dependency_links_path, "w") as f:
            # Empty file by default
            pass

    def _write_requires_txt(self, egg_info_dir: Path) -> None:
        """
        Write the requires.txt file.

        Args:
            egg_info_dir: Path to the .egg-info directory
        """
        requires_path = egg_info_dir / "requires.txt"

        with open(requires_path, "w") as f:
            # Empty file by default
            pass

    def add_metadata(self, key: str, value: str) -> None:
        """
        Add metadata to the egg info.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def tagged_version(self) -> str:
        """
        Get the tagged version.

        Returns:
            Tagged version string
        """
        version = self.version

        if self.tag_build:
            version += self.tag_build

        if self.tag_date:
            version += time.strftime("%Y%m%d")

        return version


class FileList:
    """
    Class for managing file lists for packaging.
    """

    def __init__(self) -> None:
        """Initialize the file list."""
        self.files: Set[str] = set()

    def include(self, pattern: str) -> None:
        """
        Include files matching the pattern.

        Args:
            pattern: Glob pattern
        """
        for path in Path().glob(pattern):
            if path.is_file():
                self.files.add(str(path))

    def exclude(self, pattern: str) -> None:
        """
        Exclude files matching the pattern.

        Args:
            pattern: Glob pattern
        """
        for path in Path().glob(pattern):
            if str(path) in self.files:
                self.files.remove(str(path))

    def recursive_include(self, dir_path: str, pattern: str) -> None:
        """
        Recursively include files matching the pattern in the directory.

        Args:
            dir_path: Directory path
            pattern: Glob pattern
        """
        for path in Path(dir_path).glob(f"**/{pattern}"):
            if path.is_file():
                self.files.add(str(path))

    def recursive_exclude(self, dir_path: str, pattern: str) -> None:
        """
        Recursively exclude files matching the pattern in the directory.

        Args:
            dir_path: Directory path
            pattern: Glob pattern
        """
        for path in Path(dir_path).glob(f"**/{pattern}"):
            if str(path) in self.files:
                self.files.remove(str(path))

    def get_files(self) -> List[str]:
        """
        Get the list of files.

        Returns:
            List of files
        """
        return sorted(self.files)


def create_egg_info(
    name: str, version: str, base_dir: Optional[str] = None
) -> Path:
    """
    Create an .egg-info directory.

    Args:
        name: Package name
        version: Package version
        base_dir: Base directory for the egg-info

    Returns:
        Path to the created .egg-info directory
    """
    egg_info = EggInfo(name, version, base_dir)
    return egg_info.create_egg_info()
