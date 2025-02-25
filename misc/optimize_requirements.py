#!/usr/bin/env python3
"""
SutazAI Requirements Optimizer

This script analyzes, consolidates, and optimizes various requirement files
in the SutazAI project to create a single, comprehensive requirements file
while eliminating conflicts and redundancies.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple

import pkg_resources


@dataclass
class PackageRequirement:
    """Represents a parsed package requirement with name and constraints."""

    name: str
    specifiers: List[Tuple[str, str]] = field(default_factory=list)
    extras: Set[str] = field(default_factory=set)

    def __str__(self) -> str:
        """Convert the requirement back to a string representation."""
        result = self.name
        if self.extras:
            result += "[" + ",".join(sorted(self.extras)) + "]"
        if self.specifiers:
            specs = sorted(self.specifiers, key=lambda x: (x[1], x[0]))
            result += "".join(f"{op}{version}" for op, version in specs)
        return result


class RequirementsOptimizer:
    """Analyzes and optimizes Python package requirements."""

    def __init__(self, root_dir: str):
        """Initialize the requirements optimizer with project root directory."""
        self.root_dir = root_dir
        self.parsed_requirements: Dict[str, PackageRequirement] = {}
        self.requirement_files: List[str] = []
        self.output_file = os.path.join(root_dir, "requirements.txt")
        self.backup_file = os.path.join(
            root_dir,
            f"requirements.txt.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    def find_requirement_files(self) -> None:
        """Find all requirement files in the project."""
        req_file_pattern = re.compile(r"^requirements.*\.txt$")
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if req_file_pattern.match(file) and "venv" not in root:
                    full_path = os.path.join(root, file)
                    self.requirement_files.append(full_path)

        print(f"Found {len(self.requirement_files)} requirement files:")
        for req_file in self.requirement_files:
            print(f"  - {os.path.relpath(req_file, self.root_dir)}")

    def parse_requirements(self) -> None:
        """Parse all requirements from the found files."""
        for req_file in self.requirement_files:
            try:
                with open(req_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    # Skip comments and empty lines
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Handle line continuations
                    if line.endswith("\\"):
                        continue

                    # Skip non-package lines (like -e or -r)
                    if line.startswith(("-e", "-r", "--")):
                        continue

                    try:
                        req = pkg_resources.Requirement.parse(line)

                        # Create or update the package requirement
                        if req.name.lower() not in self.parsed_requirements:
                            package_req = PackageRequirement(
                                name=req.name.lower()
                            )
                            self.parsed_requirements[req.name.lower()] = (
                                package_req
                            )
                        else:
                            package_req = self.parsed_requirements[
                                req.name.lower()
                            ]

                        # Add extras
                        package_req.extras.update(req.extras)

                        # Add version specifiers
                        for spec in req.specs:
                            if spec not in package_req.specifiers:
                                package_req.specifiers.append(spec)

                    except Exception as e:
                        print(
                            f"Warning: Could not parse {line} from {req_file}: {e}"
                        )

            except Exception as e:
                print(f"Error processing {req_file}: {e}")

    def resolve_conflicts(self) -> None:
        """Resolve version conflicts in the parsed requirements."""
        for pkg_name, pkg_req in self.parsed_requirements.items():
            spec_by_op = {}

            # Group specs by operator
            for op, version in pkg_req.specifiers:
                if op not in spec_by_op:
                    spec_by_op[op] = []
                spec_by_op[op].append(version)

            # Simplify equivalent constraints
            new_specs = []

            # Handle >= and >
            if ">=" in spec_by_op:
                new_specs.append((">=", max(spec_by_op[">="])))
            elif ">" in spec_by_op:
                new_specs.append((">", max(spec_by_op[">"])))

            # Handle <= and <
            if "<=" in spec_by_op:
                new_specs.append(("<=", min(spec_by_op["<="])))
            elif "<" in spec_by_op:
                new_specs.append(("<", min(spec_by_op["<"])))

            # Keep == and ~= as they are special
            if "==" in spec_by_op:
                # If multiple exact versions, keep the newest
                newest_version = max(spec_by_op["=="])
                new_specs.append(("==", newest_version))

            if "~=" in spec_by_op:
                for version in spec_by_op["~="]:
                    new_specs.append(("~=", version))

            pkg_req.specifiers = new_specs

    def generate_optimized_requirements(self) -> None:
        """Generate an optimized requirements file."""
        if os.path.exists(self.output_file):
            # Backup the existing file
            print(
                f"Backing up existing requirements.txt to {self.backup_file}"
            )
            os.rename(self.output_file, self.backup_file)

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("# SutazAI Optimized Requirements\n")
            f.write(
                f"# Generated by RequirementsOptimizer on "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                "# This file consolidates all requirements from multiple files\n"
            )
            f.write("\n")

            # Sort packages alphabetically
            sorted_packages = sorted(
                self.parsed_requirements.values(), key=lambda x: x.name
            )

            for pkg_req in sorted_packages:
                f.write(f"{str(pkg_req)}\n")

        print(f"✅ Generated optimized requirements file: {self.output_file}")
        print(
            f"   Consolidated {len(self.parsed_requirements)} packages "
            f"from {len(self.requirement_files)} files."
        )
        print(f"   Original requirements backed up to {self.backup_file}")


def main():
    """Main function to run the requirements optimizer."""
    # Get the project root directory
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Optimizing requirements for project at: {root_dir}")
    optimizer = RequirementsOptimizer(root_dir)

    # Find and parse requirements
    optimizer.find_requirement_files()
    optimizer.parse_requirements()

    # Resolve conflicts and generate optimized file
    optimizer.resolve_conflicts()
    optimizer.generate_optimized_requirements()

    print("\nRequirements optimization complete! 🎉")


if __name__ == "__main__":
    main()
