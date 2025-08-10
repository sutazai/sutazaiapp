#!/usr/bin/env python3
"""
ULTRA-MASSIVE DOCKERFILE CONSOLIDATION SCRIPT
Author: Ultra System Architect
Date: August 10, 2025
Purpose: Migrate 587 Dockerfiles to use consolidated base images
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import json

class DockerfileMigrator:
    def __init__(self):
        self.base_dir = Path("/opt/sutazaiapp")
        self.archive_dir = self.base_dir / "archive" / "dockerfiles" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            "total_dockerfiles": 0,
            "python_migrated": 0,
            "nodejs_migrated": 0,
            "skipped": 0,
            "errors": 0,
            "already_migrated": 0
        }
        
    def find_all_dockerfiles(self):
        """Find all Dockerfiles in the codebase"""
        dockerfiles = []
        for dockerfile in self.base_dir.rglob("Dockerfile*"):
            if dockerfile.is_file():
                dockerfiles.append(dockerfile)
        return dockerfiles
    
    def detect_dockerfile_type(self, dockerfile_path):
        """Detect if Dockerfile is Python, Node.js, or other"""
        try:
            content = dockerfile_path.read_text()
            
            # Skip if already using base images
            if "python-agent-master" in content or "nodejs-agent-master" in content:
                return "already_migrated"
            
            # Check base image
            from_match = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE)
            if not from_match:
                return "unknown"
            
            base_image = from_match.group(1).lower()
            
            if "python" in base_image:
                return "python"
            elif "node" in base_image:
                return "nodejs"
            elif any(x in base_image for x in ["alpine", "ubuntu", "debian"]):
                # Check if Python or Node is installed
                if "python" in content.lower() or "pip" in content.lower():
                    return "python"
                elif "node" in content.lower() or "npm" in content.lower():
                    return "nodejs"
            
            return "other"
        except Exception as e:
            print(f"Error analyzing {dockerfile_path}: {e}")
            return "error"
    
    def migrate_python_dockerfile(self, dockerfile_path):
        """Migrate Python Dockerfile to use base image"""
        try:
            content = dockerfile_path.read_text()
            
            # Extract custom requirements if any
            requirements_match = re.search(r"COPY.*requirements.*\.txt", content)
            custom_requirements = requirements_match.group(0) if requirements_match else None
            
            # Extract application code copy commands
            app_copies = re.findall(r"^COPY\s+(?!.*requirements).*$", content, re.MULTILINE)
            
            # Extract custom environment variables
            env_vars = re.findall(r"^ENV\s+.*$", content, re.MULTILINE)
            
            # Extract EXPOSE port
            expose_match = re.search(r"^EXPOSE\s+(\d+)", content, re.MULTILINE)
            port = expose_match.group(1) if expose_match else "8080"
            
            # Extract CMD or ENTRYPOINT
            cmd_match = re.search(r"^(CMD|ENTRYPOINT)\s+(.*)$", content, re.MULTILINE)
            cmd = cmd_match.group(0) if cmd_match else 'CMD ["python", "-u", "app.py"]'
            
            # Build new Dockerfile
            new_dockerfile = f"""# Migrated to use consolidated base image
# Original archived at: {self.archive_dir.relative_to(self.base_dir)}/{dockerfile_path.name}
# Migration date: {datetime.now().isoformat()}

FROM sutazai-python-agent-master:latest

# Service-specific configuration
ENV SERVICE_PORT={port}

# Copy application code
WORKDIR /app
"""
            
            # Add app copies
            for copy_cmd in app_copies:
                new_dockerfile += f"{copy_cmd}\n"
            
            # Add custom requirements if exist
            if custom_requirements:
                new_dockerfile += f"\n# Install additional requirements\n"
                new_dockerfile += f"{custom_requirements}\n"
                new_dockerfile += f"RUN pip install --no-cache-dir -r requirements.txt\n"
            
            # Add custom environment variables
            if env_vars:
                new_dockerfile += f"\n# Custom environment variables\n"
                for env in env_vars:
                    if "SERVICE_PORT" not in env:  # Skip if already set
                        new_dockerfile += f"{env}\n"
            
            # Add expose and command
            new_dockerfile += f"\nEXPOSE {port}\n"
            new_dockerfile += f"{cmd}\n"
            
            return new_dockerfile
            
        except Exception as e:
            print(f"Error migrating Python Dockerfile {dockerfile_path}: {e}")
            return None
    
    def migrate_nodejs_dockerfile(self, dockerfile_path):
        """Migrate Node.js Dockerfile to use base image"""
        try:
            content = dockerfile_path.read_text()
            
            # Extract package.json copy
            package_match = re.search(r"COPY.*package.*\.json", content)
            
            # Extract application code copy commands
            app_copies = re.findall(r"^COPY\s+(?!.*package).*$", content, re.MULTILINE)
            
            # Extract EXPOSE port
            expose_match = re.search(r"^EXPOSE\s+(\d+)", content, re.MULTILINE)
            port = expose_match.group(1) if expose_match else "3000"
            
            # Extract CMD or ENTRYPOINT
            cmd_match = re.search(r"^(CMD|ENTRYPOINT)\s+(.*)$", content, re.MULTILINE)
            cmd = cmd_match.group(0) if cmd_match else 'CMD ["node", "index.js"]'
            
            # Build new Dockerfile
            new_dockerfile = f"""# Migrated to use consolidated base image
# Original archived at: {self.archive_dir.relative_to(self.base_dir)}/{dockerfile_path.name}
# Migration date: {datetime.now().isoformat()}

FROM sutazai-nodejs-agent-master:latest

# Service-specific configuration
ENV SERVICE_PORT={port}

# Copy application code
WORKDIR /app
"""
            
            # Add package.json if exists
            if package_match:
                new_dockerfile += f"\n# Copy service-specific package.json\n"
                new_dockerfile += f"{package_match.group(0)}\n"
                new_dockerfile += f"RUN npm ci --only=production && npm cache clean --force\n"
            
            # Add app copies
            for copy_cmd in app_copies:
                new_dockerfile += f"{copy_cmd}\n"
            
            # Add expose and command
            new_dockerfile += f"\nEXPOSE {port}\n"
            new_dockerfile += f"{cmd}\n"
            
            return new_dockerfile
            
        except Exception as e:
            print(f"Error migrating Node.js Dockerfile {dockerfile_path}: {e}")
            return None
    
    def archive_original(self, dockerfile_path):
        """Archive original Dockerfile"""
        try:
            # Create archive directory structure
            relative_path = dockerfile_path.relative_to(self.base_dir)
            archive_path = self.archive_dir / relative_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy original file
            shutil.copy2(dockerfile_path, archive_path)
            return True
        except Exception as e:
            print(f"Error archiving {dockerfile_path}: {e}")
            return False
    
    def migrate_dockerfile(self, dockerfile_path):
        """Migrate a single Dockerfile"""
        dockerfile_type = self.detect_dockerfile_type(dockerfile_path)
        
        if dockerfile_type == "already_migrated":
            self.stats["already_migrated"] += 1
            return True
        
        if dockerfile_type == "error":
            self.stats["errors"] += 1
            return False
        
        if dockerfile_type not in ["python", "nodejs"]:
            self.stats["skipped"] += 1
            return True
        
        # Archive original
        if not self.archive_original(dockerfile_path):
            self.stats["errors"] += 1
            return False
        
        # Migrate based on type
        if dockerfile_type == "python":
            new_content = self.migrate_python_dockerfile(dockerfile_path)
            if new_content:
                dockerfile_path.write_text(new_content)
                self.stats["python_migrated"] += 1
                return True
        elif dockerfile_type == "nodejs":
            new_content = self.migrate_nodejs_dockerfile(dockerfile_path)
            if new_content:
                dockerfile_path.write_text(new_content)
                self.stats["nodejs_migrated"] += 1
                return True
        
        self.stats["errors"] += 1
        return False
    
    def run_migration(self):
        """Run the full migration process"""
        print("=" * 80)
        print("ULTRA-MASSIVE DOCKERFILE CONSOLIDATION")
        print("=" * 80)
        
        # Find all Dockerfiles
        dockerfiles = self.find_all_dockerfiles()
        self.stats["total_dockerfiles"] = len(dockerfiles)
        
        print(f"Found {len(dockerfiles)} Dockerfiles to process")
        print(f"Archive directory: {self.archive_dir}")
        print("-" * 80)
        
        # Process each Dockerfile
        for i, dockerfile in enumerate(dockerfiles, 1):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(dockerfiles)} processed...")
            
            # Skip base images themselves
            if "docker/base" in str(dockerfile):
                continue
            
            self.migrate_dockerfile(dockerfile)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate migration report"""
        print("\n" + "=" * 80)
        print("MIGRATION COMPLETE - FINAL REPORT")
        print("=" * 80)
        
        print(f"Total Dockerfiles found: {self.stats['total_dockerfiles']}")
        print(f"Already using base images: {self.stats['already_migrated']}")
        print(f"Python Dockerfiles migrated: {self.stats['python_migrated']}")
        print(f"Node.js Dockerfiles migrated: {self.stats['nodejs_migrated']}")
        print(f"Skipped (other types): {self.stats['skipped']}")
        print(f"Errors encountered: {self.stats['errors']}")
        
        total_migrated = self.stats['python_migrated'] + self.stats['nodejs_migrated']
        if self.stats['total_dockerfiles'] > 0:
            reduction_pct = (total_migrated / self.stats['total_dockerfiles']) * 100
            print(f"\nConsolidation achieved: {reduction_pct:.1f}% of Dockerfiles")
        
        # Save report to file
        report_path = self.base_dir / "reports" / "dockerfile-dedup" / f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
                "archive_dir": str(self.archive_dir)
            }, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        print(f"Original Dockerfiles archived at: {self.archive_dir}")

if __name__ == "__main__":
    migrator = DockerfileMigrator()
    migrator.run_migration()