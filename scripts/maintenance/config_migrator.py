#!/usr/bin/env python3
"""
Configuration Migration Tool
Converts hardcoded values to environment variables
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

logger = logging.getLogger(__name__)

class ConfigMigrator:
    """Migrates hardcoded configuration to environment variables"""
    
    # Patterns to detect and replace
    HARDCODE_PATTERNS = [
        # URLs and addresses
        (r'"http://localhost:(\d+)"', r'os.getenv("SERVICE_URL_{port}", "http://localhost:{port}")'),
        (r'"127\.0\.0\.1:(\d+)"', r'os.getenv("SERVICE_HOST_{port}", "127.0.0.1:{port}")'),
        (r'localhost:(\d+)', r'{host}:{port}'),
        
        # Database connections
        (r'password\s*=\s*["\']([^"\']+)["\']', r'password=os.getenv("DB_PASSWORD", "{value}")'),
        (r'user\s*=\s*["\']([^"\']+)["\']', r'user=os.getenv("DB_USER", "{value}")'),
        
        # API keys and secrets
        (r'api_key\s*=\s*["\']([^"\']+)["\']', r'api_key=os.getenv("API_KEY", "{value}")'),
        (r'secret\s*=\s*["\']([^"\']+)["\']', r'secret=os.getenv("SECRET_KEY", "{value}")'),
        
        # Common service ports
        (r'10010', 'int(os.getenv("BACKEND_PORT", "10010"))'),
        (r'10011', 'int(os.getenv("FRONTEND_PORT", "10011"))'),
        (r'10006', 'int(os.getenv("CONSUL_PORT", "10006"))'),
        (r'10015', 'int(os.getenv("KONG_ADMIN_PORT", "10015"))'),
        (r'10005', 'int(os.getenv("KONG_PROXY_PORT", "10005"))'),
    ]
    
    # Files to exclude from migration
    EXCLUDE_PATTERNS = [
        "node_modules/",
        ".git/",
        "__pycache__/",
        ".pyc",
        ".env",
        "config_migrator.py"
    ]
    
    def __init__(self, root_directory: str, dry_run: bool = True):
        self.root_directory = Path(root_directory)
        self.dry_run = dry_run
        self.changes = []
        self.env_vars = set()
        
    def migrate_directory(self) -> Dict[str, any]:
        """Migrate all files in directory"""
        results = {
            "files_processed": 0,
            "files_changed": 0,
            "changes_made": 0,
            "env_vars_created": set(),
            "errors": []
        }
        
        for file_path in self._get_files_to_process():
            try:
                file_results = self.migrate_file(file_path)
                results["files_processed"] += 1
                
                if file_results["changes_made"] > 0:
                    results["files_changed"] += 1
                    results["changes_made"] += file_results["changes_made"]
                    results["env_vars_created"].update(file_results["env_vars"])
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                
        return results
        
    def migrate_file(self, file_path: Path) -> Dict[str, any]:
        """Migrate a single file"""
        results = {
            "changes_made": 0,
            "env_vars": set()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # Apply all patterns
            for pattern, replacement in self.HARDCODE_PATTERNS:
                content, changes = self._apply_pattern(content, pattern, replacement)
                results["changes_made"] += changes
                
            # If changes were made
            if content != original_content:
                if not self.dry_run:
                    # Add import if needed
                    if "import os" not in content and "os.getenv" in content:
                        content = self._add_os_import(content, file_path)
                        
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
                logger.info(f"{'Would migrate' if self.dry_run else 'Migrated'} {file_path}: {results['changes_made']} changes")
                
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
            raise
            
        return results
        
    def _get_files_to_process(self) -> List[Path]:
        """Get list of files to process"""
        files = []
        
        for pattern in ["**/*.py", "**/*.js", "**/*.ts", "**/*.yml", "**/*.yaml"]:
            for file_path in self.root_directory.glob(pattern):
                if self._should_process_file(file_path):
                    files.append(file_path)
                    
        return files
        
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        file_str = str(file_path)
        
        for exclude in self.EXCLUDE_PATTERNS:
            if exclude in file_str:
                return False
                
        return file_path.is_file()
        
    def _apply_pattern(self, content: str, pattern: str, replacement: str) -> Tuple[str, int]:
        """Apply a single pattern replacement"""
        import re
        
        matches = list(re.finditer(pattern, content))
        if not matches:
            return content, 0
            
        changes = 0
        new_content = content
        
        # Process matches in reverse order to maintain positions
        for match in reversed(matches):
            if self._is_valid_replacement_context(content, match):
                # Extract values from match groups
                if match.groups():
                    groups = match.groups()
                    if len(groups) == 1:
                        # Single group (like port number)
                        if "{port}" in replacement:
                            new_replacement = replacement.format(port=groups[0])
                        elif "{value}" in replacement:
                            new_replacement = replacement.format(value=groups[0])
                        else:
                            new_replacement = replacement
                    else:
                        new_replacement = replacement
                else:
                    new_replacement = replacement
                    
                # Apply replacement
                start, end = match.span()
                new_content = new_content[:start] + new_replacement + new_content[end:]
                changes += 1
                
        return new_content, changes
        
    def _is_valid_replacement_context(self, content: str, match) -> bool:
        """Check if replacement is valid in this context"""
        # Get surrounding context
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end]
        
        # Skip if already an environment variable
        if "os.getenv" in context or "process.env" in context:
            return False
            
        # Skip if in comments
        lines = context.split('\n')
        match_line = None
        for line in lines:
            if match.group() in line:
                match_line = line.strip()
                break
                
        if match_line:
            if match_line.startswith('#') or match_line.startswith('//'):
                return False
                
        return True
        
    def _add_os_import(self, content: str, file_path: Path) -> str:
        """Add os import to Python files"""
        if not file_path.suffix == '.py':
            return content
            
        lines = content.split('\n')
        
        # Find insertion point (after existing imports)
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_end = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
                
        # Insert import
        lines.insert(import_end, "import os")
        return '\n'.join(lines)
        
    def generate_env_file(self, env_file_path: str = ".env.generated"):
        """Generate .env file with detected variables"""
        env_vars = {
            # Service URLs
            "BACKEND_PORT": "10010",
            "FRONTEND_PORT": "10011",
            "CONSUL_PORT": "10006",
            "KONG_ADMIN_PORT": "10015", 
            "KONG_PROXY_PORT": "10005",
            
            # Database
            "DB_PASSWORD": "change_me",
            "DB_USER": "sutazai_user",
            
            # Security
            "API_KEY": "your_api_key_here",
            "SECRET_KEY": "your_secret_key_here",
        }
        
        with open(env_file_path, 'w') as f:
            f.write("# Generated environment variables\n")
            f.write("# Customize these values for your environment\n\n")
            
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
                
        logger.info(f"Generated environment file: {env_file_path}")
        

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Migrate hardcoded configuration to environment variables")
    parser.add_argument("directory", help="Root directory to migrate")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes (overrides dry-run)")
    parser.add_argument("--generate-env", action="store_true", help="Generate .env file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    dry_run = args.dry_run and not args.apply
    
    migrator = ConfigMigrator(args.directory, dry_run=dry_run)
    results = migrator.migrate_directory()
    
    print(f"\nMigration Results:")
    print(f"Files processed: {results['files_processed']}")
    print(f"Files changed: {results['files_changed']}")
    print(f"Total changes: {results['changes_made']}")
    print(f"Environment variables: {len(results['env_vars_created'])}")
    
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
            
    if args.generate_env:
        migrator.generate_env_file()
        
        
if __name__ == "__main__":
    main()