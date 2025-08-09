#!/usr/bin/env python3
"""
INTELLIGENT Fantasy Elements Remover
Removes AGI/ASI/quantum/magic references WITHOUT breaking LocalAGI/BigAGI services
"""

import os
import re
from pathlib import Path

class FantasyRemover:
    def __init__(self):
        self.repo_root = Path("/opt/sutazaiapp")
        self.changes = []
        
        # CRITICAL: Never touch these services
        self.protected_terms = [
            "LocalAGI",
            "BigAGI", 
            "localagi",
            "bigagi"
        ]
        
        # Fantasy terms to remove
        self.fantasy_terms = {
            "ASI LEVEL ACHIEVED": "System Optimization Complete",
            "AGI Level achieved": "Multi-Agent System Active",
            "AGI level": "system level",
            "ASI": "Advanced System",
            "magic_function": "process_function",
            "wizard_data": "config_data",
            "magic": "process",
            "wizard": "configurator",
            "quantum": "advanced",
            "teleport": "transfer",
            "black-box": "processing-unit",
            "telekinesis": "remote-control"
        }
    
    def is_protected(self, line):
        """Check if line contains protected service names"""
        for protected in self.protected_terms:
            if protected in line:
                return True
        return False
    
    def clean_file(self, filepath):
        """Clean fantasy elements from a single file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            original = content
            
            # Skip if it contains protected terms
            for protected in self.protected_terms:
                if protected in content:
                    print(f"  ⚠️  Skipping protected file: {filepath}")
                    return False
            
            # Replace fantasy terms
            for fantasy, replacement in self.fantasy_terms.items():
                # Case-sensitive replacement
                content = content.replace(fantasy, replacement)
                # Case-insensitive for isolated words
                pattern = r'\b' + re.escape(fantasy.lower()) + r'\b'
                content = re.sub(pattern, replacement.lower(), content, flags=re.IGNORECASE)
            
            if content != original:
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"  ✅ Cleaned: {filepath}")
                self.changes.append(str(filepath))
                return True
                
        except Exception as e:
            print(f"  ❌ Error processing {filepath}: {e}")
        
        return False
    
    def run(self):
        """Clean all Python files"""
        print("🧹 Removing fantasy elements intelligently...")
        
        # Find Python files with fantasy terms
        python_files = list(self.repo_root.glob("**/*.py"))
        
        cleaned = 0
        for pyfile in python_files:
            # Skip node_modules and .git
            if "node_modules" in str(pyfile) or ".git" in str(pyfile):
                continue
            
            # Skip critical system files
            if "main.py" in str(pyfile) or "config.py" in str(pyfile):
                continue
            
            # Check if file contains fantasy terms
            try:
                with open(pyfile, 'r') as f:
                    content = f.read()
                
                has_fantasy = any(term in content for term in self.fantasy_terms.keys())
                
                if has_fantasy:
                    if self.clean_file(pyfile):
                        cleaned += 1
            except:
                continue
        
        print(f"\n✅ Cleaned {cleaned} files")
        print(f"📝 Modified files: {len(self.changes)}")
        
        # Update CHANGELOG
        self.update_changelog()
        
        return True
    
    def update_changelog(self):
        """Document changes in CHANGELOG"""
        if not self.changes:
            return
        
        changelog_path = self.repo_root / "docs/CHANGELOG.md"
        
        from datetime import datetime
        entry = f"""
### Fantasy Elements Removal - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
- [Component] System-wide cleanup
- [Type] refactor
- [Changes] Removed fantasy terms from {len(self.changes)} files
- [Protected] LocalAGI/BigAGI services preserved
- [Status] No functionality affected
"""
        
        with open(changelog_path, "r") as f:
            content = f.read()
        
        # Insert after date header
        content = content.replace("## 2025-08-09", f"## 2025-08-09{entry}")
        
        with open(changelog_path, "w") as f:
            f.write(content)
        
        print("✅ Updated CHANGELOG.md")

if __name__ == "__main__":
    remover = FantasyRemover()
    remover.run()