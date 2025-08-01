#!/usr/bin/env python3
"""
Validate all agent configuration files in .claude/agents directory
"""

import os
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

class AgentConfigValidator:
    def __init__(self):
        self.base_path = Path("/opt/sutazaiapp")
        self.agents_path = self.base_path / ".claude" / "agents"
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
        self.agent_registry = {}
        
    def load_agent_file(self, filepath: Path) -> Tuple[Dict, str]:
        """Load and parse an agent markdown file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split frontmatter and content
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                    return frontmatter, body
            
            return None, None
            
        except Exception as e:
            return None, str(e)
    
    def validate_frontmatter(self, agent_id: str, frontmatter: Dict) -> List[str]:
        """Validate agent frontmatter"""
        errors = []
        
        # Check required fields
        required_fields = ['name', 'description', 'model']
        for field in required_fields:
            if field not in frontmatter:
                errors.append(f"Missing required field: {field}")
        
        # Validate name matches filename
        if frontmatter.get('name') != agent_id:
            errors.append(f"Name mismatch: {frontmatter.get('name')} != {agent_id}")
        
        # Validate description format
        desc = frontmatter.get('description', '')
        if not desc.strip():
            errors.append("Empty description")
        elif not desc.startswith("Use this agent when you need to:"):
            errors.append("Description should start with 'Use this agent when you need to:'")
        
        # Validate model
        valid_models = ['sonnet', 'opus', 'haiku']
        if frontmatter.get('model') not in valid_models:
            errors.append(f"Invalid model: {frontmatter.get('model')}. Must be one of: {valid_models}")
            
        return errors
    
    def validate_content(self, content: str) -> List[str]:
        """Validate agent content"""
        warnings = []
        
        # Check for required sections
        required_sections = [
            "Core Responsibilities",
            "Technical Implementation",
            "Integration Points"
        ]
        
        for section in required_sections:
            if section not in content:
                warnings.append(f"Missing section: {section}")
        
        # Check content length
        if len(content) < 500:
            warnings.append("Content seems too short (< 500 chars)")
        
        # Check for system prompt
        if not content.strip().startswith("You are"):
            warnings.append("Content should start with system prompt 'You are...'")
            
        return warnings
    
    def extract_capabilities(self, description: str) -> List[str]:
        """Extract capabilities from description"""
        capabilities = []
        
        lines = description.split('\\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') and len(line) > 2:
                capabilities.append(line[2:])
        
        return capabilities
    
    def test_agent(self, agent_file: Path) -> Dict:
        """Test a single agent file"""
        agent_id = agent_file.stem
        result = {
            "agent_id": agent_id,
            "file": str(agent_file),
            "status": "unknown",
            "errors": [],
            "warnings": [],
            "capabilities": []
        }
        
        # Load agent file
        frontmatter, content = self.load_agent_file(agent_file)
        
        if frontmatter is None:
            result["status"] = "failed"
            result["errors"].append(f"Failed to load agent file: {content}")
            return result
        
        # Validate frontmatter
        errors = self.validate_frontmatter(agent_id, frontmatter)
        if errors:
            result["errors"].extend(errors)
        
        # Validate content
        warnings = self.validate_content(content)
        if warnings:
            result["warnings"].extend(warnings)
        
        # Extract capabilities
        if 'description' in frontmatter:
            capabilities = self.extract_capabilities(frontmatter['description'])
            result["capabilities"] = capabilities
            
            # Store in registry
            self.agent_registry[agent_id] = {
                "name": frontmatter.get('name'),
                "model": frontmatter.get('model'),
                "capabilities": capabilities,
                "capability_count": len(capabilities)
            }
        
        # Determine status
        if result["errors"]:
            result["status"] = "failed"
        elif result["warnings"]:
            result["status"] = "warning"
        else:
            result["status"] = "passed"
            
        return result
    
    def validate_all_agents(self):
        """Validate all agent files"""
        print("🔍 Validating all agent configurations...")
        print("=" * 80)
        
        agent_files = sorted(self.agents_path.glob("*.md"))
        
        for agent_file in agent_files:
            result = self.test_agent(agent_file)
            
            # Display result
            status_icon = {
                "passed": "✅",
                "warning": "⚠️",
                "failed": "❌"
            }.get(result["status"], "❓")
            
            caps_count = len(result["capabilities"])
            print(f"{status_icon} {result['agent_id']:<35} ({caps_count} capabilities)")
            
            if result["errors"]:
                for error in result["errors"]:
                    print(f"   ❌ {error}")
                    
            if result["warnings"]:
                for warning in result["warnings"]:
                    print(f"   ⚠️  {warning}")
            
            # Track results
            if result["status"] == "passed":
                self.results["passed"].append(result["agent_id"])
            elif result["status"] == "failed":
                self.results["failed"].append(result["agent_id"])
            else:
                self.results["warnings"].append(result["agent_id"])
        
        print("=" * 80)
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary"""
        total = len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["warnings"])
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total agents validated: {total}")
        print(f"   ✅ Passed: {len(self.results['passed'])}")
        print(f"   ⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"   ❌ Failed: {len(self.results['failed'])}")
        
        # Show failed agents
        if self.results["failed"]:
            print(f"\n❌ Failed agents:")
            for agent in self.results["failed"]:
                print(f"   - {agent}")
        
        # Show top agents by capabilities
        print(f"\n🏆 Top agents by capability count:")
        sorted_agents = sorted(
            self.agent_registry.items(), 
            key=lambda x: x[1]['capability_count'], 
            reverse=True
        )[:10]
        
        for agent_id, info in sorted_agents:
            print(f"   - {agent_id:<35} ({info['capability_count']} capabilities)")
        
        # Create detailed report
        self.create_report()
    
    def create_report(self):
        """Create a detailed validation report"""
        report_path = self.base_path / "agent_validation_report.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["warnings"]),
                "passed": len(self.results["passed"]),
                "warnings": len(self.results["warnings"]),
                "failed": len(self.results["failed"])
            },
            "results": self.results,
            "agent_registry": self.agent_registry
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Also create a capability matrix
        self.create_capability_matrix()
    
    def create_capability_matrix(self):
        """Create agent capability matrix"""
        matrix_path = self.base_path / "agent_capability_matrix.md"
        
        with open(matrix_path, 'w') as f:
            f.write("# Agent Capability Matrix\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group agents by model
            by_model = defaultdict(list)
            for agent_id, info in self.agent_registry.items():
                by_model[info.get('model', 'unknown')].append((agent_id, info))
            
            for model in sorted(by_model.keys()):
                f.write(f"\n## {model.upper()} Model Agents\n\n")
                
                agents = sorted(by_model[model], key=lambda x: x[0])
                for agent_id, info in agents:
                    f.write(f"### {agent_id}\n")
                    f.write(f"**Capabilities:** {info['capability_count']}\n\n")
                    
                    # Write first 5 capabilities
                    for i, cap in enumerate(info['capabilities'][:5]):
                        f.write(f"- {cap}\n")
                    if len(info['capabilities']) > 5:
                        f.write(f"- ... and {len(info['capabilities']) - 5} more\n")
                    f.write("\n")
        
        print(f"📊 Capability matrix saved to: {matrix_path}")

def main():
    validator = AgentConfigValidator()
    validator.validate_all_agents()
    
    # Exit with error code if any failures
    if validator.results["failed"]:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()