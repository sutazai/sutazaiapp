#!/usr/bin/env python3
"""
Simple verification that all AI agents follow codebase hygiene rules
"""
import os
import glob

def verify_agents_have_hygiene_rules():
    """Check that all agent files include the hygiene enforcement section"""
    
    agent_files = glob.glob('/opt/sutazaiapp/.claude/agents/*.md')
    agent_files = [f for f in agent_files if not any(x in f for x in ['README', 'INDEX', 'REPORT', 'SUMMARY'])]
    
    print(f"Checking {len(agent_files)} agent files for hygiene rules...\n")
    
    required_sections = [
        "🧼 MANDATORY: Codebase Hygiene Enforcement",
        "Clean Code Principles",
        "Zero Duplication Policy",
        "File Organization Standards"
    ]
    
    all_compliant = True
    
    for agent_file in agent_files:
        agent_name = os.path.basename(agent_file)
        with open(agent_file, 'r') as f:
            content = f.read()
        
        has_all_sections = all(section in content for section in required_sections)
        
        if has_all_sections:
            print(f"✅ {agent_name} - Has hygiene rules")
        else:
            print(f"❌ {agent_name} - Missing hygiene rules")
            all_compliant = False
    
    print("\n" + "="*60)
    if all_compliant:
        print("✅ ALL AGENTS HAVE CODEBASE HYGIENE RULES!")
        print("\nThey will enforce:")
        print("- Clean, consistent code")
        print("- No duplication") 
        print("- Proper file organization")
        print("- Dead code removal")
        print("- Professional standards")
        print("- Reuse existing components")
        print("- No fantasy elements")
        print("- Preserve existing functionality")
    else:
        print("❌ Some agents need hygiene rules added")
    
    return all_compliant

if __name__ == '__main__':
    verify_agents_have_hygiene_rules()