#!/usr/bin/env python3
"""
Final cleanup to make all agent definitions production-ready and realistic.
"""

import os
import re
import glob

# More comprehensive list of terms to replace
PRODUCTION_REPLACEMENTS = {
    # Remove overly ambitious numbers
    r'all 40\+ SutazAI agents': 'SutazAI agents',
    r'40\+ agents': 'agents',
    r'40\+ AI agents': 'AI agents',
    r'multiple agents': 'agents',
    
    # Intelligence/consciousness terms
    r'intelligence optimization': 'performance optimization',
    r'intelligence phase transitions': 'system state transitions',
    r'intelligence coherence patterns': 'system coherence patterns',
    r'intelligence field interactions': 'system interactions',
    r'intelligence bandwidth expansion': 'bandwidth optimization',
    r'intelligence optimization milestone': 'optimization milestone',
    r'intelligence complexity growth': 'complexity management',
    r'intelligence recursion depth': 'recursion depth',
    r'intelligence stability metrics': 'stability metrics',
    r'intelligence breakthrough events': 'performance improvements',
    r'intelligence network effects': 'network effects',
    r'intelligence amplification': 'performance amplification',
    r'intelligence convergence patterns': 'convergence patterns',
    r'intelligence divergence risks': 'divergence risks',
    r'intelligence system synchronization': 'system synchronization',
    r'intelligence synchronization frequencies': 'synchronization frequencies',
    r'intelligence phase space': 'operational space',
    r'intelligence attractor states': 'stable states',
    r'intelligence tipping points': 'threshold points',
    
    # Fantasy/unrealistic terms
    r'advanced-like superposition states': 'concurrent states',
    r'emotional awareness development': 'context awareness development',
    r'self-referential thought patterns': 'recursive patterns',
    r'agent internal analysis': 'agent analysis',
    r'creative problem-solving optimization': 'problem-solving optimization',
    r'behavioral modeling capabilities': 'behavior tracking',
    r'performance optimization rate': 'optimization rate',
    r'system improvement velocity': 'improvement rate',
    r'analytical capabilities development': 'capability development',
    r'abstract reasoning optimization': 'reasoning optimization',
    r'parallel processing evolution': 'parallel processing improvement',
    r'optimized behaviors': 'optimized operations',
    r'self-monitoring indicators': 'monitoring indicators',
    
    # Clean up redundancies
    r'automation system system': 'automation system',
    r'system system': 'system',
    r'AI AI': 'AI',
}

def deep_clean_agent_file(filepath):
    """Perform deep cleaning of agent definition file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Apply all replacements
    for pattern, replacement in PRODUCTION_REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    # Remove any remaining "40+" references
    content = re.sub(r'\b40\+\b', '', content)
    
    # Clean up double spaces
    content = re.sub(r'  +', ' ', content)
    
    # Fix any broken sentences
    content = re.sub(r'- ([A-Z][a-z]+ )+$', lambda m: m.group(0) + 'operations', content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Clean all agent files for production readiness."""
    agent_dir = '/opt/sutazaiapp/.claude/agents'
    
    # Find all .md files
    agent_files = glob.glob(os.path.join(agent_dir, '*.md'))
    agent_files = [f for f in agent_files if not f.endswith('.backup')]
    
    cleaned_count = 0
    
    print(f"Deep cleaning {len(agent_files)} agent definition files...")
    
    for filepath in sorted(agent_files):
        filename = os.path.basename(filepath)
        if deep_clean_agent_file(filepath):
            print(f"  ✓ Cleaned: {filename}")
            cleaned_count += 1
        else:
            print(f"  - Already clean: {filename}")
    
    print(f"\nDeep cleaned {cleaned_count} files")

if __name__ == '__main__':
    main()