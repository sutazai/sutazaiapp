#!/usr/bin/env python3
"""
Purpose: Verify Ollama/GPT-OSS usage compliance (Rule 16 enforcement)
Usage: python check-llm-usage.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import re
from pathlib import Path
from typing import List

# Forbidden LLM patterns
FORBIDDEN_PATTERNS = [
    r'openai\.ChatCompletion',
    r'anthropic\.Client',
    r'gpt-[34]',
    r'claude-\d',
    r'palm\.generate',
    r'cohere\.Client',
    r'huggingface_hub',
    r'transformers\.pipeline.*model=(?!tinyllama)',
]

# Allowed patterns
ALLOWED_PATTERNS = [
    r'ollama',
    r'tinyllama',
    r'localhost:10104',  # Ollama default port
]

def check_llm_usage(filepath: Path) -> List[str]:
    """Check for non-Ollama LLM usage."""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            # Check for forbidden patterns
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(
                        f"{filepath}:{line_num}: Forbidden LLM usage detected - use Ollama with GPT-OSS"
                    )
            
            # Check for direct model loading without Ollama
            if 'load_model' in line or 'from_pretrained' in line:
                if not any(re.search(allowed, line, re.IGNORECASE) for allowed in ALLOWED_PATTERNS):
                    violations.append(
                        f"{filepath}:{line_num}: Direct model loading - use Ollama framework"
                    )
                    
    except:
        pass
    
    return violations

def main():
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    all_violations = []
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.suffix in ['.py', '.yaml', '.yml']:
            violations = check_llm_usage(filepath)
            all_violations.extend(violations)
    
    if all_violations:
        print("❌ Rule 16 Violation: Non-Ollama LLM usage detected")
        for violation in all_violations:
            print(f"  - {violation}")
        print("\n📋 Fix: Use 'ollama run tinyllama' for all local LLM tasks")
        return 1
    
    print("✅ Rule 16: LLM usage check passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())