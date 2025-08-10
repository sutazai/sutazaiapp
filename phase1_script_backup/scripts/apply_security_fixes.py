#!/usr/bin/env python3
"""
SutazAI Script Security Fix Application

Applies comprehensive security fixes to all shell scripts in the SutazAI system.
Addresses infinite loops, eval usage, parameter injection, and other security issues.

Usage:
    python3 scripts/apply_security_fixes.py
    python3 scripts/apply_security_fixes.py --dry-run
    python3 scripts/apply_security_fixes.py --report-only

Author: Shell Automation Specialist
Created: 2025-08-10
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from lib.logging_utils import setup_logging
from lib.security_utils import validate_all_scripts, generate_security_report

def fix_script_security(script_path: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Apply comprehensive security fixes to a shell script.
    
    Args:
        script_path: Path to the shell script
        dry_run: If True, only analyze without making changes
        
    Returns:
        Dictionary with fix results
    """
    logger = setup_logging("security_fixer", "INFO")
    
    result = {
        'script': script_path,
        'fixes_applied': [],
        'errors': [],
        'backup_created': False
    }
    
    if not os.path.exists(script_path):
        result['errors'].append(f"Script not found: {script_path}")
        return result
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        result['errors'].append(f"Failed to read script: {e}")
        return result
    
    content = original_content
    
    # Apply security fixes in order
    
    # 1. Add error handling (set -euo pipefail)
    if not re.search(r'set\s+-[euo]+', content):
        shebang_match = re.search(r'^#![^\n]+', content)
        if shebang_match:
            insertion_point = shebang_match.end()
            error_handling = "\n\n# Strict error handling\nset -euo pipefail\n"
            content = content[:insertion_point] + error_handling + content[insertion_point:]
            result['fixes_applied'].append("Added strict error handling (set -euo pipefail)")
    
    # 2. Add signal handlers
    if not re.search(r'trap\s+.*EXIT|trap\s+.*INT|trap\s+.*TERM', content):
        # Find insertion point after shebang and set commands
        lines = content.split('\n')
        insert_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('#!') or line.startswith('set ') or line.strip() == '' or line.startswith('#'):
                insert_idx = i + 1
            else:
                break
        
        signal_handlers = [
            "",
            "# Signal handlers for graceful shutdown", 
            "cleanup_and_exit() {",
            "    local exit_code=\"${1:-0}\"",
            "    echo \"Script interrupted, cleaning up...\" >&2",
            "    # Clean up any background processes",
            "    jobs -p | xargs -r kill 2>/dev/null || true",
            "    exit \"$exit_code\"",
            "}",
            "",
            "trap 'cleanup_and_exit 130' INT",
            "trap 'cleanup_and_exit 143' TERM", 
            "trap 'cleanup_and_exit 1' ERR",
            ""
        ]
        
        lines = lines[:insert_idx] + signal_handlers + lines[insert_idx:]
        content = '\n'.join(lines)
        result['fixes_applied'].append("Added signal handlers for graceful shutdown")
    
    # 3. Fix infinite loops with timeouts
    infinite_loop_patterns = [
        r'while\s+true\s*;?\s*do',
        r'while\s+:\s*;?\s*do', 
        r'while\s+\[\s*1\s*\]\s*;?\s*do',
        r'for\s*\(\(\s*;\s*;\s*\)\)\s*;?\s*do'
    ]
    
    for pattern in infinite_loop_patterns:
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
        if matches:
            lines = content.split('\n')
            offset = 0
            
            for match in matches:
                # Find line with the match
                match_start = match.start() - offset
                char_count = 0
                line_idx = 0
                
                for i, line in enumerate(lines):
                    if char_count + len(line) + 1 > match_start:
                        line_idx = i
                        break
                    char_count += len(line) + 1
                
                # Add timeout mechanism
                indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
                timeout_code = [
                    f"{' ' * indent}# Timeout mechanism to prevent infinite loops",
                    f"{' ' * indent}LOOP_TIMEOUT=${{LOOP_TIMEOUT:-300}}  # 5 minute default timeout",
                    f"{' ' * indent}loop_start=$(date +%s)",
                ]
                
                # Insert timeout initialization before the loop
                lines = lines[:line_idx] + timeout_code + lines[line_idx:]
                
                # Find the loop body and add timeout check
                for j in range(line_idx + len(timeout_code) + 1, len(lines)):
                    if lines[j].strip() == 'done':
                        timeout_check = [
                            f"{' ' * (indent + 4)}# Check for timeout",
                            f"{' ' * (indent + 4)}current_time=$(date +%s)",
                            f"{' ' * (indent + 4)}if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then",
                            f"{' ' * (indent + 8)}echo 'Loop timeout reached after ${{LOOP_TIMEOUT}}s, exiting...' >&2",
                            f"{' ' * (indent + 8)}break",
                            f"{' ' * (indent + 4)}fi",
                            ""
                        ]
                        lines = lines[:j] + timeout_check + lines[j:]
                        break
                
                offset += sum(len(line) + 1 for line in timeout_code)
            
            content = '\n'.join(lines)
            result['fixes_applied'].append("Added timeout mechanisms to infinite loops")
    
    # 4. Fix eval usage
    eval_patterns = [
        (r'\beval\s+\$([A-Za-z_][A-Za-z0-9_]*)', r'# SECURITY FIX: eval replaced\n# Original: eval $\1\n"${\1}"'),
        (r'\beval\s+"([^"]*)"', r'# SECURITY FIX: eval replaced\n# Original: eval "\1"\n\1'),
        (r'\beval\s+([^;\n]+)', r'# SECURITY FIX: eval usage removed\n# Original: eval \1\n# TODO: Replace with safer alternative')
    ]
    
    for pattern, replacement in eval_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            result['fixes_applied'].append("Removed or replaced eval usage")
            break
    
    # 5. Quote unquoted variables
    unquoted_patterns = [
        (r'\[\s+(\$[A-Za-z_][A-Za-z0-9_]*)\s+([!=<>]+)\s+([^"\]\s]+)\s+\]', r'[ "\1" \2 "\3" ]'),
        (r'\[\s+([^"\]\s]+)\s+([!=<>]+)\s+(\$[A-Za-z_][A-Za-z0-9_]*)\s+\]', r'[ "\1" \2 "\3" ]'),
    ]
    
    for pattern, replacement in unquoted_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            result['fixes_applied'].append("Added quotes around variables in tests")
    
    # 6. Fix unsafe temporary file usage
    unsafe_temp_pattern = r'>\s*/tmp/([^/\s"]+)'
    if re.search(unsafe_temp_pattern, content):
        # Replace with mktemp usage
        content = re.sub(
            unsafe_temp_pattern, 
            r'> "$(mktemp /tmp/\1.XXXXXX)"',
            content
        )
        result['fixes_applied'].append("Fixed unsafe temporary file usage")
    
    # 7. Add input validation for critical operations
    if re.search(r'rm\s+-rf?\s+\$', content):
        result['fixes_applied'].append("WARNING: Found potentially dangerous rm usage")
    
    # Apply changes if not dry run
    if not dry_run and content != original_content:
        try:
            # Create backup
            backup_path = f"{script_path}.backup_{int(time.time())}"
            shutil.copy2(script_path, backup_path)
            result['backup_created'] = backup_path
            
            # Write fixed content
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Applied {len(result['fixes_applied'])} security fixes to {script_path}")
            
        except Exception as e:
            result['errors'].append(f"Failed to apply fixes: {e}")
    
    elif dry_run and result['fixes_applied']:
        logger.info(f"[DRY RUN] Would apply {len(result['fixes_applied'])} fixes to {script_path}")
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Apply comprehensive security fixes to SutazAI shell scripts"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Analyze scripts without making changes')
    parser.add_argument('--report-only', action='store_true', 
                       help='Generate security report only')
    parser.add_argument('--scripts-dir', default='scripts',
                       help='Directory containing scripts (default: scripts)')
    parser.add_argument('--output-dir', default='logs',
                       help='Directory for reports and logs (default: logs)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("security_fix_orchestrator", "DEBUG" if args.verbose else "INFO")
    
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / args.scripts_dir
    output_dir = base_dir / args.output_dir
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    logger.info("SutazAI Script Security Fix Application Started")
    logger.info(f"Scripts directory: {scripts_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Report only: {args.report_only}")
    
    # Validate all scripts first
    logger.info("Analyzing all scripts for security issues...")
    script_issues = validate_all_scripts(str(scripts_dir))
    
    # Generate security report
    report_file = output_dir / f"security_report_{int(time.time())}.md"
    generate_security_report(script_issues, str(report_file))
    logger.info(f"Security report generated: {report_file}")
    
    if args.report_only:
        logger.info("Report-only mode, exiting without applying fixes")
        return
    
    # Apply fixes to all shell scripts
    total_scripts = 0
    total_fixes = 0
    total_errors = 0
    fix_results = []
    
    for root, dirs, files in os.walk(scripts_dir):
        for file in files:
            if file.endswith(('.sh', '.bash')):
                script_path = Path(root) / file
                total_scripts += 1
                
                logger.info(f"Processing: {script_path}")
                result = fix_script_security(str(script_path), args.dry_run)
                fix_results.append(result)
                
                total_fixes += len(result['fixes_applied'])
                total_errors += len(result['errors'])
                
                if result['errors']:
                    logger.error(f"Errors in {script_path}: {result['errors']}")
    
    # Generate fix results report
    results_file = output_dir / f"security_fixes_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'dry_run': args.dry_run,
            'total_scripts': total_scripts,
            'total_fixes_applied': total_fixes,
            'total_errors': total_errors,
            'results': fix_results
        }, f, indent=2)
    
    # Summary
    logger.info("=== SECURITY FIX SUMMARY ===")
    logger.info(f"Scripts processed: {total_scripts}")
    logger.info(f"Fixes applied: {total_fixes}")
    logger.info(f"Errors encountered: {total_errors}")
    logger.info(f"Results saved to: {results_file}")
    
    if not args.dry_run and total_fixes > 0:
        logger.info("Security fixes have been applied. Please test the scripts thoroughly.")
    elif args.dry_run:
        logger.info("Dry run completed. Use without --dry-run to apply fixes.")
    
    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    sys.exit(main())