# Garbage Collection and Cleanup Enforcer

## Rule 13: No Garbage, No Rot

The Garbage Collection and Cleanup Enforcer is a comprehensive tool that implements Rule 13 from the CLAUDE.md guidelines, ensuring zero tolerance for digital clutter in the codebase.

## Features

### 🔍 Intelligent Detection
- **Pattern-based detection** with confidence scoring
- **AST analysis** for dead code detection
- **Content hash comparison** for duplicate files
- **Reference analysis** to prevent false positives
- **Git integration** for history-aware decisions
- **Multi-threaded scanning** for performance

### 🛡️ Safety First
- **Risk assessment** for every detected item
- **Reference checking** before removal
- **Automatic archiving** before deletion
- **Git status integration** to avoid breaking changes
- **Configurable safety thresholds**
- **Dry-run mode** for safe testing

### 🗑️ Comprehensive Garbage Types
1. **Temporary Files**: `*.tmp`, `*.temp`, `*.swp`, cache files
2. **Backup Files**: `*.bak`, `*.backup`, `*~`, `*.old`, copies
3. **Build Artifacts**: `*.pyc`, `__pycache__/`, `dist/`, `node_modules/`
4. **Log Files**: `*.log`, debug logs, application logs
5. **Cache Files**: `.DS_Store`, `Thumbs.db`, browser caches
6. **Dead Code**: Unused functions, classes, imports
7. **Duplicate Files**: Identical content with different names
8. **Commented Code**: Large blocks of commented-out code
9. **Old Versions**: `*_v1`, `*_final`, `*_new` files
10. **Stale Configs**: Unused configuration files

### 📊 Detailed Reporting
- **Confidence-based categorization**
- **Risk-level assessment**
- **Space recovery calculations**
- **Top violations by size**
- **Duplicate file detection**
- **Git command suggestions**
- **Rollback instructions**

## Usage

### Basic Usage

```bash
# Dry run (safe - no changes made)
python scripts/agents/garbage-collection-enforcer.py --dry-run

# Live cleanup with moderate risk
python scripts/agents/garbage-collection-enforcer.py --live --risk-threshold moderate

# High-confidence cleanup only
python scripts/agents/garbage-collection-enforcer.py --live --confidence-threshold 0.9
```

### Advanced Options

```bash
# Custom project path
python scripts/agents/garbage-collection-enforcer.py --project-root /path/to/project --dry-run

# Conservative cleanup (safe items only)
python scripts/agents/garbage-collection-enforcer.py --live --risk-threshold safe

# Disable git integration
python scripts/agents/garbage-collection-enforcer.py --dry-run --no-git

# Skip archiving (not recommended)
python scripts/agents/garbage-collection-enforcer.py --live --no-archive

# Custom output location
python scripts/agents/garbage-collection-enforcer.py --dry-run --output /tmp/cleanup-report.json

# Verbose logging
python scripts/agents/garbage-collection-enforcer.py --dry-run --verbose
```

## Configuration

The enforcer uses `garbage-collection-config.yaml` for advanced configuration:

```yaml
# Detection thresholds
detection:
  confidence_threshold: 0.7
  risk_threshold: "moderate"
  
# Safety settings
safety:
  enable_git_integration: true
  archive_before_delete: true
  protected_paths:
    - ".git/"
    - "tests/"
    - "docs/"

# Custom patterns
custom_garbage_patterns:
  - "*.debug"
  - "*_testing.*"
  - "experimental_*"
```

## Safety Features

### Risk Levels
- **SAFE**: Build artifacts, temp files, cache files - auto-removable
- **MODERATE**: Files without references, old versions - requires confidence
- **RISKY**: Files with references, source code files - manual review
- **DANGEROUS**: Configuration files, core system files - never auto-remove

### Reference Checking
The enforcer uses `ripgrep` for fast codebase searching to find references:
- File name references in imports
- Path references in configs
- Symbol references in code
- Documentation references

### Git Integration
- Check if files are tracked in git
- Identify modified/staged files
- Generate appropriate git commands
- Provide rollback instructions

### Archiving System
All removed files are archived with:
- Original directory structure preserved
- Timestamped archive directories
- Metadata about removal reason
- Easy restoration process

## Report Structure

The enforcer generates comprehensive JSON reports:

```json
{
  "metadata": {
    "rule": "Rule 13: No Garbage, No Rot",
    "timestamp": "2025-08-03T20:52:44",
    "session_id": "20250803_205202"
  },
  "statistics": {
    "files_scanned": 4789,
    "garbage_items_found": 1628,
    "actionable_items": 30,
    "space_recovered_bytes": 0
  },
  "analysis": {
    "items_by_type": { ... },
    "items_by_risk": { ... },
    "confidence_distribution": { ... }
  },
  "findings": {
    "top_violations_by_size": [ ... ],
    "duplicate_files": [ ... ]
  },
  "recommendations": [ ... ],
  "audit_trail": { ... }
}
```

## Integration

### Pre-commit Hook
```bash
# Add to .pre-commit-config.yaml
- repo: local
  hooks:
    - id: garbage-check
      name: Check for garbage files
      entry: python scripts/agents/garbage-collection-enforcer.py --dry-run --confidence-threshold 0.9
      language: system
      pass_filenames: false
```

### CI/CD Pipeline
```yaml
# GitHub Actions example
- name: Garbage Collection Check
  run: |
    python scripts/agents/garbage-collection-enforcer.py --dry-run
    if [ $? -ne 0 ]; then
      echo "Garbage files detected - please clean up"
      exit 1
    fi
```

### Automated Cleanup
```bash
# Weekly cleanup cron job
0 2 * * 0 cd /opt/sutazaiapp && python scripts/agents/garbage-collection-enforcer.py --live --risk-threshold safe
```

## Examples

### Typical Scan Results
```
🧹 GARBAGE COLLECTION ENFORCER - SUMMARY
============================================================
Mode: DRY RUN
Items Found: 1628
Actionable Items: 30
Space Recovered: 0.84 MB
Scan Duration: 41.36s

📊 Top Violations by Size:
  • tests/comprehensive_test_report_final.py (0.02MB) - old_version
  • data/langflow/.cache/langflow/profile_pictures/Space/042-space shuttle.svg (0.02MB) - temporary_file

📁 Duplicate Files: 70
```

### Common Findings
- **Dead Code**: 1346 unused functions/classes found
- **Temporary Files**: 82 cache and temp files
- **Duplicate Files**: 70 identical files with different names
- **Log Files**: 31 old log files consuming space
- **Empty Files**: 69 zero-byte files

## Best Practices

### Before Running
1. **Always start with dry-run** to review findings
2. **Commit current work** to git before live cleanup
3. **Review the report** thoroughly before proceeding
4. **Check protected paths** are configured correctly

### During Cleanup
1. **Use conservative thresholds** initially
2. **Increase confidence requirements** for risky operations
3. **Verify archives** are created properly
4. **Monitor space recovery** metrics

### After Cleanup
1. **Run tests** to ensure nothing broke
2. **Review git diff** for unexpected changes
3. **Check application functionality**
4. **Update .gitignore** to prevent future garbage

## Troubleshooting

### Common Issues

**Performance Issues**
```bash
# Reduce scan scope
python scripts/agents/garbage-collection-enforcer.py --project-root /opt/sutazaiapp/specific-dir

# Increase confidence threshold
python scripts/agents/garbage-collection-enforcer.py --confidence-threshold 0.9
```

**False Positives**
```bash
# Use safer risk threshold
python scripts/agents/garbage-collection-enforcer.py --risk-threshold safe

# Check specific file
rg "filename.ext" . --type py
```

**Git Issues**
```bash
# Disable git integration if problematic
python scripts/agents/garbage-collection-enforcer.py --no-git

# Check git status first
git status --porcelain
```

### Recovery

**Restore Archived Files**
```bash
# Files are archived to:
ls -la archive/garbage_cleanup_YYYYMMDD_HHMMSS/

# Restore specific file
cp archive/garbage_cleanup_20250803_205202/path/to/file.py .
```

**Git Rollback**
```bash
# If cleanup was committed
git reset --hard HEAD~1

# Restore specific files
git checkout HEAD~1 -- path/to/file.py
```

## Performance

### Metrics
- **Scan Speed**: ~4800 files in 41 seconds
- **Memory Usage**: Minimal with streaming analysis
- **CPU Usage**: Multi-threaded for optimal performance
- **I/O Optimization**: Batch file operations

### Limitations
- **Large codebases**: May require time limits
- **Network filesystems**: Slower due to I/O latency
- **Binary files**: Limited analysis capabilities
- **Encrypted files**: Cannot analyze content

## Development

### Extending Detection
Add new garbage patterns in the `GARBAGE_PATTERNS` dictionary:

```python
GARBAGE_PATTERNS[GarbageType.MY_TYPE] = [
    "*.myext",
    "my_pattern_*",
    "*_my_suffix.*"
]
```

### Custom Risk Assessment
Override the `_assess_risk` method for custom logic:

```python
async def _assess_risk(self, file_path: Path, garbage_type: GarbageType) -> RiskLevel:
    # Custom risk logic here
    return RiskLevel.SAFE
```

### Adding New Scan Types
Implement additional scan methods:

```python
async def _scan_my_type(self) -> List[GarbageItem]:
    # Custom scanning logic
    return items
```

## License

This tool is part of the SUTAZAI system and follows the same license terms.

## Support

For issues, questions, or contributions:
1. Check the generated reports for detailed analysis
2. Review the logs at `/opt/sutazaiapp/logs/garbage-collection-enforcer.log`
3. Use verbose mode for debugging: `--verbose`
4. Test with dry-run before live operations