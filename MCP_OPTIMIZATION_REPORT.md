# MCP Configuration Optimization Report
**Date:** November 17, 2025  
**Issue:** Large MCP tools context (~147,063 tokens > 25,000 recommended limit)

## Problem
Claude doctor reported excessive token usage from 15 enabled MCP servers:
- github-project-manager: ~29,510 tokens
- gitmcp-sutazai: ~22,583 tokens  
- gitmcp-anthropic: ~22,556 tokens
- gitmcp-docs: ~22,529 tokens
- github: ~18,123 tokens
- Plus 10 more servers

**Total:** ~147,063 tokens (5.9x over the recommended 25,000 limit)

## Solution Applied
Reduced enabled MCP servers from 15 to **5 essential servers**:

### ✅ Enabled (Core Functionality)
1. **filesystem** - File system operations (essential)
2. **github** - GitHub integration (essential for git operations)
3. **memory** - Context memory management (essential)
4. **sequential-thinking** - Advanced reasoning (essential)
5. **gitmcp-sutazai** - Project-specific git tools (essential)

### ⛔ Disabled (Optional/Specialized)
- **removed** - Duplicate/unused entry
- **extended-memory** - Redundant with memory
- **ddg** - DuckDuckGo search (use when needed)
- **http-fetch** - HTTP requests (use when needed)
- **playwright** - Browser automation (specialized use)
- **gitmcp-anthropic** - Secondary git repo (not needed for main project)
- **gitmcp-docs** - Documentation git repo (not needed for main project)
- **context7** - Specialized context tool (redundant)
- **memory-bank** - Redundant with memory
- **github-project-manager** - Advanced GitHub features (can enable when needed)

### Already Disabled
- **everything** - Everything search
- **code-index** - Code indexing

## Expected Impact
- **Token reduction:** ~147k → ~25-30k tokens (80% reduction)
- **Performance:** Faster context loading and response times
- **Cost:** Reduced API token usage per request

## How to Re-enable Servers
If you need a disabled server temporarily:

1. Edit `.claude/settings.local.json`
2. Move the server from `disabledMcpjsonServers` to `enabledMcpjsonServers`
3. Restart Claude CLI

Example:
```json
"enabledMcpjsonServers": [
  "filesystem",
  "github",
  "memory",
  "sequential-thinking",
  "gitmcp-sutazai",
  "playwright"  // Add when needed
]
```

## Verification
Run as root to verify:
```bash
su
claude doctor
```

Expected: Context usage should now be under 25,000 tokens with no warnings.

## Files Modified
- `.claude/settings.local.json` - Updated enabled/disabled server lists

## Recommendations
1. Keep only essential servers enabled by default
2. Enable specialized servers (playwright, http-fetch, ddg) only when needed
3. Monitor `claude doctor` output periodically
4. Consider disabling additional servers if still over 25k tokens
