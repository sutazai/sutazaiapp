# MCP Server Testing Procedures

## Overview

This document provides clear instructions for properly testing Model Context Protocol (MCP) servers in the SutazAI system.

## Testing Methods

### 1. Simple Self-Check (Recommended)

Test all MCP servers at once:
```bash
./scripts/mcp/selfcheck_all.sh
```

Test a specific MCP server:
```bash
./scripts/mcp/wrappers/<server-name>.sh --selfcheck
```

Examples:
```bash
./scripts/mcp/wrappers/playwright-mcp.sh --selfcheck
./scripts/mcp/wrappers/memory-bank-mcp.sh --selfcheck
```

### 2. List Available Servers

```bash
./scripts/mcp/test_all_mcp_servers.sh --list-servers
```

### 3. Comprehensive Testing

Test all servers with detailed reporting:
```bash
./scripts/mcp/test_all_mcp_servers.sh
```

Test a specific server:
```bash
./scripts/mcp/test_all_mcp_servers.sh --server <server-name>
```

## Common Issues and Solutions

### PLAYWRIGHT-MCP

**Issue**: "unknown option '--selfcheck'" when running `npx -y @playwright/mcp --selfcheck`

**Solution**: Use the wrapper script instead:
```bash
./scripts/mcp/wrappers/playwright-mcp.sh --selfcheck
```

**Explanation**: The --selfcheck option is handled by our wrapper scripts, not by the MCP servers themselves.

### MEMORY-BANK-MCP

**Issue**: "python module missing" warning

**Solution**: This is expected behavior. The system falls back to npx and works correctly.

**Explanation**: The wrapper tries Python first, then falls back to npx if the Python module isn't installed.

## Currently Registered MCP Servers (17 total)

1. **compass-mcp** - MCP server discovery and recommendations
2. **context7** - Documentation and library context
3. **ddg** - DuckDuckGo search integration
4. **extended-memory** - Extended memory capabilities
5. **files** - File system operations
6. **github** - GitHub integration
7. **http** - HTTP fetching capabilities
8. **knowledge-graph-mcp** - Knowledge graph operations
9. **language-server** - Language server protocol integration
10. **mcp_ssh** - SSH operations
11. **memory-bank-mcp** - Memory bank management
12. **nx-mcp** - Nx workspace integration
13. **playwright-mcp** - Browser automation with Playwright
14. **postgres** - PostgreSQL database operations
15. **puppeteer-mcp** - Browser automation with Puppeteer
16. **sequentialthinking** - Sequential thinking patterns
17. **ultimatecoder** - Advanced coding operations

## Debugging Failed Tests

1. Check the detailed logs:
   ```bash
   ls -la /opt/sutazaiapp/logs/mcp_*.log
   tail -n 50 /opt/sutazaiapp/logs/mcp_selfcheck_*.log
   ```

2. Test wrapper syntax:
   ```bash
   bash -n ./scripts/mcp/wrappers/<server-name>.sh
   ```

3. Check if dependencies are installed:
   ```bash
   # For Node.js based servers
   npm list -g | grep <package-name>
   
   # For Python based servers
   python3 -c "import <module-name>"
   ```

## Do NOT Do This

❌ **Wrong**: `npx -y @playwright/mcp --selfcheck`
✅ **Correct**: `./scripts/mcp/wrappers/playwright-mcp.sh --selfcheck`

❌ **Wrong**: Testing MCP servers directly without wrappers
✅ **Correct**: Always use the wrapper scripts in `/opt/sutazaiapp/scripts/mcp/wrappers/`

## Wrapper Script Locations

All wrapper scripts are located in:
```
/opt/sutazaiapp/scripts/mcp/wrappers/
```

Each server has a corresponding `.sh` file that handles:
- Dependency checking
- Version verification
- Proper command resolution
- Error handling
- Self-check capabilities

## Exit Codes

- `0`: Success
- `1`: Test failure
- `127`: Command not found or dependency missing

## Logging

- Self-check logs: `/opt/sutazaiapp/logs/mcp_selfcheck_*.log`
- Comprehensive test logs: `/opt/sutazaiapp/logs/mcp_comprehensive_test_*.log`
- Test reports (JSON): `/opt/sutazaiapp/logs/mcp_test_report_*.json`