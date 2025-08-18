# CHANGELOG - UltimateCoderMCP Directory

All notable changes to the MCP infrastructure will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2025-08-17] - MCP Server Cleanup

### Removed
- **postgres-mcp**: Removed failed PostgreSQL MCP server from configuration
  - Stopped and removed Docker container `postgres-mcp-485297-1755469768`
  - Removed from `.mcp.json` configuration
  - Removed from `backend/config/mcp_mesh_registry.yaml`
  - Service was causing system instability and connection failures
  
- **puppeteer-mcp**: Removed non-functional Puppeteer MCP server
  - Removed from `.mcp.json` configuration
  - Cleaned up orphaned pid file from `/run/mcp/puppeteer-mcp.pid`
  - Wrapper script `/scripts/mcp/wrappers/puppeteer-mcp.sh` was already missing
  - Service was marked as "no longer in use" but still present in configs

### Changed
- Updated `CLAUDE.md` to reflect new MCP server count (21 → 19 servers)
- Updated all references to MCP container counts in documentation
- Cleaned up backend MCP mesh registry configuration

### Impact
- Reduced system resource consumption by removing failed services
- Eliminated potential connection retry loops from failed servers
- Improved overall system stability by removing non-functional components
- No impact on operational functionality (services were already failed/unused)

### Technical Details
- Container removed: `postgres-mcp-485297-1755469768` (was running but failing)
- PID file cleanup: `/opt/sutazaiapp/run/mcp/puppeteer-mcp.pid`
- Configuration files updated: `.mcp.json`, `CLAUDE.md`, `backend/config/mcp_mesh_registry.yaml`

### Rule Compliance
- **Rule 4**: Investigated existing files before making changes
- **Rule 5**: Applied professional standards to cleanup operation
- **Rule 18**: Created mandatory CHANGELOG.md for directory
- **Rule 20**: Protected remaining MCP servers while removing failed ones