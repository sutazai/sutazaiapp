# Scripts Directory Changelog

All notable changes to the scripts directory structure and organization.

## [2.0.0] - 2025-08-29

### Added
- Created categorical subdirectories for better organization:
  - `deploy/` - Deployment and startup scripts
  - `monitoring/` - Monitoring and health check scripts
  - `maintenance/` - Maintenance and cleanup scripts
  - `dev/` - Development and debugging scripts
  - `data/` - Data management and migration scripts
  - `utils/` - Utility and helper scripts
  - `test/` - Testing and validation scripts
- Added README.md to each subdirectory with documentation
- Created `check-compliance.sh` for compliance checking
- Created `fix-ollama-semgrep.sh` for service health fixes
- Added comprehensive documentation for all scripts

### Changed
- Reorganized all scripts into appropriate subdirectories
- Moved deployment scripts to `deploy/`
- Moved monitoring scripts to `monitoring/`
- Moved maintenance scripts to `maintenance/`
- Improved script naming consistency

### Fixed
- Fixed Ollama service health issues
- Fixed Semgrep service health issues
- Resolved script organization chaos

## [1.0.0] - 2025-08-28

### Added
- Initial script collection
- Basic monitoring scripts
- Deployment automation
- MCP server wrappers