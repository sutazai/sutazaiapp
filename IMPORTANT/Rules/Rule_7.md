📌 Rule 7: Script Organization & Control
Requirement: Maintain centralized, documented, and reusable scripts that eliminate chaos and provide reliable automation across all environments.
✅ Complete Script Organization Structure:
/scripts/
├── README.md                  # Directory overview & usage guide
├── dev/                       # Local development tools
│   ├── setup/
│   │   ├── install-deps.sh    # Install development dependencies
│   │   ├── setup-env.sh       # Configure development environment
│   │   ├── init-database.sh   # Initialize local database
│   │   └── setup-hooks.sh     # Install git hooks & pre-commit
│   ├── database/
│   │   ├── reset-db.sh        # Reset database to clean state
│   │   ├── seed-data.sh       # Load test/development data
│   │   ├── backup-local.sh    # Backup local development data
│   │   └── migrate.sh         # Run database migrations
│   ├── testing/
│   │   ├── run-tests.sh       # Execute test suites
│   │   ├── coverage.sh        # Generate coverage reports
│   │   ├── lint.sh            # Run code quality checks
│   │   └── format.sh          # Auto-format codebase
│   ├── services/
│   │   ├── start-services.sh  # Start local development services
│   │   ├── stop-services.sh   # Stop all local services
│   │   ├── restart.sh         # Restart specific services
│   │   └── logs.sh            # View service logs
│   └── cleanup/
│       ├── clean-cache.sh     # Clear development caches
│       ├── clean-logs.sh      # Remove old log files
│       ├── clean-deps.sh      # Clean dependency caches
│       └── reset-workspace.sh # Full workspace reset
├── deploy/                    # Deployment automation
│   ├── environments/
│   │   ├── staging.sh         # Deploy to staging environment
│   │   ├── production.sh      # Deploy to production environment
│   │   ├── development.sh     # Deploy to development environment
│   │   └── rollback.sh        # Rollback deployment
│   ├── infrastructure/
│   │   ├── provision.sh       # Provision infrastructure resources
│   │   ├── configure.sh       # Configure infrastructure settings
│   │   ├── scale.sh           # Scale infrastructure components
│   │   └── destroy.sh         # Destroy infrastructure (with safeguards)
│   ├── containers/
│   │   ├── build-images.sh    # Build container images
│   │   ├── push-images.sh     # Push images to registry
│   │   ├── deploy-containers.sh # Deploy containerized services
│   │   └── update-containers.sh # Update running containers
│   ├── database/
│   │   ├── migrate-prod.sh    # Production database migrations
│   │   ├── backup-prod.sh     # Production database backups
│   │   ├── restore-prod.sh    # Production database restoration
│   │   └── sync-environments.sh # Sync data between environments
│   ├── security/
│   │   ├── rotate-secrets.sh  # Rotate secrets and credentials
│   │   ├── update-certs.sh    # Update SSL certificates
│   │   ├── scan-vulnerabilities.sh # Security vulnerability scans
│   │   └── audit-permissions.sh # Audit access permissions
│   └── monitoring/
│       ├── setup-monitoring.sh # Configure monitoring systems
│       ├── deploy-alerts.sh   # Deploy alerting configurations
│       ├── health-check.sh    # Verify system health
│       └── performance-test.sh # Run performance benchmarks
├── data/                      # Data management utilities
│   ├── backup/
│   │   ├── backup-all.sh      # Comprehensive system backup
│   │   ├── backup-database.sh # Database-specific backups
│   │   ├── backup-files.sh    # File system backups
│   │   └── verify-backups.sh  # Validate backup integrity
│   ├── migration/
│   │   ├── export-data.sh     # Export data for migration
│   │   ├── import-data.sh     # Import migrated data
│   │   ├── transform-data.sh  # Data transformation scripts
│   │   └── validate-migration.sh # Validate migration success
│   ├── processing/
│   │   ├── etl-pipeline.sh    # Extract, transform, load pipeline
│   │   ├── batch-processing.sh # Batch data processing jobs
│   │   ├── data-validation.sh # Data quality validation
│   │   └── generate-reports.sh # Generate data reports
│   ├── maintenance/
│   │   ├── cleanup-old-data.sh # Remove obsolete data
│   │   ├── optimize-database.sh # Database optimization
│   │   ├── reindex-search.sh  # Rebuild search indexes
│   │   └── compress-logs.sh   # Compress and archive logs
│   └── sync/
│       ├── sync-environments.sh # Synchronize data between environments
│       ├── sync-external.sh   # Sync with external data sources
│       ├── sync-cdn.sh        # Sync assets to CDN
│       └── sync-backups.sh    # Sync backups to remote storage
├── utils/                     # General purpose helpers
│   ├── system/
│   │   ├── check-dependencies.sh # Verify system dependencies
│   │   ├── update-system.sh   # System updates and patches
│   │   ├── monitor-resources.sh # Monitor system resources
│   │   └── generate-ssl.sh    # Generate SSL certificates
│   ├── network/
│   │   ├── test-connectivity.sh # Test network connectivity
│   │   ├── check-ports.sh     # Verify port accessibility
│   │   ├── proxy-setup.sh     # Configure proxy settings
│   │   └── dns-check.sh       # Verify DNS configuration
│   ├── security/
│   │   ├── scan-ports.sh      # Security port scanning
│   │   ├── check-permissions.sh # File permission auditing
│   │   ├── encrypt-files.sh   # File encryption utilities
│   │   └── generate-keys.sh   # Generate encryption keys
│   ├── validation/
│   │   ├── validate-config.sh # Configuration validation
│   │   ├── check-syntax.sh    # Code syntax validation
│   │   ├── verify-links.sh    # Link validation utilities
│   │   └── test-endpoints.sh  # API endpoint testing
│   └── maintenance/
│       ├── log-rotation.sh    # Log rotation and archival
│       ├── disk-cleanup.sh    # Disk space management
│       ├── service-health.sh  # Service health monitoring
│       └── generate-docs.sh   # Documentation generation
├── test/                      # Testing automation
│   ├── unit/
│   │   ├── run-unit-tests.sh  # Execute unit test suites
│   │   ├── coverage-unit.sh   # Unit test coverage analysis
│   │   ├── test-specific.sh   # Run specific test files
│   │   └── watch-tests.sh     # Continuous test execution
│   ├── integration/
│   │   ├── run-integration.sh # Integration test execution
│   │   ├── test-api.sh        # API integration testing
│   │   ├── test-database.sh   # Database integration testing
│   │   └── test-services.sh   # Service integration testing
│   ├── e2e/
│   │   ├── run-e2e.sh         # End-to-end test execution
│   │   ├── test-workflows.sh  # User workflow testing
│   │   ├── test-browsers.sh   # Cross-browser testing
│   │   └── test-mobile.sh     # Mobile platform testing
│   ├── performance/
│   │   ├── load-test.sh       # Load testing execution
│   │   ├── stress-test.sh     # Stress testing scenarios
│   │   ├── benchmark.sh       # Performance benchmarking
│   │   └── profile.sh         # Performance profiling
│   ├── security/
│   │   ├── security-scan.sh   # Security vulnerability scanning
│   │   ├── penetration-test.sh # Penetration testing
│   │   ├── auth-test.sh       # Authentication testing
│   │   └── compliance-test.sh # Compliance validation
│   └── automation/
│       ├── setup-test-env.sh  # Test environment setup
│       ├── teardown-env.sh    # Test environment cleanup
│       ├── generate-test-data.sh # Test data generation
│       └── parallel-tests.sh  # Parallel test execution
├── monitoring/                # System monitoring & alerting
│   ├── health-checks.sh       # System health verification
│   ├── performance-monitor.sh # Performance monitoring
│   ├── log-analysis.sh        # Log analysis and parsing
│   ├── alert-management.sh    # Alert configuration management
│   └── metrics-collection.sh  # Metrics gathering and reporting
├── maintenance/               # Regular maintenance tasks
│   ├── daily-maintenance.sh   # Daily maintenance routines
│   ├── weekly-maintenance.sh  # Weekly maintenance tasks
│   ├── monthly-maintenance.sh # Monthly maintenance procedures
│   └── emergency-procedures.sh # Emergency response procedures
└── templates/                 # Script templates & examples
    ├── script-template.sh     # Standard script template
    ├── python-template.py     # Python script template
    ├── batch-template.bat     # Windows batch template
    └── examples/              # Example implementations
	
Script Organization & Control
✅ Script Standards:

Descriptive, hyphenated naming conventions for all scripts
Header documentation with purpose, usage, requirements, and examples
Proper argument handling using argparse/click for Python or getopts for shell
Error handling with meaningful exit codes and descriptive error messages
No hardcoded values or secrets - use configuration files or environment variables
Comprehensive logging with appropriate log levels and structured output
Input validation and sanitization for all user-provided parameters
Timeout handling for long-running operations with appropriate defaults
Proper signal handling for graceful shutdown and cleanup procedures
Cross-platform compatibility considerations and environment detection
Dependency checking with clear error messages for missing requirements
Version checking and compatibility validation for external tools
Lock file management to prevent concurrent execution conflicts
Progress indicators and status reporting for long-running operations
Rollback capabilities for scripts that make significant system changes
Dry-run options for testing script behavior without making changes
Verbose and quiet modes to control output detail levels
Configuration file support for complex parameter management
Environment-specific behavior with clear environment detection
Resource cleanup procedures to prevent resource leaks
Performance monitoring and optimization for resource-intensive scripts
Security considerations including input sanitization and privilege management
Documentation generation capabilities for self-documenting scripts
Integration with CI/CD pipelines and automation frameworks
Monitoring and alerting integration for production script execution
Backup and recovery procedures for scripts that modify critical data
Testing frameworks and validation procedures for script reliability
Internationalization support for user-facing messages and output
Accessibility considerations for scripts with user interfaces
Version control integration and change tracking capabilities

🚫 Forbidden Practices:

Creating scripts without proper documentation and usage examples
Using hardcoded paths, URLs, credentials, or environment-specific values
Implementing scripts without proper error handling and exit codes
Creating scripts that can't be safely executed multiple times (not idempotent)
Using global variables without clear necessity and documentation
Creating scripts without input validation and sanitization
Implementing long-running operations without progress indicators or timeouts
Using deprecated or unsupported system commands or utilities
Creating scripts without proper logging and debugging capabilities
Implementing destructive operations without confirmation prompts or dry-run modes
Using shell scripts for complex logic better suited to higher-level languages
Creating scripts without considering cross-platform compatibility requirements
Implementing scripts without proper dependency checking and requirements validation
Using temporary files without proper cleanup and error handling
Creating scripts that require manual intervention during automated execution
Implementing scripts without proper backup and rollback capabilities
Using scripts to handle sensitive data without proper security measures
Creating monolithic scripts that perform multiple unrelated functions
Implementing scripts without proper testing and validation procedures
Using scripts in production without proper monitoring and alerting integration
Creating scripts without considering resource usage and performance implications
Implementing scripts without proper version control and change management
Using scripts for critical operations without proper approval and review processes
Creating scripts that violate security policies or compliance requirements
Implementing scripts without considering maintenance and long-term support needs
Using scripts without proper integration with existing automation frameworks
Creating scripts that don't follow established coding standards and conventions
Implementing scripts without considering internationalization and accessibility needs
Using scripts without proper documentation of assumptions and limitations
Creating scripts that can't be safely interrupted or resumed

Script Quality Assurance:
Code Quality Standards:

Follow shell scripting best practices including proper quoting and variable expansion
Use shellcheck or equivalent static analysis tools for shell scripts
Implement comprehensive error handling with trap statements for cleanup
Use proper exit codes following standard conventions (0 for success, non-zero for errors)
Implement input validation with clear error messages for invalid parameters
Use consistent indentation and formatting throughout all scripts
Follow naming conventions for variables, functions, and file names
Implement proper logging with timestamps and structured output formats
Use comments to explain complex logic and business requirements
Implement proper signal handling for graceful shutdown and cleanup

Security and Safety Standards:

Validate all user inputs to prevent injection attacks and system compromise
Use proper file permissions and ownership for all scripts and data files
Implement proper secret management without storing credentials in scripts
Use secure temporary file creation with proper cleanup procedures
Implement proper privilege escalation only when necessary with clear justification
Use secure communication protocols for any network operations
Implement proper audit logging for security-critical operations
Use input sanitization to prevent command injection and path traversal attacks
Implement proper access controls and authentication for sensitive scripts
Use encryption for any sensitive data processed or stored by scripts

Testing and Validation Procedures:

Implement unit testing for complex script functions and logic
Use integration testing to validate script interactions with external systems
Implement smoke testing for critical deployment and maintenance scripts
Use load testing for scripts that will handle high-volume operations
Implement security testing to validate input handling and access controls
Use compatibility testing across different operating systems and environments
Implement regression testing to prevent introduction of new bugs
Use performance testing to validate resource usage and execution time
Implement failure testing to validate error handling and recovery procedures
Use user acceptance testing for scripts with user-facing functionality

Operational Integration:

Integrate scripts with monitoring and alerting systems for production use
Implement proper logging integration with centralized log management systems
Use configuration management systems for script deployment and updates
Implement proper backup and recovery procedures for script-managed data
Use scheduling systems for automated script execution with proper error handling
Implement proper notification systems for script execution status and results
Use metrics collection for script performance and reliability monitoring
Implement proper dependency management for external tools and libraries
Use proper version control and change management for script updates
Implement proper documentation and knowledge management for operational procedures

Validation Criteria:

All scripts follow established templates and coding standards
Comprehensive documentation exists for all scripts including usage examples
Error handling and exit codes are properly implemented and tested
Input validation prevents security vulnerabilities and system errors
All scripts are idempotent and can be safely executed multiple times
Proper logging and monitoring integration is functional and tested
Security measures are implemented and validated through testing
Performance characteristics are documented and within acceptable limits
Integration with CI/CD pipelines is functional and reliable
Backup and rollback procedures are tested and documented
Cross-platform compatibility requirements are met and verified
Dependency management is automated and reliable
Testing procedures are comprehensive and regularly executed
Documentation is current and accessible to all relevant team members
Compliance and security requirements are met and audited
Operational integration with monitoring and alerting systems is functional
Version control and change management procedures are followed
Team training on script usage and maintenance has been completed
Script execution metrics are collected and analyzed for continuous improvement
Emergency procedures and incident response capabilities are tested and documented

*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/