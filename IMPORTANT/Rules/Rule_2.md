Never Break Existing Functionality
Requirement: Every change must preserve or improve current behavior - zero tolerance for regressions.
✅ Required Practices:

Investigation first: Understand what exists before changing anything
Backwards compatibility: Support legacy or provide graceful migration paths
Test coverage: Both old and new logic before merging
Document impact: Clear communication of risks and changes
Maintain API contracts and data schemas across versions
Implement feature flags for risky changes to enable quick rollback
Use semantic versioning for all public interfaces and APIs
Create comprehensive migration guides for breaking changes
Establish deprecation timelines with clear end-of-life dates
Monitor system behavior before, during, and after changes
Validate all external integrations continue to function
Test database migrations with realistic data volumes
Verify performance characteristics remain within acceptable bounds
Ensure error handling patterns are preserved or improved
Maintain security posture and access control mechanisms
Preserve existing logging and monitoring capabilities
Test edge cases and boundary conditions thoroughly
Validate internationalization and localization features
Ensure accessibility standards are maintained
Test across all supported browsers and platforms
Verify mobile and responsive design functionality
Maintain existing caching and optimization strategies
Test disaster recovery and backup procedures
Validate compliance with regulatory requirements
Ensure third-party service integrations remain functional
Test user workflows and critical business processes
Maintain existing configuration and environment variables
Preserve existing data export and import capabilities
Test concurrent user scenarios and load handling
Validate existing automation and scheduled tasks

🚫 Forbidden Practices:

Making changes without understanding current functionality
Removing features without providing migration paths
Changing API responses without versioning or deprecation
Deploying code without comprehensive testing
Breaking external client integrations silently
Removing error handling or validation logic
Changing data formats without backward compatibility
Modifying database schemas without migration scripts
Altering authentication or authorization mechanisms
Breaking existing user workflows or processes
Removing monitoring, logging, or alerting capabilities
Changing performance characteristics without analysis
Breaking existing caching or optimization strategies
Removing configuration options without alternatives
Changing error messages or codes without documentation
Breaking existing import/export functionality
Altering security mechanisms without security review
Removing existing accessibility features
Breaking internationalization or localization
Changing existing URL structures without redirects
Removing existing CLI commands or arguments
Breaking existing webhook or callback mechanisms
Changing existing file formats or protocols
Removing existing backup or recovery capabilities
Breaking existing reporting or analytics features
Changing existing notification or messaging systems
Removing existing search or filtering capabilities
Breaking existing batch processing or scheduling
Changing existing rate limiting or throttling behavior
Removing existing documentation or help systems
Breaking existing integration tests or end-to-end tests
Changing existing deployment or rollback procedures
Removing existing health checks or status endpoints
Breaking existing load balancing or failover mechanisms

Pre-change Investigation Checklist:

Trace usage with grep -R "functionName" . across entire codebase
Search for dynamic references (reflection, string-based calls)
Review Git history to understand original implementation intent
Identify all direct and indirect consumers of the functionality
Map data flow and dependencies throughout the system
Review existing test coverage and identify gaps
Analyze performance implications and resource usage
Check for external documentation or API references
Verify integration with third-party services and systems
Review error handling and edge case scenarios
Identify configuration dependencies and environment variables
Check database schema dependencies and relationships
Review security implications and access control requirements
Analyze impact on existing monitoring and alerting
Verify compatibility with existing automation and scripts
Check impact on existing user workflows and processes
Review compliance and regulatory implications
Identify potential rollback complexities and risks

Testing & Validation Requirements:

Run full automated test suite with 100% pass rate
Execute manual testing of all affected user workflows
Perform integration testing with all dependent services
Test backward compatibility with previous API versions
Validate database migration scripts with production-like data
Execute performance testing to ensure no degradation
Test error handling and recovery scenarios
Validate security testing and vulnerability scans
Execute load testing with realistic traffic patterns
Test failover and disaster recovery procedures
Validate monitoring and alerting functionality
Test configuration changes across all environments
Execute accessibility and usability testing
Test internationalization and localization features
Validate mobile and cross-browser compatibility
Test concurrent user scenarios and race conditions
Execute end-to-end workflow testing
Test rollback procedures and recovery mechanisms

Documentation Requirements:

Update API documentation with change details
Create migration guides for affected consumers
Document breaking changes with timeline and alternatives
Update configuration and environment variable documentation
Revise troubleshooting and FAQ sections
Update deployment and operational procedures
Document new error codes and messages
Update security and compliance documentation
Revise user guides and help documentation
Update development and testing procedures

Rollback Planning:

Prepare exact rollback procedures with step-by-step instructions
Test rollback procedures in staging environment
Identify rollback triggers and decision criteria
Document data recovery procedures if needed
Prepare communication plan for rollback scenarios
Ensure monitoring and alerting for rollback detection
Plan for gradual rollback using feature flags or canary deployments
Document post-rollback validation procedures

Validation Criteria:

All automated tests pass with no new failures
Manual testing confirms no regression in functionality
Performance metrics remain within acceptable bounds
All identified consumers continue to function correctly
Error handling maintains existing behavior patterns
Security scanning shows no new vulnerabilities
Load testing demonstrates stable performance characteristics
Integration testing confirms all external dependencies work
Rollback procedures have been tested and documented
Migration scripts execute successfully with test data
Documentation accurately reflects all changes made
Monitoring and alerting continue to function properly
Compliance requirements continue to be met
User workflows complete successfully end-to-end
API contracts remain intact or properly versioned
Database integrity is maintained throughout changes
Configuration changes deploy successfully across environments
Third-party integrations continue to function as expected
Backup and recovery procedures remain operational
Accessibility standards are maintained or improved


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/