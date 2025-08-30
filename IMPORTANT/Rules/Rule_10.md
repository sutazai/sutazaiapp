Rule 10: Functionality-First Cleanup - Never Delete Blindly
✅ Required Practices:
Investigation and Analysis Phase:

Search for all references and dependencies across the entire codebase, documentation, and configuration
Understand purpose and current usage through code analysis, Git history, and stakeholder consultation
Test system functionality without it in isolated environments before production removal
Archive before deletion if uncertain with detailed metadata and restoration procedures
Document what was removed and why with comprehensive impact analysis and decision rationale
Perform recursive dependency analysis to identify indirect dependencies and usage patterns
Review Git commit history to understand original implementation intent and evolution
Analyze code metrics and usage statistics to determine actual utilization patterns
Interview original developers or maintainers when possible for historical context
Review bug reports and incident logs for any references to the target functionality
Check external documentation, wikis, and knowledge bases for usage references
Analyze API logs and monitoring data for actual runtime usage patterns
Review configuration files and environment variables for hidden dependencies
Check deployment scripts and infrastructure code for usage references
Analyze database schemas and data migrations for structural dependencies
Review test suites for both direct and indirect test coverage
Check continuous integration pipelines for build and deployment dependencies
Analyze monitoring and alerting configurations for observability dependencies
Review backup and recovery procedures for data and configuration dependencies
Check compliance and audit requirements for retention and functionality needs

Risk Assessment and Safety Measures:

Categorize cleanup impact as low, medium, or high risk based on comprehensive analysis
Identify all stakeholders who might be affected by the removal
Assess potential business impact and user experience degradation
Evaluate security implications of removing security-related functionality
Analyze performance impact of removing optimization or caching mechanisms
Assess compliance and regulatory implications of removing audit or logging functionality
Identify potential data loss or corruption risks from removing data processing logic
Evaluate integration impact on external systems and third-party services
Assess rollback complexity and time requirements for emergency restoration
Identify required approvals and sign-offs for different risk categories
Plan staged removal approach for high-risk cleanup operations
Establish monitoring and alerting for post-cleanup validation
Prepare communication plans for stakeholders and affected teams
Create contingency plans for different failure scenarios
Establish clear go/no-go criteria based on risk assessment results

Testing and Validation Procedures:

Create isolated test environments that mirror production configurations
Execute comprehensive functional testing without the target functionality
Perform integration testing to validate external system interactions
Run performance testing to ensure no degradation in system performance
Execute security testing to validate that removal doesn't introduce vulnerabilities
Perform user acceptance testing for user-facing functionality removal
Run load testing to ensure system stability under normal and peak conditions
Execute disaster recovery testing to validate backup and recovery procedures
Perform compliance testing to ensure regulatory requirements are still met
Run automated test suites to catch any regression in existing functionality
Test rollback procedures to ensure rapid restoration capability if needed
Validate monitoring and alerting systems continue to function properly
Test configuration management and deployment procedures
Perform data integrity validation for any database or storage changes
Execute cross-platform and cross-browser testing for user interface changes

Documentation and Communication Requirements:

Create detailed removal plans with step-by-step procedures and timelines
Document all discovered dependencies and their resolution or migration strategies
Record decision rationale including risk assessment and mitigation strategies
Update architectural documentation to reflect system changes
Revise user documentation and help guides to remove references
Update API documentation and change logs for external consumers
Communicate planned removals to all affected teams and stakeholders
Create migration guides for users or systems dependent on removed functionality
Document rollback procedures with exact restoration steps and time estimates
Update monitoring and alerting documentation for any observability changes
Revise troubleshooting guides and runbooks to reflect system changes
Create post-removal validation checklists and acceptance criteria
Document lessons learned and improve cleanup procedures for future use
Update team knowledge bases and training materials
Record cleanup metrics and timelines for process improvement

🚫 Forbidden Practices:

Deleting code without comprehensive investigation of its purpose and dependencies
Removing functionality based on assumptions without validating actual usage patterns
Cleaning up during high-traffic periods or critical business operations
Removing code without proper backup and restoration procedures in place
Deleting functionality without stakeholder notification and approval processes
Removing security or compliance-related code without proper security review
Cleaning up production systems without testing in staging environments first
Deleting code that affects external integrations without partner notification
Removing monitoring or logging functionality without replacement observability
Cleaning up during deployment freezes or change control restriction periods
Deleting code without understanding the original problem it was designed to solve
Removing functionality that may be seasonally used without considering usage cycles
Cleaning up code that affects data integrity or regulatory compliance requirements
Deleting functionality without considering backward compatibility requirements
Removing code that may be referenced in legal or contractual obligations
Cleaning up without proper version control and change tracking procedures
Deleting functionality during major system migrations or infrastructure changes
Removing code without validating that replacement functionality is fully operational
Cleaning up critical path functionality without comprehensive redundancy validation
Deleting code that affects disaster recovery or business continuity procedures
Removing functionality without proper deprecation periods and user notification
Cleaning up code that may be needed for debugging or forensic analysis
Deleting functionality that affects system performance or resource utilization
Removing code without considering the impact on automated testing and CI/CD pipelines

Cleanup Methodology by Component Type:
Code and Logic Cleanup:

Analyze function and method usage through static code analysis tools
Use runtime profiling to identify actually executed code paths
Review code coverage reports to identify untested and potentially unused code
Analyze import dependencies and module usage patterns
Check for dynamic code execution that might not be caught by static analysis
Review reflection and metaprogramming usage that might create hidden dependencies
Validate that interfaces and abstract classes are not implemented elsewhere
Check for configuration-driven code execution that might be environment-specific
Analyze plugin and extension points that might dynamically load removed code
Review serialization and deserialization code that might depend on removed classes

Database and Schema Cleanup:

Analyze table and column usage through query log analysis
Review foreign key constraints and referential integrity requirements
Check stored procedures, triggers, and functions for dependencies
Analyze data migration scripts and backup procedures for structural dependencies
Review application code for dynamic SQL generation that might reference removed elements
Check reporting and analytics queries for dependencies on removed schema elements
Validate that removed schema elements are not used in compliance or audit procedures
Review data archival and retention policies for removed table dependencies
Analyze replication and synchronization procedures for schema dependencies
Check disaster recovery procedures for dependencies on removed database elements

Configuration and Infrastructure Cleanup:

Review environment-specific configuration files for usage of removed elements
Analyze infrastructure as code for dependencies on removed components
Check monitoring and alerting configurations for references to removed services
Review load balancer and proxy configurations for removed endpoint dependencies
Analyze security policies and access controls for removed resource references
Check backup and recovery procedures for dependencies on removed infrastructure
Review CI/CD pipelines for dependencies on removed configuration elements
Analyze container and orchestration configurations for removed service dependencies
Check network configurations and firewall rules for removed service references
Review capacity planning and resource allocation for removed component dependencies

Documentation and Knowledge Cleanup:

Review all user documentation for references to removed functionality
Check API documentation and developer guides for removed endpoint references
Analyze training materials and onboarding documentation for removed procedure references
Review troubleshooting guides and runbooks for removed diagnostic procedures
Check architectural documentation and system diagrams for removed component references
Analyze decision records and design documents for removed functionality context
Review compliance documentation for removed control or procedure references
Check security documentation for removed threat model or mitigation references
Analyze operational procedures for removed monitoring or maintenance tasks
Review vendor documentation and contracts for removed integration dependencies

Post-Cleanup Validation and Monitoring:
Immediate Validation:

Execute comprehensive smoke tests to validate core system functionality
Monitor error rates and system performance metrics for anomalies
Validate that all critical user workflows continue to function properly
Check external integrations and API consumers for any disruption
Monitor security scanning and vulnerability detection systems
Validate backup and recovery procedures continue to function correctly
Check compliance monitoring and reporting systems for any issues
Monitor application logs for any errors related to removed functionality
Validate that monitoring and alerting systems continue to function properly
Check deployment and rollback procedures for any issues

Extended Validation:

Monitor system performance and resource utilization for trends
Analyze user behavior and usage patterns for any negative impact
Review error logs and incident reports for any issues related to cleanup
Monitor external partner and integration feedback for any problems
Validate that business metrics and KPIs remain within acceptable ranges
Check regulatory compliance and audit trail completeness
Monitor security posture and vulnerability scanning results
Analyze cost and resource utilization changes from cleanup activities
Review customer support tickets and feedback for any cleanup-related issues
Validate that disaster recovery and business continuity plans remain effective

Long-term Monitoring:

Analyze trends in system performance and stability metrics
Review business impact and ROI from cleanup activities
Monitor technical debt reduction and code maintainability improvements
Analyze development velocity and deployment frequency changes
Review security posture improvements from cleanup activities
Monitor cost savings and resource optimization from infrastructure cleanup
Analyze team productivity and developer experience improvements
Review knowledge transfer effectiveness and documentation quality
Monitor compliance and audit efficiency improvements
Evaluate cleanup process effectiveness and identify improvement opportunities

Validation Criteria:

Comprehensive investigation has been completed with documented findings
Risk assessment has been performed with appropriate mitigation strategies
All affected stakeholders have been identified and consulted
Testing has been completed in environments that mirror production
Backup and rollback procedures have been tested and validated
Documentation has been updated to reflect all changes
Monitoring and alerting continue to function properly
No critical functionality has been inadvertently removed
Performance and security posture remain within acceptable parameters
Compliance and regulatory requirements continue to be met
External integrations and partner systems remain functional
User workflows and business processes continue to operate normally
Development and deployment procedures remain functional
Knowledge transfer has been completed for any procedural changes
Post-cleanup monitoring shows stable system behavior
Rollback procedures are available and tested for emergency restoration
Team training has been completed for any operational changes
Cost and resource utilization improvements are documented and validated
Technical debt reduction goals have been achieved
Process improvements have been identified and documented for future cleanup efforts


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/