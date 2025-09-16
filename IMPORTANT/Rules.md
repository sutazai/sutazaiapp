COMPLETE PROFESSIONAL CODEBASE STANDARDS - FULL INDEX
📑 COMPLETE 20 RULES + ALL REQUIREMENTS
📌 Rule 1: Real Implementation Only - No Fantasy Code
Requirement: Every line of code must work today, on current systems, with existing dependencies.
✅ Required Practices:

Use concrete, descriptive names: emailSender, userValidator, paymentProcessor
Import actual, installed libraries: import nodemailer from 'nodemailer'
Reference real APIs with documented endpoints and authentication
Write functions that have actual implementations, not placeholder stubs
Use environment variables that exist and are documented
Reference database tables, columns, and schemas that actually exist
Use file paths that are valid and accessible in target environments
Implement error handling for real, documented failure scenarios
Log to actual, configured log destinations (files, services, streams)
Pin dependencies to specific, tested version numbers
Use realistic test data that represents actual use cases
Reference configuration keys that are defined and documented
Call network endpoints that are reachable and monitored
Implement authentication mechanisms that are actually deployed
Use error codes and messages that correspond to real system responses
Reference documentation links that are valid and current
Handle timeouts and rate limits for actual service constraints
Use database connections that are configured and tested
Implement caching with actual cache stores (Redis, Memcached)
Reference monitoring and alerting systems that are operational
Use SSL certificates that are valid and not self-signed in production
Implement backup and recovery procedures that are tested
Use load balancers and reverse proxies that are configured
Reference container registries and image repositories that exist
Implement health checks that actually validate service status
Use message queues and event streams that are configured
Reference secrets management systems that are operational
Implement rate limiting using actual throttling mechanisms
Use CDN endpoints that are configured and accessible

#### Use Testing Protocol (Use Playwright MCP) for every testing aspect
```bash
npx playwright test
npx playwright test --ui
npx playwright test --project=chromium
npx playwright test example
npx playwright test --debug
npx playwright codegen

## Critical Instructions
- Start again from the beginning if you have to
- Make sure to follow these steps
- Make sure to fully test every change properly every step of the way

## Testing Requirements - Use Playwright MCP for Proper Testing
```bash
npx playwright test          # Runs the end-to-end tests
npx playwright test --ui     # Starts the interactive UI mode
npx playwright test --project=chromium  # Runs the tests only on Desktop Chrome
npx playwright test example  # Runs the tests in a specific file
npx playwright test --debug  # Runs the tests in debug mode
npx playwright codegen      # Auto generate tests with Codegen
🚫 Forbidden Practices:

Abstract service names: mailService, automationHandler, intelligentSystem
Placeholder comments: // TODO: add AI automation here, // magic happens
Fictional integrations: imports from non-existent packages or "future" APIs
Theoretical abstractions: code that assumes capabilities we don't have
Imaginary infrastructure: references to systems that don't exist
Mock implementations in production code paths
Hardcoded localhost or development URLs in production builds
References to non-existent database tables or columns
Placeholder data that doesn't represent real scenarios
Comments suggesting features that aren't implemented
Abstract interfaces without concrete implementations available
Theoretical error codes that don't exist in the system
References to undefined or missing configuration keys
Imports from local development paths not in production
Usage of experimental or unstable API versions
Assumptions about "future" system capabilities
References to non-existent environment variables
Calls to endpoints that don't exist or aren't documented
Usage of libraries not listed in dependency manifests
References to monitoring systems that aren't configured
Placeholder authentication or authorization mechanisms
File operations on paths that don't exist in target systems
Database queries against non-existent schemas
Integration with services that aren't accessible
Magic strings or numbers without defined constants
Assumptions about infinite resources or perfect networks
References to "eventual" or "planned" infrastructure
Theoretical scaling patterns not implemented
Imaginary performance characteristics
References to non-existent security policies
Placeholder encryption or hashing algorithms
Assumptions about zero-latency operations
References to ideal-world deployment scenarios
Theoretical data consistency models not implemented

Validation Criteria:

All imports resolve to actual installed packages
All API calls reference documented, accessible endpoints
All environment variables are defined in configuration
All functions contain working implementations
No TODOs referencing non-existent systems or capabilities
All database references point to existing tables and columns
All file paths are accessible in deployment environments
All network calls have proper error handling and timeouts
All configuration keys are documented in environment configs
All test scenarios use realistic, representative data
All dependencies are pinned to stable, tested versions
All external services have health checks and monitoring
All error messages reference real, documented error conditions
All logging destinations are configured and accessible
All documentation links resolve to valid, current resources
All authentication mechanisms are implemented and tested
All cache references point to configured cache stores
All monitoring and alerting integrations are operational
All rate limiting and timeout values match actual service limits
All database connections are properly configured and pooled
All SSL/TLS configurations use valid certificates
All backup procedures are documented and tested
All load balancing configurations are verified and functional
All container images exist in accessible registries
All health check endpoints return meaningful status information
All message queue configurations are tested and monitored
All secrets are stored in configured management systems
All throttling mechanisms are implemented and tested
All CDN configurations are verified and accessible
All CI/CD pipeline steps are functional and repeatable

📌 Rule 2: Never Break Existing Functionality
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

📌 Rule 3: Comprehensive Analysis Required
Requirement: Conduct thorough, systematic review of the entire application ecosystem before any change is made.
✅ Required Practices:

Analyze complete codebase structure and dependencies before any modifications
Map all data flows and system interactions across components
Review all configuration files, environment variables, and settings
Examine all database schemas, relationships, and data integrity constraints
Investigate all API endpoints, webhooks, and external integrations
Analyze all deployment pipelines, CI/CD processes, and automation
Review all monitoring, logging, and alerting configurations
Examine all security policies, authentication, and authorization mechanisms
Investigate all performance characteristics and resource utilization
Analyze all user workflows and business process dependencies
Review all documentation, including technical specs and user guides
Examine all test coverage, including unit, integration, and end-to-end tests
Investigate all third-party service dependencies and integrations
Analyze all infrastructure components and their configurations
Review all disaster recovery and backup procedures
Examine all compliance requirements and regulatory constraints
Investigate all error handling and failure recovery mechanisms
Analyze all caching strategies and data storage patterns
Review all communication protocols and message formats
Examine all scheduled tasks, cron jobs, and background processes
Investigate all load balancing and scaling configurations
Analyze all networking and firewall rules
Review all container and orchestration configurations
Examine all secrets management and encryption implementations
Investigate all audit trails and logging mechanisms
Analyze all performance bottlenecks and optimization opportunities
Review all user interface components and interaction patterns
Examine all data migration and synchronization processes
Investigate all version control and branching strategies
Analyze all quality assurance and testing procedures

🚫 Forbidden Practices:

Making assumptions about system behavior without verification
Skipping analysis of "obvious" or "simple" components
Ignoring indirect dependencies and side effects
Focusing only on direct code changes without system context
Overlooking configuration and environment dependencies
Ignoring performance and scalability implications
Skipping security and compliance impact analysis
Overlooking user experience and workflow implications
Ignoring infrastructure and deployment considerations
Skipping analysis of error handling and edge cases
Overlooking monitoring and observability requirements
Ignoring disaster recovery and business continuity impacts
Skipping analysis of third-party service dependencies
Overlooking data consistency and integrity requirements
Ignoring regulatory and compliance implications
Skipping analysis of existing technical debt
Overlooking cross-team and cross-system dependencies
Ignoring historical context and design decisions
Skipping analysis of testing and quality assurance gaps
Overlooking documentation and knowledge management needs
Ignoring resource utilization and cost implications
Skipping analysis of maintenance and operational overhead
Overlooking accessibility and internationalization requirements
Ignoring mobile and cross-platform considerations
Skipping analysis of concurrent access and race conditions
Overlooking backup and recovery implications
Ignoring vendor lock-in and technology risks
Skipping analysis of team knowledge and skill requirements

System Component Analysis:
Frontend Analysis:

Component architecture and reusability patterns
State management and data flow patterns
Routing and navigation structures
Asset management and optimization strategies
Browser compatibility and progressive enhancement
Accessibility compliance and usability patterns
Performance characteristics and loading strategies
Error handling and user feedback mechanisms
Internationalization and localization support
Mobile responsiveness and touch interactions

Backend Analysis:

Service architecture and communication patterns
Database design and relationship structures
API design and versioning strategies
Authentication and authorization mechanisms
Error handling and logging patterns
Performance characteristics and bottlenecks
Caching strategies and data storage patterns
Background job processing and queuing
External service integrations and dependencies
Security implementations and vulnerability assessments

Infrastructure Analysis:

Server configurations and resource allocations
Network architecture and security rules
Load balancing and failover mechanisms
Container orchestration and scaling policies
Storage systems and backup procedures
Monitoring and alerting configurations
Disaster recovery and business continuity plans
Cost optimization and resource efficiency
Compliance and regulatory requirements
Vendor dependencies and service level agreements

Data Analysis:

Database schema design and relationships
Data integrity constraints and validation rules
Migration scripts and versioning strategies
Backup and recovery procedures
Data retention and archival policies
Privacy and compliance requirements
Performance characteristics and indexing strategies
Replication and synchronization mechanisms
Data access patterns and query optimization
Analytics and reporting capabilities

Security Analysis:

Authentication and authorization mechanisms
Encryption implementations and key management
Input validation and sanitization procedures
Access control and privilege management
Audit logging and compliance tracking
Vulnerability assessment and penetration testing
Security monitoring and incident response
Data privacy and protection measures
Third-party security dependencies
Regulatory compliance and certification requirements

Performance Analysis:

Response time characteristics and bottlenecks
Resource utilization patterns and optimization opportunities
Caching strategies and effectiveness
Database query performance and optimization
Network latency and bandwidth considerations
Concurrent user handling and scaling limits
Memory usage patterns and garbage collection
CPU utilization and processing efficiency
Storage I/O patterns and optimization
CDN and edge caching effectiveness

Integration Analysis:

Third-party service dependencies and SLAs
API contracts and versioning strategies
Webhook and callback mechanisms
Message queuing and event streaming
Data synchronization and consistency patterns
Error handling and retry mechanisms
Rate limiting and throttling configurations
Authentication and authorization for integrations
Monitoring and alerting for external dependencies
Fallback and degradation strategies

Documentation Requirements:

Create comprehensive system analysis report include exact time and date in report name
Document all discovered dependencies and relationships
Map all data flows and system interactions
Record all configuration dependencies and requirements
Document all security and compliance considerations
Create risk assessment and mitigation strategies
Document all performance characteristics and bottlenecks
Record all infrastructure and deployment dependencies
Document all user workflow and business process impacts
Create detailed change impact analysis
Record all testing and validation requirements
Document all rollback and recovery procedures
Create timeline and resource requirements for changes
Document all stakeholder communication needs
Record all training and knowledge transfer requirements

Validation Criteria:

All system components have been analyzed and documented
All dependencies and relationships have been mapped
All configuration requirements have been identified
All security implications have been assessed
All performance impacts have been evaluated
All compliance requirements have been verified
All user workflow impacts have been analyzed
All infrastructure dependencies have been mapped
All testing requirements have been identified
All rollback procedures have been planned
All stakeholder impacts have been communicated
All resource requirements have been estimated
All timeline constraints have been identified
All risk mitigation strategies have been developed
All knowledge gaps have been identified and addressed
All documentation has been reviewed and approved
All analysis findings have been validated by subject matter experts
All change recommendations have been prioritized and planned
All implementation strategies have been evaluated
All success criteria have been defined and agreed upon

📌 Rule 4: Investigate Existing Files & Consolidate First
Requirement: Exhaustively search for existing files and consolidate improvements into them rather than creating duplicates.
✅ Required Practices:

MANDATORY FIRST STEP: Read and review CHANGELOG.md thoroughly - this contains every record of changes
/opt/sutazaiapp/TODO.md      # Deeply understand ALL pending tasks, priorities, known issues
/opt/sutazaiapp/changelog.md # Deeply review complete change history - understand EVERYTHING done
ALWAYS investigate existing files before creating new ones
ALWAYS prefer editing existing files to creating new files
Search entire codebase for existing documentation, scripts, or code
Consolidate improvements into current files rather than duplicating
Only create new files when absolutely necessary for the goal
Apply DRY principles consistently across all content
Study CHANGELOG.md to understand historical context and decision rationale
Review all previous changes to understand why existing files were created or modified
Analyze CHANGELOG.md patterns to understand team conventions and standards
Use CHANGELOG.md to identify recent changes that might affect your work
Cross-reference CHANGELOG.md entries with current codebase state
Understand from CHANGELOG.md which files are actively maintained vs deprecated
Use comprehensive search tools to find similar functionality across the codebase
Review Git history to understand why existing files were created
Analyze existing file patterns and naming conventions before additions
Check for related functionality in different directories and modules
Investigate configuration files and environment-specific variations
Search for similar documentation in wikis, README files, and docs folders
Review existing test files before creating new test suites
Check for similar utility functions across different modules
Investigate existing build scripts and automation before creating new ones
Search for existing database migration scripts and schema changes
Review existing API endpoints before creating new routes
Check for existing component libraries and reusable UI elements
Investigate existing error handling patterns and logging mechanisms
Search for existing validation and sanitization functions
Review existing authentication and authorization implementations
Check for existing monitoring and alerting configurations
Investigate existing deployment and infrastructure scripts
Search for existing data processing and transformation functions
Review existing integration patterns with third-party services
Check for existing caching and optimization implementations
Investigate existing backup and recovery procedures
Search for existing compliance and security implementations
Review existing performance testing and benchmarking tools
Check for existing documentation templates and style guides

🚫 Forbidden Practices:

Skipping CHANGELOG.md review before making any changes or creating files
Creating new files without thorough investigation of existing ones
Duplicating functionality that already exists elsewhere in the codebase
Ignoring historical context and decision rationale from CHANGELOG.md
Ignoring similar implementations in different parts of the system
Creating new documentation when existing docs could be updated
Writing new utility functions without checking for existing ones
Creating new configuration files without reviewing existing patterns
Implementing new features without checking for existing similar features
Creating new test files without reviewing existing test patterns
Writing new scripts without checking for existing automation
Creating new API endpoints without reviewing existing route patterns
Implementing new database schemas without checking existing structures
Creating new components without reviewing existing UI libraries
Writing new error handling without checking existing patterns
Creating new logging mechanisms without reviewing existing implementations
Implementing new authentication without checking existing systems
Creating new monitoring without reviewing existing observability tools
Writing new deployment scripts without checking existing procedures
Creating new data processing without reviewing existing pipelines
Implementing new integrations without checking existing patterns
Creating new caching without reviewing existing optimization strategies
Writing new backup procedures without checking existing systems
Creating new compliance implementations without reviewing existing controls
Implementing new security measures without checking existing protections
Creating new performance tools without reviewing existing benchmarks
Writing new documentation templates without checking existing styles
Creating new build processes without reviewing existing automation
Implementing new migration scripts without checking existing patterns
Creating new validation functions without reviewing existing implementations
Writing new transformation logic without checking existing processors
Creating new notification systems without reviewing existing mechanisms
Making decisions that contradict established patterns in CHANGELOG.md without justification
Repeating mistakes or approaches that failed according to CHANGELOG.md history

Investigation Methodology:
MANDATORY FIRST STEP - CHANGELOG.md Analysis:

Read CHANGELOG.md completely before any investigation or changes
Study every entry to understand historical context and decision rationale
Identify patterns in file creation, modification, and deletion decisions
Understand which files are actively maintained vs deprecated or archived
Review recent changes that might impact your planned modifications
Identify team members or agents responsible for similar changes
Understand established conventions and standards from historical entries
Cross-reference CHANGELOG.md entries with current codebase state
Note any discrepancies between CHANGELOG.md and actual file state
Identify related changes that were made together historically
Understand the reasoning behind previous consolidation or separation decisions
Review any rollback or reversal entries to understand what didn't work
Identify dependencies between files based on change history
Understand performance or security considerations from historical entries
Note any compliance or regulatory considerations mentioned in entries

Code Search Techniques:

Use grep -r "pattern" . for text-based searches across all files
Use find . -name "*.ext" -exec grep -l "pattern" {} \; for specific file types
Use IDE global search with regex patterns for complex searches
Search for function names, class names, and variable patterns
Use git grep for version-controlled content searches
Search for import statements and dependency usage
Use code analysis tools to find similar functions and patterns
Search for configuration keys and environment variable usage
Use database query tools to find similar schema patterns
Search for API endpoint patterns and route definitions

Documentation Search Strategies:

Search README files in all directories and subdirectories
Check wiki pages and internal documentation systems
Review confluence, notion, or other documentation platforms
Search for markdown files throughout the entire repository
Check comments and inline documentation in code files
Review API documentation and specification files
Search for architectural decision records (ADRs)
Check design documents and technical specifications
Review user guides and help documentation
Search for troubleshooting guides and FAQ sections

Configuration and Infrastructure Search:
[Content continues with all remaining search strategies and validation criteria...]
📌 Rule 5: Professional Project Standards
Requirement: Approach every task with enterprise-grade discipline and long-term thinking as if this were a mission-critical production system.
✅ Required Practices:

No trial-and-error in main branches
Respect structure and long-term maintainability
Every decision must be intentional and reviewed
Follow established software engineering best practices and design patterns
Implement comprehensive code review processes with multiple reviewers
Maintain detailed documentation for all architectural decisions and rationale
Use feature branches and pull requests for all changes, no direct commits to main
Implement proper testing strategies with unit, integration, and end-to-end coverage
Follow security-first development practices with regular vulnerability assessments
Maintain performance benchmarks and monitor for regressions continuously
Use semantic versioning and proper release management procedures
Implement proper error handling, logging, and monitoring throughout the system
Follow accessibility guidelines and ensure inclusive design practices
Maintain consistent coding standards enforced through automated tooling
Use infrastructure as code for all environment management and deployment
Implement proper backup and disaster recovery procedures with regular testing
Maintain comprehensive API documentation and change management
Follow data privacy and compliance requirements (GDPR, HIPAA, etc.)
Implement proper secret management and credential rotation policies
Use dependency management with security scanning and license compliance
Maintain staging environments that mirror production configurations
Implement proper change management with rollback procedures
Follow established incident response and post-mortem procedures
Maintain technical debt tracking and regular remediation cycles
Implement proper capacity planning and resource monitoring
Use established communication protocols for technical decisions
Maintain knowledge management systems and team documentation
Follow proper project planning with clear milestones and deliverables
Implement stakeholder communication and progress reporting
Use established risk assessment and mitigation procedures

🚫 Forbidden Practices:

Experimental or untested code in production branches
Making changes without proper planning and impact assessment
Skipping code reviews or approval processes for any changes
Implementing features without proper requirements and acceptance criteria
Using undocumented or unsupported third-party libraries and dependencies
Hardcoding configuration values or credentials in source code
Deploying changes without proper testing in staging environments
Making breaking changes without proper versioning and migration plans
Implementing security measures without proper review and testing
Using deprecated or end-of-life technologies and frameworks
Skipping performance testing and capacity planning for new features
Making architectural decisions without proper consultation and documentation
Implementing user-facing changes without proper usability testing
Using personal or development credentials in production environments
Skipping backup verification and disaster recovery testing
Making database changes without proper migration scripts and rollback plans
Implementing monitoring and alerting without proper threshold setting
Using manual deployment processes without automation and validation
Making compliance-related changes without proper legal and security review
Skipping accessibility testing and inclusive design considerations
Using unlicensed or incompatible software components
Making infrastructure changes without proper change control
Implementing data processing without proper privacy and security controls
Using experimental or beta software in production environments
Making cross-team changes without proper coordination and communication
Skipping documentation updates for system changes and new features
Using personal development tools without security and compliance approval
Making performance optimizations without proper benchmarking and validation
Implementing new technologies without proper evaluation and approval
Skipping team training and knowledge transfer for new systems
Using quick fixes or workarounds without proper long-term solutions

[Content continues with all Quality Assurance Standards, Security and Compliance Standards, Performance and Scalability Standards, Documentation and Communication Standards, Project Management Standards, Risk Management and Governance, Team Collaboration Standards, and Validation Criteria...]
📌 Rule 6: Centralized Documentation
Requirement: Maintain comprehensive, organized, and current documentation as a critical component of the codebase infrastructure.
✅ Complete Documentation Structure:
/docs/
├── overview.md                 # Project summary & goals
├── setup/
│   ├── local_dev.md           # Development environment setup
│   ├── environments.md        # Configuration & secrets management
│   ├── dependencies.md        # System requirements & package installation
│   ├── troubleshooting.md     # Common setup issues & solutions
│   └── tools.md               # Required development tools & IDEs
├── architecture/
│   ├── system_design.md       # High-level architecture overview
│   ├── api_reference.md       # Endpoint specifications & examples
│   ├── data_flow.md           # Information flow diagrams
│   ├── database_schema.md     # Database design & relationships
│   ├── security_model.md      # Authentication & authorization design
│   ├── integration_patterns.md # External service integration approaches
│   ├── caching_strategy.md    # Caching layers & invalidation policies
│   └── scalability_plan.md    # Performance & scaling considerations
├── development/
│   ├── coding_standards.md    # Style guides & best practices
│   ├── git_workflow.md        # Branching strategy & commit conventions
│   ├── testing_strategy.md    # Testing approaches & frameworks
│   ├── code_review.md         # Review process & checklists
│   ├── debugging_guide.md     # Debugging tools & techniques
│   └── performance_tuning.md  # Optimization guidelines & profiling
├── operations/
│   ├── deployment/
│   │   ├── pipeline.md        # CI/CD processes & automation
│   │   ├── procedures.md      # Manual deployment steps
│   │   ├── rollback.md        # Emergency rollback procedures
│   │   └── environments.md    # Production, staging, dev configs
│   ├── monitoring/
│   │   ├── observability.md   # Logging, metrics, & tracing
│   │   ├── alerts.md          # Alert configurations & responses
│   │   ├── dashboards.md      # Monitoring dashboard guides
│   │   └── incident_response.md # Incident handling procedures
│   ├── infrastructure/
│   │   ├── provisioning.md    # Infrastructure setup & management
│   │   ├── networking.md      # Network configuration & security
│   │   ├── storage.md         # Data storage & backup strategies
│   │   └── disaster_recovery.md # DR procedures & testing
│   └── maintenance/
│       ├── backups.md         # Backup procedures & restoration
│       ├── upgrades.md        # System upgrade procedures
│       ├── security.md        # Security maintenance & patching
│       └── capacity_planning.md # Resource planning & scaling
├── user_guides/
│   ├── getting_started.md     # New user onboarding
│   ├── feature_guides/        # Feature-specific documentation
│   ├── tutorials/             # Step-by-step learning materials
│   ├── faq.md                 # Frequently asked questions
│   ├── best_practices.md      # User best practices & tips
│   └── migration_guides/      # Version upgrade instructions
├── api/
│   ├── authentication.md      # Auth methods & token management
│   ├── endpoints/             # Individual endpoint documentation
│   ├── examples/              # Code examples & use cases
│   ├── error_codes.md         # Error handling & status codes
│   ├── rate_limiting.md       # API limits & throttling
│   └── webhooks.md            # Webhook configuration & handling
├── compliance/
│   ├── security_policies.md   # Security requirements & policies
│   ├── privacy_policy.md      # Data privacy & GDPR compliance
│   ├── audit_logs.md          # Audit trail requirements
│   ├── regulatory.md          # Industry-specific compliance
│   └── certifications.md      # Security certifications & assessments
├── team/
│   ├── onboarding.md          # New team member guide
│   ├── roles_responsibilities.md # Team structure & ownership
│   ├── communication.md       # Communication protocols & channels
│   ├── decision_making.md     # Technical decision processes
│   └── knowledge_sharing.md   # Learning & development practices
├── reference/
│   ├── glossary.md            # Technical terms & definitions
│   ├── acronyms.md            # Abbreviations & their meanings
│   ├── external_links.md      # Useful external resources
│   ├── vendor_docs.md         # Third-party service documentation
│   └── standards.md           # Industry standards & specifications
├── templates/
│   ├── adr_template.md        # Architectural Decision Record template
│   ├── runbook_template.md    # Operational runbook template
│   ├── incident_report.md     # Incident report template
│   └── feature_spec.md        # Feature specification template
└── changelog.md               # Release history & changes
✅ Documentation Standards:

Lowercase, hyphen-separated filenames for consistency
Consistent Markdown formatting with standardized headers
Clear ownership and last-updated dates in document headers
Update with every change that affects behavior or procedures
Use standardized document templates for consistency
Include table of contents for documents longer than 10 sections
Use consistent code block formatting with language specification
Include cross-references and linking between related documents
Use standardized image formats and alt text for accessibility
Implement consistent terminology and avoid jargon without explanation
Include examples and practical use cases in all procedural documentation
Use consistent date formats (YYYY-MM-DD) throughout all documentation
Include version numbers for API and feature documentation
Use standardized warning and note callout formats
Implement consistent formatting for commands, paths, and configuration
Include prerequisites and assumptions at the beginning of procedures
Use numbered lists for sequential procedures and bullet points for options
Include validation steps and expected outcomes for all procedures
Use consistent heading hierarchy (H1 for title, H2 for major sections)
Include related documentation links in see-also sections

🚫 Forbidden Practices:

Creating documentation in multiple scattered locations
Using inconsistent formatting or style across documents
Leaving outdated or incorrect information without updates
Creating documents without clear ownership or maintenance responsibility
Using technical jargon without providing definitions or context
Creating documentation that duplicates existing content unnecessarily
Using personal or temporary links that may become inaccessible
Creating documents without proper version control and change tracking
Using screenshots without alt text or descriptions for accessibility
Creating procedures without testing or validation steps
Using absolute file paths or environment-specific references
Creating documentation without considering different user skill levels
Using inconsistent terminology or naming conventions
Creating documents that assume undocumented prerequisite knowledge
Using placeholder content or "TODO" sections in published documentation
Creating documents without proper review and approval processes
Using external dependencies without backup or alternative references
Creating documentation that violates security or compliance requirements
Using copyrighted content without proper attribution or licensing
Creating documents without considering internationalization requirements

[Content continues with all remaining Document Lifecycle Management practices...]
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
	
Rule 7: Script Organization & Control
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

📌 Rule 8: Python Script Excellence
✅ Required Practices:

Clear location and purpose with logical organization within script directories
Comprehensive docstrings following PEP 257 and Google/NumPy documentation styles
CLI argument support using argparse or click with comprehensive help text
Proper logging instead of print statements with configurable log levels
Production-ready code quality following PEP 8 and modern Python best practices
Remove all debugging/experimental scripts from production repositories
Use type hints throughout all functions and classes for better code clarity
Implement comprehensive error handling with specific exception types
Follow single responsibility principle with focused, modular functions
Use virtual environments and requirements.txt for dependency management
Implement proper configuration management using config files or environment variables
Use context managers for resource management (files, connections, locks)
Implement proper input validation and sanitization for all user inputs
Use dataclasses or Pydantic models for structured data handling
Implement proper testing with unittest, pytest, or similar frameworks
Use proper package structure with init.py files and clear module organization
Implement proper signal handling for graceful shutdown and cleanup
Use async/await patterns for I/O-bound operations where appropriate
Implement proper memory management and resource cleanup
Use established design patterns (Factory, Observer, Strategy) where appropriate
Implement proper security measures for handling sensitive data
Use proper versioning and compatibility checking for dependencies
Implement proper performance monitoring and optimization
Use established linting tools (pylint, flake8, black) for code quality
Implement proper internationalization support for user-facing messages
Use f-strings for string formatting and avoid old-style formatting methods
Implement proper exception chaining and context preservation
Use pathlib for file system operations instead of os.path
Implement proper concurrent execution with threading or multiprocessing
Use proper database connection pooling and transaction management
Implement proper caching strategies for expensive operations
Use proper serialization formats (JSON, MessagePack) for data exchange
Implement proper retry mechanisms with exponential backoff
Use proper HTTP client libraries with session management and connection pooling
Implement proper metrics collection and performance monitoring
Use proper secrets management and credential handling
Implement proper backup and recovery procedures for data operations
Use proper progress indicators for long-running operations
Implement proper health checks and status reporting
Use proper documentation generation with Sphinx or similar tools

🚫 Forbidden Practices:

Using print() statements for logging or user feedback in production scripts
Creating scripts without comprehensive documentation and usage examples
Implementing CLI interfaces without proper argument validation and help text
Using global variables without clear necessity and proper encapsulation
Creating scripts without proper error handling and exception management
Using hardcoded values for configuration, paths, URLs, or credentials
Implementing scripts without proper testing and validation procedures
Creating monolithic scripts that perform multiple unrelated functions
Using deprecated Python features or libraries without upgrade plans
Implementing scripts without proper logging configuration and management
Creating scripts that don't follow PEP 8 style guidelines and conventions
Using bare except clauses that catch all exceptions without specificity
Implementing file operations without proper error handling and cleanup
Creating scripts without proper virtual environment and dependency management
Using eval() or exec() functions without extreme necessity and security measures
Implementing scripts without proper input validation and sanitization
Creating scripts that modify global state without clear documentation
Using string concatenation for building file paths instead of pathlib
Implementing database operations without proper connection management
Creating scripts without proper signal handling and graceful shutdown
Using synchronous operations for I/O-bound tasks that could be asynchronous
Implementing scripts without proper memory management and resource cleanup
Creating scripts that don't handle different operating systems appropriately
Using mutable default arguments in function definitions
Implementing scripts without proper version checking and compatibility validation
Creating scripts that expose sensitive information in logs or error messages
Using shell=True in subprocess calls without proper input sanitization
Implementing scripts without proper timeout handling for external operations
Creating scripts that don't follow established security best practices
Using pickle for serialization of untrusted data without security considerations
Implementing scripts without proper internationalization and localization support
Creating scripts that don't handle Unicode and encoding issues properly
Using outdated libraries or dependencies with known security vulnerabilities
Implementing scripts without proper documentation of assumptions and limitations
Creating scripts that can't be safely interrupted or resumed

Script Structure and Organization:
Standard Script Template:
python#!/usr/bin/env python3
"""
Script Name: descriptive_script_name.py
Purpose: Clear description of what this script does and why it exists
Author: Team/Individual responsible for maintenance
Created: YYYY-MM-DD HH:MM:SS UTC
Last Modified: YYYY-MM-DD HH:MM:SS UTC
Version: X.Y.Z

Usage:
    python descriptive_script_name.py [options]
    
Examples:
    python descriptive_script_name.py --input data.csv --output results.json
    python descriptive_script_name.py --config config.yaml --dry-run
    
Requirements:
    - Python 3.8+
    - Required packages listed in requirements.txt
    - Environment variables: VAR1, VAR2
    - External dependencies: database access, API credentials

Execution History:
    - 2024-01-15 10:30:45 UTC: Initial creation and basic functionality
    - 2024-01-16 14:22:30 UTC: Added error handling and logging
    - 2024-01-17 09:15:12 UTC: Implemented configuration management
    - 2024-01-18 16:45:38 UTC: Added CLI argument validation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import time
import datetime

# Script execution start time for tracking
SCRIPT_START_TIME = datetime.datetime.now(tz=datetime.timezone.utc)
EXECUTION_ID = f"exec_{SCRIPT_START_TIME.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

# Configure logging with precise timestamps
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s.%(msecs)03d UTC - {EXECUTION_ID} - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'script_{SCRIPT_START_TIME.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log script initiation with exact timestamp
logger.info(f"Script initiated at {SCRIPT_START_TIME.isoformat()} with execution ID: {EXECUTION_ID}")

class ScriptError(Exception):
    """Custom exception for script-specific errors."""
    pass

class ConfigurationManager:
    """Manages script configuration from files and environment variables."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from file and environment."""
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        logger.info(f"Loading configuration at {start_time.isoformat()}")
        
        # Implementation details...
        
        end_time = datetime.datetime.now(tz=datetime.timezone.utc)
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Configuration loaded in {duration:.6f}s at {end_time.isoformat()}")

def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        shutdown_time = datetime.datetime.now(tz=datetime.timezone.utc)
        total_runtime = (shutdown_time - SCRIPT_START_TIME).total_seconds()
        logger.info(f"Received signal {signum} at {shutdown_time.isoformat()}, shutting down gracefully after {total_runtime:.6f}s runtime...")
        # Cleanup operations...
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive validation."""
    parser = argparse.ArgumentParser(
        description="Detailed description of script functionality",
        epilog="Additional usage information and examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input file path (required)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (default: results.json)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode without making changes'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity level'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()

def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input arguments and prerequisites."""
    validation_start = datetime.datetime.now(tz=datetime.timezone.utc)
    logger.info(f"Starting input validation at {validation_start.isoformat()}")
    
    if not args.input.exists():
        raise ScriptError(f"Input file does not exist: {args.input}")
    
    if not args.input.is_file():
        raise ScriptError(f"Input path is not a file: {args.input}")
    
    # Additional validation logic...
    
    validation_end = datetime.datetime.now(tz=datetime.timezone.utc)
    validation_duration = (validation_end - validation_start).total_seconds()
    logger.info(f"Input validation completed in {validation_duration:.6f}s at {validation_end.isoformat()}")

def main() -> int:
    """Main script execution function."""
    try:
        # Set up signal handlers
        setup_signal_handlers()
        
        # Parse arguments
        args_start = datetime.datetime.now(tz=datetime.timezone.utc)
        args = parse_arguments()
        args_end = datetime.datetime.now(tz=datetime.timezone.utc)
        logger.info(f"Arguments parsed in {(args_end - args_start).total_seconds():.6f}s at {args_end.isoformat()}")
        
        # Configure logging level based on verbosity
        if args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.verbose >= 1:
            logging.getLogger().setLevel(logging.INFO)
        
        # Validate inputs
        validate_inputs(args)
        
        # Initialize configuration
        config_manager = ConfigurationManager(args.config)
        
        main_logic_start = datetime.datetime.now(tz=datetime.timezone.utc)
        logger.info(f"Starting main script logic at {main_logic_start.isoformat()}")
        
        # Main script logic here...
        
        main_logic_end = datetime.datetime.now(tz=datetime.timezone.utc)
        total_runtime = (main_logic_end - SCRIPT_START_TIME).total_seconds()
        main_logic_duration = (main_logic_end - main_logic_start).total_seconds()
        
        logger.info(f"Main logic completed in {main_logic_duration:.6f}s at {main_logic_end.isoformat()}")
        logger.info(f"Script completed successfully with total runtime {total_runtime:.6f}s")
        return 0
        
    except ScriptError as e:
        error_time = datetime.datetime.now(tz=datetime.timezone.utc)
        runtime = (error_time - SCRIPT_START_TIME).total_seconds()
        logger.error(f"Script error at {error_time.isoformat()} after {runtime:.6f}s runtime: {e}")
        return 1
    except KeyboardInterrupt:
        interrupt_time = datetime.datetime.now(tz=datetime.timezone.utc)
        runtime = (interrupt_time - SCRIPT_START_TIME).total_seconds()
        logger.info(f"Script interrupted by user at {interrupt_time.isoformat()} after {runtime:.6f}s runtime")
        return 130
    except Exception as e:
        error_time = datetime.datetime.now(tz=datetime.timezone.utc)
        runtime = (error_time - SCRIPT_START_TIME).total_seconds()
        logger.exception(f"Unexpected error at {error_time.isoformat()} after {runtime:.6f}s runtime: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
Code Quality Standards:
Documentation Requirements:

Module-level docstrings explaining purpose, usage, and requirements
Function and class docstrings following Google or NumPy style conventions
Inline comments for complex logic and business rules
Type hints for all function parameters and return values
Clear variable and function names that express intent
README files for complex scripts with setup and usage instructions
Example usage and common use cases documented
Error codes and troubleshooting guide included
Performance characteristics and limitations documented
Dependencies and external requirements clearly listed

Error Handling and Logging:

Use specific exception types instead of generic Exception class
Implement proper exception chaining to preserve error context
Log errors with appropriate severity levels and structured information
Include correlation IDs for tracking related operations
Implement proper retry mechanisms with exponential backoff
Use circuit breaker patterns for external service dependencies
Log performance metrics and execution statistics
Implement proper audit logging for security-critical operations
Use structured logging formats (JSON) for machine readability
Include contextual information in all log messages

Performance and Scalability:

Use generators and iterators for memory-efficient data processing
Implement proper connection pooling for database and HTTP operations
Use async/await for I/O-bound operations where appropriate
Implement proper caching strategies for expensive computations
Use multiprocessing or threading for CPU-bound parallel operations
Monitor memory usage and implement garbage collection strategies
Use profiling tools to identify performance bottlenecks
Implement proper batch processing for large datasets
Use streaming APIs for large file processing
Implement proper resource limits and timeout handling

Security and Safety:

Validate and sanitize all user inputs to prevent injection attacks
Use proper authentication and authorization for external services
Implement proper secrets management without hardcoding credentials
Use secure communication protocols (HTTPS, TLS) for network operations
Implement proper input size limits to prevent resource exhaustion
Use parameterized queries for database operations
Implement proper file permission checking and access controls
Use secure temporary file creation with proper cleanup
Implement proper logging that doesn't expose sensitive information
Use encryption for sensitive data storage and transmission

Testing and Quality Assurance:
Unit Testing Requirements:

Comprehensive test coverage for all functions and methods
Use pytest or unittest framework with proper test organization
Implement test fixtures and mocks for external dependencies
Use property-based testing for complex logic validation
Implement performance regression testing for critical operations
Use code coverage tools to ensure adequate test coverage
Implement integration testing for external service interactions
Use mutation testing to validate test quality and effectiveness
Implement load testing for scripts handling high-volume operations
Use continuous integration to run tests automatically

Code Quality Tools:

Use black for automatic code formatting and style consistency
Implement pylint or flake8 for static code analysis and linting
Use mypy for static type checking and type safety validation
Implement bandit for security vulnerability scanning
Use isort for import organization and consistency
Implement pre-commit hooks for automated quality checking
Use dependency scanning tools for security vulnerability detection
Implement code complexity analysis with tools like radon
Use documentation linting tools to ensure documentation quality
Implement automated security scanning in CI/CD pipelines

Deployment and Operations:
Package Management:

Use requirements.txt with pinned versions for reproducible environments
Implement proper virtual environment management
Use pip-tools for dependency management and conflict resolution
Implement proper package building and distribution procedures
Use Docker containers for consistent execution environments
Implement proper environment-specific configuration management
Use proper secrets management systems for credential handling
Implement proper logging configuration for different environments
Use monitoring and alerting integration for production deployments
Implement proper backup and recovery procedures for script data

Validation Criteria:

All scripts follow established templates and coding standards
Comprehensive documentation exists with clear usage examples
Type hints are implemented throughout all functions and classes
Error handling covers all expected failure scenarios
Logging is properly configured with appropriate levels and formats
CLI interfaces provide comprehensive help and validation
Testing coverage meets established quality thresholds
Security measures are implemented and validated
Performance characteristics meet established requirements
Code quality tools pass without critical issues
Dependencies are properly managed and regularly updated
Documentation is current and accessible to team members
Integration with CI/CD pipelines is functional and reliable
Monitoring and alerting integration works correctly
Backup and recovery procedures are tested and documented
Team training on script usage and maintenance is completed
Scripts integrate properly with existing automation frameworks
Compliance and security requirements are met and audited
Emergency procedures and incident response are documented
Long-term maintenance and support procedures are established

📌  Rule 9: Single Source Frontend/Backend - Zero Duplication Architecture
Requirement: Maintain absolute architectural clarity with single, authoritative frontend and backend directories, eliminating all duplication and confusion through disciplined version control and feature management.
CRITICAL: One Source of Truth for Each Layer

Single Frontend: Only /frontend directory for all UI code
Single Backend: Only /backend directory for all server code
Zero Tolerance: No v1/, v2/, old/, backup/, deprecated/, or duplicate directories
Version Control: Git branches and tags for versioning, not directory duplication
Feature Management: Feature flags for experiments, not separate codebases

✅ Required Practices:
Mandatory Investigation Before Consolidation:
bash# Comprehensive duplicate detection
find . -type d \( -name "*frontend*" -o -name "*backend*" -o -name "*client*" -o -name "*server*" -o -name "*api*" -o -name "*ui*" \) | grep -v node_modules

# Search for version indicators
find . -type d | grep -E "(v[0-9]+|old|backup|deprecated|legacy|archive|previous|copy)"

# Analyze directory contents for duplication
for dir in $(find . -name "package.json" -o -name "requirements.txt"); do
    echo "Found project root: $(dirname $dir)"
done

# Git history analysis for branching points
git log --all --graph --decorate --oneline | grep -E "(frontend|backend)"
Consolidation Process:
bashconsolidate_to_single_source() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    # Step 1: Inventory all variants
    echo "[$timestamp] Starting consolidation inventory..."
    find . -type d -name "*frontend*" > frontend_variants.txt
    find . -type d -name "*backend*" > backend_variants.txt
    
    # Step 2: Analyze each variant
    for variant in $(cat frontend_variants.txt backend_variants.txt); do
        analyze_variant_purpose "$variant"
        check_unique_features "$variant"
        assess_migration_complexity "$variant"
    done
    
    # Step 3: Create consolidation plan
    create_migration_strategy
    document_feature_differences
    plan_git_branch_structure
    
    # Step 4: Execute consolidation
    merge_to_canonical_directories
    implement_feature_flags
    archive_deprecated_versions
    
    # Step 5: Validate consolidation
    run_comprehensive_tests
    verify_no_functionality_lost
    update_all_references
}
Feature Flag Implementation:
javascript// frontend/src/config/features.js
const FEATURES = {
  EXPERIMENTAL_UI: process.env.REACT_APP_EXPERIMENTAL_UI === 'true',
  BETA_DASHBOARD: process.env.REACT_APP_BETA_DASHBOARD === 'true',
  NEW_AUTH_FLOW: process.env.REACT_APP_NEW_AUTH_FLOW === 'true',
  ADVANCED_ANALYTICS: process.env.REACT_APP_ADVANCED_ANALYTICS === 'true'
};

// Usage in components
if (FEATURES.EXPERIMENTAL_UI) {
  return <ExperimentalComponent />;
} else {
  return <StableComponent />;
}
python# backend/app/config/features.py
from enum import Enum
import os

class FeatureFlags(Enum):
    EXPERIMENTAL_API = os.getenv('EXPERIMENTAL_API', 'false').lower() == 'true'
    BETA_ENDPOINTS = os.getenv('BETA_ENDPOINTS', 'false').lower() == 'true'
    NEW_AUTH_SYSTEM = os.getenv('NEW_AUTH_SYSTEM', 'false').lower() == 'true'
    ADVANCED_CACHING = os.getenv('ADVANCED_CACHING', 'false').lower() == 'true'

# Usage in routes
if FeatureFlags.EXPERIMENTAL_API.value:
    app.include_router(experimental_routes)
Git Branch Strategy:
bash# Main branches
main           # Production-ready code
develop        # Integration branch
staging        # Pre-production testing

# Feature branches
feature/new-dashboard
feature/api-v2-endpoints
experiment/ai-integration
hotfix/critical-bug-fix

# Version tags instead of directories
git tag -a v1.0.0 -m "Version 1.0.0 release"
git tag -a v2.0.0-beta -m "Version 2.0.0 beta"
Directory Structure Enforcement:
/opt/sutazaiapp/
├── frontend/                    # ONLY frontend directory
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API service layers
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Utility functions
│   │   ├── config/            # Configuration including features
│   │   └── experimental/      # Feature-flagged experimental code
│   ├── public/
│   ├── package.json
│   └── README.md
│
├── backend/                     # ONLY backend directory
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── models/            # Data models
│   │   ├── services/          # Business logic
│   │   ├── utils/             # Utility functions
│   │   ├── config/            # Configuration including features
│   │   └── experimental/      # Feature-flagged experimental code
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
└── [NO OTHER FRONTEND/BACKEND DIRECTORIES ALLOWED]
Migration from Duplicates:
bashmigrate_duplicate_codebases() {
    local source_dir="$1"
    local target_dir="$2"
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    # Create migration record
    cat >> MIGRATION_LOG.md << EOF
## Migration: $source_dir -> $target_dir
**Date**: $timestamp
**Reason**: Consolidating duplicate codebases per Rule 9

### Pre-Migration Analysis
- Unique features identified: [list]
- Dependencies specific to source: [list]
- Configuration differences: [list]

### Migration Steps
1. Backed up source to: /archives/$timestamp/$(basename $source_dir)
2. Identified unique features for feature flags
3. Merged code into $target_dir
4. Updated all references and imports
5. Tested all functionality preserved

### Post-Migration Validation
- All tests passing: ✓
- Feature flags working: ✓
- No functionality lost: ✓
EOF

    # Perform migration
    create_archive "$source_dir"
    extract_unique_features "$source_dir"
    merge_into_target "$target_dir"
    update_all_references
    run_validation_tests
}
🚫 Forbidden Practices:
Directory Duplication Violations:

Creating frontend_v2/, backend_old/, or any versioned directories
Maintaining multiple frontend or backend directories simultaneously
Using directory names to indicate versions or experiments
Creating "backup" directories instead of using Git
Duplicating code instead of using shared libraries
Creating separate directories for different deployment environments
Using copy/paste for experiments instead of branches
Maintaining deprecated code in separate directories

Version Control Violations:

Not using Git branches for experiments and features
Creating new directories for major version changes
Avoiding Git tags for version marking
Not documenting branch purposes and lifespans
Keeping experimental code in main/master branch
Using comments to "version" code instead of Git
Not cleaning up merged feature branches
Creating "archive" directories instead of using Git history

Feature Management Violations:

Hardcoding experimental features without flags
Creating separate apps for A/B testing
Not documenting feature flag purposes
Leaving feature flags permanently enabled
Using compile-time instead of runtime flags
Not having a feature flag retirement plan
Creating duplicate components without flags
Not centralizing feature flag configuration

Investigation Methodology:
Duplicate Detection Process:
bashdetect_all_duplicates() {
    echo "=== Searching for Frontend Duplicates ==="
    find . -type d -name "*frontend*" -o -name "*client*" -o -name "*ui*" | \
        grep -v node_modules | while read dir; do
        echo "Found: $dir"
        echo "  - Size: $(du -sh "$dir" | cut -f1)"
        echo "  - Last modified: $(stat -c %y "$dir" | cut -d' ' -f1)"
        echo "  - Package.json: $([ -f "$dir/package.json" ] && echo "Yes" || echo "No")"
    done
    
    echo "=== Searching for Backend Duplicates ==="
    find . -type d -name "*backend*" -o -name "*server*" -o -name "*api*" | \
        grep -v node_modules | while read dir; do
        echo "Found: $dir"
        echo "  - Size: $(du -sh "$dir" | cut -f1)"
        echo "  - Last modified: $(stat -c %y "$dir" | cut -d' ' -f1)"
        echo "  - Requirements.txt: $([ -f "$dir/requirements.txt" ] && echo "Yes" || echo "No")"
    done
}
Unique Feature Analysis:
bashanalyze_unique_features() {
    local dir1="$1"
    local dir2="$2"
    
    echo "Comparing $dir1 vs $dir2"
    
    # Compare file structures
    diff -qr "$dir1" "$dir2" | grep "Only in" > unique_files.txt
    
    # Compare dependencies
    if [ -f "$dir1/package.json" ] && [ -f "$dir2/package.json" ]; then
        diff <(jq -S '.dependencies' "$dir1/package.json") \
             <(jq -S '.dependencies' "$dir2/package.json")
    fi
    
    # Compare configurations
    find "$dir1" -name "*.config.*" -o -name "*.env*" > config1.txt
    find "$dir2" -name "*.config.*" -o -name "*.env*" > config2.txt
    diff config1.txt config2.txt
    
    # Document findings
    document_unique_features_for_migration
}
Documentation Requirements:
CHANGELOG.md Entry for Consolidation:
markdown### [2024-12-20 15:45:30 UTC] - v2.0.0 - ARCHITECTURE - MAJOR - Frontend/Backend Consolidation
**Who**: DevOps Team (devops@company.com)
**Why**: Eliminate confusion from multiple frontend/backend directories per Rule 9
**What**: 
  - Consolidated frontend_v1/, frontend_v2/, frontend_old/ into /frontend
  - Merged backend/, backend_v2/, api_old/ into /backend  
  - Implemented feature flags for experimental features
  - Created Git tags for version history
  - Archived deprecated code with restoration procedures
**Impact**: 
  - All imports and references updated
  - CI/CD pipelines reconfigured
  - Documentation updated
  - No functionality lost
**Validation**: 
  - All tests passing
  - Feature flags tested
  - Performance benchmarks maintained
Validation Criteria:
Structure Validation:

✓ Only one /frontend directory exists
✓ Only one /backend directory exists
✓ No versioned directories (v1, v2, old, etc.)
✓ No duplicate codebases
✓ Git branches used for versions
✓ Feature flags properly implemented
✓ All references updated
✓ Documentation current

Functionality Validation:

✓ All features preserved during consolidation
✓ No regression in functionality
✓ Feature flags working correctly
✓ Performance maintained or improved
✓ All tests passing
✓ Build processes working
✓ Deployment successful
✓ No broken imports or references

Process Validation:

✓ Investigation completed before consolidation
✓ Unique features identified and preserved
✓ Migration plan documented
✓ Backups created before changes
✓ Team notified of changes
✓ Git history preserved
✓ Feature flag documentation complete
✓ Rollback procedures tested

This expanded Rule 9 provides comprehensive guidance for maintaining single-source architecture while preserving the ability to experiment and version through proper Git usage and feature flags.

📌 Rule 10: Functionality-First Cleanup - Never Delete Blindly
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

📌 Rule 11: Docker Excellence
Docker Standards:
All Docker diagrams must include the complete directory structure
Reference these specific diagrams in /opt/sutazaiapp/IMPORTANT/diagrams:

Dockerdiagram-core.md - Core container architecture
Dockerdiagram-self-coding.md - Self-coding service containers
Dockerdiagram-training.md - Training environment containers
Dockerdiagram.md - Main Docker architecture overview
PortRegistry.md - Port allocation and service registry

Every container architecture must follow the structure shown in /opt/sutazaiapp/IMPORTANT/diagrams
All Docker configurations go in /docker/ but reference the diagrams for architecture decisions
Port assignments must follow PortRegistry.md specifications
✅ Required Practices:
Configuration Standards:

All Docker configurations centralized in /docker/ directory only
Reference architecture diagrams in /opt/sutazaiapp/IMPORTANT/diagrams before any container changes
Multi-stage Dockerfiles with development and production variants
Non-root user execution with proper USER directives (never run as root)
Pinned base image versions (never use latest tags)
Comprehensive HEALTHCHECK instructions for all services
Proper .dockerignore files to optimize build context
Docker Compose files for each environment (dev/staging/prod)

Security & Scanning:

Container vulnerability scanning in CI/CD pipeline
Secrets managed externally (never in images or ENV vars)
Security validation before any production deployment
Read-only root filesystem where applicable
Capability dropping with minimal required permissions

Resource & Performance:

Resource limits and requests defined for all containers
Memory limits: Explicit values (e.g., 512m)
CPU limits: Decimal notation (e.g., cpus: '0.5')
Structured logging with proper log levels and formats
Log rotation configured to prevent disk exhaustion

Orchestration & Integration:

Container orchestration with proper service mesh integration
Service discovery through Consul or equivalent
Network isolation using Docker networks per environment
Inter-service communication via service mesh
Health checks integrated with orchestration platform

🚫 Forbidden Practices:
Configuration Violations:

Creating Docker files outside /docker/ directory
Using latest or unpinned image tags in any environment
Running containers as root user without explicit security review
Storing secrets, credentials, or sensitive data in container images

Build & Deployment Violations:

Building images without vulnerability scanning and security validation
Creating monolithic containers that violate single responsibility principle
Using development configurations or debugging tools in production images
Deploying without comprehensive security scanning

Operational Violations:

Implementing containers without proper health checks and monitoring
Creating containers without proper resource limits and quotas
Using containers that don't handle graceful shutdown (SIGTERM)
Ignoring service mesh integration requirements

Cross-Rule References:

See Rule 1: All Docker configurations must be real and working
See Rule 4: Investigate existing Docker configurations before creating new ones
See Rule 12: Docker setup integrated with universal deployment script
See Rule 19: Document all Docker changes in CHANGELOG.md

Validation Criteria:

All containers pass security scans with zero high-severity vulnerabilities
Docker configurations follow established patterns in /docker/ directory
Architecture decisions align with diagrams in /opt/sutazaiapp/IMPORTANT/diagrams
Containers start reliably and handle graceful shutdown
Resource usage is optimized and within defined limits
All services have functional health checks and monitoring
Documentation is current and matches actual container behavior
Service mesh integration verified and functional
Container orchestration properly configured
Security scanning integrated into CI/CD pipeline

Implementation Checklist:
bash# Pre-deployment validation
□ Dockerfile in /docker/ directory
□ Multi-stage build implemented
□ Base image version pinned
□ USER directive sets non-root (1000:1000)
□ HEALTHCHECK defined with timings
□ .dockerignore optimized
□ Resource limits defined
□ Security scan passed
□ Secrets externalized
□ Graceful shutdown tested
□ Service mesh configured
□ Port matches PortRegistry.md
□ Architecture matches diagrams
□ CHANGELOG.md updated

📌 Rule 12: Universal Deployment Script
CRITICAL: Complete End-to-End Automation - One Command Does Everything:

Zero Interruption Deployment: Script must run from start to finish without ANY manual intervention
Complete System Setup: Must install, configure, and start the entire system to fully operational state
One Command Success: ./deploy.sh --env=prod results in complete running system - NO EXCEPTIONS
No Partial Deployments: Script either completes 100% successfully or rolls back completely
No Manual Steps: Zero requirement for human intervention during any phase of deployment
Full Stack Deployment: Database, backend, frontend, networking, monitoring - everything operational
Production Ready: System must be serving traffic and fully functional at script completion
No Post-Deployment Tasks: No additional configuration, setup, or manual steps required
Complete Validation: Script validates entire system is operational before declaring success
Zero Mistakes Tolerance: Any error triggers automatic rollback to previous state

CRITICAL: Complete Self-Sufficiency - Auto-Install Everything:

Dependency Detection: Automatically detect all missing tools, packages, and dependencies
Auto-Installation: Install all missing prerequisites without user intervention
Platform Intelligence: Detect OS and use appropriate package managers (apt, yum, brew, apk, etc.)
Version Validation: Ensure installed versions meet minimum requirements for deployment
Tool Chain Setup: Install complete toolchain (Docker, Git, curl, wget, systemctl, etc.)
Runtime Installation: Install all required runtimes (Node.js, Python, Java, etc.) with correct versions
Database Installation: Install and configure required databases (MySQL, PostgreSQL, Redis, etc.)
Web Server Installation: Install and configure web servers (nginx, Apache) with optimal settings
Monitoring Tools: Install monitoring stack (Prometheus, Grafana, or alternatives)
Security Tools: Install security tools (fail2ban, ufw, SSL certificate management)
Network Tools: Install network utilities (iptables, netstat, ss, iperf) for optimization
Backup Tools: Install backup utilities and configure automated backup systems

CRITICAL: Hardware and Network Resource Detection and Optimization:

Hardware Detection: Detect CPU cores, RAM, disk space, and network interfaces using standard tools
Resource Optimization: Configure Docker, databases, and services based on detected hardware
Docker Configuration: Set Docker daemon limits, container resources based on available CPU/RAM
Network Optimization: Configure connection pools, timeouts based on detected network capacity
Storage Optimization: Configure disk I/O, cache sizes based on available storage type and space
Memory Allocation: Set JVM heaps, database buffers, cache sizes based on available RAM
CPU Utilization: Configure worker processes, thread pools based on available CPU cores
Pre-flight Validation: Test hardware meets minimum requirements before deployment begins

MANDATORY FIRST STEP - Investigation and Consolidation:

ALWAYS search for existing deployment scripts across entire codebase before creating new ones
Use comprehensive search: find . -name "*deploy*" -o -name "*build*" -o -name "*install*" -o -name "*.sh"
Investigate all found scripts: purpose, functionality, dependencies, and current usage
Analyze Git history to understand why each script was created and its evolution
Test each existing script in isolated environment to understand full functionality
Map all unique capabilities and consolidate into single ./deploy.sh
Preserve all working functionality - never lose capabilities during consolidation
Document what was consolidated from each script and why
Test consolidated script thoroughly in all environments before removing originals
Archive original scripts with restoration procedures before deletion
Update all references, documentation, and CI/CD pipelines to use new unified script
Validate that team members can execute same workflows with consolidated script
Only remove original scripts after 100% validation that consolidated version works

✅ Required Practices:
Complete Dependency Management and Auto-Installation:

Operating System Detection: Detect OS type and version using uname, /etc/os-release, lsb_release
Package Manager Detection: Identify available package managers (apt, yum, dnf, zypper, brew, pkg)
Dependency Scanning: Scan for all required tools and packages before starting deployment
Auto-Installation Logic: Install missing dependencies using appropriate package manager
Version Checking: Verify installed versions meet minimum requirements, upgrade if necessary
Alternative Installation: Use alternative installation methods if package manager fails (wget, curl, compile)
Tool Validation: Test that installed tools work correctly before proceeding
Path Configuration: Ensure all installed tools are in PATH and accessible
Permission Setup: Configure appropriate permissions for installed tools and services
Service Configuration: Configure installed services for automatic startup and optimal operation

Essential Tool Installation:
bash# Core system tools
install_core_tools() {
    local tools=("curl" "wget" "git" "unzip" "tar" "gzip" "systemctl" "ps" "netstat" "ss")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            install_package "$tool"
        fi
    done
}

# Docker installation
install_docker() {
    if ! command -v docker &> /dev/null; then
        case "$OS" in
            "ubuntu"|"debian")
                curl -fsSL https://get.docker.com | sh
                ;;
            "centos"|"rhel"|"fedora")
                curl -fsSL https://get.docker.com | sh
                ;;
            "alpine")
                apk add docker docker-compose
                ;;
        esac
        systemctl enable docker
        systemctl start docker
    fi
    validate_docker_installation
}

# Database installation
install_databases() {
    if [[ "$REQUIRES_MYSQL" == "true" ]] && ! command -v mysql &> /dev/null; then
        install_mysql_server
        configure_mysql_optimal_settings
    fi
    
    if [[ "$REQUIRES_POSTGRESQL" == "true" ]] && ! command -v psql &> /dev/null; then
        install_postgresql_server
        configure_postgresql_optimal_settings
    fi
    
    if [[ "$REQUIRES_REDIS" == "true" ]] && ! command -v redis-cli &> /dev/null; then
        install_redis_server
        configure_redis_optimal_settings
    fi
}

# Runtime installation
install_runtimes() {
    if [[ "$REQUIRES_NODEJS" == "true" ]] && ! command -v node &> /dev/null; then
        install_nodejs_lts
        validate_nodejs_version
    fi
    
    if [[ "$REQUIRES_PYTHON" == "true" ]] && ! command -v python3 &> /dev/null; then
        install_python3_and_pip
        validate_python_version
    fi
    
    if [[ "$REQUIRES_JAVA" == "true" ]] && ! command -v java &> /dev/null; then
        install_openjdk
        validate_java_version
    fi
}
Platform-Specific Installation Logic:

Ubuntu/Debian: Use apt update && apt install -y for package installation
CentOS/RHEL/Fedora: Use yum install -y or dnf install -y for package installation
Alpine Linux: Use apk add for package installation
macOS: Use brew install if Homebrew available, otherwise use manual installation
Amazon Linux: Use yum install -y with Amazon Linux-specific repositories
SUSE/openSUSE: Use zypper install -y for package installation
Arch Linux: Use pacman -S for package installation
FreeBSD: Use pkg install for package installation
Generic Unix: Use manual installation methods (wget, curl, compile from source)

Hardware Resource Detection and Optimization:

CPU Detection: Use nproc, /proc/cpuinfo to detect cores, architecture, features
Memory Detection: Use free, /proc/meminfo to detect total/available RAM
Storage Detection: Use df, lsblk to detect disk space, filesystem types, mount points
Network Detection: Use ip, ethtool to detect network interfaces and capabilities
Performance Testing: Run basic performance tests (dd for disk, iperf for network) to establish baselines
Resource Calculation: Calculate optimal settings based on detected resources (e.g., DB buffer = 25% RAM)
Docker Daemon Config: Set --default-ulimit, --storage-driver based on system capabilities
Container Limits: Set --memory, --cpus for each container based on available resources
Database Tuning: Configure MySQL/PostgreSQL buffer pools, connection limits based on RAM
Web Server Config: Set worker processes, connection limits based on CPU cores

Docker and Container Optimization:

Docker Installation: Auto-install Docker if not present using official installation scripts
Docker Compose Installation: Install Docker Compose if not available
Storage Driver Selection: Choose optimal storage driver (overlay2, devicemapper) based on filesystem
Memory Limits: Set container memory limits to prevent OOM kills while maximizing usage
CPU Limits: Set CPU limits to ensure fair resource sharing without waste
Container Placement: Use Docker Compose resource constraints for optimal container placement
Health Checks: Configure comprehensive health checks for all containers
Restart Policies: Set appropriate restart policies based on service criticality
Network Configuration: Configure Docker networks for optimal performance and security
Volume Configuration: Configure volumes with appropriate permissions and performance settings
Registry Configuration: Configure container registry with optimal caching and authentication

Network Resource Optimization:

Network Tools Installation: Install iperf3, netstat, ss, tcpdump for network analysis
Bandwidth Detection: Test available bandwidth using simple tools (wget, curl with large files)
Latency Testing: Test network latency to external services and databases
Connection Pool Sizing: Configure database connection pools based on CPU cores and expected load
Timeout Configuration: Set timeouts based on measured network latency + buffer
Load Balancer Config: Configure nginx/HAProxy with optimal worker processes and connections
DNS Configuration: Configure DNS caching and resolution for optimal performance
Firewall Configuration: Configure iptables/ufw rules with performance impact
TCP Tuning: Tune TCP buffer sizes and window scaling for optimal throughput
SSL Configuration: Configure SSL with optimal cipher suites and session management
CDN Configuration: Configure CDN settings based on geographic deployment location

Pre-flight Checks and Validation:

Tool Availability: Verify all required tools are installed and functional
Version Compatibility: Check all tool versions meet minimum requirements
Permission Validation: Verify script has necessary permissions for all operations
Minimum Requirements: Verify minimum CPU, RAM, disk space, network connectivity
Port Availability: Check all required ports are available before starting services
External Connectivity: Test connectivity to external services, databases, APIs
Storage Performance: Test disk I/O performance meets minimum requirements
Memory Availability: Ensure sufficient memory available for all planned services
Network Connectivity: Validate network connectivity and DNS resolution
Security Requirements: Verify system meets basic security requirements
Backup Validation: Verify backup storage is accessible and has sufficient space

Complete End-to-End Automation:

System Preparation: Install all required packages, create users, configure system services
Database Setup: Install, configure, and initialize databases with schemas and initial data
Application Deployment: Build, deploy, and configure all application services
Web Server Setup: Install and configure nginx/Apache with SSL and optimal settings
Security Configuration: Configure firewalls, SSL certificates, basic security hardening
Monitoring Setup: Deploy basic monitoring (e.g., Prometheus, Grafana) with essential metrics
Backup Configuration: Setup automated backups with tested restore procedures
Service Integration: Configure and test all service-to-service communication
Health Validation: Validate all services are running and responding correctly
Performance Testing: Run basic performance tests to ensure system meets requirements
Documentation Generation: Generate basic operational documentation and access information
Team Notification: Send notification with deployment status and access information

Investigation and Consolidation (MANDATORY FIRST STEP):

Search for existing scripts: deployment, build, install, setup, CI/CD scripts
Fresh Server Scenario: If no existing scripts found, proceed directly with deployment
Existing Scripts Scenario: If scripts found, require consolidation before proceeding
Analyze each script's functionality using standard tools (grep, awk, bash analysis)
Map dependencies and understand integration points with existing systems
Test existing scripts in isolated environments to understand behavior
Consolidate functionality by merging working code into single script
Preserve all working functionality during consolidation process
Document consolidation decisions and archive original scripts safely
Test consolidated script thoroughly before removing originals
Update all references and documentation to point to consolidated script

Zero Assumptions Architecture:

Package Management: Auto-detect and use appropriate package manager (apt, yum, brew)
Dependency Installation: Install all required system dependencies automatically
User Management: Create required system users with appropriate permissions
Directory Structure: Create all required directories with correct permissions
Service Configuration: Configure all system services for automatic startup
Environment Setup: Configure environment variables and system paths
Security Setup: Configure basic security (firewall, user permissions, SSL)
Database Initialization: Initialize databases with required schemas and users
Application Configuration: Generate and deploy all application configuration files
Network Setup: Configure network interfaces, routing, and DNS as needed

Environment Management:

Environment Detection: Detect environment based on hostname, network, or explicit flags
Configuration Loading: Load environment-specific configurations from standard locations
Resource Allocation: Adjust resource allocation based on environment (dev uses less resources)
Security Policies: Apply appropriate security policies for each environment
Monitoring Configuration: Configure monitoring appropriate for environment sensitivity
Backup Policies: Apply backup and retention policies appropriate for environment
Performance Settings: Apply performance tuning appropriate for environment load
Integration Configuration: Configure external service integrations per environment
Feature Flags: Configure feature flags and toggles per environment
Scaling Policies: Configure auto-scaling appropriate for environment requirements

Self-Update Mechanism:

Version Checking: Check for script updates against source repository
Backup Current: Backup current script before any updates
Download and Validate: Download new version and validate syntax
Integrity Verification: Verify download integrity using checksums
Update and Restart: Update script and restart with new version
Configuration Updates: Update configuration files and templates
Dependency Updates: Check and update system dependencies
Container Updates: Pull new container images and update configurations
Rollback Capability: Rollback to previous version if update fails
Change Documentation: Document what changed and why

Comprehensive Error Handling:

Structured Exit Codes: Use specific exit codes for different failure types
Error Logging: Log all errors with timestamps and context information
Retry Logic: Implement retry logic for transient failures (network, etc.)
Graceful Degradation: Continue with non-critical failures, abort on critical ones
Rollback Triggers: Automatically trigger rollback on specific error conditions
Error Notification: Send notifications for errors requiring human attention
Recovery Procedures: Document and automate recovery procedures for common failures
Timeout Handling: Handle timeouts for all external operations
Resource Cleanup: Clean up resources on failure to prevent resource leaks
State Validation: Validate system state before and after critical operations

Rollback and Recovery:

State Snapshots: Create snapshots of critical system state before changes
Database Backups: Backup databases before schema changes or data operations
Configuration Backups: Backup all configuration files before modifications
Service State Tracking: Track which services were started/modified for rollback
Atomic Operations: Use atomic operations where possible (database transactions, etc.)
Rollback Scripts: Generate rollback scripts during deployment for easy recovery
Validation During Rollback: Validate system state during rollback process
Cleanup Procedures: Clean up any resources created during failed deployment
Service Restoration: Restore services to previous running state
Data Integrity: Ensure data integrity during rollback operations

Validation and Reporting:

Pre-deployment Checks: Validate system requirements and dependencies
Deployment Monitoring: Monitor deployment progress with regular status updates
Service Health Checks: Validate all services are healthy and responding
Performance Validation: Basic performance testing to ensure acceptable response times
Security Validation: Basic security checks (open ports, permissions, SSL)
Integration Testing: Test critical integrations with external services
End-to-End Testing: Run basic end-to-end tests of critical user workflows
Deployment Report: Generate comprehensive deployment report with metrics
Access Information: Provide access URLs, credentials, and operational information
Next Steps: Document any recommended next steps or optimizations

🚫 Forbidden Practices:
Dependency Management Violations:

Assuming any tools or packages are pre-installed on target systems
Requiring manual installation of dependencies before script execution
Using tools without checking if they're available and installing if missing
Failing to validate that installed tools work correctly before using them
Using hardcoded paths to tools without checking if they exist
Ignoring different package managers across different operating systems
Installing packages without verifying successful installation
Using deprecated or insecure installation methods
Installing packages without configuring them appropriately
Failing to handle installation failures gracefully with alternatives

Realistic Constraint Violations:

Using theoretical or experimental technologies not available in standard distributions
Requiring specialized hardware or software not commonly available
Implementing features that require months of development work
Using AI/ML capabilities that don't exist in standard deployment tools
Requiring manual installation of complex custom tools before script execution
Using cloud-specific features that don't work on-premises or other clouds
Implementing security features that require specialized security hardware
Using performance optimization techniques that require kernel modifications
Requiring enterprise licenses or paid tools for basic functionality
Implementing features that require specialized networking equipment

Hardware and Resource Optimization Violations:

Using hardcoded resource limits that ignore available hardware
Deploying Docker without configuring resource limits and usage
Ignoring available CPU cores when configuring worker processes
Using fixed memory allocations regardless of available RAM
Configuring databases without considering available memory
Using default network settings without testing available bandwidth
Ignoring disk performance characteristics when configuring storage
Setting timeouts without considering actual network latency
Using single-threaded processes on multi-core systems
Configuring services without considering system resource capacity

Complete Automation Violations:

Requiring manual input, confirmation, or intervention during deployment
Deploying partial systems that require additional manual setup
Using deployment procedures that pause and wait for human action
Creating deployments that require manual post-deployment configuration
Using scripts that fail silently and require manual troubleshooting
Deploying without validating that all services are actually working
Creating systems that are deployed but not ready to serve traffic
Using deployment procedures that require specialized knowledge to execute
Deploying without comprehensive error handling and automatic recovery
Creating deployments that leave systems in inconsistent or broken states

Investigation and Consolidation Violations:

Creating new deployment scripts without searching for existing ones
Ignoring existing deployment automation when building new solutions
Consolidating scripts without understanding their complete functionality
Removing existing scripts without thorough testing of replacements
Creating consolidated scripts that lose important existing capabilities
Skipping analysis of why existing scripts were created originally
Consolidating without documenting what functionality came from where
Removing scripts without proper backup and restoration procedures
Creating solutions that break existing team workflows and procedures
Ignoring integration with existing CI/CD and operational tools

Validation Criteria:
Dependency Management and Auto-Installation Validation:

All required tools and packages automatically detected and installed
Installation works correctly across different operating systems and package managers
Version validation ensures all tools meet minimum requirements
Alternative installation methods work when package managers fail
Tool functionality validated after installation before proceeding with deployment
PATH and environment properly configured for all installed tools
Installation failures handled gracefully with appropriate error messages
No manual intervention required for any dependency installation
Installation process documented in deployment logs for troubleshooting
Rollback procedures can cleanly remove installed dependencies if needed

Hardware and Resource Optimization Validation:

System hardware completely detected and documented (CPU, RAM, disk, network)
Docker configured with appropriate resource limits based on available hardware
Database configurations optimized for available memory and CPU
Web server configurations optimized for available CPU cores
Network configurations optimized for detected bandwidth and latency
Container resource limits prevent resource contention while maximizing utilization
Performance testing validates optimization settings improve performance
Resource monitoring confirms optimal utilization without resource exhaustion
System performs better than default configurations under realistic load
Resource scaling works correctly as system load increases

Complete End-to-End Validation:

Single command execution results in fully functional system serving requests
Database is operational with all required schemas, users, and initial data
Application services are running and responding to requests correctly
Web server is serving content with proper SSL configuration
All required system services are running and configured for auto-start
Monitoring is operational and collecting metrics from all services
Backup systems are configured and initial backups have been created
Security configurations are applied and basic security tests pass
Integration with external services is working correctly
System can handle expected production traffic loads
No manual intervention was required during any phase of deployment
System is ready for production use immediately after script completion

Investigation and Consolidation Validation:

Comprehensive search completed for all existing deployment-related scripts
Fresh Server: No existing scripts found, deployment proceeds normally
Existing Scripts: All scripts analyzed and consolidated successfully
All functionality from original scripts preserved in consolidated version
Consolidated script tested thoroughly in all relevant environments
Team workflows continue to work with consolidated script
All references and documentation updated to use consolidated script
Original scripts properly archived with tested restoration procedures
No regression in deployment capabilities or functionality
Team training completed on consolidated script usage

Functional Validation:

Script executes successfully on fresh systems without any manual setup
All environment configurations deploy correctly with appropriate settings
Self-update mechanism works reliably with proper backup and validation
Error handling provides clear information and triggers appropriate recovery
Rollback procedures successfully restore system to previous working state
All validation checks accurately detect problems before they cause issues
Deployment reporting provides comprehensive status and access information
Script integrates properly with existing CI/CD and monitoring systems
Performance meets established requirements under realistic load conditions
Security configurations meet basic security requirements for environment

Operational Validation:

Any team member can execute deployment without specialized training
Deployment logs provide sufficient detail for troubleshooting any issues
Monitoring provides real-time visibility into deployment progress and system health
Documentation is complete and enables effective operational management
Emergency procedures work correctly for common failure scenarios
System maintenance procedures are documented and tested
Team has been trained on operational procedures and troubleshooting
Integration with existing operational tools and procedures is functional
Compliance requirements are met for the target environment
Long-term maintenance and support procedures are established and documented

📌 Rule 13: Zero Tolerance for Waste
CRITICAL: Everything Must Serve a Purpose:

Active Contribution: Every line of code, file, asset, and configuration must actively contribute to system functionality
Zero Dead Code: No unused functions, classes, variables, or code blocks permitted in codebase
No Abandoned Features: Remove incomplete features, experimental code, and abandoned implementations
Purpose Validation: Every component must have clear, documented purpose and active usage
Regular Purging: Systematic removal of waste through automated detection and manual review
Prevention Focus: Prevent waste accumulation through development practices and automated checks
Team Accountability: Every team member responsible for identifying and removing waste
Continuous Monitoring: Ongoing monitoring and alerts for waste accumulation

MANDATORY: Investigation Before Removal - Never Delete Blindly:

Root Cause Analysis: Investigate WHY each piece of code/asset exists before considering removal
Purpose Discovery: Understand original intent and current potential value through code analysis and Git history
Integration Opportunity Assessment: Determine if "unused" code belongs elsewhere and should be moved/integrated
Dependency Mapping: Map all dependencies and relationships before making removal decisions
Alternative Usage Investigation: Search for dynamic usage, reflection, configuration-driven execution
Historical Context Research: Analyze Git history, commit messages, and PR discussions for context
Team Knowledge Gathering: Consult with original authors and team members about purpose and usage
Documentation Cross-Reference: Check documentation, comments, and external references for usage patterns
Runtime Analysis: Analyze runtime behavior and usage patterns that static analysis might miss
Future Need Assessment: Evaluate if code might be needed for planned features or migrations
Only Remove After Confirmation: Remove only after confirming absolutely no purpose or integration opportunity

✅ Required Practices:
Mandatory Investigation Process Before Any Removal:

Comprehensive Code Analysis: Use grep, ripgrep, and IDE search to find all references and usage patterns
Git History Investigation: Analyze commit history, blame annotations, and PR discussions to understand purpose
Dynamic Usage Detection: Search for string-based references, reflection usage, and configuration-driven calls
Cross-Repository Search: Search across all repositories for references and dependencies
Documentation Cross-Reference: Check all documentation for references to the code or asset
Configuration Analysis: Analyze configuration files for dynamic references and conditional usage
Database Reference Check: Search database schemas and data for references to code functionality
External Integration Analysis: Check for external service integrations and API usage
Test Code Analysis: Examine test files for usage patterns and expected functionality
Build System Analysis: Check build scripts, deployment configs, and CI/CD for usage

Integration Opportunity Assessment:

Consolidation Potential: Identify if unused code should be consolidated with similar functionality elsewhere
Relocation Opportunities: Determine if code belongs in different modules, services, or repositories
Abstraction Opportunities: Assess if code should be abstracted into reusable utilities or libraries
Service Migration: Evaluate if code should be moved to different services or microservices
Library Extraction: Determine if code should be extracted into shared libraries or packages
Configuration Migration: Assess if hardcoded values should be moved to configuration systems
Documentation Migration: Evaluate if code comments should become external documentation
Test Migration: Determine if code should be moved to test utilities or fixtures
Tool Migration: Assess if code should become development tools or scripts
Framework Migration: Evaluate if code should be migrated to different frameworks or patterns

Dead Code Detection and Investigation:

Automated Dead Code Analysis: Use static analysis tools to detect potentially unused functions, classes, and variables
Manual Verification: Manually verify automated detection results through comprehensive investigation
Import/Dependency Analysis: Identify unused imports but investigate before removal for dynamic usage
Function Call Analysis: Track function usage but verify through runtime analysis and dynamic calls
Variable Usage Analysis: Detect unused variables but check for reflection, serialization, or configuration usage
Class and Interface Analysis: Identify unused classes but investigate for framework usage, plugins, or extensions
Asset Usage Analysis: Detect unused assets but verify through dynamic loading and content management systems
Database Schema Analysis: Identify unused schema elements but verify through data analysis and migrations
API Endpoint Analysis: Detect unused endpoints but verify through external usage and documentation
Configuration Analysis: Identify unused config keys but verify through environment-specific and optional usage

TODO and Task Investigation:

TODO Origin Investigation: Investigate why each TODO was created and what problem it was meant to solve
TODO Context Analysis: Analyze surrounding code and related functionality to understand TODO purpose
TODO Priority Assessment: Evaluate business impact and technical debt implications of TODO items
TODO Integration Opportunities: Assess if TODO functionality should be integrated into existing features
TODO Timeline Analysis: Understand original timeline expectations and current relevance
TODO Ownership Investigation: Identify original authors and current stakeholders for TODO items
TODO Business Value Assessment: Evaluate if TODO represents valuable functionality worth implementing
TODO Technical Assessment: Analyze technical complexity and integration requirements
TODO Alternative Solutions: Investigate if TODO problem has been solved differently elsewhere
TODO Resolution Path: Determine appropriate resolution: implement, integrate, convert to issue, or remove

Commented Code Investigation:

Comment History Analysis: Investigate Git history to understand why code was commented out
Comment Context Evaluation: Analyze surrounding code to understand commented code's original purpose
Comment Alternative Investigation: Determine if commented functionality exists elsewhere in codebase
Comment Integration Assessment: Evaluate if commented code should be integrated into current functionality
Comment Value Analysis: Assess if commented code provides examples, documentation, or reference value
Comment Temporal Analysis: Understand when code was commented and if circumstances have changed
Comment Author Consultation: Contact original authors to understand commenting rationale
Comment Testing Investigation: Determine if commented code was removed due to testing or functionality issues
Comment Recovery Assessment: Evaluate if commented code should be recovered and integrated properly
Comment Documentation Value: Assess if commented code should become documentation or examples

Asset and Resource Investigation:

Asset Usage Pattern Analysis: Investigate how assets were intended to be used and current usage patterns
Asset Reference Search: Search for dynamic asset loading, content management, and configuration-driven usage
Asset Historical Context: Analyze when assets were added and original purpose or requirements
Asset Integration Opportunities: Determine if assets should be consolidated or moved to content management
Asset Quality Assessment: Evaluate asset quality and potential value for future use
Asset Replacement Analysis: Investigate if assets have been replaced by better alternatives
Asset Licensing Investigation: Verify asset licensing and legal requirements for retention or removal
Asset Performance Impact: Assess performance impact of asset retention vs removal
Asset Migration Opportunities: Evaluate if assets should be moved to CDN, cloud storage, or asset management
Asset Documentation Value: Determine if assets provide documentation or reference value

Legacy and Abandoned Feature Investigation:

Feature Usage Analytics: Analyze actual feature usage through logs, analytics, and user behavior data
Feature Business Value Assessment: Evaluate business value and user impact of potentially unused features
Feature Integration Opportunities: Assess if feature functionality should be integrated into active features
Feature Migration Assessment: Determine if features should be migrated to new frameworks or systems
Feature Stakeholder Consultation: Consult with product managers, users, and stakeholders about feature value
Feature Technical Debt Analysis: Evaluate technical debt and maintenance cost of retaining features
Feature Replacement Investigation: Determine if features have been replaced by better alternatives
Feature Future Need Assessment: Evaluate if features might be needed for planned functionality
Feature Compliance Requirements: Assess if features are required for regulatory or compliance reasons
Feature Documentation and Training Value: Evaluate if features provide learning or reference value

Systematic Investigation Workflow:
bash# Investigation workflow before removal
investigate_before_removal() {
    local target="$1"
    local target_type="$2"  # code, asset, config, etc.
    
    log_info "Starting investigation for: $target"
    
    # Step 1: Comprehensive search for references
    search_all_references "$target"
    
    # Step 2: Git history analysis
    analyze_git_history "$target"
    
    # Step 3: Dynamic usage detection
    detect_dynamic_usage "$target"
    
    # Step 4: Integration opportunity assessment
    assess_integration_opportunities "$target"
    
    # Step 5: Stakeholder consultation
    consult_stakeholders "$target"
    
    # Step 6: Business/technical value assessment
    assess_value "$target"
    
    # Step 7: Make decision: remove, integrate, or keep
    make_removal_decision "$target"
}

# Only remove after thorough investigation
safe_removal() {
    local target="$1"
    
    if investigate_before_removal "$target"; then
        if confirm_no_purpose_or_integration "$target"; then
            archive_before_removal "$target"
            remove_with_documentation "$target"
            log_removal_decision "$target"
        else
            log_info "Keeping $target - found purpose or integration opportunity"
        fi
    else
        log_warn "Investigation incomplete for $target - not removing"
    fi
}
Documentation and Comment Investigation:

Documentation Relevance Analysis: Investigate if documentation refers to current or planned functionality
Documentation Integration Assessment: Determine if documentation should be consolidated elsewhere
Documentation Historical Value: Evaluate if documentation provides valuable historical context
Documentation Reference Investigation: Search for external references to documentation content
Comment Purpose Analysis: Investigate purpose and value of code comments before removal
Comment Integration Opportunities: Assess if comments should become external documentation
Comment Historical Context: Analyze comment history and evolution for context and value
Comment Code Relationship: Investigate relationship between comments and surrounding code
Comment Business Context: Understand business rules and requirements documented in comments
Comment Maintenance Assessment: Evaluate ongoing maintenance requirements for documentation

🚫 Forbidden Practices:
Investigation Process Violations:

Removing any code, assets, or configuration without thorough investigation of purpose and usage
Using only automated tools to determine if code is "dead" without manual verification
Skipping Git history analysis to understand why code was created and its evolution
Removing code without searching for dynamic references, reflection, or configuration-driven usage
Failing to consult with original authors or team members about code purpose
Removing code without assessing integration opportunities with existing functionality
Skipping cross-repository and cross-service searches for dependencies and references
Removing code during critical periods without proper investigation time
Making removal decisions based solely on static analysis without runtime verification
Removing code without documenting investigation findings and decision rationale

Removal Process Violations:

Deleting code immediately upon detection as "unused" without investigation period
Removing code without creating proper archives and restoration procedures
Skipping testing and validation after code removal to ensure no functionality is broken
Removing shared code without coordinating with all dependent teams and services
Failing to update documentation and references when removing code
Removing code without following established change management procedures
Deleting code without considering impact on external integrations and APIs
Removing configuration or assets without checking environment-specific usage
Skipping rollback testing after code removal to ensure recovery procedures work
Removing code that might be needed for data migrations or system transitions

Integration Assessment Violations:

Failing to evaluate if unused code should be consolidated with similar functionality
Removing code that could be abstracted into reusable utilities or libraries
Skipping assessment of whether code should be moved to different services or modules
Failing to consider if hardcoded values should become configurable parameters
Removing code without considering if it should become development tools or scripts
Skipping evaluation of whether code should be migrated to newer frameworks
Failing to assess if code functionality is needed elsewhere in the system
Removing code without considering its value for future features or requirements
Skipping analysis of whether code should become external libraries or packages
Failing to evaluate if code provides valuable examples or reference implementations

Team and Process Violations:

Making removal decisions without team consultation and stakeholder input
Removing code without assigning ownership for investigation and decision-making
Skipping peer review of removal decisions and investigation findings
Failing to document investigation process and decision rationale
Removing code without proper communication to affected teams and stakeholders
Skipping training team members on proper investigation and removal procedures
Failing to track investigation metrics and improvement in removal decision quality
Removing code without considering organizational standards and policies
Skipping integration with change management and approval processes
Failing to establish clear criteria and procedures for removal decisions

Validation Criteria:
Investigation Process Validation:

Every potential removal has documented investigation with findings and rationale
Git history analysis completed for all code targeted for removal
Comprehensive search completed across all repositories and services for references
Dynamic usage patterns investigated through runtime analysis and testing
Integration opportunities assessed and documented for all removal candidates
Stakeholder consultation completed for shared code and significant functionality
Business and technical value assessment documented for all removal decisions
Investigation timeline appropriate for complexity and scope of removal candidate
Investigation documentation accessible and reviewable by team members
Investigation process follows established procedures and organizational standards

Integration Assessment Validation:

All removal candidates evaluated for consolidation opportunities with existing code
Relocation opportunities assessed and documented for code that might belong elsewhere
Abstraction opportunities identified and evaluated for reusable functionality
Service and module migration opportunities assessed for organizational improvements
Configuration migration opportunities evaluated for hardcoded values and settings
Library extraction opportunities assessed for code that could be shared
Documentation migration opportunities evaluated for comments and inline documentation
Tool migration opportunities assessed for code that could become development utilities
Framework migration opportunities evaluated for code that should be modernized
Integration decisions documented with rationale and implementation timeline

Removal Decision Validation:

All removal decisions based on thorough investigation rather than automated detection alone
Removal decisions include clear documentation of investigation findings
Integration opportunities either implemented or documented with justification for not pursuing
Stakeholder approval obtained for removal of shared code and significant functionality
Removal timeline appropriate for complexity and coordination requirements
Backup and restoration procedures tested and validated before removal
Impact assessment completed for all affected systems and teams
Change management procedures followed for all significant removals
Rollback procedures tested and validated after removal completion
Post-removal validation confirms no functionality regression or integration issues

Team Process Validation:

All team members trained on investigation and removal procedures
Investigation responsibilities clearly assigned and consistently executed
Peer review process established and followed for removal decisions
Investigation metrics tracked and show improvement in decision quality over time
Team feedback collected and incorporated into investigation procedures
Documentation of investigation process current and accessible to team
Integration with organizational change management and approval processes functional
Communication procedures established and followed for removal decisions
Continuous improvement in investigation and removal processes demonstrated
Team competency in investigation tools and techniques validated and maintained



Rule 14: Specialized AI Agent Usage - Claude Code Sub-Agent Orchestration
CRITICAL: Claude Code Sub-Agent System
These are specifications for Claude Code's internal agent selection system. Claude Code uses these sub-agents from /opt/sutazaiapp/.claude/agents/ to apply specialized domain knowledge during task execution.
AI AGENT SELECTION ALGORITHM (Claude Code Internal):
pythonclass ClaudeCodeAgentSelector:
    def __init__(self):
        self.agents = self.load_agents_from_directory('/opt/sutazaiapp/.claude/agents/')
        self.changelog = self.parse_changelog_history()
        
    def select_optimal_agent(self, task_specification):
        # Task Analysis
        domain = self.identify_primary_domain(task_specification)
        complexity = self.assess_task_complexity(task_specification)
        technical_requirements = self.extract_technical_requirements(task_specification)
        
        # Historical Performance
        previous_success = self.check_changelog_for_similar_tasks(task_specification)
        
        # Agent Matching
        domain_specialists = self.filter_by_domain(domain)
        complexity_capable = self.filter_by_complexity(domain_specialists, complexity)
        requirement_matches = self.match_technical_requirements(complexity_capable, technical_requirements)
        
        # Selection
        optimal_agent = self.select_best_match(requirement_matches, previous_success)
        
        return optimal_agent
COMPLETE TASK MATCHING MATRIX:
yamlclaude_code_agent_matrix:
  backend_development:
    api_design:
      simple: "backend-architect.md"
      complex: ["backend-architect.md", "system-architect.md"]
      graphql: "graphql-architect.md"
      security_critical: ["backend-architect.md", "security-auditor.md"]
      
    microservices:
      architecture: "system-architect.md"
      implementation: "backend-architect.md"
      python: "python-pro.md"
      golang: "golang-pro.md"
      rust: "rust-pro.md"
      
    database_work:
      schema_design: "database-architect.md"
      admin: "database-admin.md"
      optimization: "database-optimizer.md"
      performance: "database-optimization.md"
      complex_queries: "sql-pro.md"
      nosql: "nosql-specialist.md"
      supabase: ["supabase-schema-architect.md", "supabase-realtime-optimizer.md"]
      
  frontend_development:
    react_projects:
      simple: "frontend-developer.md"
      complex: "frontend-architect.md"
      performance: "react-performance-optimization.md"
      
    ui_ux:
      design: "ui-ux-designer.md"
      architecture: "frontend-architect.md"
      accessibility: "web-accessibility-checker.md"
      vitals: "web-vitals-optimizer.md"
      cli: "cli-ui-designer.md"
      
    mobile:
      cross_platform: "mobile-developer.md"
      ios_specific: "ios-developer.md"
      
  testing_quality:
    test_strategy:
      lead: "test-engineer.md"
      automation: "test-automator.md"
      quality: "quality-engineer.md"
      
    specialized_testing:
      performance: "performance-engineer.md"
      profiling: "performance-profiler.md"
      load: "load-testing-specialist.md"
      mcp: "mcp-testing-engineer.md"
      code_review: "code-reviewer.md"
      
  devops_infrastructure:
    deployment:
      automation: "deployment-engineer.md"
      devops: "devops-engineer.md"
      troubleshooting: "devops-troubleshooter.md"
      
    infrastructure:
      cloud: "cloud-architect.md"
      terraform: "terraform-specialist.md"
      migration: "cloud-migration-specialist.md"
      monitoring: "monitoring-specialist.md"
      network: "network-engineer.md"
      incidents: "incident-responder.md"
      
  security_compliance:
    security:
      audit: "security-auditor.md"
      engineering: "security-engineer.md"
      penetration: "penetration-tester.md"
      api: "api-security-audit.md"
      mcp: "mcp-security-auditor.md"
      
    specialized:
      smart_contracts: "smart-contract-auditor.md"
      compliance: "compliance-specialist.md"
      
  data_ml:
    data:
      analysis: "data-analyst.md"
      engineering: "data-engineer.md"
      science: "data-scientist.md"
      quant: "quant-analyst.md"
      
    ml_ai:
      ai: "ai-engineer.md"
      ml: "ml-engineer.md"
      mlops: "mlops-engineer.md"
      nlp: "nlp-engineer.md"
      vision: "computer-vision-engineer.md"
      evaluation: "model-evaluator.md"
      ethics: "ai-ethics-advisor.md"
      prompts: "prompt-engineer.md"
      
  documentation:
    technical: "technical-writer.md"
    expert: "documentation-expert.md"
    api: "api-documenter.md"
    docusaurus: "docusaurus-expert.md"
    changelog: "changelog-generator.md"
    reports: "report-generator.md"
    
  specialized_domains:
    mcp_protocol:
      expert: "mcp-expert.md"
      protocol: "mcp-protocol-specialist.md"
      integration: "mcp-integration-engineer.md"
      deployment: "mcp-deployment-orchestrator.md"
      registry: "mcp-registry-navigator.md"
      server: "mcp-server-architect.md"
      
    web3_blockchain:
      contracts: "smart-contract-specialist.md"
      integration: "web3-integration-specialist.md"
      
    media:
      audio: ["audio-mixer.md", "audio-quality-controller.md"]
      video: "video-editor.md"
      podcast: ["podcast-transcriber.md", "podcast-content-analyzer.md", "podcast-metadata-specialist.md"]
      social: ["social-media-clip-creator.md", "tiktok-strategist.md"]
      
    ocr_processing:
      preprocessing: "ocr-preprocessing-optimizer.md"
      quality: "ocr-quality-assurance.md"
      grammar: "ocr-grammar-fixer.md"
      visual: "visual-analysis-ocr.md"
MULTI-AGENT COORDINATION PATTERNS:
yamlclaude_code_workflows:
  sequential_workflow:
    description: "Claude Code executes agents in sequence"
    example_full_feature:
      - stage: "requirements_analysis"
        agent: "business-analyst.md"
        output: "requirements_document"
        
      - stage: "system_design"
        agent: "system-architect.md"
        input: "requirements_document"
        output: "architecture_specification"
        
      - stage: "api_design"
        agent: "backend-architect.md"
        input: "architecture_specification"
        output: "api_specification"
        
      - stage: "database_design"
        agent: "database-architect.md"
        input: "api_specification"
        output: "database_schema"
        
      - stage: "implementation"
        agent: "python-pro.md"
        input: ["api_specification", "database_schema"]
        output: "working_code"
        
      - stage: "testing"
        agent: "test-automator.md"
        input: "working_code"
        output: "test_suite"
        
      - stage: "security_review"
        agent: "security-auditor.md"
        input: ["working_code", "test_suite"]
        output: "security_report"
        
      - stage: "deployment"
        agent: "deployment-engineer.md"
        input: ["working_code", "security_report"]
        output: "deployed_system"
        
  parallel_workflow:
    description: "Claude Code executes multiple agents simultaneously"
    coordination: "shared_api_contract"
    parallel_tracks:
      backend:
        agent: "backend-architect.md"
        artifact: "backend_implementation"
        
      frontend:
        agent: "frontend-developer.md"
        artifact: "frontend_implementation"
        
      database:
        agent: "database-admin.md"
        artifact: "database_setup"
        
      testing:
        agent: "test-engineer.md"
        artifact: "test_framework"
        
    integration:
      agent: "fullstack-developer.md"
      responsibility: "integrate_all_tracks"
      
  expert_consultation:
    description: "Claude Code consults specialists during execution"
    primary: "system-architect.md"
    consultations:
      - trigger: "performance_issue"
        specialist: "performance-engineer.md"
        
      - trigger: "security_concern"
        specialist: "security-auditor.md"
        
      - trigger: "database_complexity"
        specialist: "database-optimizer.md"
        
      - trigger: "ui_ux_decision"
        specialist: "ui-ux-designer.md"
PERFORMANCE TRACKING:
yamlclaude_code_metrics:
  task_completion:
    accuracy: "requirement_satisfaction"
    completeness: "all_requirements_met"
    quality: "code_standards_adherence"
    efficiency: "execution_time"
    
  agent_effectiveness:
    domain_expertise: "specialized_knowledge_applied"
    technical_accuracy: "correct_implementation"
    best_practices: "standards_followed"
    
  workflow_success:
    handoff_quality: "clean_artifacts_between_stages"
    integration_success: "components_work_together"
    overall_quality: "final_deliverable_quality"
✅ Required Practices:
For Claude Code Execution:

Ensure /opt/sutazaiapp/.claude/agents/ directory is accessible
Check CHANGELOG.md for previous agent usage patterns
Document agent selection in execution logs
Track performance metrics for optimization
Maintain agent files with current specifications
Use most specific agent for task domain
Design multi-agent workflows for complex tasks
Implement validation between agent handoffs
Monitor agent effectiveness over time
Share successful patterns via CHANGELOG.md

Agent Selection Priority:

Check if task matches previous successful patterns in CHANGELOG.md
Select most specialized agent for domain
Add complementary agents for multi-faceted tasks
Document selection rationale
Track effectiveness for future reference

🚫 Forbidden Practices:

Using agents not in /opt/sutazaiapp/.claude/agents/
Manually executing agent instructions outside Claude Code
Modifying agent files without understanding dependencies
Removing actively used agents
Using generic agents when specialists exist
Skipping CHANGELOG.md documentation
Ignoring performance tracking
Breaking established successful patterns
Using outdated agent references
Mixing Claude Code agents with other AI systems

Validation Criteria:

Agent exists in directory
Selection matches task requirements
Workflow properly sequenced
Handoffs clearly defined
Performance tracked
Documentation complete
Patterns reusable
Knowledge captured

CHANGELOG.md Entry Format:
markdown### [YYYY-MM-DD HH:MM:SS UTC] - Version - Component - Change Type
**Who**: Claude Code [Agent: agent-name.md]
**Why**: Task requirement and agent selection rationale
**What**: Specific work completed using agent expertise
**Impact**: Systems affected and dependencies
**Validation**: Testing performed and results
**Performance**: Execution metrics and effectiveness

-------

 Rule 15: Documentation Quality - Perfect Information Architecture
MISSION-CRITICAL: Perfect Documentation Excellence - Zero Ambiguity, Maximum Clarity:

Single Source of Truth: One authoritative location for each piece of information with zero duplication or conflicting content
Actionable Intelligence: Every document must provide clear next steps, decision criteria, and implementation guidance
Real-Time Currency: Documentation automatically updated and validated to remain current with system reality
Precise Temporal Tracking: Exact timestamps for all document creation, updates, and reviews with full audit trail
Instant Accessibility: Information discoverable and accessible within seconds through intelligent organization and search
Zero Knowledge Gaps: Complete coverage of all systems, processes, and decisions without missing critical information
Context-Aware Guidance: Documentation adapts to user context, role, and immediate needs for maximum relevance
Measurable Impact: Documentation effectiveness measured through user success, onboarding velocity, and decision accuracy
Continuous Intelligence: Documentation system learns and improves through usage patterns and feedback loops

CRITICAL: Mandatory Timestamp Requirements:

Creation Timestamp: Exact date and time (with timezone) when document was originally created
Last Modified Timestamp: Precise timestamp of most recent content modification
Review Timestamps: Timestamps for all formal reviews, approvals, and validations
Author Attribution: Clear identification of who created or modified content with timestamps
Change History: Complete audit trail of all changes with timestamps and change descriptions
Access Timestamps: Tracking of when content was last accessed for relevance analysis
Validation Timestamps: Timestamps for all automated and manual content validation checks
Retirement Timestamps: Exact timestamps when content is deprecated or archived

✅ Required Practices:
Comprehensive Timestamp Management:

Mandatory Header Information: Every document must include standardized header with complete timestamp data
Automated Timestamp Generation: Automated systems capture exact timestamps for all document lifecycle events
Timezone Standardization: All timestamps in UTC with clear timezone indication for global accessibility
Precision Requirements: Timestamps accurate to the second (YYYY-MM-DD HH:MM:SS UTC) for precise tracking
Change Attribution: Every change includes author identification and exact timestamp of modification
Review Cycle Tracking: Complete timestamp tracking of review cycles, approvals, and stakeholder sign-offs
System Integration: Timestamps automatically generated through version control and content management systems
Manual Override Prevention: Systems prevent manual timestamp manipulation to ensure accuracy and integrity
Audit Trail Preservation: Complete preservation of all timestamp data throughout document lifecycle
Access Pattern Tracking: Timestamps for content access patterns to identify popular and stale content

Standardized Document Header Format:
markdown---
document_id: "DOC-YYYY-NNNN"
title: "Document Title"
created_date: "2024-12-20 15:30:45 UTC"
created_by: "author.name@company.com"
last_modified: "2024-12-20 16:45:22 UTC"
modified_by: "editor.name@company.com"
last_reviewed: "2024-12-20 14:20:10 UTC"
reviewed_by: "reviewer.name@company.com"
next_review_due: "2025-03-20 14:20:10 UTC"
version: "2.1.0"
status: "active" | "draft" | "deprecated" | "archived"
owner: "team.name@company.com"
category: "architecture | process | api | user-guide"
tags: ["tag1", "tag2", "tag3"]
last_validation: "2024-12-20 13:15:30 UTC"
validation_status: "passed" | "failed" | "pending"
---

# Document Content Begins Here
📌 Rule 16: Local LLM Operations - Intelligent Hardware-Aware AI Management
MISSION-CRITICAL: Intelligent Resource-Aware AI Operations:

Intelligent Hardware Detection: Automated detection and assessment of current hardware capabilities and constraints
Real-Time Resource Monitoring: Continuous monitoring of system resources with predictive capacity analysis
Automated Model Selection: AI-powered decision making for optimal model selection based on task complexity and available resources
Dynamic Safety Management: Real-time safety checks and automatic model switching based on system health
Predictive Resource Management: Predictive analysis of resource requirements before model activation
Self-Healing Operations: Automatic recovery and optimization when resource constraints are detected
Zero Manual Intervention: Fully automated model management with human oversight only for critical decisions
Hardware-Optimized Performance: Continuous optimization based on detected hardware capabilities and performance patterns

CRITICAL: Automated Hardware Assessment and Decision System:
pythonclass HardwareIntelligenceSystem:
    def __init__(self):
        self.hardware_profile = self.detect_hardware_capabilities()
        self.performance_baselines = self.establish_performance_baselines()
        self.safety_thresholds = self.calculate_safety_thresholds()
        
    def detect_hardware_capabilities(self):
        """Comprehensive hardware detection and profiling"""
        return {
            'cpu': {
                'cores': self.get_cpu_cores(),
                'architecture': self.get_cpu_architecture(),
                'frequency': self.get_cpu_frequency(),
                'cache_size': self.get_cache_sizes(),
                'instruction_sets': self.get_supported_instructions(),
                'thermal_design_power': self.get_tdp(),
                'current_utilization': self.get_cpu_utilization(),
                'temperature': self.get_cpu_temperature()
            },
            'memory': {
                'total_ram': self.get_total_memory(),
                'available_ram': self.get_available_memory(),
                'memory_type': self.get_memory_type(),
                'memory_speed': self.get_memory_speed(),
                'memory_bandwidth': self.get_memory_bandwidth(),
                'swap_available': self.get_swap_space()
            },
            'gpu': {
                'gpu_present': self.detect_gpu(),
                'gpu_model': self.get_gpu_model(),
                'gpu_memory': self.get_gpu_memory(),
                'gpu_utilization': self.get_gpu_utilization(),
                'gpu_temperature': self.get_gpu_temperature(),
                'compute_capability': self.get_compute_capability()
            }
        }
📌 Rule 17: Canonical Documentation Authority - Ultimate Source of Truth
MISSION-CRITICAL: Absolute Information Authority:

Single Source of Truth: /opt/sutazaiapp/IMPORTANT/ serves as the ultimate authority that overrides all conflicting information
Continuous Document Migration: Systematic identification and migration of important documents to canonical authority location
Perpetual Currency: Continuous review and validation to ensure all authority documents remain current and accurate
Complete Temporal Tracking: Comprehensive timestamp tracking for creation, migration, updates, and all document lifecycle events
Hierarchical Authority: Clear authority hierarchy with /opt/sutazaiapp/IMPORTANT/ at the apex of all documentation systems
Automatic Conflict Resolution: Systematic detection and resolution of information conflicts with authority precedence
Real-Time Synchronization: All downstream documentation automatically synchronized with canonical sources
Universal Compliance: All teams, systems, and processes must comply with canonical authority without exception

Authority Document Standards with Migration Tracking:
markdown---
AUTHORITY_LEVEL: "CANONICAL_SOURCE_OF_TRUTH"
document_id: "AUTH-YYYY-NNNN"
title: "Canonical Authority Document Title"

# CREATION TRACKING
original_creation_date: "2024-01-15 10:30:45 UTC"
original_creation_by: "original.author@company.com"
original_creation_location: "/team_docs/architecture/system_design.md"
original_discovery_date: "2024-12-20 09:15:30 UTC"
discovered_by: "document.curator@company.com"

# MIGRATION TRACKING  
migration_date: "2024-12-20 16:45:22 UTC"
migration_from: "/team_docs/architecture/system_design.md"
migration_to: "/opt/sutazaiapp/IMPORTANT/architecture/system_architecture_authority.md"
migration_by: "document.curator@company.com"
migration_reason: "Critical system architecture document requires authority status"
migration_approved_by: "chief.architect@company.com"
migration_validation: "Content integrity verified, references updated"

# CURRENT STATUS
last_modified: "2024-12-20 16:45:22 UTC"
modified_by: "authority.owner@company.com"
last_authority_review: "2024-12-20 16:45:22 UTC"
authority_reviewer: "chief.architect@company.com"
next_authority_review: "2025-01-20 16:45:22 UTC"
currency_status: "current"
last_currency_check: "2024-12-20 17:00:00 UTC"

# AUTHORITY METADATA
version: "1.0.0"
status: "CANONICAL_AUTHORITY"
authority_scope: "Complete system architecture design and standards"
override_precedence: "ABSOLUTE"
conflict_resolution_owner: "chief.architect@company.com"
emergency_contact: "architecture.team@company.com"
---
📌 Rule 18: Mandatory Documentation Review - Comprehensive Knowledge Acquisition
Mandatory Review Sequence (Must be completed in order):

CHANGELOG.md Audit and Creation (FIRST PRIORITY)

Scan all directories in work scope for CHANGELOG.md files
Create missing CHANGELOG.md files using standardized template
Review existing CHANGELOG.md files for currency and completeness
Identify any gaps in change documentation and flag for investigation
Validate CHANGELOG.md format consistency across all directories
Update any outdated or incomplete CHANGELOG.md files immediately


Primary Authority Sources (/opt/sutazaiapp/IMPORTANT/)

Line-by-line review of complete document including recent updates
Cross-reference with CHANGELOG.md to understand rule evolution
Note any updates since last review with timestamps
Document understanding of all 20 fundamental rules
Identify any rule changes or additions since last work
Validate understanding of specialized AI agent requirements


Canonical Authority Documentation (/opt/sutazaiapp/IMPORTANT/*)

Complete review of all documents in authority hierarchy
Review corresponding CHANGELOG.md files for change context
Reference architecture diagrams and validate understanding
Review PortRegistry.md for any port allocation changes
Validate Docker architecture requirements and constraints
Cross-reference authority documents for consistency



📌 Rule 19: Change Tracking Requirements - Comprehensive Change Intelligence System
Mandatory CHANGELOG.md Entry Format (Enhanced and Comprehensive):
markdown### [YYYY-MM-DD HH:MM:SS.fff UTC] - [SemVer] - [COMPONENT] - [CHANGE_TYPE] - [Brief Description]
**Change ID**: CHG-YYYY-NNNNNN (auto-generated unique identifier)
**Execution Time**: [YYYY-MM-DD HH:MM:SS.fff UTC] (precise execution timestamp)
**Duration**: [XXX.XXXs] (time taken to implement change)
**Trigger**: [manual/automated/scheduled/incident_response/security_patch]

**Who**: [AI Agent (agent-name.md) OR Human (full.name@company.com)]
**Approval**: [approver.name@company.com] (for changes requiring approval)
**Review**: [reviewer1@company.com, reviewer2@company.com] (peer reviewers)

**Why**: [Comprehensive business/technical justification]
- **Business Driver**: [Business requirement, user need, compliance requirement]
- **Technical Rationale**: [Technical debt, performance, security, scalability]
- **Risk Mitigation**: [What risks does this change address]
- **Success Criteria**: [How will success be measured]

**What**: [Detailed technical description of changes]
- **Files Modified**: [List of all files with line count changes]
- **Database Changes**: [Schema, data, index modifications]
- **Configuration Changes**: [Environment, deployment, infrastructure]
- **Dependencies**: [New, updated, or removed dependencies]
- **API Changes**: [Endpoint modifications, breaking changes]
- **UI/UX Changes**: [User interface modifications]

**Impact Analysis**: [Comprehensive impact assessment]
- **Downstream Systems**: [Systems that depend on this change]
- **Upstream Dependencies**: [Systems this change depends on]
- **User Impact**: [End user experience changes]
- **Performance Impact**: [Performance characteristics affected]
- **Security Impact**: [Security posture changes]
- **Compliance Impact**: [Regulatory or policy compliance effects]

**Testing and Validation**: [Comprehensive testing information]
- **Test Coverage**: [Unit: XX%, Integration: XX%, E2E: XX%]
- **Test Types**: [Unit, integration, performance, security, accessibility]
- **Test Results**: [Pass/fail status, performance metrics]
- **Manual Testing**: [Manual test scenarios executed]
- **User Acceptance**: [UAT results, stakeholder sign-off]

**Rollback Planning**: [Comprehensive rollback information]
- **Rollback Procedure**: [Step-by-step rollback instructions]
- **Rollback Trigger Conditions**: [When to initiate rollback]
- **Rollback Time Estimate**: [Expected time to complete rollback]
- **Rollback Testing**: [Validation that rollback procedures work]
- **Data Recovery**: [Data backup and recovery procedures]
📌 Rule 20: MCP Server Protection - Critical Infrastructure Safeguarding
MISSION-CRITICAL: Absolute MCP Infrastructure Protection:

Absolute Protection: MCP servers are protected infrastructure that must never be modified without explicit user authorization
Comprehensive Investigation: All MCP issues must be thoroughly investigated and documented before any action is taken
Proactive Monitoring: Continuous monitoring and health checking of all MCP server components and configurations
Emergency Procedures: Established emergency procedures for MCP server failures that prioritize restoration over removal
Change Control: Rigorous change control procedures specifically designed for MCP server infrastructure
Knowledge Preservation: Comprehensive documentation and knowledge management for all MCP server configurations
Team Training: Mandatory training for all team members on MCP server protection and management procedures
Business Continuity: MCP server protection ensures business continuity and AI system functionality

Mandatory Protection Requirements:

Zero Unauthorized Changes: No modifications to MCP servers, configurations, or wrapper scripts without explicit user permission
Preservation First: Always preserve existing MCP server integrations when making any system changes
Investigation Over Removal: Investigate and report MCP issues rather than removing or disabling servers
Configuration Immutability: Treat .mcp.json and wrapper scripts as immutable without explicit authorization
Integration Preservation: Ensure all system changes maintain MCP server functionality and integration
Backup and Recovery: Maintain comprehensive backups and recovery procedures for all MCP configurations
Access Control: Implement strict access controls and audit trails for MCP server administration
Change Documentation: Document all authorized MCP changes with comprehensive rationale and approval