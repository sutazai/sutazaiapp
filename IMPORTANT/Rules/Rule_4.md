Rule 4: Investigate Existing Files & Consolidate First
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


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/