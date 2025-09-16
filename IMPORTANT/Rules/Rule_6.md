Rule 6: Centralized Documentation
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

*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/

