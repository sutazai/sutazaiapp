Rule 13: Zero Tolerance for Waste
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


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/