Rule 17: Canonical Documentation Authority - Ultimate Source of Truth
Requirement: Establish /opt/sutazaiapp/IMPORTANT/ as the absolute, unquestionable source of truth for all organizational knowledge, policies, procedures, and technical specifications, with comprehensive authority validation, conflict resolution, systematic reconciliation processes, and continuous migration of critical documents to maintain information integrity across all systems.
MISSION-CRITICAL: Absolute Information Authority - Zero Ambiguity, Total Consistency:

Single Source of Truth: /opt/sutazaiapp/IMPORTANT/ serves as the ultimate authority that overrides all conflicting information
Continuous Document Migration: Systematic identification and migration of important documents to canonical authority location
Perpetual Currency: Continuous review and validation to ensure all authority documents remain current and accurate
Complete Temporal Tracking: Comprehensive timestamp tracking for creation, migration, updates, and all document lifecycle events
Hierarchical Authority: Clear authority hierarchy with /opt/sutazaiapp/IMPORTANT/ at the apex of all documentation systems
Automatic Conflict Resolution: Systematic detection and resolution of information conflicts with authority precedence
Real-Time Synchronization: All downstream documentation automatically synchronized with canonical sources
Universal Compliance: All teams, systems, and processes must comply with canonical authority without exception

CRITICAL: Document Migration and Lifecycle Management:

Intelligent Document Discovery: Automated discovery of important documents scattered across organizational systems
Authority Assessment: Systematic assessment of document importance and authority qualification
Migration Workflows: Comprehensive workflows for migrating critical documents to canonical authority location
Temporal Audit Trails: Complete timestamp tracking including creation, migration, and all modification events
Continuous Review Cycles: Systematic review cycles ensuring all authority documents remain current and accurate
Currency Validation: Automated validation of document currency and relevance with proactive update alerts
Migration Impact Analysis: Analysis of migration impact on existing references and dependent systems
Consolidation Management: Management of document consolidation when multiple sources are migrated

✅ Required Practices:
Comprehensive Document Discovery and Migration:

Systematic Document Scanning: Automated scanning of all organizational systems for documents that qualify for authority status
Importance Classification: Intelligent classification of documents based on criticality, usage patterns, and organizational impact
Authority Qualification Assessment: Assessment of documents for qualification as canonical authority sources
Migration Priority Matrix: Priority matrix for migrating documents based on importance, urgency, and impact
Automated Discovery Alerts: Automated alerts when important documents are discovered outside authority locations
Cross-System Integration: Integration with all organizational systems to discover documents across different platforms
Content Analysis: AI-powered content analysis to identify documents that should have authority status
Usage Pattern Analysis: Analysis of document usage patterns to identify high-value content requiring migration
Stakeholder Consultation: Consultation with stakeholders to validate document importance and migration decisions
Migration Workflow Automation: Automated workflows for streamlined document migration processes

Document Migration Workflow System:
pythonclass DocumentMigrationSystem:
    def __init__(self):
        self.discovery_engine = DocumentDiscoveryEngine()
        self.migration_workflow = MigrationWorkflowManager()
        self.authority_validator = AuthorityValidator()
        
    def discover_and_migrate_important_documents(self):
        """Comprehensive document discovery and migration process"""
        
        # Phase 1: Discovery
        discovered_documents = self.discovery_engine.scan_all_systems([
            '/home/*/documents/',
            '/shared/team_docs/',
            '/project_docs/',
            '/wiki_exports/',
            '/confluence_backup/',
            '/sharepoint_sync/',
            '/google_drive_sync/',
            '/slack_files/',
            '/email_attachments/',
            '/version_control_docs/'
        ])
        
        # Phase 2: Importance Assessment
        for document in discovered_documents:
            importance_score = self.assess_document_importance(document)
            if importance_score >= self.AUTHORITY_THRESHOLD:
                migration_candidate = {
                    'source_path': document.path,
                    'importance_score': importance_score,
                    'content_type': document.content_type,
                    'usage_frequency': document.usage_stats,
                    'stakeholder_references': document.stakeholder_count,
                    'last_modified': document.last_modified,
                    'creation_date': document.creation_date,
                    'discovered_date': datetime.utcnow(),
                    'migration_priority': self.calculate_migration_priority(document)
                }
                
                # Phase 3: Migration Execution
                self.execute_migration(migration_candidate)
    
    def execute_migration(self, migration_candidate):
        """Execute document migration with complete audit trail"""
        
        migration_record = {
            'migration_id': self.generate_migration_id(),
            'source_path': migration_candidate['source_path'],
            'target_path': self.determine_target_path(migration_candidate),
            'migration_timestamp': datetime.utcnow(),
            'migration_reason': self.document_migration_reason(migration_candidate),
            'original_creation_date': migration_candidate['creation_date'],
            'migration_approved_by': self.get_migration_approver(),
            'content_validation': self.validate_content_integrity(),
            'reference_updates_required': self.identify_reference_updates()
        }
        
        # Execute migration with full tracking
        self.migration_workflow.migrate_with_tracking(migration_record)
Continuous Review and Currency Management:

Automated Review Scheduling: Automated scheduling of review cycles based on document type, criticality, and age
Currency Monitoring: Real-time monitoring of document currency with alerts for outdated or stale content
Proactive Update Alerts: Proactive alerts to document owners when content may need updating
Review Assignment: Intelligent assignment of review tasks to appropriate subject matter experts
Review Workflow Management: Comprehensive workflow management for document review processes
Currency Validation: Automated validation of document currency against system state and external changes
Update Tracking: Detailed tracking of all updates and changes with complete audit trails
Review Quality Assurance: Quality assurance processes for review completeness and accuracy
Escalation Procedures: Escalation procedures for overdue reviews and unresolved currency issues
Review Performance Metrics: Performance metrics for review processes and reviewer effectiveness

Comprehensive Temporal Tracking System:
yamlmandatory_document_metadata:
  temporal_tracking:
    original_creation_date: "YYYY-MM-DD HH:MM:SS UTC"
    original_creation_by: "creator.email@company.com"
    original_creation_location: "/original/path/to/document"
    
    migration_history:
      - migration_date: "YYYY-MM-DD HH:MM:SS UTC"
        migration_from: "/previous/location/path"
        migration_to: "/opt/sutazaiapp/IMPORTANT/category/"
        migration_by: "migrator.email@company.com"
        migration_reason: "Document identified as critical authority source"
        migration_approval: "chief.architect@company.com"
        
    modification_history:
      - modification_date: "YYYY-MM-DD HH:MM:SS UTC"
        modified_by: "editor.email@company.com"
        modification_type: "content_update" | "metadata_update" | "structural_change"
        modification_summary: "Brief description of changes made"
        approval_required: true | false
        approved_by: "approver.email@company.com"
        
    review_history:
      - review_date: "YYYY-MM-DD HH:MM:SS UTC"
        reviewed_by: "reviewer.email@company.com"
        review_type: "scheduled" | "triggered" | "emergency"
        review_outcome: "current" | "needs_update" | "major_revision"
        next_review_due: "YYYY-MM-DD HH:MM:SS UTC"
        review_notes: "Reviewer comments and recommendations"
        
    currency_validation:
      last_currency_check: "YYYY-MM-DD HH:MM:SS UTC"
      currency_status: "current" | "stale" | "outdated" | "critical"
      currency_validated_by: "validator.email@company.com"
      next_currency_check: "YYYY-MM-DD HH:MM:SS UTC"
      automated_checks_enabled: true | false
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

# DEPENDENCIES AND REFERENCES
downstream_dependencies:
  - "/docs/api/api_design_standards.md"
  - "/docs/deployment/deployment_architecture.md"
  - "/docs/security/security_architecture.md"
related_authorities:
  - "AUTH-2024-0001 (Security Architecture Authority)"
  - "AUTH-2024-0003 (Data Architecture Authority)"
reference_updates_completed: true
broken_references_fixed: true
---

# CANONICAL AUTHORITY NOTICE
This document serves as the CANONICAL SOURCE OF TRUTH for system architecture.
All conflicting information in other documents is superseded by this authority.
Any discrepancies must be reported immediately for reconciliation.

**MIGRATION NOTICE**: This document was migrated from `/team_docs/architecture/system_design.md` 
on 2024-12-20 16:45:22 UTC due to its critical importance as organizational authority.

## Document History Summary
- **Originally Created**: January 15, 2024 by original.author@company.com
- **Discovered for Migration**: December 20, 2024 during systematic authority review
- **Migrated to Authority Status**: December 20, 2024 with full validation and reference updates
- **Authority Status Confirmed**: Chief Architect approval on December 20, 2024
Continuous Review and Update Management:

Review Schedule Automation: Automated scheduling of reviews based on document criticality and change frequency
Multi-Tier Review Process: Multi-tier review process with different intervals for different document types
Stakeholder Review Coordination: Coordination of reviews involving multiple stakeholders and subject matter experts
Review Quality Standards: Quality standards for review thoroughness and documentation
Update Trigger Identification: Identification of external changes that trigger need for document updates
Review Performance Tracking: Tracking of review performance and reviewer effectiveness
Review Backlog Management: Management of review backlogs and overdue reviews
Emergency Review Procedures: Emergency review procedures for critical updates and urgent changes
Review Integration: Integration of reviews with change management and development processes
Continuous Improvement: Continuous improvement of review processes based on effectiveness metrics

Document Currency Validation System:

Automated Currency Checks: Automated checks for document currency against system state and external changes
Currency Indicators: Clear currency indicators visible to all users of authority documents
Staleness Detection: Detection of document staleness based on usage patterns and external changes
Update Recommendations: Automated recommendations for document updates based on currency analysis
Currency Metrics: Comprehensive metrics on document currency and update frequency
Currency Alerts: Alert systems for documents approaching or exceeding currency thresholds
Validation Workflows: Workflows for validating document currency and scheduling updates
Currency Reporting: Regular reporting on document currency status across all authority documents
Predictive Currency Analysis: Predictive analysis of when documents will need updates
Currency Integration: Integration of currency management with other document management processes

Migration Impact Management:

Reference Discovery: Discovery of all references to documents being migrated to authority status
Reference Update Automation: Automated updating of references when documents are migrated
Link Validation: Validation that all links and references work correctly after migration
Notification Management: Notification of all stakeholders when documents are migrated
Access Control Migration: Migration of appropriate access controls with document authority elevation
Tool Integration: Integration with development tools and systems to update references automatically
Backup and Recovery: Backup of original documents before migration with recovery procedures
Migration Validation: Validation that migrations completed successfully without data loss
Integration Testing: Testing to ensure migrated documents integrate properly with existing systems
User Training: Training for users on new document locations and authority status

🚫 Forbidden Practices:
Document Migration Violations:

Leaving important documents in non-authority locations when they qualify for canonical authority status
Migrating documents without proper temporal tracking and audit trail documentation
Failing to update references and links when documents are migrated to authority locations
Migrating documents without proper approval and stakeholder notification processes
Creating duplicate copies instead of properly migrating documents to authority locations
Ignoring discovered important documents and failing to assess them for migration needs
Migrating documents without proper content validation and integrity checking
Failing to document migration reasons and decision-making rationale
Migrating documents without considering impact on existing workflows and processes
Bypassing migration procedures for "urgent" or "temporary" document moves

Review and Currency Violations:

Ignoring scheduled review cycles and allowing authority documents to become outdated
Conducting superficial reviews without proper validation of content currency and accuracy
Failing to update documents when reviews identify needed changes or corrections
Allowing authority documents to remain stale without proper staleness indicators
Skipping review approval processes and making unauthorized changes to authority documents
Ignoring currency alerts and automated recommendations for document updates
Conducting reviews without proper documentation of review outcomes and decisions
Failing to schedule appropriate review cycles based on document criticality and change frequency
Making document changes without proper review and approval workflows
Ignoring review performance metrics and failing to improve review processes

Temporal Tracking Violations:

Creating or modifying documents without proper timestamp documentation and audit trails
Failing to document original creation dates and authorship information during migration
Modifying timestamp information manually or bypassing automated timestamp generation
Missing migration date documentation when documents are moved to authority status
Failing to track modification history and change attribution throughout document lifecycle
Ignoring temporal tracking requirements in automated systems and document management tools
Creating documents without proper metadata structure and temporal tracking compliance
Failing to preserve temporal tracking information during document format changes or migrations
Bypassing temporal tracking for "minor" changes or administrative modifications
Using inconsistent timestamp formats or failing to use UTC standardization

Authority Management Violations:

Maintaining important documents outside authority structure without proper migration assessment
Creating alternative authority sources without proper integration and hierarchy management
Ignoring authority precedence when conflicts exist between canonical and non-canonical sources
Making authority-related decisions without consulting current authority documents
Implementing changes that contradict established authority without proper exception procedures
Bypassing authority review and approval processes for content modifications
Creating unofficial documentation that duplicates or conflicts with canonical authority
Using outdated authority documents when current versions are available
Ignoring authority migration requirements when documents qualify for canonical status
Failing to maintain authority document quality and currency standards

Validation Criteria:
Document Discovery and Migration Excellence:

Automated document discovery operational and identifying important documents across all organizational systems
Migration assessment comprehensive and accurately identifying documents requiring authority status
Migration workflows efficient and ensuring seamless transition of documents to authority locations
Temporal tracking complete and preserving all historical information throughout migration process
Reference updating automated and ensuring all links and dependencies remain functional after migration
Stakeholder notification comprehensive and ensuring all affected parties are informed of migrations
Migration approval processes functional and ensuring appropriate governance and oversight
Content validation thorough and ensuring document integrity throughout migration process
Impact analysis comprehensive and addressing all effects of document authority elevation
Migration performance metrics positive and demonstrating efficient and effective migration processes

Continuous Review Excellence:

Review scheduling automated and ensuring all authority documents receive appropriate review frequency
Review quality high and demonstrating thorough validation of content currency and accuracy
Review completion rates excellent and meeting established targets for review cycle adherence
Currency monitoring comprehensive and providing real-time visibility into document staleness
Update processes efficient and ensuring rapid response to identified currency issues
Review assignment intelligent and matching reviewers with appropriate expertise and availability
Review documentation complete and providing clear audit trails for all review activities
Review performance optimization continuous and improving review effectiveness and efficiency
Review integration seamless and connecting with broader change management and development processes
Stakeholder satisfaction high with review quality, timing, and communication

Temporal Tracking Excellence:

Timestamp accuracy complete and providing precise tracking of all document lifecycle events
Audit trail preservation comprehensive and maintaining complete historical record of all document changes
Metadata standards consistently applied across all authority documents and systems
Temporal tracking automation functional and preventing manual timestamp manipulation
Historical preservation complete and maintaining access to all versions and change history
Migration tracking detailed and documenting complete migration process and decision-making
Modification tracking granular and capturing all changes with appropriate attribution and approval
Review tracking comprehensive and documenting all review activities and outcomes
Currency tracking real-time and providing current visibility into document staleness and update needs
Integration with version control systems seamless and maintaining consistency across all tracking systems

Authority Management Excellence:

Authority hierarchy clearly established and consistently respected across all organizational systems
Document quality exceptional and maintaining high standards for all canonical authority sources
Currency maintenance systematic and ensuring all authority documents remain current and accurate
Conflict resolution efficient and rapidly addressing discrepancies between authority and other sources
Change management comprehensive and ensuring all authority modifications follow proper procedures
Stakeholder adoption complete and demonstrating consistent consultation of authority documents
System integration seamless and ensuring authority documents are accessible and usable across all platforms
Governance processes robust and providing appropriate oversight and control for authority management
Performance metrics positive and demonstrating continuous improvement in authority system effectiveness
Business value demonstrated through measurable improvements in decision-making quality and organizational alignment

Document Lifecycle Management Excellence:

Creation standards consistently applied and ensuring all new authority documents meet quality requirements
Migration procedures standardized and enabling efficient elevation of important documents to authority status
Review cycles optimized and balancing currency needs with resource efficiency
Update processes streamlined and enabling rapid response to changing requirements and external factors
Retirement procedures systematic and ensuring obsolete authority documents are properly archived
Version control comprehensive and maintaining clear lineage and change tracking throughout document lifecycle
Access control appropriate and ensuring proper security while enabling necessary access for authority consultation
Backup and recovery robust and protecting against data loss and ensuring business continuity
Integration testing thorough and ensuring authority documents work correctly with all dependent systems
User experience excellent and enabling efficient discovery, access, and utilization of authority information

Enhanced Document Header Template with Migration Tracking:
markdown---
# AUTHORITY DESIGNATION
AUTHORITY_LEVEL: "CANONICAL_SOURCE_OF_TRUTH"
document_id: "AUTH-2024-0156"
title: "Complete Document Title with Authority Designation"

# TEMPORAL TRACKING - CREATION
original_creation_date: "2024-01-15 10:30:45 UTC"
original_creation_by: "jane.developer@company.com"
original_creation_location: "/team_docs/processes/deployment_guide.md"
original_discovery_date: "2024-12-20 09:15:30 UTC"
discovered_by: "document.curator@company.com"
discovery_method: "automated_scan" | "manual_identification" | "stakeholder_nomination"

# TEMPORAL TRACKING - MIGRATION
migration_date: "2024-12-20 16:45:22 UTC"
migration_from: "/team_docs/processes/deployment_guide.md"
migration_to: "/opt/sutazaiapp/IMPORTANT/processes/deployment_process_authority.md"
migration_by: "document.curator@company.com"
migration_reason: "Critical deployment process requires canonical authority status"
migration_approved_by: "chief.architect@company.com"
migration_validation_completed: true
references_updated: true
stakeholders_notified: true

# TEMPORAL TRACKING - MODIFICATIONS
last_modified: "2024-12-20 16:45:22 UTC"
modified_by: "process.owner@company.com"
modification_type: "content_update"
modification_summary: "Updated deployment procedures for new container platform"
modification_approved_by: "chief.architect@company.com"

# TEMPORAL TRACKING - REVIEWS
last_authority_review: "2024-12-20 16:45:22 UTC"
authority_reviewer: "chief.architect@company.com"
review_outcome: "current"
next_authority_review: "2025-01-20 16:45:22 UTC"
review_frequency: "monthly" | "quarterly" | "semi-annual" | "annual"

# TEMPORAL TRACKING - CURRENCY
currency_status: "current" | "stale" | "outdated" | "critical"
last_currency_check: "2024-12-20 17:00:00 UTC"
currency_validated_by: "automated_system" | "manual_reviewer"
next_currency_check: "2024-12-27 17:00:00 UTC"
currency_triggers: ["system_changes", "policy_updates", "external_changes"]

# AUTHORITY METADATA
version: "2.1.0"
status: "CANONICAL_AUTHORITY"
authority_scope: "Complete deployment process and procedures"
override_precedence: "ABSOLUTE"
conflict_resolution_owner: "devops.lead@company.com"
emergency_contact: "devops.team@company.com"


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/
