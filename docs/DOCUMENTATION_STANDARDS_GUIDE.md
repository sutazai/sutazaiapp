# DOCUMENTATION STANDARDS GUIDE
## SutazAI Knowledge Management & Quality Assurance Framework

**Created**: 2025-08-16 06:30:00 UTC  
**Authority**: System Knowledge Curator  
**Purpose**: Establish comprehensive documentation standards for knowledge consistency  
**Scope**: All SutazAI documentation including CLAUDE.md, AGENTS.md, and technical guides

---

## 1. DOCUMENTATION ACCURACY STANDARDS

### 1.1 Reality Verification Requirements

#### **MANDATORY VERIFICATION PROCEDURES**
```yaml
Service Documentation:
  Container Status: docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
  Health Checks: curl -f http://localhost:[port]/health
  Configuration: Verify config file existence and syntax
  Port Availability: netstat -tulpn | grep [port]

Agent Documentation:
  Registry Verification: Check agents/agent_registry.json entries
  Implementation Status: Verify code exists in respective directories
  Container Status: Confirm containerized agents in docker-compose
  Health Endpoints: Test agent health check responses

API Documentation:
  Endpoint Testing: curl -X GET http://localhost:10010/api/v1/[endpoint]
  Response Validation: Verify actual response matches documentation
  Authentication: Test auth requirements and token handling
  Error Codes: Document actual error responses
```

#### **VERIFICATION AUTOMATION**
```bash
#!/bin/bash
# Documentation Validation Script
# Location: /scripts/validation/validate_documentation.sh

validate_services() {
    local services_file="$1"
    local errors=0
    
    while IFS= read -r service_line; do
        if [[ $service_line =~ Port[[:space:]]+([0-9]+) ]]; then
            port="${BASH_REMATCH[1]}"
            if ! nc -z localhost "$port"; then
                echo "ERROR: Service port $port not accessible"
                ((errors++))
            fi
        fi
    done < "$services_file"
    
    return $errors
}

validate_agent_status() {
    local agent_registry="/opt/sutazaiapp/agents/agent_registry.json"
    local docker_services
    docker_services=$(docker ps --format "{{.Names}}" | grep agent)
    
    # Cross-reference registry with running containers
    # Implementation details...
}
```

### 1.2 Accuracy Classification System

#### **ACCURACY LEVELS**
```yaml
VERIFIED:
  Definition: Confirmed against running system within 24 hours
  Requirements: Automated verification passed
  Indicators: ✅ VERIFIED badge, timestamp
  Example: "Kong API Gateway (Port 10005) ✅ VERIFIED 2025-08-16"

IMPLEMENTED:
  Definition: Code exists and functional, may not be containerized
  Requirements: Code review + manual testing
  Indicators: ⚙️ IMPLEMENTED badge
  Example: "Service Mesh v2 ⚙️ IMPLEMENTED (not yet containerized)"

DEFINED:
  Definition: Specified in configuration, not yet implemented
  Requirements: Configuration file verification
  Indicators: 📋 DEFINED badge
  Example: "Advanced ML Agent 📋 DEFINED (in registry, awaiting implementation)"

PLANNED:
  Definition: Roadmap items without implementation
  Requirements: Architectural documentation
  Indicators: 🗓️ PLANNED badge
  Example: "Multi-cloud deployment 🗓️ PLANNED Q2 2025"
```

---

## 2. CONTENT ORGANIZATION STANDARDS

### 2.1 Document Structure Templates

#### **SERVICE DOCUMENTATION TEMPLATE**
```markdown
# Service Name
**Status**: [VERIFIED/IMPLEMENTED/DEFINED] | **Last Verified**: YYYY-MM-DD HH:MM:SS UTC

## Overview
Brief description of service purpose and role in system architecture.

## Configuration
- **Container Name**: sutazai-[service-name]
- **Ports**: [external]:[internal] - Purpose description
- **Image**: [image:tag] - Specific version
- **Dependencies**: List of required services
- **Health Check**: curl http://localhost:[port]/health

## Integration Points
- **API Endpoints**: List of exposed endpoints
- **Service Dependencies**: Services this depends on
- **Dependent Services**: Services that depend on this
- **Communication Protocols**: REST/gRPC/messaging

## Operations
- **Startup Command**: docker-compose up [service-name]
- **Logs**: docker logs sutazai-[service-name]
- **Troubleshooting**: Common issues and solutions
- **Monitoring**: Metrics and alerts

## Examples
Working examples with actual curl commands and expected responses.
```

#### **AGENT DOCUMENTATION TEMPLATE**
```markdown
# Agent Name
**Type**: [Ultra-Tier/Core-Operational/Specialist] | **Status**: [OPERATIONAL/IMPLEMENTED/DEFINED/PLANNED]  
**Last Updated**: YYYY-MM-DD HH:MM:SS UTC

## Specialization
Specific domain expertise and use cases.

## Implementation Status
- **Registry Entry**: ✅/❌ Present in agent_registry.json
- **Code Implementation**: ✅/❌ Code exists in /agents/[agent-name]/
- **Containerized**: ✅/❌ Running as Docker container
- **Health Endpoint**: http://localhost:[port]/health

## Capabilities
```yaml
capabilities:
  - capability_1: Description
  - capability_2: Description
```

## Usage Patterns
When to use this agent vs alternatives.

## Integration
- **Communication**: REST/messaging/direct
- **Dependencies**: Required services/agents
- **Coordination**: How it works with other agents

## Examples
Real usage examples with working commands.
```

### 2.2 Cross-Reference Standards

#### **LINKING CONVENTIONS**
```yaml
Internal Links:
  Services: [Service Name](./CLAUDE.md#service-name)
  Agents: [Agent Name](./AGENTS.md#agent-name)  
  Configs: [Config File](/path/to/config.yml)
  Code: [Implementation](/path/to/code.py)

External References:
  Docker Images: [image:tag](https://hub.docker.com/r/[image])
  Documentation: [External Doc](https://external-site.com/docs)
  GitHub: [Repository](https://github.com/org/repo)

Status References:
  Always include current status and last verification timestamp
  Link to health check endpoints where available
  Reference configuration files and code locations
```

#### **CROSS-REFERENCE MATRIX**
```yaml
Service -> Agent Relationships:
  Document which agents use which services
  Include dependency chains and communication patterns
  Show optional vs required dependencies

Agent -> Service Relationships:
  Show which services agents expose
  Document inter-agent communication requirements
  Map agent orchestration patterns

Configuration Relationships:
  Link services to their config files
  Reference environment variables and settings
  Show config inheritance and overrides
```

---

## 3. TERMINOLOGY & NAMING STANDARDS

### 3.1 Standardized Terminology

#### **SERVICE TERMINOLOGY**
```yaml
Preferred Terms:
  - "Service" (not "microservice", "component", "module")
  - "Container" (not "instance", "deployment")
  - "Health Check" (not "health endpoint", "status check")
  - "Configuration" (not "config", "settings", "options")
  - "Dependencies" (not "requirements", "prereqs")

Status Terms:
  - OPERATIONAL: Currently running and healthy
  - DEGRADED: Running but with issues
  - STOPPED: Not running
  - FAILED: Failed to start or crashed
  - MAINTENANCE: Temporarily disabled for updates
```

#### **AGENT TERMINOLOGY**
```yaml
Preferred Terms:
  - "Agent" (not "AI agent", "bot", "service")
  - "Capabilities" (not "skills", "functions", "abilities")
  - "Specialization" (not "domain", "expertise area")
  - "Implementation" (not "code", "logic", "algorithm")
  - "Orchestration" (not "coordination", "management")

Agent Types:
  - Ultra-Tier: Supreme coordinators (2 agents)
  - Core-Operational: Essential running agents (8 containers)
  - Specialist: Domain-specific implementations (200+ definitions)
  - Utility: Support and helper agents
```

#### **ARCHITECTURE TERMINOLOGY**
```yaml
Infrastructure Terms:
  - "Service Mesh" (Kong + Consul + service discovery)
  - "Message Broker" (RabbitMQ for async communication)
  - "API Gateway" (Kong for external access)
  - "Service Discovery" (Consul for health/registration)
  - "Vector Database" (ChromaDB/Qdrant/FAISS)
  - "Monitoring Stack" (Prometheus/Grafana/Loki/exporters)

Avoid Ambiguous Terms:
  - "System" (too vague - specify service/agent/infrastructure)
  - "Platform" (too broad - specify component)
  - "Framework" (specify exact framework type)
  - "Engine" (specify processing/coordination/analysis engine)
```

### 3.2 Naming Conventions

#### **FILE & DIRECTORY NAMING**
```yaml
Documentation Files:
  - UPPERCASE.md for system-level docs (CLAUDE.md, AGENTS.md)
  - lowercase-kebab-case.md for guides (user-guide.md)
  - Title-Case-With-Underscores.md for reports (System_Analysis_Report.md)

Agent Definitions:
  - lowercase-kebab-case.md (ai-senior-engineer.md)
  - No spaces, underscores, or special characters
  - Descriptive but concise names

Configuration Files:
  - lowercase-kebab-case.yml/json (agent-registry.json)
  - Environment-specific suffixes (.dev.yml, .prod.yml)
  - Service-specific prefixes (kong-config.yml)
```

#### **PORT & SERVICE NAMING**
```yaml
Port Allocation:
  - 10000-10099: Core infrastructure
  - 10100-10199: Vector & AI services  
  - 10200-10299: Monitoring stack
  - 11000-11999: Agent services

Container Naming:
  - sutazai-[service-name] (sutazai-postgres)
  - sutazai-[agent-name] (sutazai-hardware-optimizer)
  - No random suffixes or prefixes

Environment Variables:
  - UPPERCASE_SNAKE_CASE (POSTGRES_PASSWORD)
  - Service prefix for clarity (KONG_ADMIN_LISTEN)
  - Consistent naming across services
```

---

## 4. QUALITY ASSURANCE FRAMEWORK

### 4.1 Review Process Standards

#### **DOCUMENTATION REVIEW LEVELS**
```yaml
Level 1 - Automated Validation:
  Frequency: Every commit
  Checks: Link validation, format compliance, spelling
  Tools: markdownlint, link-checker, spell-checker
  Threshold: 100% pass rate required

Level 2 - Technical Review:
  Frequency: Weekly
  Checks: Technical accuracy, completeness, currency
  Reviewers: Domain specialists
  Threshold: Technical reviewer approval required

Level 3 - Architecture Review:
  Frequency: Monthly
  Checks: Architectural alignment, integration consistency
  Reviewers: System architects
  Threshold: Architecture team approval required

Level 4 - Stakeholder Review:
  Frequency: Quarterly
  Checks: Business alignment, user experience, strategic fit
  Reviewers: Product stakeholders
  Threshold: Stakeholder sign-off required
```

#### **REVIEW CHECKLIST TEMPLATES**
```yaml
Technical Accuracy Review:
  - [ ] All service ports verified against running containers
  - [ ] All API endpoints tested with actual responses
  - [ ] All agent statuses confirmed (operational/implemented/defined)
  - [ ] All configuration paths verified to exist
  - [ ] All commands tested in clean environment
  - [ ] All dependencies confirmed and documented
  - [ ] All version numbers current and accurate
  - [ ] All health checks functional

Content Quality Review:
  - [ ] Information organized logically
  - [ ] Examples clear and working
  - [ ] Troubleshooting covers common issues
  - [ ] Cross-references accurate and helpful
  - [ ] Terminology consistent throughout
  - [ ] Target audience appropriate
  - [ ] Learning objectives met
  - [ ] Action items clear and specific

Architecture Consistency Review:
  - [ ] Service relationships accurately depicted
  - [ ] Agent orchestration patterns documented
  - [ ] Integration points clearly defined
  - [ ] Dependencies properly mapped
  - [ ] Communication protocols specified
  - [ ] Security considerations addressed
  - [ ] Performance characteristics documented
  - [ ] Scalability factors addressed
```

### 4.2 Automated Quality Controls

#### **VALIDATION AUTOMATION**
```python
#!/usr/bin/env python3
"""
Documentation Quality Validator
Location: /scripts/validation/doc_quality_validator.py
"""

import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class DocumentationValidator:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []
        
    def validate_service_endpoints(self) -> List[str]:
        """Validate all documented service endpoints are accessible"""
        errors = []
        
        # Parse CLAUDE.md for service definitions
        claude_md = self.base_path / "CLAUDE.md"
        if claude_md.exists():
            with open(claude_md, 'r') as f:
                content = f.read()
                
            # Extract port definitions
            import re
            port_pattern = r'Port\s+(\d+)'
            ports = re.findall(port_pattern, content)
            
            for port in ports:
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if response.status_code != 200:
                        errors.append(f"Service on port {port} health check failed")
                except requests.RequestException:
                    errors.append(f"Service on port {port} not accessible")
                    
        return errors
    
    def validate_agent_registry(self) -> List[str]:
        """Validate agent registry consistency"""
        errors = []
        
        registry_path = self.base_path / "agents" / "agent_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                
            # Check each agent has corresponding documentation
            for agent_id, agent_data in registry.get('agents', {}).items():
                agent_file = self.base_path / ".claude" / "agents" / f"{agent_id}.md"
                if not agent_file.exists():
                    errors.append(f"Agent {agent_id} missing documentation file")
                    
        return errors
    
    def validate_cross_references(self) -> List[str]:
        """Validate all internal links work"""
        errors = []
        
        # Implementation for link validation
        # Check all markdown links point to existing files/sections
        
        return errors
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        return {
            'timestamp': '2025-08-16T06:30:00Z',
            'service_endpoint_errors': self.validate_service_endpoints(),
            'agent_registry_errors': self.validate_agent_registry(),
            'cross_reference_errors': self.validate_cross_references(),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings)
        }

if __name__ == "__main__":
    validator = DocumentationValidator()
    report = validator.generate_report()
    print(json.dumps(report, indent=2))
```

---

## 5. CONTENT MAINTENANCE PROCEDURES

### 5.1 Currency Management

#### **UPDATE FREQUENCY STANDARDS**
```yaml
Critical Information (Daily Updates):
  - Service health status
  - Agent operational status  
  - API endpoint availability
  - Security alerts and patches

Important Information (Weekly Updates):
  - Configuration changes
  - New agent deployments
  - Feature releases
  - Performance metrics

Standard Information (Monthly Updates):
  - Architecture diagrams
  - Integration guides
  - Best practices
  - Troubleshooting procedures

Reference Information (Quarterly Updates):
  - Historical data
  - Deprecated features
  - Archive documentation
  - Strategic roadmaps
```

#### **AUTOMATED FRESHNESS MONITORING**
```yaml
Freshness Indicators:
  - Last verified timestamp on each section
  - Automated staleness warnings (>7 days)
  - Missing verification alerts
  - Drift detection between docs and reality

Update Triggers:
  - Container status changes
  - New service deployments
  - Agent additions/removals
  - Configuration modifications
  - API endpoint changes
  - Security updates
```

### 5.2 Version Control Standards

#### **DOCUMENTATION VERSIONING**
```yaml
Version Numbering:
  - Major.Minor.Patch (e.g., 1.2.3)
  - Major: Architectural changes
  - Minor: New services/agents
  - Patch: Corrections/clarifications

Branching Strategy:
  - main: Production documentation
  - develop: Staging updates
  - feature/[description]: Specific updates
  - hotfix/[issue]: Critical corrections

Commit Message Format:
  docs(scope): brief description
  
  Longer description if needed
  
  - Bullet points for changes
  - Reference issue numbers
  - Include verification status
```

#### **CHANGE TRACKING REQUIREMENTS**
```yaml
CHANGELOG.md Updates:
  - Every documentation change must have CHANGELOG entry
  - Include timestamp, author, change type, impact
  - Reference specific files and sections modified
  - Note verification status and testing performed

Audit Trail:
  - Who made changes (author/reviewer)
  - When changes were made (timestamp)
  - Why changes were made (rationale)
  - What was changed (specific details)
  - How changes were verified (validation method)

Review History:
  - Track all review comments and resolutions
  - Maintain reviewer approvals
  - Document validation results
  - Archive superseded versions
```

---

## 6. USER EXPERIENCE STANDARDS

### 6.1 Audience-Specific Guidelines

#### **DEVELOPER DOCUMENTATION**
```yaml
Requirements:
  - Code examples that actually work
  - Complete API references with real responses
  - Clear setup and configuration instructions
  - Troubleshooting for common issues
  - Performance considerations
  - Security best practices

Format Preferences:
  - Step-by-step procedures
  - Copy-paste commands
  - Expected outputs shown
  - Error handling examples
  - Integration patterns
  - Testing approaches
```

#### **ADMINISTRATOR DOCUMENTATION**
```yaml
Requirements:
  - Deployment procedures
  - Monitoring and alerting setup
  - Backup and recovery procedures
  - Security configuration
  - Performance tuning
  - Troubleshooting runbooks

Format Preferences:
  - Checklists and workflows
  - Decision trees for troubleshooting
  - Resource requirement specifications
  - Capacity planning guidelines
  - Incident response procedures
  - Maintenance schedules
```

#### **ARCHITECT DOCUMENTATION**
```yaml
Requirements:
  - System architecture diagrams
  - Component interaction patterns
  - Scalability considerations
  - Integration capabilities
  - Technical constraints
  - Design rationale

Format Preferences:
  - High-level overviews
  - Detailed technical specifications
  - Trade-off analysis
  - Future roadmap alignment
  - Risk assessments
  - Decision frameworks
```

### 6.2 Accessibility Standards

#### **CONTENT ACCESSIBILITY**
```yaml
Structure Requirements:
  - Logical heading hierarchy (H1 > H2 > H3)
  - Descriptive section titles
  - Clear table headers
  - Alt text for diagrams
  - Color-blind friendly formatting

Navigation Requirements:
  - Table of contents for long documents
  - Breadcrumb navigation
  - Internal linking for related concepts
  - Search-friendly formatting
  - Mobile-responsive layout

Language Requirements:
  - Plain language principles
  - Technical jargon explained
  - Consistent terminology
  - Active voice preferred
  - International English standards
```

---

## 7. TOOLS & AUTOMATION

### 7.1 Documentation Toolchain

#### **REQUIRED TOOLS**
```yaml
Validation Tools:
  - markdownlint: Formatting consistency
  - link-checker: Internal/external link validation
  - spell-checker: Spelling and grammar
  - vale: Style guide enforcement
  - docker-compose: Service validation

Generation Tools:
  - mermaid: Diagram generation
  - swagger-codegen: API documentation
  - jsonschema: Configuration validation
  - git-changelog: Automated changelog generation

Quality Tools:
  - lighthouse: Documentation performance
  - axe: Accessibility checking
  - broken-link-checker: Link health monitoring
  - doc-analytics: Usage tracking
```

#### **AUTOMATION WORKFLOWS**
```yaml
Pre-commit Hooks:
  - Markdown linting
  - Spell checking
  - Link validation
  - Format consistency
  - Git commit message validation

CI/CD Integration:
  - Automated testing of documented procedures
  - Service endpoint validation
  - Configuration file verification
  - Cross-reference checking
  - Documentation deployment

Scheduled Checks:
  - Daily: Service health validation
  - Weekly: Link health monitoring
  - Monthly: Content freshness audit
  - Quarterly: Comprehensive review
```

---

## 8. COMPLIANCE & GOVERNANCE

### 8.1 Documentation Governance Framework

#### **OWNERSHIP & RESPONSIBILITY**
```yaml
System Knowledge Curator:
  - Overall documentation strategy
  - Quality standards enforcement
  - Cross-component consistency
  - Architecture documentation accuracy

Domain Specialists:
  - Technical accuracy within domain
  - Domain-specific best practices
  - Integration documentation
  - Troubleshooting procedures

Service Owners:
  - Service-specific documentation
  - Configuration accuracy
  - Operational procedures
  - Performance characteristics

Agent Specialists:
  - Agent capability documentation
  - Implementation status accuracy
  - Usage patterns and examples
  - Integration requirements
```

#### **QUALITY GATES**
```yaml
Documentation Release Gates:
  1. Automated validation passes (100%)
  2. Technical review approved
  3. Architecture review approved (if applicable)
  4. Stakeholder review approved (if applicable)
  5. Testing verification completed
  6. CHANGELOG updated
  7. Cross-references validated

Documentation Standards Compliance:
  - Terminology consistency checked
  - Format standards verified
  - Accuracy verification completed
  - Accessibility requirements met
  - User experience validated
  - Maintenance procedures documented
```

---

## 9. SUCCESS METRICS

### 9.1 Quality Metrics

#### **ACCURACY METRICS**
```yaml
Technical Accuracy:
  - Service endpoint success rate: >95%
  - Agent status accuracy: >98%
  - Configuration path validation: 100%
  - API response matching: >95%

Content Quality:
  - Broken link rate: <1%
  - Spelling error rate: <0.1%
  - Format compliance: 100%
  - Style guide adherence: >95%

Freshness Metrics:
  - Documentation staleness: <7 days average
  - Verification frequency: Daily for critical info
  - Update response time: <24 hours for critical changes
  - Review completion rate: >90% within SLA
```

### 9.2 User Experience Metrics

#### **USABILITY METRICS**
```yaml
Task Completion:
  - Time to find information: <2 minutes average
  - Task completion success rate: >90%
  - User satisfaction score: >4.0/5.0
  - Documentation usage frequency: Tracked weekly

Accessibility Metrics:
  - Accessibility compliance score: >95%
  - Mobile usability score: >90%
  - Cross-browser compatibility: 100%
  - Search effectiveness: >85% success rate
```

---

## 10. IMPLEMENTATION TIMELINE

### 10.1 Phase 1: Foundation (Week 1)
- [ ] Implement automated validation tools
- [ ] Establish quality review processes
- [ ] Create documentation templates
- [ ] Set up monitoring and alerting

### 10.2 Phase 2: Content Updates (Week 2-3)
- [ ] Update CLAUDE.md with standards compliance
- [ ] Update AGENTS.md with accuracy standards
- [ ] Implement cross-reference validation
- [ ] Deploy automated freshness monitoring

### 10.3 Phase 3: Process Integration (Week 4)
- [ ] Integrate with CI/CD pipelines
- [ ] Train team on new standards
- [ ] Implement feedback collection
- [ ] Launch user experience monitoring

### 10.4 Phase 4: Optimization (Month 2)
- [ ] Analyze usage patterns
- [ ] Optimize based on feedback
- [ ] Enhance automation
- [ ] Scale quality processes

---

**Document Authority**: System Knowledge Curator  
**Compliance**: 20 Codebase Rules + Enforcement Rules  
**Review Frequency**: Monthly  
**Next Review**: 2025-09-16 06:30:00 UTC