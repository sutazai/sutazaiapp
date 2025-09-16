# Rule 14: Specialized AI Agent Usage - Claude Code Sub-Agent Orchestration

## CRITICAL: Claude Code Sub-Agent System
These are specifications for Claude Code's internal agent selection system. Claude Code uses these sub-agents from `/opt/sutazaiapp/.claude/agents/` to apply specialized domain knowledge during task execution.

## AI AGENT SELECTION ALGORITHM (Claude Code Internal):
```python
class ClaudeCodeAgentSelector:
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
```

## COMPLETE TASK MATCHING MATRIX:

### Backend Development
```yaml
backend_development:
  api_design:
    simple: "backend-architect.md"
    complex: ["backend-architect.md", "system-architect.md"]
    graphql: ["graphql-architect.md", "graphql-performance-optimizer.md", "graphql-security-specialist.md"]
    security_critical: ["backend-architect.md", "security-auditor.md", "api-security-audit.md"]
    openapi_spec: "docs-api-openapi.md"
    api_testing: "api-tester.md"
    development: "dev-backend-api.md"
    
  microservices:
    architecture: ["system-architect.md", "arch-system-design.md"]
    implementation: "backend-architect.md"
    python: ["python-pro.md", "python-expert.md"]
    golang: "golang-pro.md"
    rust: "rust-pro.md"
    javascript: "javascript-pro.md"
    typescript: "typescript-pro.md"
    php: "php-pro.md"
    c_sharp: "c-sharp-pro.md"
    c: "c-pro.md"
    cpp: "cpp-pro.md"
    
  database_work:
    schema_design: "database-architect.md"
    admin: "database-admin.md"
    optimization: ["database-optimizer.md", "database-optimization.md"]
    complex_queries: "sql-pro.md"
    nosql: "nosql-specialist.md"
    supabase: ["supabase-schema-architect.md", "supabase-realtime-optimizer.md"]
```

### Frontend Development
```yaml
frontend_development:
  react_projects:
    simple: "frontend-developer.md"
    complex: "frontend-architect.md"
    performance: "react-performance-optimization.md"
    mobile: "spec-mobile-react-native.md"
    
  ui_ux:
    design: ["ui-ux-designer.md", "ui-designer.md"]
    architecture: "frontend-architect.md"
    accessibility: "web-accessibility-checker.md"
    vitals: "web-vitals-optimizer.md"
    cli: "cli-ui-designer.md"
    research: "ux-researcher.md"
    
  mobile:
    cross_platform: "mobile-developer.md"
    ios_specific: "ios-developer.md"
    app_builder: "mobile-app-builder.md"
    app_store: "app-store-optimizer.md"
    
  fullstack:
    development: "fullstack-developer.md"
```

### Research & Analysis
```yaml
research_analysis:
  academic:
    researcher: "academic-researcher.md"
    general: "researcher.md"
    technical: "technical-researcher.md"
    trends: "trend-researcher.md"
    
  coordination:
    orchestrator: "research-orchestrator.md"
    coordinator: "research-coordinator.md"
    synthesizer: "research-synthesizer.md"
    brief_generator: "research-brief-generator.md"
    
  business:
    analyst: "business-analyst.md"
    requirements: "requirements-analyst.md"
    competitive: "competitive-intelligence-analyst.md"
    attribution: "marketing-attribution-analyst.md"
    
  data_analysis:
    general: "data-analyst.md"
    quantitative: "quant-analyst.md"
    risk: "risk-manager.md"
```

### Testing & Quality
```yaml
testing_quality:
  test_strategy:
    lead: "test-engineer.md"
    automation: "test-automator.md"
    quality: "quality-engineer.md"
    writer_fixer: "test-writer-fixer.md"
    general: "tester.md"
    
  specialized_testing:
    performance: ["performance-engineer.md", "performance-analyzer.md"]
    profiling: "performance-profiler.md"
    benchmarking: ["performance-benchmarker.md", "benchmark-suite.md"]
    monitoring: "performance-monitor.md"
    load: "load-testing-specialist.md"
    mcp: "mcp-testing-engineer.md"
    results_analysis: "test-results-analyzer.md"
    
  code_quality:
    review: "code-reviewer.md"
    analysis: ["code-analyzer.md", "analyze-code-quality.md"]
    refactoring: "refactoring-expert.md"
    debugging: "debugger.md"
    error_detection: "error-detective.md"
    root_cause: "root-cause-analyst.md"
    comparison: "text-comparison-validator.md"
    
  validation:
    production: "production-validator.md"
    url_context: "url-context-validator.md"
```

### DevOps & Infrastructure
```yaml
devops_infrastructure:
  deployment:
    automation: ["deployment-engineer.md", "devops-automator.md"]
    devops: ["devops-engineer.md", "devops-architect.md"]
    troubleshooting: "devops-troubleshooter.md"
    cicd: "ops-cicd-github.md"
    
  infrastructure:
    cloud: "cloud-architect.md"
    terraform: "terraform-specialist.md"
    migration: ["cloud-migration-specialist.md", "migration-plan.md"]
    monitoring: "monitoring-specialist.md"
    network: "network-engineer.md"
    incidents: "incident-responder.md"
    
  optimization:
    load_balancing: "load-balancer.md"
    vault: "vault-optimizer.md"
    workflow: "workflow-optimizer.md"
    dx: "dx-optimizer.md"
```

### Security & Compliance
```yaml
security_compliance:
  security:
    audit: ["security-auditor.md", "api-security-audit.md"]
    engineering: "security-engineer.md"
    management: "security-manager.md"
    penetration: "penetration-tester.md"
    mcp_security: "mcp-security-auditor.md"
    
  specialized:
    smart_contracts: ["smart-contract-auditor.md", "smart-contract-specialist.md"]
    compliance: "compliance-specialist.md"
    legal: "legal-advisor.md"
```

### Data Science & ML/AI
```yaml
data_ml:
  data:
    analysis: "data-analyst.md"
    engineering: "data-engineer.md"
    science: "data-scientist.md"
    modeling: "data-ml-model.md"
    
  ml_ai:
    ai_engineering: "ai-engineer.md"
    ml_engineering: "ml-engineer.md"
    mlops: "mlops-engineer.md"
    nlp: "nlp-engineer.md"
    vision: "computer-vision-engineer.md"
    evaluation: "model-evaluator.md"
    ethics: "ai-ethics-advisor.md"
    prompts: "prompt-engineer.md"
    experiment_tracking: "experiment-tracker.md"
```

### Content & Documentation
```yaml
content_documentation:
  documentation:
    technical: "technical-writer.md"
    expert: "documentation-expert.md"
    api: ["api-documenter.md", "docs-api-openapi.md"]
    docusaurus: "docusaurus-expert.md"
    changelog: "changelog-generator.md"
    reports: "report-generator.md"
    
  content_creation:
    creator: "content-creator.md"
    curator: "content-curator.md"
    marketer: "content-marketer.md"
    storyteller: "visual-storyteller.md"
    whimsy: "whimsy-injector.md"
    
  structure:
    document_analyzer: "document-structure-analyzer.md"
    markdown_formatter: "markdown-syntax-formatter.md"
    fact_checker: "fact-checker.md"
```

### Marketing & Growth
```yaml
marketing_growth:
  strategy:
    product: "product-strategist.md"
    growth: "growth-hacker.md"
    brand: "brand-guardian.md"
    seo: "seo-analyzer.md"
    attribution: "marketing-attribution-analyst.md"
    hackathon: "hackathon-ai-strategist.md"
    
  social_media:
    reddit: "reddit-community-builder.md"
    instagram: "instagram-curator.md"
    twitter: "twitter-engager.md"
    tiktok: "tiktok-strategist.md"
    clips: "social-media-clip-creator.md"
    
  customer:
    support: "customer-support.md"
    sales: "sales-automator.md"
```

### MCP Protocol Specialists
```yaml
mcp_protocol:
  expert: "mcp-expert.md"
  protocol: "mcp-protocol-specialist.md"
  integration: "mcp-integration-engineer.md"
  deployment: "mcp-deployment-orchestrator.md"
  registry: "mcp-registry-navigator.md"
  server: "mcp-server-architect.md"
  testing: "mcp-testing-engineer.md"
  security: "mcp-security-auditor.md"
```

### Media Processing
```yaml
media_processing:
  audio:
    mixer: "audio-mixer.md"
    quality: "audio-quality-controller.md"
    
  video:
    editor: "video-editor.md"
    
  podcast:
    transcriber: "podcast-transcriber.md"
    content_analyzer: "podcast-content-analyzer.md"
    metadata: "podcast-metadata-specialist.md"
    
  ocr:
    preprocessing: "ocr-preprocessing-optimizer.md"
    quality: "ocr-quality-assurance.md"
    grammar: "ocr-grammar-fixer.md"
    visual: "visual-analysis-ocr.md"
```

### Web3 & Blockchain
```yaml
web3_blockchain:
  contracts: ["smart-contract-specialist.md", "smart-contract-auditor.md"]
  integration: "web3-integration-specialist.md"
```

### Project Management
```yaml
project_management:
  planning:
    planner: "planner.md"
    sprint: "sprint-prioritizer.md"
    shipper: "project-shipper.md"
    
  coordination:
    task_decomposition: "task-decomposition-expert.md"
    requirements: "requirements-analyst.md"
    feedback: "feedback-synthesizer.md"
```

### Distributed Systems & Coordination
```yaml
distributed_systems:
  consensus:
    byzantine: "byzantine-coordinator.md"
    raft: "raft-manager.md"
    quorum: "quorum-manager.md"
    
  synchronization:
    crdt: "crdt-synchronizer.md"
    gossip: "gossip-coordinator.md"
    
  optimization:
    topology: "topology-optimizer.md"
    resource: "resource-allocator.md"
```

### Specialized Tools & Utilities
```yaml
specialized_tools:
  automation:
    smart: "automation-smart-agent.md"
    rapid_prototype: "rapid-prototyper.md"
    
  system_management:
    dependency: "dependency-manager.md"
    context: "context-manager.md"
    memory: "memory-coordinator.md"
    legacy: "legacy-modernizer.md"
    modernizer: "architecture-modernizer.md"
    
  extraction:
    url_links: "url-link-extractor.md"
    timestamp: "timestamp-precision-specialist.md"
    
  integration:
    payment: "payment-integration.md"
    tool_evaluation: "tool-evaluator.md"
```

### Meta Agents & Orchestration
```yaml
meta_orchestration:
  experts:
    agent_expert: "agent-expert.md"
    agent_overview: "agent-overview.md"
    
  orchestrators:
    task: "orchestrator-task.md"
    swarm_init: "coordinator-swarm-init.md"
    sparc: ["sparc-coordinator.md", "implementer-sparc-coder.md"]
    
  specialized_coordinators:
    memory: "memory-coordinator.md"
    hive_mind: "hive-mind/"
    swarm: "swarm/"
    consensus: "consensus/"
    
  review:
    architect: "architect-review.md"
    reviewer: "reviewer.md"
    review_agent: "review-agent.md"
```

### Specialized Domains
```yaml
specialized_domains:
  search:
    specialist: "search-specialist.md"
    
  command:
    expert: "command-expert.md"
    shell_scripting: "shell-scripting-pro.md"
    
  education:
    mentor: "socratic-mentor.md"
    learning_guide: "learning-guide.md"
    
  studio:
    producer: "studio-producer.md"
    operations: "studio-operations/"
    
  github:
    pr_manager: "github-pr-manager.md"
    
  metadata:
    agent: "metadata-agent.md"
    tag_agent: "tag-agent.md"
    
  misc:
    coder: "coder.md"
    connection: "connection-agent.md"
    moc: "moc-agent.md"
    query_clarifier: "query-clarifier.md"
    llms_maintainer: "llms-maintainer.md"
    tdd_london: "tdd-london-swarm.md"
```

## MULTI-AGENT COORDINATION PATTERNS:

### Sequential Workflow
```yaml
sequential_workflow:
  description: "Claude Code executes agents in sequence"
  example_full_feature:
    - stage: "requirements_analysis"
      agents: ["business-analyst.md", "requirements-analyst.md"]
      output: "requirements_document"
      
    - stage: "research"
      agents: ["technical-researcher.md", "research-coordinator.md"]
      input: "requirements_document"
      output: "research_findings"
      
    - stage: "system_design"
      agents: ["system-architect.md", "arch-system-design.md"]
      input: ["requirements_document", "research_findings"]
      output: "architecture_specification"
      
    - stage: "api_design"
      agents: ["backend-architect.md", "api-documenter.md"]
      input: "architecture_specification"
      output: "api_specification"
      
    - stage: "database_design"
      agents: ["database-architect.md", "database-optimizer.md"]
      input: "api_specification"
      output: "database_schema"
      
    - stage: "implementation"
      agents: ["python-pro.md", "code-analyzer.md"]
      input: ["api_specification", "database_schema"]
      output: "working_code"
      
    - stage: "testing"
      agents: ["test-automator.md", "test-writer-fixer.md"]
      input: "working_code"
      output: "test_suite"
      
    - stage: "performance"
      agents: ["performance-engineer.md", "performance-profiler.md"]
      input: ["working_code", "test_suite"]
      output: "performance_report"
      
    - stage: "security_review"
      agents: ["security-auditor.md", "penetration-tester.md"]
      input: ["working_code", "test_suite"]
      output: "security_report"
      
    - stage: "documentation"
      agents: ["technical-writer.md", "documentation-expert.md"]
      input: ["working_code", "api_specification"]
      output: "documentation"
      
    - stage: "deployment"
      agents: ["deployment-engineer.md", "devops-engineer.md"]
      input: ["working_code", "security_report", "performance_report"]
      output: "deployed_system"
      
    - stage: "monitoring"
      agents: ["monitoring-specialist.md", "incident-responder.md"]
      input: "deployed_system"
      output: "monitoring_dashboard"
```

### Parallel Workflow
```yaml
parallel_workflow:
  description: "Claude Code executes multiple agents simultaneously"
  coordination: "shared_api_contract"
  parallel_tracks:
    backend:
      agents: ["backend-architect.md", "python-pro.md"]
      artifact: "backend_implementation"
      
    frontend:
      agents: ["frontend-developer.md", "react-performance-optimization.md"]
      artifact: "frontend_implementation"
      
    database:
      agents: ["database-admin.md", "database-optimizer.md"]
      artifact: "database_setup"
      
    testing:
      agents: ["test-engineer.md", "test-automator.md"]
      artifact: "test_framework"
      
    documentation:
      agents: ["technical-writer.md", "api-documenter.md"]
      artifact: "documentation"
      
  integration:
    agents: ["fullstack-developer.md", "architect-review.md"]
    responsibility: "integrate_all_tracks"
```

### Expert Consultation
```yaml
expert_consultation:
  description: "Claude Code consults specialists during execution"
  primary: "system-architect.md"
  consultations:
    - trigger: "performance_issue"
      specialists: ["performance-engineer.md", "performance-profiler.md"]
      
    - trigger: "security_concern"
      specialists: ["security-auditor.md", "penetration-tester.md"]
      
    - trigger: "database_complexity"
      specialists: ["database-optimizer.md", "sql-pro.md"]
      
    - trigger: "ui_ux_decision"
      specialists: ["ui-ux-designer.md", "ux-researcher.md"]
      
    - trigger: "distributed_system"
      specialists: ["byzantine-coordinator.md", "raft-manager.md"]
      
    - trigger: "ml_requirement"
      specialists: ["ml-engineer.md", "data-scientist.md"]
      
    - trigger: "code_quality"
      specialists: ["code-reviewer.md", "refactoring-expert.md"]
```

### Swarm Intelligence
```yaml
swarm_intelligence:
  description: "Multiple agents work as a coordinated swarm"
  coordinator: "coordinator-swarm-init.md"
  swarm_types:
    research_swarm:
      agents: ["researcher.md", "technical-researcher.md", "trend-researcher.md"]
      synthesizer: "research-synthesizer.md"
      
    testing_swarm:
      agents: ["test-engineer.md", "test-automator.md", "test-writer-fixer.md"]
      analyzer: "test-results-analyzer.md"
      
    optimization_swarm:
      agents: ["performance-engineer.md", "database-optimizer.md", "workflow-optimizer.md"]
      coordinator: "performance-monitor.md"
```

## PERFORMANCE TRACKING:
```yaml
claude_code_metrics:
  task_completion:
    accuracy: "requirement_satisfaction"
    completeness: "all_requirements_met"
    quality: "code_standards_adherence"
    efficiency: "execution_time"
    
  agent_effectiveness:
    domain_expertise: "specialized_knowledge_applied"
    technical_accuracy: "correct_implementation"
    best_practices: "standards_followed"
    collaboration: "inter_agent_communication"
    
  workflow_success:
    handoff_quality: "clean_artifacts_between_stages"
    integration_success: "components_work_together"
    overall_quality: "final_deliverable_quality"
    time_to_completion: "total_workflow_duration"
```

## ✅ Required Practices:

### For Claude Code Execution:
1. Ensure `/opt/sutazaiapp/.claude/agents/` directory is accessible
2. Check CHANGELOG.md for previous agent usage patterns
3. Document agent selection in execution logs
4. Track performance metrics for optimization
5. Maintain agent files with current specifications
6. Use most specific agent for task domain
7. Design multi-agent workflows for complex tasks
8. Implement validation between agent handoffs
9. Monitor agent effectiveness over time
10. Share successful patterns via CHANGELOG.md

### Agent Selection Priority:
1. Check if task matches previous successful patterns in CHANGELOG.md
2. Select most specialized agent for domain
3. Add complementary agents for multi-faceted tasks
4. Consider swarm approaches for complex problems
5. Document selection rationale
6. Track effectiveness for future reference

## 🚫 Forbidden Practices:
- Using agents not in `/opt/sutazaiapp/.claude/agents/`
- Manually executing agent instructions outside Claude Code
- Modifying agent files without understanding dependencies
- Removing actively used agents
- Using generic agents when specialists exist
- Skipping CHANGELOG.md documentation
- Ignoring performance tracking
- Breaking established successful patterns
- Using outdated agent references
- Mixing Claude Code agents with other AI systems

## Validation Criteria:
- Agent exists in directory
- Selection matches task requirements
- Workflow properly sequenced
- Handoffs clearly defined
- Performance tracked
- Documentation complete
- Patterns reusable
- Knowledge captured

## CHANGELOG.md Entry Format:
```markdown
### [YYYY-MM-DD HH:MM:SS UTC] - Version - Component - Change Type
**Who**: Claude Code [Agent: agent-name.md]
**Why**: Task requirement and agent selection rationale
**What**: Specific work completed using agent expertise
**Impact**: Systems affected and dependencies
**Validation**: Testing performed and results
**Performance**: Execution metrics and effectiveness
```

---

## Critical Compliance Requirement:
**ALL AGENTS MUST FOLLOW ALL 20 PROFESSIONAL CODEBASE STANDARDS RULES**

Every agent in `/opt/sutazaiapp/.claude/agents/` is required to:
- Adhere to all 20 rules defined in the Complete Professional Codebase Standards
- Maintain real implementation standards (Rule 1)
- Never break existing functionality (Rule 2)
- Conduct comprehensive analysis before changes (Rule 3)
- Investigate and consolidate existing files (Rule 4)
- Follow professional project standards (Rule 5)
- Maintain centralized documentation (Rule 6)
- Follow script organization standards (Rules 7-8)
- Maintain single source architecture (Rule 9)
- Never delete blindly (Rule 10)
- Follow Docker excellence standards (Rule 11)
- Support universal deployment (Rule 12)
- Maintain zero tolerance for waste (Rule 13)
- Use this Rule 14 agent orchestration properly
- Maintain perfect documentation (Rules 15-18)
- Follow comprehensive change tracking (Rule 19)
- Protect MCP infrastructure (Rule 20)

**Enforcement**: Claude Code automatically validates agent compliance with all rules before execution.

---

*Last Updated: 2025-08-30 00:00:00 UTC - Based on current agents in `/opt/sutazaiapp/.claude/agents/`*