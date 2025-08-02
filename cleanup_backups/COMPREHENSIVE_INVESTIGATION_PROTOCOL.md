# Comprehensive System Investigation Protocol

## MANDATORY FOR ALL AGENTS - Updated for 69+ Agent System

Every AI agent in the SutazAI system MUST follow this comprehensive investigation protocol when analyzing any component, fixing issues, or making changes. This includes all 69+ agents with special emphasis on coordination between senior-* agents and infrastructure-devops-manager.

## Essential Agent Hierarchy

### Tier 1 - Critical Infrastructure (Always Active)
- **senior-ai-engineer** - AI/ML architecture oversight
- **senior-backend-developer** - Backend system integrity
- **senior-frontend-developer** - Frontend consistency
- **infrastructure-devops-manager** - Infrastructure stability
- **autonomous-system-controller** - System orchestration
- **ai-agent-orchestrator** - Multi-agent coordination
- **self-healing-orchestrator** - System resilience

### Tier 2 - Core Specialists
- **agi-system-architect** - AGI/ASI design
- **deployment-automation-master** - Deployment processes
- **security-pentesting-specialist** - Security validation
- **testing-qa-validator** - Quality assurance

### Tier 3 - Domain Experts
- All other specialized agents (60+)

## Core Investigation Framework

```python
class ComprehensiveSystemInvestigator:
    """
    Base class that ALL agents must inherit and use for system analysis
    Updated for 69+ agent system with enhanced coordination
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent_tier = self._determine_agent_tier()
        self.investigation_log = []
        self.findings = {
            'critical_issues': [],
            'performance_bottlenecks': [],
            'security_vulnerabilities': [],
            'code_quality_issues': [],
            'architectural_flaws': [],
            'dependency_conflicts': [],
            'resource_inefficiencies': [],
            'agent_coordination_issues': [],
            'infrastructure_problems': []
        }
        self.essential_agents = {
            'senior-ai-engineer',
            'senior-backend-developer',
            'senior-frontend-developer',
            'infrastructure-devops-manager'
        }
        
    def _determine_agent_tier(self) -> int:
        """Determine agent tier for priority handling"""
        tier1_agents = {
            'senior-ai-engineer', 'senior-backend-developer',
            'senior-frontend-developer', 'infrastructure-devops-manager',
            'autonomous-system-controller', 'ai-agent-orchestrator',
            'self-healing-orchestrator'
        }
        tier2_agents = {
            'agi-system-architect', 'deployment-automation-master',
            'security-pentesting-specialist', 'testing-qa-validator'
        }
        
        if self.agent_name in tier1_agents:
            return 1
        elif self.agent_name in tier2_agents:
            return 2
        else:
            return 3
            
    def coordinate_with_essential_agents(self, action_type: str):
        """Coordinate with essential agents before major actions"""
        if self.agent_tier > 1:  # Non-critical agents must coordinate
            # Check with senior agents based on action type
            if 'infrastructure' in action_type:
                self.request_approval('infrastructure-devops-manager')
            if 'backend' in action_type or 'api' in action_type:
                self.request_approval('senior-backend-developer')
            if 'frontend' in action_type or 'ui' in action_type:
                self.request_approval('senior-frontend-developer')
            if 'model' in action_type or 'ai' in action_type:
                self.request_approval('senior-ai-engineer')
        
    def conduct_comprehensive_investigation(self, target_path: str = "/opt/sutazaiapp"):
        """
        Conduct thorough systematic investigation of the entire application
        Now includes multi-agent coordination and essential agent oversight
        """
        
        # Phase 0: Agent Coordination Check
        self.log_investigation("=== PHASE 0: AGENT COORDINATION CHECK ===")
        self.verify_essential_agents_present()
        self.check_agent_health_status()
        self.establish_communication_channels()
        
        # Phase 1: Complete System Scan
        self.log_investigation("=== PHASE 1: COMPLETE SYSTEM SCAN ===")
        self.scan_entire_codebase(target_path)
        self.analyze_directory_structure(target_path)
        self.identify_all_dependencies()
        self.map_system_architecture()
        self.inventory_all_agents()
        
        # Phase 2: Deep Component Analysis
        self.log_investigation("=== PHASE 2: DEEP COMPONENT ANALYSIS ===")
        self.analyze_all_scripts()
        self.examine_configuration_files()
        self.inspect_docker_configurations()
        self.review_service_definitions()
        
        # Phase 3: Cross-Reference Analysis
        self.log_investigation("=== PHASE 3: CROSS-REFERENCE ANALYSIS ===")
        self.identify_circular_dependencies()
        self.find_duplicate_functionality()
        self.detect_conflicting_services()
        self.analyze_resource_competition()
        
        # Phase 4: Performance & Security Audit
        self.log_investigation("=== PHASE 4: PERFORMANCE & SECURITY AUDIT ===")
        self.profile_performance_bottlenecks()
        self.scan_security_vulnerabilities()
        self.check_resource_utilization()
        self.validate_error_handling()
        
        # Phase 5: Fix Implementation
        self.log_investigation("=== PHASE 5: FIX IMPLEMENTATION ===")
        self.prioritize_issues()
        self.implement_fixes()
        self.validate_fixes()
        self.document_changes()
        
    def scan_entire_codebase(self, path: str):
        """Scan every file in the codebase"""
        
        import os
        import ast
        import json
        import yaml
        
        for root, dirs, files in os.walk(path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    # Analyze based on file type
                    if file.endswith('.py'):
                        self.analyze_python_file(file_path)
                    elif file.endswith('.js') or file.endswith('.ts'):
                        self.analyze_javascript_file(file_path)
                    elif file.endswith('.yml') or file.endswith('.yaml'):
                        self.analyze_yaml_file(file_path)
                    elif file.endswith('.json'):
                        self.analyze_json_file(file_path)
                    elif file.endswith('.sh'):
                        self.analyze_shell_script(file_path)
                    elif file.endswith('Dockerfile'):
                        self.analyze_dockerfile(file_path)
                        
                except Exception as e:
                    self.findings['critical_issues'].append({
                        'type': 'file_analysis_error',
                        'file': file_path,
                        'error': str(e),
                        'severity': 'high'
                    })
                    
    def verify_essential_agents_present(self):
        """Verify all essential agents are present and operational"""
        import os
        
        missing_agents = []
        for agent in self.essential_agents:
            agent_file = f"/opt/sutazaiapp/.claude/agents/{agent}.md"
            if not os.path.exists(agent_file):
                missing_agents.append(agent)
                self.findings['agent_coordination_issues'].append({
                    'type': 'missing_essential_agent',
                    'agent': agent,
                    'severity': 'critical',
                    'impact': 'System stability compromised'
                })
                
        if missing_agents:
            self.log_investigation(f"CRITICAL: Missing essential agents: {missing_agents}")
            
    def check_agent_health_status(self):
        """Check health status of all agents"""
        # Check for agent configuration files
        agent_configs = [
            "/opt/sutazaiapp/config/agent_communication.json",
            "/opt/sutazaiapp/.claude/agents/sutazai_agents.json"
        ]
        
        for config_file in agent_configs:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                    # Validate agent configurations
                    self.validate_agent_configs(data)
                except Exception as e:
                    self.findings['agent_coordination_issues'].append({
                        'type': 'agent_config_error',
                        'file': config_file,
                        'error': str(e),
                        'severity': 'high'
                    })
                    
    def inventory_all_agents(self):
        """Create complete inventory of all 69+ agents"""
        agents_dir = "/opt/sutazaiapp/.claude/agents"
        agent_files = []
        
        for file in os.listdir(agents_dir):
            if file.endswith('.md') and not file.endswith('-detailed.md'):
                if file != 'COMPREHENSIVE_INVESTIGATION_PROTOCOL.md':
                    agent_files.append(file.replace('.md', ''))
                    
        self.log_investigation(f"Total agents found: {len(agent_files)}")
        
        # Verify we have all expected agents
        if len(agent_files) < 69:
            self.findings['agent_coordination_issues'].append({
                'type': 'insufficient_agents',
                'found': len(agent_files),
                'expected': 69,
                'severity': 'high'
            })
    
    def analyze_python_file(self, file_path: str):
        """Deep analysis of Python files"""
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check for syntax errors
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.findings['code_quality_issues'].append({
                'type': 'syntax_error',
                'file': file_path,
                'line': e.lineno,
                'error': str(e),
                'severity': 'critical'
            })
            return
            
        # Analyze imports
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.check_import_conflicts(imports, file_path)
        
        # Check for code smells
        self.detect_code_smells(tree, file_path)
        
        # Security analysis
        self.scan_python_security(tree, file_path)
        
        # Performance analysis
        self.analyze_python_performance(tree, file_path)
        
    def identify_circular_dependencies(self):
        """Detect circular dependencies in the system"""
        
        import networkx as nx
        
        # Build dependency graph
        dep_graph = nx.DiGraph()
        
        # Add nodes and edges based on imports
        for finding in self.findings.get('imports', []):
            dep_graph.add_edge(finding['from'], finding['to'])
            
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(dep_graph))
            for cycle in cycles:
                self.findings['dependency_conflicts'].append({
                    'type': 'circular_dependency',
                    'cycle': cycle,
                    'severity': 'high'
                })
        except:
            pass
            
    def detect_conflicting_services(self):
        """Find services that conflict with each other"""
        
        services = {}
        ports_in_use = {}
        
        # Scan docker-compose files
        compose_files = self.find_files_by_pattern("**/docker-compose*.yml")
        
        for compose_file in compose_files:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
                
            if 'services' in compose_data:
                for service_name, service_config in compose_data['services'].items():
                    # Check for duplicate services
                    if service_name in services:
                        self.findings['dependency_conflicts'].append({
                            'type': 'duplicate_service',
                            'service': service_name,
                            'files': [services[service_name], compose_file],
                            'severity': 'critical'
                        })
                    else:
                        services[service_name] = compose_file
                        
                    # Check for port conflicts
                    if 'ports' in service_config:
                        for port_mapping in service_config['ports']:
                            host_port = str(port_mapping).split(':')[0]
                            if host_port in ports_in_use:
                                self.findings['dependency_conflicts'].append({
                                    'type': 'port_conflict',
                                    'port': host_port,
                                    'services': [ports_in_use[host_port], service_name],
                                    'severity': 'critical'
                                })
                            else:
                                ports_in_use[host_port] = service_name
                                
    def implement_fixes(self):
        """Implement fixes for all discovered issues"""
        
        # Sort issues by severity
        critical_issues = [f for f in self.findings['critical_issues'] if f['severity'] == 'critical']
        high_issues = [f for f in self.findings['critical_issues'] if f['severity'] == 'high']
        
        # Fix critical issues first
        for issue in critical_issues:
            self.fix_issue(issue)
            
        # Then high priority
        for issue in high_issues:
            self.fix_issue(issue)
            
        # Fix performance issues
        for bottleneck in self.findings['performance_bottlenecks']:
            self.optimize_performance(bottleneck)
            
        # Fix security vulnerabilities
        for vulnerability in self.findings['security_vulnerabilities']:
            self.patch_security_issue(vulnerability)
            
    def fix_issue(self, issue: dict):
        """Fix a specific issue"""
        
        if issue['type'] == 'syntax_error':
            self.fix_syntax_error(issue)
        elif issue['type'] == 'duplicate_service':
            self.resolve_duplicate_service(issue)
        elif issue['type'] == 'port_conflict':
            self.resolve_port_conflict(issue)
        elif issue['type'] == 'circular_dependency':
            self.break_circular_dependency(issue)
        elif issue['type'] == 'memory_leak':
            self.fix_memory_leak(issue)
        elif issue['type'] == 'security_vulnerability':
            self.patch_security_vulnerability(issue)
            
    def document_changes(self):
        """Document all changes made during investigation"""
        
        documentation = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'investigation_summary': {
                'total_files_analyzed': len(self.investigation_log),
                'critical_issues_found': len(self.findings['critical_issues']),
                'issues_fixed': len([i for i in self.findings['critical_issues'] if i.get('fixed')]),
                'performance_improvements': len(self.findings['performance_bottlenecks']),
                'security_patches': len(self.findings['security_vulnerabilities'])
            },
            'detailed_findings': self.findings,
            'changes_made': self.get_all_changes(),
            'recommendations': self.generate_recommendations()
        }
        
        # Save documentation
        doc_path = f"/opt/sutazaiapp/investigations/{self.agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        
        with open(doc_path, 'w') as f:
            json.dump(documentation, f, indent=2)
            
        return doc_path
```

## Mandatory Investigation Checklist

Every agent MUST complete this checklist:

### 1. Pre-Investigation
- [ ] Verify all 69+ agents are present
- [ ] Check essential agents (senior-*, infrastructure-devops-manager) are operational
- [ ] Establish communication with ai-agent-orchestrator
- [ ] Check system resources (CPU, Memory, Disk)
- [ ] Identify running processes and services
- [ ] Map the complete directory structure
- [ ] List all dependencies and versions
- [ ] Verify agent tier and permissions

### 2. Agent Coordination
- [ ] Request approval from relevant senior agents for major changes
- [ ] Coordinate with infrastructure-devops-manager for deployment changes
- [ ] Sync with autonomous-system-controller for system-wide impacts
- [ ] Check for ongoing operations by other agents
- [ ] Register investigation intent with ai-agent-orchestrator

### 3. Code Analysis
- [ ] Scan ALL Python files for errors
- [ ] Analyze ALL configuration files
- [ ] Check ALL Docker configurations
- [ ] Review ALL shell scripts
- [ ] Examine ALL service definitions

### 3. Dependency Analysis
- [ ] Map all import relationships
- [ ] Identify circular dependencies
- [ ] Find version conflicts
- [ ] Check for missing dependencies
- [ ] Verify compatibility matrix

### 4. Performance Analysis
- [ ] Profile CPU usage patterns
- [ ] Analyze memory consumption
- [ ] Check disk I/O bottlenecks
- [ ] Identify slow queries/operations
- [ ] Find resource-intensive loops

### 5. Security Audit
- [ ] Scan for hardcoded credentials
- [ ] Check for SQL injection risks
- [ ] Identify XSS vulnerabilities
- [ ] Review authentication mechanisms
- [ ] Validate input sanitization

### 6. Service Analysis
- [ ] Map all running services
- [ ] Check for port conflicts
- [ ] Identify duplicate services
- [ ] Verify service dependencies
- [ ] Check health endpoints

### 7. Fix Implementation
- [ ] Prioritize by severity
- [ ] Create backup before changes
- [ ] Implement fixes systematically
- [ ] Test each fix thoroughly
- [ ] Document all changes

### 8. Validation
- [ ] Run comprehensive tests
- [ ] Verify system stability
- [ ] Check performance metrics
- [ ] Validate security patches
- [ ] Ensure no regressions

### 9. Documentation
- [ ] Document all findings
- [ ] Record all changes made
- [ ] Create fix recommendations
- [ ] Update system documentation
- [ ] Generate investigation report

## Integration Requirements

ALL agents must:

1. **Inherit from ComprehensiveSystemInvestigator**
2. **Call conduct_comprehensive_investigation() before any major changes**
3. **Log all findings to the central investigation system**
4. **Coordinate with other agents to avoid conflicts**
5. **Document every change with full context**
6. **Respect agent tier hierarchy - lower tier agents must defer to higher tier**
7. **Essential agents (senior-*, infrastructure-devops-manager) have veto power**
8. **All infrastructure changes require infrastructure-devops-manager approval**
9. **All AI/ML changes require senior-ai-engineer review**
10. **Multi-agent operations must be orchestrated through ai-agent-orchestrator**

## Performance Standards

- Investigation must complete within 30 minutes
- Memory usage must stay below 4GB during investigation
- All critical issues must be fixed immediately
- System must remain operational during investigation
- No data loss or corruption allowed
- Essential agents (Tier 1) have priority resource allocation

## Resource Allocation (69+ Agent System)

### Conservative Resource Strategy
- **Idle Mode**: 6-8GB RAM, <10% CPU (tiered agent activation)
- **Active Mode**: 12-16GB RAM, 30-40% CPU (standard operations)
- **Burst Mode**: 20GB RAM, 50% CPU, 2GB GPU (intensive tasks)

### Agent Activation Tiers
- **Tier 1 (Always Active)**: senior-*, infrastructure-devops-manager, orchestrators
- **Tier 2 (On-Demand)**: Core specialists, validators, architects
- **Tier 3 (Task-Specific)**: Domain experts, specialized tools

## Code Quality Standards

All code must achieve:
- 10/10 code rating
- 0 critical issues
- <5ms response time for core operations
- 100% error handling coverage
- Complete documentation
- Full coordination with essential agents

## System Integrity Rules

1. **Never remove or disable essential agents** (senior-*, infrastructure-devops-manager)
2. **Always maintain agent count at 69+ for full AGI/ASI capability**
3. **Respect the agent hierarchy and coordination protocols**
4. **Infrastructure changes must go through infrastructure-devops-manager**
5. **Code changes must be reviewed by appropriate senior developer**
6. **AI/ML changes must be approved by senior-ai-engineer**

Remember: EVERY action must be preceded by thorough investigation and proper agent coordination. NO exceptions.