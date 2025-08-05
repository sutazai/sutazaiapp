# System Component Registry

## Overview
This document catalogs the concrete system components and services that make up our infrastructure. Each component has specific responsibilities and integration points.

## Core Services

### Architecture Components
- `system-service` - Core system service coordinator
- `problem-analyzer` - Technical issue analysis and resolution
- `process-monitor` - Process monitoring and management
- `resource-manager` - System resource optimization

### Development Services
- `ml-service` - Machine learning model management
- `api-service` - REST API and backend services
- `ui-service` - Frontend application service
- `code-analyzer` - Code quality analysis

### Process Management
- `task-scheduler` - Task scheduling and execution
- `service-orchestrator` - Service coordination
- `resource-planner` - Resource allocation
- `model-manager` - ML model deployment

### Infrastructure Services
- `infrastructure-service` - Infrastructure management
- `deployment-service` - Automated deployment pipeline
- `resource-monitor` - Hardware resource monitoring
- `model-server` - Model serving infrastructure

### Specialized Services
- `security-scanner` - Security analysis and testing
- `workflow-service` - Process workflow management
- `document-service` - Documentation management

## Supporting Services

### Security Components
- `code-scanner` - Static code analysis
- `penetration-tester` - Security testing
- `test-runner` - Automated testing service

### Automation Services
- `task-executor` - Task execution service
- `workflow-automation` - Process automation
- `pipeline-manager` - Data pipeline management
- `web-automation` - Web service automation
- `script-runner` - Shell script execution

### Coordination Services
- `task-coordinator` - Task scheduling and coordination
- `process-orchestrator` - Process management
- `project-tracker` - Project management
- `task-distributor` - Workload distribution

### Integration Services
- `system-monitor` - System monitoring
- `proxy-service` - API proxy management
- `code-generator` - Code generation service
- `optimization-service` - Performance optimization

### Analysis Services
- `voice-service` - Voice processing
- `finance-analyzer` - Financial data analysis
- `data-analyzer` - Secure data analysis

## Service Integration Examples

### System Management
```python
# Resource monitoring
from system.services import ResourceMonitor
monitor = ResourceMonitor()
metrics = monitor.get_system_metrics()

# Process management
from system.services import ProcessManager
manager = ProcessManager()
manager.start_service('api-service')

# Security scanning
from security.scanner import CodeScanner
scanner = CodeScanner()
results = scanner.scan_directory('./src')
```

### Task Automation
```python
# Task execution
from automation.tasks import TaskExecutor
executor = TaskExecutor()
result = executor.run_task('data-processing')

# Service coordination
from automation.services import ServiceOrchestrator
orchestrator = ServiceOrchestrator()
orchestrator.coordinate_services(['api', 'ui'])
```

## 🎯 Agent Selection Strategy

### Use OPUS agents when you need:
- Strategic planning and architecture
- Complex problem solving
- Creative solutions
- System-wide optimization
- Leadership and high-level design

### Use SONNET agents when you need:
- Specific tool usage
- Routine automation
- Defined workflows
- Integration tasks
- Specialized analysis

## Service Registry

| Service Name | Type | Primary Function |
|--------------|------|-----------------|
| system-service | core | System coordination |
| api-service | core | API management |
| ui-service | core | UI rendering |
| resource-manager | core | Resource allocation |
| ml-service | core | Model management |
| task-scheduler | core | Task scheduling |
| infrastructure-service | core | Infrastructure management |
| deployment-service | core | Deployment pipeline |
| security-scanner | security | Code analysis |
| test-runner | testing | Automated testing |
| task-executor | automation | Task execution |
| workflow-automation | automation | Process automation |
| pipeline-manager | automation | Data pipelines |
| web-automation | automation | Web services |
| script-runner | automation | Script execution |
| task-coordinator | coordination | Task management |
| process-orchestrator | coordination | Process management |
| project-tracker | coordination | Project tracking |
| system-monitor | monitoring | System metrics |
| code-generator | tools | Code generation |
| optimization-service | tools | Performance tuning |
| voice-service | analysis | Voice processing |
| finance-analyzer | analysis | Financial analysis |
| data-analyzer | analysis | Data processing |

## Service Dependencies

Each service is designed to operate independently while supporting integration through standardized APIs. Refer to individual service documentation for specific integration details and requirements.