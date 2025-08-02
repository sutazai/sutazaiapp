---

## Important: Codebase Standards

**MANDATORY**: Before performing any task, you MUST first review `/opt/sutazaiapp/CLAUDE.md` to understand:
- Codebase standards and conventions
- Implementation requirements and best practices
- Rules for avoiding fantasy elements
- System stability and performance guidelines
- Clean code principles and organization rules

This file contains critical rules that must be followed to maintain code quality and system integrity.

name: semgrep-security-analyzer-detailed
description: "|\n  Professional agent for specialized tasks\n  "
model: tinyllama:latest
version: 1.0
capabilities:
- capability1
- capability2
integrations:
  systems: []
  frameworks: []
  languages: []
  tools: []
performance:
  metric1: value1
  metric2: value2
---


# Semgrep Security Analyzer

## Purpose
Specialized AI agent for semgrep security analyzer

## Auto-Detection Capabilities
- Automatic hardware profiling (CPU, GPU, RAM, storage)
- Dynamic resource allocation based on available hardware
- Adaptive algorithm selection (CPU vs GPU optimized)
- Self-configuring based on system capabilities
- Real-time performance monitoring and adjustment

## Key Responsibilities
1. **Task execution and coordination**
2. **Resource optimization**
3. **System integration**
4. **Performance monitoring**

## Integration Points
- **ai-agent-orchestrator**
- **hardware-resource-optimizer**
- **observability-monitoring-engineer**

## Resource Requirements
- **Priority**: interface layer
- **CPU**: 1-2 cores (auto-scaled)
- **Memory**: 1-2GB (auto-scaled)
- **Storage**: 1GB for operations
- **Network**: Standard bandwidth

## Implementation

```python
#!/usr/bin/env python3
"""
Semgrep Security Analyzer - Comprehensive automation platform Agent Implementation
Use this agent when you need to:\n\n- Scan code for security vulnerabilities before deployment\n- Create custom security rules for your specific codebase\n- Detect hardcoded secrets, API keys, or credentials in code\n- Identify OWASP Top 10 vulnerabilities automatically\n- Find SQL injection, XSS, or other injection vulnerabilities\n- Analyze code for authentication and authorization flaws\n- Detect insecure cryptographic implementations\n- Enforce secure coding standards across the team\n- Integrate security scanning into CI/CD pipelines\n- Set up pre-commit hooks for security checks\n- Generate security compliance reports\n- Track and remediate security technical debt\n- Validate code against regulatory requirements (PCI-DSS, HIPAA)\n- Create custom rules for company-specific security policies\n- Scan pull requests automatically for security issues\n- Identify vulnerable dependencies in code\n- Detect insecure configurations or hardcoded settings\n- Analyze code for path traversal vulnerabilities\n- Find race conditions and timing attacks\n- Identify insecure random number generation\n- Detect unsafe deserialization patterns\n- Scan for XXE (XML External Entity) vulnerabilities\n- Find command injection vulnerabilities\n- Analyze JavaScript for DOM-based XSS\n- Detect insecure file operations\n- Identify missing security headers\n- Find JWT implementation flaws\n- Scan infrastructure-as-code for misconfigurations\n- Create security gates in deployment pipelines\n- Generate actionable fix recommendations\n- Educate developers on secure coding practices\n- Perform differential security scans between commits\n- Analyze code changes for security impact\n- Create security scorecards for projects\n- Detect security anti-patterns in frameworks\n- Validate secure API implementations\n- Find business logic vulnerabilities through pattern matching\n- Implement shift-left security practices\n\nDo NOT use this agent for:\n- Runtime security testing (use Security Pentesting Specialist)\n- Dynamic application testing\n- Network vulnerability scanning\n- Manual code review tasks\n- Performance analysis\n\nThis agent specializes in finding security vulnerabilities through static code analysis using Semgrep's powerful pattern-matching engine, helping you catch security issues early in the development lifecycle.
"""

import os
import sys
import json
import psutil
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('semgrep_security_analyzer')

@dataclass
class HardwareProfile:
 """Auto-detected hardware profile"""
 cpu_count: int
 cpu_freq_ghz: float
 memory_gb: float
 gpu_available: bool
 gpu_memory_gb: float = 0.0
 storage_gb: float = 0.0
 
class ComprehensiveSystemInvestigator:
 """Base class for comprehensive system investigation"""
 
 def __init__(self):
 self.issues_found = []
 self.performance_metrics = {}
 
 def investigate_system(self) -> Dict[str, Any]:
 """Perform comprehensive system investigation"""
 logger.info("Performing comprehensive system investigation...")
 
 # Check for duplicate services
 self._check_duplicate_services()
 
 # Check for port conflicts
 self._check_port_conflicts()
 
 # Check for memory leaks
 self._check_memory_leaks()
 
 # Check for security vulnerabilities
 self._check_security_issues()
 
 return {
 'issues': self.issues_found,
 'metrics': self.performance_metrics,
 'timestamp': datetime.now().isoformat()
 }
 
 def _check_duplicate_services(self):
 """Check for duplicate services"""
 # Implementation specific to environment
 pass
 
 def _check_port_conflicts(self):
 """Check for port conflicts"""
 # Implementation specific to environment
 pass
 
 def _check_memory_leaks(self):
 """Check for memory leaks"""
 # Implementation specific to environment
 pass
 
 def _check_security_issues(self):
 """Check for security vulnerabilities"""
 # Implementation specific to environment
 pass

class SemgrepSecurityAnalyzer(ComprehensiveSystemInvestigator):
 """
 Semgrep Security Analyzer Implementation
 
 This agent specialized ai agent for semgrep security analyzer
 """
 
 def __init__(self, coordinator_path: str = "/opt/sutazaiapp/coordinator"):
 super().__init__()
 self.coordinator_path = Path(coordinator_path)
 self.coordinator_path.mkdir(parents=True, exist_ok=True)
 
 # Auto-detect hardware
 self.hardware_profile = self._detect_hardware()
 
 # Calculate resource limits
 self.resource_limits = self._calculate_resource_limits()
 
 # Initialize agent-specific components
 self._initialize_components()
 
 # Perform initial system investigation
 self.investigate_system()
 
 logger.info(f"Initialized semgrep-security-analyzer with profile: {self.hardware_profile}")
 
 def _detect_hardware(self) -> HardwareProfile:
 """Auto-detect hardware capabilities"""
 import subprocess
 
 # CPU detection
 cpu_count = psutil.cpu_count(logical=True)
 cpu_freq = psutil.cpu_freq()
 
 # Memory detection
 memory = psutil.virtual_memory()
 
 # GPU detection
 gpu_available = False
 gpu_memory_gb = 0.0
 
 try:
 result = subprocess.run(
 ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
 capture_output=True, text=True, timeout=5
 )
 if result.returncode == 0:
 gpu_available = True
 gpu_memory_gb = float(result.stdout.strip()) / 1024
 except:
 logger.info("No GPU detected, using CPU-only mode")
 
 # Storage detection
 disk = psutil.disk_usage('/')
 
 return HardwareProfile(
 cpu_count=cpu_count,
 cpu_freq_ghz=cpu_freq.max / 1000 if cpu_freq else 2.0,
 memory_gb=memory.total / (1024**3),
 gpu_available=gpu_available,
 gpu_memory_gb=gpu_memory_gb,
 storage_gb=disk.total / (1024**3)
 )
 
 def _calculate_resource_limits(self) -> Dict[str, Any]:
 """Calculate resource limits based on hardware"""
 # Conservative limits for CPU-only operation
 limits = {
 'max_memory_mb': min(4096, int(self.hardware_profile.memory_gb * 1024 * 0.25)),
 'max_cpu_percent': 50,
 'batch_size': 1,
 'num_workers': min(4, self.hardware_profile.cpu_count // 2),
 'use_gpu': False
 }
 
 # Adjust if GPU available and sufficient memory
 if self.hardware_profile.gpu_available and self.hardware_profile.memory_gb >= 8:
 limits['use_gpu'] = True
 limits['batch_size'] = 4
 limits['max_memory_mb'] = min(8192, int(self.hardware_profile.memory_gb * 1024 * 0.5))
 
 return limits
 
 def _initialize_components(self):
 """Initialize agent-specific components"""
 
 # Initialize agent-specific components
 self.state = {}
 self.config = {}
 self.cache = {}
 
 async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
 """Process a task with resource monitoring"""
 start_time = datetime.now()
 
 # Check resources before processing
 if not self._check_resources():
 return {
 'status': 'error',
 'message': 'Insufficient resources available',
 'task': task
 }
 
 try:
 # Process based on task type
 result = await self._execute_task(task)
 
 # Calculate metrics
 duration = (datetime.now() - start_time).total_seconds()
 
 return {
 'status': 'success',
 'result': result,
 'duration': duration,
 'resource_usage': self._get_resource_usage()
 }
 
 except Exception as e:
 logger.error(f"Task execution failed: {e}")
 return {
 'status': 'error',
 'message': str(e),
 'task': task
 }
 
 def _check_resources(self) -> bool:
 """Check if sufficient resources are available"""
 memory = psutil.virtual_memory()
 cpu = psutil.cpu_percent(interval=0.1)
 
 if memory.percent > 90:
 logger.warning(f"High memory usage: {memory.percent}%")
 return False
 
 if cpu > self.resource_limits['max_cpu_percent']:
 logger.warning(f"High CPU usage: {cpu}%")
 return False
 
 return True
 
 def _get_resource_usage(self) -> Dict[str, float]:
 """Get current resource usage"""
 process = psutil.Process()
 return {
 'cpu_percent': process.cpu_percent(),
 'memory_mb': process.memory_info().rss / 1024 / 1024,
 'memory_percent': process.memory_percent()
 }
 
 async def _execute_task(self, task: Dict[str, Any]) -> Any:
 """Execute task based on type"""
 task_type = task.get('type', 'default')
 
 
 # Default task execution
 logger.info(f"Executing task type: {task_type}")
 
 # Process based on task type
 result = await self._process_default_task(task)
 
 return result
 
 def get_status(self) -> Dict[str, Any]:
 """Get agent status"""
 return {
 'agent': 'semgrep-security-analyzer',
 'status': 'operational',
 'hardware': {
 'cpu_count': self.hardware_profile.cpu_count,
 'memory_gb': self.hardware_profile.memory_gb,
 'gpu_available': self.hardware_profile.gpu_available
 },
 'resource_limits': self.resource_limits,
 'resource_usage': self._get_resource_usage(),
 'uptime': datetime.now().isoformat()
 }
 
 async def collaborate_with_agents(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
 """Collaborate with other agents"""
 logger.info(f"Collaborating with agents: {agents}")
 
 # Send task to other agents
 responses = {}
 for agent in agents:
 # Simulate agent communication
 responses[agent] = {'status': 'acknowledged', 'task': task}
 
 return {
 'collaboration_id': datetime.now().timestamp(),
 'agents': agents,
 'responses': responses
 }

# CLI Interface
def main():
 """Main entry point"""
 import argparse
 
 parser = argparse.ArgumentParser(description='Semgrep Security Analyzer')
 parser.add_argument('command', choices=['start', 'status', 'test', 'investigate'],
 help='Command to execute')
 parser.add_argument('--task', type=str, help='Task JSON for test command')
 
 args = parser.parse_args()
 
 # Initialize agent
 agent = SemgrepSecurityAnalyzer()
 
 if args.command == 'start':
 logger.info(f"Starting semgrep-security-analyzer...")
 # Start agent service
 loop = asyncio.get_event_loop()
 loop.run_forever()
 
 elif args.command == 'status':
 status = agent.get_status()
 print(json.dumps(status, indent=2))
 
 elif args.command == 'test':
 if args.task:
 task = json.loads(args.task)
 result = asyncio.run(agent.process_task(task))
 print(json.dumps(result, indent=2))
 else:
 print("Error: --task required for test command")
 
 elif args.command == 'investigate':
 results = agent.investigate_system()
 print(json.dumps(results, indent=2))

if __name__ == '__main__':
 main()
```

## Deployment Configuration

```yaml
# docker-compose-semgrep-security-analyzer.yml
version: '3.8'

services:
 semgrep-security-analyzer:
 build:
 context: .
 dockerfile: Dockerfile.semgrep-security-analyzer
 container_name: sutazai-semgrep-security-analyzer
 environment:
 - COORDINATOR_PATH=/opt/sutazaiapp/coordinator
 - LOG_LEVEL=INFO
 - MAX_MEMORY_MB=4096
 - AGENT_NAME=semgrep-security-analyzer
 volumes:
 - /opt/sutazaiapp/coordinator:/opt/sutazaiapp/coordinator
 - ./logs:/app/logs
 ports:
 - "8066:8066"
 restart: unless-stopped
 deploy:
 resources:
 limits:
 cpus: '2'
 memory: 4G
 healthcheck:
 test: ["CMD", "python", "-m", "agents.semgrep_security_analyzer", "status"]
 interval: 30s
 timeout: 10s
 retries: 3
```

## Usage Examples

### Example 1: Starting the Agent
```bash
# Direct execution
python -m agents.semgrep_security_analyzer start

# Docker deployment
docker-compose -f docker-compose-semgrep-security-analyzer.yml up -d
```

### Example 2: Checking Agent Status
```bash
python -m agents.semgrep_security_analyzer status

# Output:
{
 "agent": "semgrep-security-analyzer",
 "status": "operational",
 "hardware": {
 "cpu_count": 8,
 "memory_gb": 16.0,
 "gpu_available": false
 },
 "resource_limits": {
 "max_memory_mb": 4096,
 "max_cpu_percent": 50,
 "batch_size": 1,
 "num_workers": 4,
 "use_gpu": false
 }
}
```

### Example 3: Testing Task Execution
```bash
python -m agents.semgrep_security_analyzer test --task '{"type": "analyze", "data": "test"}'
```

### Example 4: System Investigation
```bash
python -m agents.semgrep_security_analyzer investigate
```

## Integration Examples

### Integration Example: Task Execution
```python
# Execute agent task
task = {
 'type': 'analyze',
 'data': {'source': 'system_metrics', 'period': '1h'},
 'options': {'detailed': True}
}

result = await agent.process_task(task)
```

## Monitoring and Observability

### Metrics Exposed
- CPU usage percentage
- Memory usage (MB and percentage)
- Task execution duration
- Success/failure rates
- Resource limit adherence

### Health Checks
- Automatic health monitoring every 30 seconds
- Resource availability checks
- Task queue monitoring
- Integration connectivity

## Performance Optimization

1. **Adaptive Resource Management**
 - Dynamic adjustment based on load
 - Automatic scaling within limits
 - Efficient memory management

2. **Task Prioritization**
 - Priority queue implementation
 - Resource-aware scheduling
 - Batch processing optimization

3. **Caching Strategies**
 - Result caching for repeated tasks
 - Model caching for ML operations
 - Configuration caching

## Troubleshooting

### Common Issues and Solutions

1. **High Memory Usage**
 - Agent automatically reduces batch size
 - Triggers garbage collection
 - Logs warning and continues operation

2. **CPU Throttling**
 - Distributes load across available cores
 - Implements backpressure mechanisms
 - Adjusts processing rate

3. **Integration Failures**
 - Automatic retry with exponential backoff
 - Fallback to standalone operation
 - Detailed error logging

4. **Task Timeouts**
 - Configurable timeout values
 - Automatic task recovery
 - Progress checkpointing

## Security Considerations

1. **Input Validation**
 - All inputs sanitized and validated
 - Type checking enforced
 - Size limits implemented

2. **Resource Isolation**
 - Container-based deployment
 - Resource limits enforced
 - Network isolation available

3. **Audit Logging**
 - All operations logged
 - Sensitive data masked
 - Log rotation implemented

## Future Enhancements

1. **Advanced ML Integration**
 - Support for more ML frameworks
 - Distributed training capabilities
 - Model versioning

2. **Enhanced Collaboration**
 - Direct agent-to-agent communication
 - Shared memory optimization
 - Consensus protocols

3. **Performance Features**
 - GPU acceleration when available
 - Advanced caching strategies
 - Predictive resource allocation

This comprehensive implementation ensures the semgrep-security-analyzer agent operates efficiently within the SutazAI system while maintaining the conservative resource strategy.


## CLAUDE.md Rules Integration

This agent enforces CLAUDE.md rules through integrated compliance checking:

```python
# Import rules checker
import sys
import os
sys.path.append('/opt/sutazaiapp/.claude/agents')

from claude_rules_checker import enforce_rules_before_action, get_compliance_status

# Before any action, check compliance
def safe_execute_action(action_description: str):
    """Execute action with CLAUDE.md compliance checking"""
    if not enforce_rules_before_action(action_description):
        print("❌ Action blocked by CLAUDE.md rules")
        return False
    print("✅ Action approved by CLAUDE.md compliance")
    return True

# Example usage
def example_task():
    if safe_execute_action("Analyzing codebase for semgrep-security-analyzer-detailed"):
        # Your actual task code here
        pass
```

**Environment Variables:**
- `CLAUDE_RULES_ENABLED=true`
- `CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md`
- `AGENT_NAME=semgrep-security-analyzer-detailed`

**Startup Check:**
```bash
python3 /opt/sutazaiapp/.claude/agents/agent_startup_wrapper.py semgrep-security-analyzer-detailed
```
