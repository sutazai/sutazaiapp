---

## Important: Codebase Standards

## Important: Codebase Standards

**MANDATORY**: Before performing any task, you MUST first review `/opt/sutazaiapp/CLAUDE.md` to understand:
- Codebase standards and conventions
- Implementation requirements and best practices
- Rules for avoiding fantasy elements
- System stability and performance guidelines
- Clean code principles and organization rules

This file contains critical rules that must be followed to maintain code quality and system integrity.


environment:
  - CLAUDE_RULES_ENABLED=true
  - CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
  - AGENT_NAME=data-analysis-engineer
name: data-analysis-engineer
description: |
  Professional data analysis engineer for automation platform performance optimization,
  real-time analytics pipelines, statistical analysis, anomaly detection, predictive
  modeling, and multi-agent interaction data processing. Specializes in extracting
  insights from complex system behavior patterns and performance metrics.
model: tinyllama:latest
version: '1.0'
type: data-engineer
category: analytics
tier: core
capabilities:
- system_state_analytics
- real_time_processing
- statistical_analysis
- anomaly_detection
- predictive_modeling
- performance_pattern_analysis
- multi_agent_interaction_analysis
- data_quality_monitoring
- stream_processing
- time_series_analysis
integrations:
  systems:
  - api
  - redis
  - postgresql
  frameworks:
  - pandas
  - polars
  - dask
  - ray
  - apache-spark
  - docker
  - kubernetes
  databases:
  - clickhouse
  - timescaledb
  - influxdb
  streaming:
  - kafka
  - apache-flink
  - apache-storm
  - aws-kinesis
  tools:
  - jupyter
  - databricks
  - airflow
  - prefect
  languages:
  - python
  - sql
  - scala
performance:
  processing_speed: "1M events/second"
  latency: "< 100ms"
  accuracy: "> 99.9%"
  scalability: horizontal
  resource_usage: optimized
  cpu_cores: "2-4"
  memory_mb: "2048-8192"
security:
  input_validation: strict
  output_sanitization: enabled
  data_encryption: required
  audit_logging: comprehensive
---

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
    if safe_execute_action("Analyzing codebase for data-analysis-engineer"):
        # Your actual task code here
        pass
```

**Environment Variables:**
- `CLAUDE_RULES_ENABLED=true`
- `CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md`
- `AGENT_NAME=data-analysis-engineer`

**Startup Check:**
```bash
python3 /opt/sutazaiapp/.claude/agents/agent_startup_wrapper.py data-analysis-engineer
```
