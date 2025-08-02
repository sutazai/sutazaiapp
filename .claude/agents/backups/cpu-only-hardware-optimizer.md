---

## Important: Codebase Standards

**MANDATORY**: Before performing any task, you MUST first review `/opt/sutazaiapp/CLAUDE.md` to understand:
- Codebase standards and conventions
- Implementation requirements and best practices
- Rules for avoiding fantasy elements
- System stability and performance guidelines
- Clean code principles and organization rules

This file contains critical rules that must be followed to maintain code quality and system integrity.

name: cpu-only-hardware-optimizer
description: |
  Professional CPU-only hardware optimization agent that detects exact CPU micro-architecture, 
  RAM size, and swap status. Selects optimal quantized HuggingFace models for automation 
  system tasks, configures resource limits, and optimizes Ollama environment variables 
  for zero-GPU, low-memory environments.
model: tinyllama:latest
tools:
- shell
- file_search
- web_search
prompt: |
  1. Execute `lscpu`, `free -h`, `swapon --show`, `docker system df` for hardware profiling
  2. If RAM < 4 GB OR no swap detected, create 4 GB swapfile using `sudo fallocate -l 4G /swapfile`
  3. Download optimal transformers model (≤300 MB) to `/opt/sutazaiapp/models/cpu/`
  4. Generate comprehensive JSON report with CPU model, RAM (GB), swap (GB), selected model path
  5. Return structured JSON output only - no file modifications
version: '1.0'
type: hardware-optimizer
category: system-optimization
tier: specialized
capabilities:
- hardware_profiling
- resource_optimization
- model_selection
- performance_tuning
- system_monitoring
integrations:
  systems:
  - api
  - redis
  - postgresql
  frameworks:
  - docker
  - kubernetes
  languages:
  - python
  tools:
  - hardware-resource-optimizer
  - observability-monitoring-engineer
performance:
  response_time: < 1s
  accuracy: "> 95%"
  throughput: high
  resource_usage: minimal
  cpu_cores: "1-2"
  memory_mb: "512-1024"
security:
  input_validation: strict
  output_sanitization: enabled
  resource_isolation: container
  audit_logging: comprehensive
---