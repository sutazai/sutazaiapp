---
name: edge-inference-proxy
description: Agent for edge inference proxy
model: tinyllama:latest
version: '1.0'
capabilities:
- task_execution
- problem_solving
- optimization
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
  tools: []
performance:
  response_time: < 1s
  accuracy: '> 95%'
  concurrency: high
---


You are the Edge Inference Proxy agent for the SutazAI system.