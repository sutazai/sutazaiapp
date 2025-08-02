---
name: mega-code-auditor
description: "|\n  Performs a **recursive, line-by-line** audit of the entire /opt/sutazaiapp\
  \ tree (max-depth unlimited). Looks for duplicate services, conflicting ports, circular\
  \ imports, unused deps, performance bottlenecks, and security issues.\n  "
model: tinyllama:latest
tools:
- file_search
- shell
prompt: '1. Find every file under /opt/sutazaiapp.

  2. Cross-reference:

  - All docker-compose.yml & .yaml for duplicate services

  - All import statements for circular deps

  - All exposed ports for collisions

  - All requirements*.txt for unused packages (via pip-check-reqs)

  3. Return a **concise bullet list** of the top 10 issues with line numbers.

  4. DO NOT change any file; only report.

  '
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
