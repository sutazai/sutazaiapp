# Professional Codebase Standards & Hygiene Guide

Mission: Maintain a clean, consistent, and scalable codebase that enables team velocity, fault tolerance, and engineering excellence.

This document is the authoritative standards guide for this repository. It is derived from the standards provided by the project owner and is considered non‑negotiable for contributions.

## Core Principles

- Stability First: Never break existing functionality
- Quality Through Automation: Automate testing, linting, and deployment
- Security by Design: Zero hardcoded secrets, mandatory scanning
- Documentation as Code: Keep docs current and centralized
- Investigate & Consolidate: Always check existing files before creating new ones
- Performance by Design: Consider performance implications from the start
- Observability Built‑in: Monitoring, logging, and tracing are mandatory
- Fail Fast, Recover Faster: Quick failure detection with automatic recovery
- Test‑Driven Quality: Comprehensive testing is required, not optional
- Reproducible Environments: Development must match production exactly
- Backwards Compatibility: Maintain API contracts and data compatibility
- Resource Efficiency: Optimize for CPU, memory, and network usage
- Maintainability First: Code must be easy to understand and modify
- Zero Trust Architecture: Never trust, always verify and validate
- Data Integrity: Protect and validate data at all application layers

## Hygiene Standards (Highlights)

- Follow existing patterns and centralize logic
- Single source of truth for APIs/components/configs/docs
- Consistent error handling, imports, comments, and naming
- Standardized tests, logging, security practices, and builds
- Unified dependency management with explicit pinning
- Forbidden duplications: APIs/components/scripts/configs/models/tests/etc.
- Project structure discipline with clear directories (src/, services/, utils/, etc.)
- Dead Code Management: investigate, consolidate, then delete ruthlessly

## The 20 Fundamental Rules

Rule 1: Real Implementation Only — No fantasy code. Every line must work today, with existing dependencies, documented endpoints, real configs, and proper error handling. No placeholders in production code paths.

The rules cover imports, APIs, env vars, functions, DB schemas, paths, network calls, configuration keys, realistic test data, dependency pinning, health checks/monitoring, auth, caching, rate limiting/timeouts, SSL, backups, load balancers, images, queues, secrets management, throttling, CDN, and CI/CD pipelines. Violations include placeholder comments, fictional integrations, theoretical abstractions/infrastructure, hardcoded localhost in prod, non‑existent schemas/env vars/endpoints/libraries, mock implementations in prod, magic strings, and ideal‑world assumptions.

See repository docs and pre‑commit hooks for automated enforcement. When in doubt, ask and align before adding new patterns.

## Enforcement

- Pre‑commit: Black, isort, Ruff, Bandit, detect‑secrets, custom hygiene checks
- Tests: Pytest (+coverage), Jest/Playwright where applicable
- CI: Compose/Make targets must pass; coverage thresholds enforced
- Security: Bandit, secret scanning; no plaintext secrets committed

## Contributing

Before opening a PR:
- Read this document and follow the checklist in the PR template
- Reuse existing modules/paths; avoid creating near‑duplicates
- Update docs and changelog entries reflecting your changes
- Ensure `make lint test` (or equivalents) pass locally

