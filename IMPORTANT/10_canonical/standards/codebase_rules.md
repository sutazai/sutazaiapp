# COMPREHENSIVE CODEBASE RULES

Added: December 19, 2024
Purpose: Establish firm engineering standards and discipline for this codebase

These rules are MANDATORY for all contributors. They ensure codebase hygiene, prevent regression, and maintain professional standards.

## Codebase Hygiene
- Enforce consistency; follow existing structure and conventions
- Centralize logic; no duplication across modules or services
- Avoid multiple versions of APIs/components/scripts/requirements/docs
- Respect project structure boundaries; propose RFC before adding new areas
- Remove dead code and temporary test code
- Mandatory tooling: linters, formatters, static analysis, dependency managers, schema enforcement, coverage; enforced in pre-commit and CI
- Commits are contracts: atomic, conventional commits, no skipping reviews/tests

## Execution Mindset
- Think like Architect/Engineer/QA/PM; prioritize clarity; leave code better than found
- Avoid anti-patterns (“for now”, “tiny change”, “clean later”, duplicate utils)

## Rule 1: No Fantasy Elements
- Only real, production-ready implementations; avoid speculative code
- Use real libraries and verifiable APIs; banned magical terms and placeholders
- Pre-commit checks for banned keywords and dependency validation

## Rule 2: Do Not Break Existing Functionality
- No regressions; maintain backward compatibility; test before merge
- Investigate full context before change; preserve advanced functionality
- Testing & safeguards: update tests, feature flags, rollback plan
- Merge criteria: green build, no new lints/types, explicit sign-off for risky areas

## Rule 3: Analyze Everything—Every Time
- Deep review of the entire application before any change; validate assumptions; document findings

## Rule 4: Reuse Before Creating
- Prefer improving existing scripts/code over duplication

## Rule 5: Professional Project, Not a Playground
- No trial-and-error; follow standards and structure

## Rule 6: Clear, Centralized, and Structured Documentation
- Centralize in `/docs/`; update docs with every change; remove outdated copies; assign ownership

## Rule 7: Eliminate Script Chaos
- Centralize under `/scripts/` with clear categories, naming, docs; remove duplicates; single-purpose scripts

## Rule 8: Python Script Sanity
- Organized locations; headers with purpose/usage; argparse; logging; production-ready; delete hacks

## Rule 9: Backend & Frontend Source of Truth
- One source of truth; remove duplicate versions; use branches/flags for experiments

## Rule 10: Functionality-First Cleanup—Never Delete Blindly
- Verify references, purpose, tests before removal; archive if uncertain; do not remove advanced working features without understanding

## Rule 11: Docker Structure Clean & Predictable
- Optimized Dockerfiles; modular compose files; proper .dockerignore; version-pin base images/deps

## Rule 12: One Self-Updating, End-to-End Deployment Script
- Maintain a single `deploy.sh` handling all envs, self-updating, with logging/rollback; documented in `/docs/deployment/`

## Rule 13: No Garbage, No Rot
- Remove abandoned code, old TODOs, commented blocks, unused imports/vars; enforce cleanup sprints

## Rule 14: Engage the Correct AI Agent
- Use specialist agents; document which agent handled which task in commits

## Rule 15: Clean, Clear, Deduplicated Documentation
- One source of truth per topic; concise, up-to-date, structured, actionable

## Rule 16: Use Local LLMs via Ollama (Default TinyLlama)
- All AI/LLM ops use Ollama local models; default `TinyLlama`; document overrides; no external APIs without explicit approval

## Rule 17: Review `/opt/sutazaiapp/IMPORTANT`
- Canonical docs in `IMPORTANT/` are the source of truth; they override conflicts

## Rule 18: Absolute Line-by-Line Deep Review of Core Docs
- Review `/opt/sutazaiapp/CLAUDE.md`, `IMPORTANT/*`, README, and architecture docs before work; document understanding and discrepancies

## Rule 19: Mandatory Change Tracking in `/opt/sutazaiapp/docs/CHANGELOG.md`
- Document every change with time/date/version/component/change type/description/owner/impact
- Undocumented changes will be reverted

All agents must study and review this file first.
