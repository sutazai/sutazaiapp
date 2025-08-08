# Repository Guidelines

This guide is authoritative and derived only from IMPORTANT/*. Treat other files as provisional.

## Project Structure & Modules
- `backend/`: API and service layer (FastAPI). Tests belong in `backend/tests`.
- `frontend/`: Streamlit UI. Source under `frontend/src` when present.
- `agents/`: AI agent implementations and orchestrators.
- `config/`, `docker/`, `scripts/`: Configuration, container definitions, and ops scripts.
- `IMPORTANT/`: Single source of truth for standards and architecture.

## Build, Test, and Run
- Containers: Use Docker Compose as the canonical interface. Start the stack with the root compose file; verify via health checks below.
- Health checks (examples):
  - `curl -f http://localhost:10010/health` (Backend)
  - `curl -f http://localhost:10104/` (Ollama tinyllama)
  - `curl -f http://localhost:10006/v1/status/leader` (Consul)
- Python tests (backend): `pytest -v backend/tests --cov=backend --cov-fail-under=80`.
- Lint/format (Python): Black, Flake8, mypy. If a Makefile exists, prefer `make lint`, `make test`.

## Coding Style & Naming
- Indentation: 4 spaces (Python), 2 spaces (JS/TS). Max line length 120.
- Naming: snake_case for Python functions/modules; PascalCase for classes; UPPER_SNAKE_CASE constants.
- Enforce formatting and imports; no commented‑out or placeholder code in main branches.

## Testing Guidelines
- Frameworks: pytest (backend), Playwright/Cypress or Newman for E2E where applicable.
- Coverage: ≥ 80% required for merges.
- Test layout: files `test_*.py` (backend); keep unit/integration markers explicit.

## Commit & PR Guidelines
- Commits: Conventional Commits format, one logical change per commit, reference issue IDs.
- PR requirements: description of change, risk/rollback plan, tests added/updated, CHANGELOG entry. Peer review is mandatory.

## Security & Configuration
- Secrets: never commit. Use environment variables; production secrets must come from the runtime.
- Network/Ports: 10000–10999 core infra; 11000–11999 agents. Verify only Ollama `tinyllama` is loaded unless changed intentionally.
- Minimum controls: input validation, security headers, rate limiting at gateway, and regular dependency scans.

## 📋 COMPREHENSIVE CODEBASE RULES

Added: December 19, 2024
Purpose: Establish firm engineering standards and discipline for this codebase

ultrathink These rules are MANDATORY for all contributors. They ensure codebase hygiene, prevent regression, and maintain professional standards.

### 🔧 Codebase Hygiene
A clean, consistent, and organized codebase is non-negotiable. It reflects engineering discipline and enables scalability, team velocity, and fault tolerance.

Every contributor is accountable for maintaining and improving hygiene—not just avoiding harm.

🧼 Enforce Consistency Relentlessly
✅ Follow the existing structure, naming patterns, and conventions. Never introduce your own style or shortcuts.

✅ Centralize logic — do not duplicate code across files, modules, or services.

🚫 Avoid multiple versions of:

APIs doing the same task (REST + GraphQL duplicating effort, for example)

UI components or CSS/SCSS modules with near-identical logic or styling

Scripts that solve the same problem in slightly different ways

Requirements files scattered across environments with conflicting dependencies

Documentation split across folders with different levels of accuracy

📂 Project Structure Discipline
📌 Never dump files or code in random or top-level folders.

📌 Place everything intentionally, following modular boundaries:

components/ for reusable UI parts

services/ or api/ for network interactions

utils/ for pure logic or helpers

hooks/ for reusable frontend logic

schemas/ or types/ for data validation

If the ideal location doesn't exist, propose a clear structure and open a small RFC (Request for Comments) before proceeding.

🗑️ Dead Code is Debt
🔥 Regularly delete unused code, legacy assets, stale test files, or experimental stubs.

❌ "Just in case" or "might be useful later" is not a valid reason to keep clutter.

🧪 Temporary test code must be removed or clearly gated (e.g. with feature flags or development-only checks).

🧪 Use Tools to Automate Discipline
✅ Mandatory for all contributors:

Linters: ESLint, Flake8, RuboCop

Formatters: Prettier, Black, gofmt

Static analysis: TypeScript, mypy, SonarQube, Bandit

Dependency managers: pip-tools, Poetry, pnpm, npm lockfiles

Schema enforcement: JSON schema, Pydantic, zod

Test coverage tooling: Jest, pytest-cov, Istanbul

🔄 Integrate these tools in pre-commit, pre-push, and CI/CD workflows:

No code gets into production branches without passing hygiene checks.

Every PR should be green and self-explanatory.

✍️ Commits Are Contracts
✅ Write atomic commits—one logical change per commit.

🧾 Follow conventional commit patterns or similar style guides (feat:, fix:, refactor:, etc.).

🧪 No skipping reviews or tests for "quick fixes." These introduce long-term chaos.

🧠 Execution Mindset: Act Like a Top-Level Engineer
🛠️ Think like an Architect, Engineer, QA, and PM—all at once.

🔬 Examine the full context of any change before writing code.

🧭 Prioritize long-term clarity over short-term speed.

🧱 Every change should make the codebase easier to maintain for someone else later.

🚩 Red Flags (Anti-Patterns to Avoid)
🔴 "I'll just put this here for now" — No, there is no "for now."

🔴 "It's just a tiny change" — That's how tech debt begins.

🔴 "We can clean this up later" — "Later" rarely comes.

🔴 Duplicate modules named utils.js, helper.py, or service.ts across packages.

🔴 PRs that include: unrelated changes, commented-out code, unreviewed temporary logs.

🧭 Final Reminder
A healthy codebase is a shared responsibility.
Every line of code you touch should be better than you found it.

🚫 Rules to Follow

-------
📌 Rule 1: No Fantasy Elements
✨ Only real, production-ready implementations are allowed.
Do not write speculative, placeholder, "in-theory," or overly abstract code unless it's been fully validated and grounded in current platform constraints.

✨ Avoid overengineering or unnecessary abstraction.
No fictional components, fake classes, dream APIs, or imaginary infrastructure. All code must reflect actual, working systems.

✨ No 'someday' solutions.
Avoid comments like // TODO: magically scale this later or // imagine this uses a future AI module. If it doesn't exist now, it doesn't go in the codebase.

✨ Be honest with the present limitations.
Code must work today, not in a hypothetical perfect setup. Assume real-world constraints like flaky hardware, latency, cold starts, and limited memory.
All code and documentation must use real, grounded constructs—no metaphors, magic terms, or hypothetical "black-box" AI.

✨ Forbidden:

Terms like wizardService, magicHandler, teleportData(), or comments such as // TODO: add telekinesis here.

Pseudo-functions that don't map to an actual library or API (e.g. superIntuitiveAI.optimize()).

✨ Mandated Practices:

Name things concretely: emailSender, not magicMailer.

Use real libraries: import from nodemailer, not from "the built-in mailer."

Link to docs in comments or README—every external API or framework must be verifiable.

✅ Pre-Commit Checks:

 Search for banned keywords (magic, wizard, black-box, etc.) in your diff.

 Verify every new dependency is in package.json (or requirements.txt) with a valid version.

 Ensure code examples in docs actually compile or run.

----
 Rule 2: Do Not Break Existing Functionality
✨ Every change must respect what already works.
Before modifying any file, component, or flow, verify exactly what it currently does and why. Don't assume anything.

✨ Regression = failure.
If your change breaks or downgrades existing features—even temporarily—it's considered a critical issue. Stability comes first.

✨ Backwards compatibility is a must.
If your refactor or feature update changes existing behavior, either support legacy use cases or migrate them gracefully.

✨ Always test before merging.
Write or update test cases to explicitly cover both new logic and old logic. Nothing ships unless it's verified to not break production behavior.

✨ Communicate impact clearly.
If there's any risk of side effects, escalate and document. Silent changes are forbidden.

🔍 Before modifying any file, investigate the full functionality and behavior of the existing code—understand what it does, how it's used, and whether it's actively supporting a feature or integration.

🧪 If a change is required, test the full end-to-end flow before and after. Confirm the logic is preserved or improved—never regressed.

🔁 Refactor only when necessary and with proper safeguards. If existing advanced functionality is present (e.g., dynamic routing, lazy loading, caching, etc.), it must be preserved or enhanced, not removed.

📊 Maintain proper version control and rollback strategies in case a new change introduces instability or conflict.

💡 Document what was changed, why, and what was verified to ensure that others won't unknowingly override or disrupt a critical flow later.

❗ Breaking changes must never be merged without full validation across all dependent systems and deployment scenarios.
Every change must preserve or improve current behavior—no regressions, ever.

✨ Investigation Steps:

Trace usage:

grep -R "functionName" .

Check import graphs or IDE "Find Usages."

Run baseline tests:

npm test, pytest, or your CI suite.

Manual sanity check of any affected UI or API endpoints.

Review consumers:

Frontend pages that call an endpoint

Cron jobs or scripts that rely on a helper

✨ Testing & Safeguards:

 Write or update tests covering both old and new behavior.

 Gate big changes behind feature flags until fully validated.

 Prepare a rollback plan—document the exact revert commit or steps.

✅ Merge Criteria:

 Green build with 100% test pass rate

 No new lint/type errors

 Explicit sign-off from the original feature owner or lead

❗ Do Not merge breaking changes without:

A clear "Breaking Change" section in the PR description

A migration or upgrade guide in CHANGELOG.md or docs

-------
📌 Rule 3: Analyze Everything—Every Time
✨ A thorough, deep review of the entire application is required before any change is made.

✨ Check all files, folders, scripts, directories, configuration files, pipelines, logs, and documentation without exception.

✨ Do not rely on assumptions—validate every piece of code logic, every dependency, every API interaction, and every test.

✨ Document what you find, and do not move forward until you have a complete understanding of the system.

-------
📌 Rule 4: Reuse Before Creating
✨ Always check if a script or piece of code already exists before creating a new one.

✨ If it exists, use it or improve it—don't duplicate it. No more script chaos where there are five different versions of the same functionality scattered across the codebase.

-------
📌 Rule 5: Treat This as a Professional Project — Not a Playground
✨ This is not a testing ground or experimental repository. Every change must be done with a professional mindset—no trial-and-error, no haphazard additions, and no skipping steps.

✨ Respect the structure, follow established standards, and treat this like you would a high-stakes production system.

-------
📌 Rule 6: Clear, Centralized, and Structured Documentation
✨ All documentation must be in a central /docs/ directory with a logical folder structure.

✨ Update documentation as part of every change—no exceptions.

✨ Do not leave outdated documentation lying around. Remove it immediately or update it to reflect the current state.

✨ Ownership and collaboration: Make it clear what each document is for, who owns it, and when it was last updated.

-------
📌 Rule 7: Eliminate Script Chaos — Clean, Consolidate, and Control
✨ We will not tolerate script sprawl. All scripts must be:

• Centralized in a single, well-organized /scripts/ directory.

• Categorized clearly (e.g., /scripts/deployment/, /scripts/testing/, /scripts/utils/).

• Named descriptively and purposefully.

• Documented with headers explaining their purpose, usage, and dependencies.

✨ Remove all unused scripts. If you find duplicates, consolidate them into one.

✨ Scripts should have one purpose and do it well. No monolithic, do-everything scripts.

-------
📌 Rule 8: Python Script Sanity — Structure, Purpose, and Cleanup
✨ Python scripts must:

• Be organized into a clear location (e.g., /scripts/python/ or within specific module directories).

• Include proper headers: purpose, author, date, usage instructions.

• Use argparse or similar for CLI arguments—no hardcoded values.

• Handle errors gracefully with logging.

• Be production-ready, not quick hacks.

✨ Delete all test scripts, debugging scripts, and one-off experiments from the repository. If you need them temporarily, use a separate branch or local environment.

-------
📌 Rule 9: Backend & Frontend Version Control — No More Duplication Chaos
✨ There should be one and only one source of truth for the backend and frontend.

✨ Remove all v1, v2, v3, old, backup, deprecated versions immediately.

✨ If you need to experiment, use branches and feature flags—not duplicate directories.

-------
📌 Rule 10: Functionality-First Cleanup — Never Delete Blindly
✨ Before removing any code, script, or file:

• Verify all references and dependencies.

• Understand its purpose and usage.

• Test the system without it to ensure nothing breaks.

• Archive before deletion if there's any doubt.

✨ Do not delete advanced functionality that works (e.g., caching, optimization, monitoring) just because you don't understand it immediately. Investigate first.

-------
📌 Rule 11: Docker Structure Must Be Clean, Modular, and Predictable
✨ All Docker-related files must follow a consistent structure:

• Dockerfiles should be optimized, multi-stage where appropriate, and well-commented.

• docker-compose.yml files must be modular and environment-specific (dev, staging, prod).

• Use .dockerignore properly to exclude unnecessary files.

• Version-pin all base images and dependencies.

-------
📌 Rule 12: One Self-Updating, Intelligent, End-to-End Deployment Script
✨ Create and maintain a single deploy.sh script that:

• Is self-sufficient and comprehensive.

• Handles all environments (dev, staging, production) with appropriate flags.

• Is self-updating—pulls the latest changes and updates itself before running.

• Provides clear logging, error handling, and rollback capabilities.

• Is documented inline and in /docs/deployment/.

✨ No more scattered deployment scripts. One script to rule them all.

-------
📌 Rule 13: No Garbage, No Rot
✨ Abandoned code, TODO comments older than 30 days, commented-out blocks, and unused imports/variables must be removed.

✨ If it's not being used, it doesn't belong in the codebase.

✨ Regular cleanup sprints will be enforced.

-------
📌 Rule 14: Engage the Correct AI Agent for Every Task
✨ We have specialized AI agents. Use them appropriately:

• Backend tasks → Backend specialist

• Frontend tasks → Frontend specialist

• DevOps tasks → DevOps specialist

• Documentation → Documentation specialist

✨ Do not use a generalist agent for specialized work when a specialist is available.

✨ Document which agent was used for which task in commit messages.

-------
📌 Rule 15: Keep Documentation Clean, Clear, and Deduplicated
✨ Documentation must be:

• Clear and concise—no rambling or redundancy.

• Up-to-date—reflects the current state of the system.

• Structured—follows a consistent format and hierarchy.

• Actionable—provides clear next steps, not just descriptions.

✨ Remove all duplicate documentation immediately. There should be one source of truth for each topic.

-------
📌 Rule 16: Use Local LLMs Exclusively via Ollama, Default to TinyLlama
✨ All AI/LLM operations must use Ollama with locally hosted models.

✨ Default model: TinyLlama (fast, efficient, sufficient for most tasks).

✨ Document any model overrides clearly in configuration and code comments.

✨ No external API calls to OpenAI, Anthropic, or other cloud providers without explicit approval and documentation.

-------
📌 Rule 17: Review and Follow All Documents in /opt/sutazaiapp/IMPORTANT
✨ The /opt/sutazaiapp/IMPORTANT directory contains canonical documentation that must be reviewed before making any changes.

✨ These documents represent the source of truth and override any conflicting information elsewhere.

✨ If you find discrepancies, the IMPORTANT/ documents win.

-------
📌 Rule 18: Absolute, Line-by-Line Deep Review of Core Documentation
✨ Before starting any work, you must perform a line-by-line review of:

• /opt/sutazaiapp/CLAUDE.md

• /opt/sutazaiapp/IMPORTANT/*

• Project README files

• Architecture documentation

✨ This is not optional. Zero tolerance for skipping this step.

✨ Document your understanding and any discrepancies found.

-------
📌 Rule 19: Mandatory Change Tracking in /opt/sutazaiapp/docs/CHANGELOG.md
✨ Every single change, no matter how small, must be documented in the CHANGELOG.

✨ Format: [Time] - [Date] - [Version] - [Component] - [Change Type] - [Description]

✨ Include:

• What was changed

• Why it was changed

• Who made the change (AI agent or human)

• Potential impact or dependencies

✨ No exceptions. Undocumented changes will be reverted.

All agents must study and review this file first
