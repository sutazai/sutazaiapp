🔧 Codebase Hygiene
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

If the ideal location doesn’t exist, propose a clear structure and open a small RFC (Request for Comments) before proceeding.

🗑️ Dead Code is Debt
🔥 Regularly delete unused code, legacy assets, stale test files, or experimental stubs.

❌ “Just in case” or “might be useful later” is not a valid reason to keep clutter.

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

🧪 No skipping reviews or tests for “quick fixes.” These introduce long-term chaos.

🧠 Execution Mindset: Act Like a Top-Level Engineer
🛠️ Think like an Architect, Engineer, QA, and PM—all at once.

🔬 Examine the full context of any change before writing code.

🧭 Prioritize long-term clarity over short-term speed.

🧱 Every change should make the codebase easier to maintain for someone else later.

🚩 Red Flags (Anti-Patterns to Avoid)
🔴 "I'll just put this here for now" — No, there is no “for now.”

🔴 "It's just a tiny change" — That’s how tech debt begins.

🔴 "We can clean this up later" — “Later” rarely comes.

🔴 Duplicate modules named utils.js, helper.py, or service.ts across packages.

🔴 PRs that include: unrelated changes, commented-out code, unreviewed temporary logs.

🧭 Final Reminder
A healthy codebase is a shared responsibility.
Every line of code you touch should be better than you found it.

🚫 Rules to Follow

-------
📌 Rule 1: No Fantasy Elements
✨ Only real, production-ready implementations are allowed.
Do not write speculative, placeholder, “in-theory,” or overly abstract code unless it's been fully validated and grounded in current platform constraints.

✨ Avoid overengineering or unnecessary abstraction.
No fictional components, fake classes, dream APIs, or concrete implementation or real example infrastructure. All code must reflect actual, working systems.

✨ No ‘specific future version or roadmap item’ solutions.
Avoid comments like // TODO: automatically, programmatically scale this later or // imagine this uses a future AI module. If it doesn’t exist now, it doesn’t go in the codebase.

✨ Be honest with the present limitations.
Code must work today, not in a hypothetical perfect setup. Assume real-world constraints like flaky hardware, latency, cold starts, and limited memory.
All code and documentation must use real, grounded constructs—no metaphors, specific implementation name (e.g., emailSender, dataProcessor) terms, or hypothetical “black-box” AI.

✨ Forbidden:

Terms like wizardService, specificHandler (e.g., authHandler, dataHandler), teleportData(), or comments such as // TODO: add telekinesis here.

Pseudo-functions that don’t map to an actual library or API (e.g. superIntuitiveAI.optimize()).

✨ Mandated Practices:

Name things concretely: emailSender, not magicMailer.

Use real libraries: import from nodemailer, not from “the built-in mailer.”

Link to docs in comments or README—every external API or framework must be verifiable.

✅ Pre-Commit Checks:

 Search for banned keywords (specific implementation name (e.g., emailSender, dataProcessor), wizard, black-box, etc.) in your diff.

 Verify every new dependency is in package.json (or requirements.txt) with a valid version.

 Ensure code examples in docs actually compile or run.

----
 Rule 2: Do Not Break Existing Functionality
✨ Every change must respect what already works.
Before modifying any file, component, or flow, verify exactly what it currently does and why. Don’t assume anything.

✨ Regression = failure.
If your change breaks or downgrades existing features—even temporarily—it’s considered a critical issue. Stability comes first.

✨ Backwards compatibility is a must.
If your refactor or feature update changes existing behavior, either support legacy use cases or migrate them gracefully.

✨ Always test before merging.
Write or update test cases to explicitly cover both new logic and old logic. Nothing ships unless it’s verified to not break production behavior.

✨ Communicate impact clearly.
If there’s any risk of side effects, escalate and document. Silent changes are forbidden.

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

Check import graphs or IDE “Find Usages.”

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

A clear “Breaking Change” section in the PR description

A migration or upgrade guide in CHANGELOG.md or docs



----
📌 Rule 3: Analyze Everything—Every Time
Always conduct a thorough and systematic review of the entire application before proceeding.

Files

 Ensure naming conventions are consistent and meaningful

 Remove redundant or obsolete files

 Verify file dependencies and imports

Folders

 Maintain a clear, logical folder structure

 Avoid duplication of modules or components

 Group related functionality properly

Scripts

 Check for reusability and maintainability

 Confirm scripts are version-controlled and documented

 Validate script execution paths and environment variables

Directories

 Ensure consistent layout across environments (dev, staging, prod)

 Remove unused or temporary directories

 Audit for unnecessary nesting and flatten where possible

Code Logic

 Scrutinize conditionals, loops, and function logic for accuracy and efficiency

 Validate edge-case handling and error management

 Check for unnecessary complexity—simplify where possible

 Confirm logic is testable and covered by unit/integration tests

Dependencies & Packages

 Validate all installed packages are in use and up to date

 Remove deprecated or unused dependencies

 Verify security and license compliance

APIs & Integrations

 Confirm external APIs are stable and well-documented

 Check rate limits, error handling, and retry logic

 Validate data mappings and transformation layers

Configuration Files

 Ensure all environment-specific settings are properly scoped

 Avoid hard-coded secrets—use secure environment variables

 Check for misconfigured flags or unused parameters

Build/Deployment Pipelines

 Audit CI/CD pipelines for completeness and reliability

 Ensure testing, linting, and rollback mechanisms are in place

 Check for consistency across branches and environments

Logs & Monitoring

 Confirm logging is present, relevant, and not excessive

 Ensure sensitive data is never logged

 Verify monitoring and alerting are properly configured

Testing Coverage

 Validate coverage reports and ensure tests are passing

 Confirm all new or modified logic is properly tested

 Check for flaky or redundant tests

 Ensure everything is functioning 100% correctly before proceeding.

---
📌 Rule 4: Reuse Before Creating
 Always check for and reuse existing scripts.

 Only create new scripts if absolutely necessary—and only when no existing solution fits.
 
 ----
 📌 Rule 5: Treat This as a Professional Project — Not a Playground
 Approach every task with a professional mindset—this is not an experiment or personal sandbox.

 Do not treat the codebase as a place for trial-and-error coding, shortcuts, or sloppy work.

 Respect the structure, standards, and long-term maintainability of the project.

 Every decision must be intentional, reviewed, and aligned with best practices.
 
 ----
 📌 Rule 6: Clear, Centralized, and Structured Documentation
Documentation is part of the codebase — it must be organized, consistent, and treated with the same discipline as source code. It should enable any contributor or stakeholder to onboard, build, debug, or deploy without confusion.

📁 Centralized Location & Structure
 Store all documentation in a single, clearly defined location, such as:

A dedicated /docs/ directory in the root of the codebase

An internal wiki (e.g., GitHub Wiki, GitLab Wiki)

A linked platform like Notion, Confluence, or ReadTheDocs

 Do not scatter docs across arbitrary folders, branches, personal notes, or unlinked platforms.

📂 Folder & File Structure
Organize content into logical subfolders for easier navigation. Example structure:

bash
Copy
Edit
/docs
  ├── overview.md                      # Project summary, purpose, goals
  ├── setup/
  │   ├── local_dev.md                 # Local dev setup (tools, dependencies)
  │   └── environments.md              # Env variables, secrets, staging/prod configs
  ├── backend/
  │   ├── architecture.md              # System design, flow diagrams
  │   ├── auth_flow.md                 # Authentication and session logic
  │   └── api_reference.md             # Endpoint specs, versioning, response codes
  ├── frontend/
  │   ├── component_structure.md       # Folder/component layout and naming
  │   └── styling.md                   # CSS, design tokens, theming
  ├── ci-cd/
  │   ├── pipeline.md                  # CI/CD flow, build triggers
  │   └── deploy_process.md            # Manual and auto deployment steps
  └── changelog.md                    # Release history, patch notes
🧹 Naming & Formatting Conventions
 Use lowercase, hyphen-separated file names (auth-flow.md, not AuthFlow or authFlow_v1).

 All documents must be in Markdown (.md) or a consistent readable format (Notion pages, if using Notion).

 Include a clear title and introduction at the top of every file.

 Use consistent headings (##), bullet styles, code blocks, and tables across all files.

 Add anchor links and table of contents in long docs.

📈 Update Discipline
 Documentation must be updated alongside any feature, refactor, or config change that affects setup, behavior, or architecture.

 Outdated documentation is worse than none — all docs must reflect current reality, not past plans.

 Major changes should include:

A changelog entry (changelog.md)

Version tags if docs refer to API versions or release states

 Every PR must include doc updates if it introduces a change requiring explanation. If no doc change is needed, explain why in the PR.

👥 Ownership & Collaboration
 Assign documentation owners for each section (e.g., backend lead owns backend/, DevOps owns ci-cd/)

 Use reviews to ensure content is technically accurate and clearly written

 Encourage contributions from all team members—clarity helps everyone

 Periodically review and refactor documentation to keep it lean and useful

🚫 What to Avoid
 ❌ Don’t leave setup instructions only in Slack messages or commit comments

 ❌ Don’t rely on tribal knowledge—write it down

 ❌ Don’t duplicate documents across folders or tools

 ❌ Don’t use inconsistent formats (e.g., some docs in .docx, others in Notion)
 
 📌 Rule 7: Eliminate Script Chaos — Clean, Consolidate, and Control
There must be no mess in scripts. All scripts must be centralized, documented, purposeful, and reusable. Treat script sprawl like tech debt — every duplicate, broken, or forgotten script is a liability that slows down the team and adds risk.

🔥 Immediate Cleanup Required
 Audit all existing scripts in the entire codebase — not just in /scripts/, but across all folders.

 Identify and remove:

 Repetitive scripts with overlapping or duplicate logic

 Outdated scripts that are no longer relevant or used

 “Temporary” scripts that were never cleaned up

 Scripts with unclear names or unknown purposes

 Keep only what is actively used, tested, and maintained.

➡ Outcome: A lean, purposeful set of scripts that can be trusted by anyone on the team.

📁 Centralized Script Directory with Clear Structure
All valid scripts must be moved into a single, organized folder structure such as:

pgsql
Copy
Edit
/scripts
  ├── README.md                  # Explains the purpose and usage of this folder
  ├── dev/                       # Local development tools
  │   ├── reset-db.sh
  │   └── start-dev.sh
  ├── deploy/
  │   ├── release.sh
  │   └── docker-rebuild.sh
  ├── data/
  │   ├── seed-users.py
  │   └── export-db.sh
  ├── utils/
  │   └── format-json.py
  └── test/
      └── run-all-tests.sh
 Never store scripts directly in root folders or scattered across feature directories

 Use subfolders by function (/dev/, /data/, /deploy/, /test/, /migrations/)

 Each script must serve a unique, justified purpose

📄 Required Standards for Each Script
Every script must follow these conventions:

 Filename: Lowercase, hyphenated, and descriptive (e.g., clear-cache.sh, not a.sh or TestReset2.sh)

 Header Comment:

bash
Copy
Edit
# Purpose: Resets local database to a known state
# Usage: ./reset-db.sh [--with-demo-data]
# Requires: Docker, Postgres running locally
 Inputs & Outputs: Accept parameters and handle them correctly. Do not hard-code logic.

 Exit Codes: Fail gracefully and return non-zero status on errors.

 Security: No hard-coded secrets, passwords, or credentials. Use env vars or vaults.

 Execution: Scripts should be executable (chmod +x) and runnable from root or docs.

 No Dead Code: Remove unused functions, print statements, and commented-out experiments.

🧼 Ongoing Script Management Rules
 Rule: One Task = One Script. If two scripts do the same thing, merge or refactor them.

 Rule: Reuse over Rewrite. If logic exists elsewhere (in another script or production code), import it or call it — don’t duplicate it.

 Rule: Everything Documented. The /scripts/README.md must include:

 Description of every folder and its purpose

 List of all scripts with a one-line summary

 Any required setup instructions

 Rule: Remove What You Don’t Use. No "just in case" or "conditional logic or feature flag this still works" files.

🚫 Absolutely Avoid
 ❌ Scripts with no comments or unclear intent

 ❌ Random script names like test1.py, temp.sh, fix.sh

 ❌ Copy-paste variants that only differ by a few lines

 ❌ Leaving multiple scripts that do the same task in slightly different ways

 ❌ Keeping personal or debug scripts in the main repo
 
--- 
 🐍 Rule 8: Python Script Sanity — Structure, Purpose, and Cleanup
All .py scripts must be purposeful, organized, and maintained like production code — no “random helpers,” one-off experiments, or duplications are allowed.

📁 Location & Structure
 All .py scripts must live in a clear, centralized location like:

bash
Copy
Edit
/scripts
  ├── data/
  │   └── load_mock_users.py
  ├── utils/
  │   └── format_validator.py
  ├── deploy/
  │   └── db_migrate.py
 Avoid having .py files:

In root directories

In random feature folders

In /tests/ unless they are test files

In /notebooks/ unless tied to Jupyter workflows

📌 Each .py Script Must:
✅ Have a clear purpose
✅ Be named descriptively (e.g., generate_config.py, not stuff.py)
✅ Include this at the top:

python
Copy
Edit
"""
Purpose: Explain what this script does in 1–2 sentences.
Usage: python path/to/script.py [--options]
Requirements: List any external dependencies or env vars.
"""
✅ Handle CLI args (use argparse, click, etc.) — do NOT hard-code values
✅ Use __name__ == "__main__" guard properly
✅ Log properly instead of using print() for important info
✅ Exit with meaningful codes for CI/CD hooks or automation
✅ Support imports from main codebase when appropriate
✅ Be tested — use mocks or unit tests if logic is non-trivial

🧼 Code Quality Expectations
 Use black or ruff to auto-format code

 No specific implementation name (e.g., emailSender, dataProcessor) numbers or hardcoded file paths — use config or arguments

 If a script does something complex, break it into modules

 Add error handling — no stack trace dumps to users

❌ You Must Not:
❌ Keep local debugging scripts like test2.py, print_stuff.py, old_fix.py

❌ Duplicate business logic that already exists in main packages

❌ Leave behind scripts from Jupyter experiments

❌ Use .py files to log into prod manually or tweak prod data without approval

❌ Push personal tools or hacks into the repo

✅ Ideal Python Script Workflow
You create a new script → scripts/data/sync_to_staging.py

You explain its purpose, usage, and requirements at the top

You use argparse to allow flags like --dry-run

You log outputs cleanly and return proper exit codes

You document it in scripts/README.md under the correct section

You remove any old sync1.py, sync2.py, or similar duplicates


---
🧹 Rule 9: Backend & Frontend Version Control — No More Duplication Chaos
All frontend and backend code must be centralized, single-source, and maintained like a shared production codebase. No forks, no loose copies, no abandoned or duplicate folders.

🧱 1. 📁 Folder Structure — One Source of Truth
Ensure the project root uses clear folder boundaries like:

arduino
Copy
Edit
/backend
  ├── app/
  ├── config/
  ├── routes/
  ├── services/
  └── tests/

/frontend
  ├── src/
  │   ├── components/
  │   ├── pages/
  │   ├── assets/
  │   └── utils/
  └── public/
✅ There must be one and only one /frontend and one /backend.
❌ No legacy folders like /web2, /old_api, /frontend-new, /backend_dev, etc.

🔥 2. Remove Deprecated or Duplicate Versions
Action Required:

 Audit the repo for duplicate or legacy frontend/backend directories

 Remove everything that is:

Not documented or referenced

Not actively maintained

Clearly replaced by newer versions

Marked with old_, copy_, final/, dev_2, or any similar naming

 If unsure, move legacy versions to a dated /archive/ folder with a README

🚦 3. Use Branches & Feature Flags — Not Clones
 Never duplicate entire frontend or backend code for testing — use Git branches

 Use feature flags or toggle systems for unstable features

 Avoid dev/, test1/, or new_feature/ folders in the codebase — everything should be version-controlled properly

🧼 4. Code Cleanliness Rules
Frontend
 Every component should have a clear file and naming pattern

 Remove all unused components, CSS files, or legacy UI fragments

 Use linting (eslint, prettier) and type-checking (tsc or prop-types)

 No dead routes, test pages, or unreferenced assets

Backend
 Each service/module should have a clear owner or purpose

 Remove all endpoints that are unused or obsolete

 Use config files, not hardcoded values

 Consolidate repeated logic into shared modules/utilities

 Avoid .bak, v2, copy.py, or temp_routes.js type of files

📋 5. Document Your Folder Layout
Inside both /frontend and /backend, maintain a top-level README.md describing:

Folder structure

How to run locally

Key services/components

Deployment instructions

📎 6. Optional — Use Git Tags to Preserve Legacy
If you're afraid of losing old versions:

 Tag them as Git snapshots (e.g., v1.0-legacy-frontend)

 Then safely delete from the mainline repo

🧠 The Philosophy
There is one backend and one frontend — no duplicates, no experiments lying around, and no guesswork about what version is live.

----
🧪 Rule 10: Functionality-First Cleanup — Never Delete Blindly
All cleanup activities — whether scripts, frontend code, backend services, or documentation — must be preceded by functional verification.

Do not remove, rename, or consolidate files unless you've confirmed their role, references, and impact.

⚠️ Common Violations (Real Examples You’re Facing)
Scripts deleted just because “they look duplicated” → broke automated flows

Components removed from frontend without checking dynamic usage

Backend routes consolidated without verifying their consumers

.py scripts renamed or removed without checking scheduled jobs or imports

---
✅ Cleanup Must Follow These Steps:
📍Locate References
Before removing or renaming anything:

 Search the entire codebase (recursive search) for references to the file/class/function

 Check pipelines, scheduled jobs, imports, and CLI calls

 Check for links in the documentation, README files, or user-facing apps

🧠 Understand the Purpose

 Read the script or module

 Identify if it’s legacy, deprecated, or just poorly named

 Ask the original author or team if unclear

✅ Verify with Tests or Manual Steps

 Check if tests cover the component

 If no tests exist, manually test the functionality

 Document what was tested and why it’s safe to delete or merge

📥 Move to /archive/ Before Permanent Deletion

 Keep a dated snapshot of any file you're removing in /archive/<date>/

 Include a note in archive/README.md explaining why it was moved

🔐 All Cleanup PRs Must Include:
 A checklist of what was removed

 Evidence that the removed files are unused or tested

 A link to the archived copy (if applicable)

 Who reviewed and signed off on the deletion

🧭 Final Principle
⚠️ "If you can’t prove a file is safe to delete, don’t delete it."
All deletions must be deliberate, documented, and tested. No "spring cleaning" allowed without validation.

---
🐳 Rule 11: Docker Structure Must Be Clean, Modular, and Predictable
Your Docker-related assets must follow a strict, predictable structure:

✅ Required Standards:
Each service must have its own Dockerfile or be clearly documented as part of a multi-service build.

All Dockerfiles must:

Use official base images where possible

Be version-pinned (FROM node:18.17, not just node)

Be multi-stage if the image supports it

Avoid installing unnecessary packages

Use .dockerignore to exclude local clutter

📂 Suggested Structure:
bash
Copy
Edit
/docker
  ├── backend/
  │   └── Dockerfile
  ├── frontend/
  │   └── Dockerfile
  ├── nginx/
  │   └── Dockerfile
  ├── docker-compose.yml
  └── .dockerignore
📦 Final Rule:
Containers must be reproducible, minimal, and contain only what’s necessary for runtime.



----
🚀 Rule 12: One Self-Updating, Intelligent, End-to-End Deployment Script
There must be exactly one canonical deployment script (deploy.sh or equivalent) that serves as the single source of truth for provisioning any environment—from bare-metal/VM to production—with a single command.

Core Qualities
Self-Sufficient

Assumes a pristine OS (Ubuntu/CentOS/RHEL) with no prior state.

Updates package repositories and installs OS-level dependencies.

Comprehensive

Installs language runtimes (Node, Python, Java, etc.).

Builds or pulls container images (Docker, Kubernetes).

Provisions databases, queues, caches, storage.

Configures networking, SSL certs, firewalls, DNS.

Runs DB migrations, seed data loads, and smoke tests.

Deterministic & Idempotent

Re-running yields the same end state without duplication.

Uses lockfiles and checksum validation to pin versions.

Resilient

Detects and recovers from partial failures (network hiccups, missing packages).

Includes retry logic and rollback hooks.

Secure

Never embeds secrets—fetches from a vault or secure store.

Logs metadata only; never credentials.

Integrates with CI/CD secrets management.

Self-Updating & Sync Awareness
Before any deployment, the script must:

Detect Changes

Monitor diffs in Dockerfiles, package.json/requirements.txt, IaC (Terraform/CloudFormation), migration scripts, and environment configs.

Run Pre-Deployment Analysis

Validate new dependencies install successfully.

Lint and schema-validate all configuration files.

Check for resource/port conflicts, missing provisions.

Fail Fast on Drift

If the script is out-of-date, exit with a clear error (e.g., “ERROR: Service X added in docker-compose.yml but not provisioned in deploy.sh.”).

Command Interface & Phases
bash
Copy
Edit
# Full deploy in one step
./deploy.sh --env production

# Granular phases
./deploy.sh --env staging --phase init    # OS & base deps
./deploy.sh --env staging --phase build   # Build & package
./deploy.sh --env staging --phase deploy  # Provision & launch
./deploy.sh --env staging --phase verify  # Smoke tests & health checks
--env: dev | staging | production

--phase (optional): init | build | deploy | verify | all (default)

Pre-Flight & Validation
Run Tests: unit, integration, and end-to-end suites.

Lint & Validate: Docker configs, YAML, JSON, IaC.

Dry-Run Mode:

bash
Copy
Edit
./deploy.sh --env production --dry-run
Outputs a full plan without executing changes.

Post-Deploy Reporting
Print service endpoints, ports, and admin URLs.

Generate a deployment manifest (deploy-manifest.json) capturing versions, commits, and timestamps.

Optionally notify stakeholders (Slack, email) with a summary.

CI/CD Integration & Enforcement
Pipeline Step: CI/CD must invoke only this script for deployments.

Staging Dry-Run: Any infra or code changes trigger an automated staging dry-run in CI.

Approval Gates: Changes to deploy.sh require Ops sign-off and two-step PR approval.

“If it isn’t in deploy.sh, it doesn’t exist.” Every stack change—new service, dependency, or config—must be handled by, detected in, or documented by this script.

---
📌 Rule 13: No Garbage, No Rot
🗑️ Absolutely no junk, clutter, or abandoned code should exist in the repository—ever.

✨ Zero-Tolerance for Mess:

No "temporary", "WIP", "old", or "copy-of" files left hanging.

No unused variables, functions, endpoints, branches, or components.

No outdated screenshots, configs, or asset folders that serve no purpose.

✨ Common Violations to Eliminate:

test_final_v2_but_real_this_time.py

old-component.jsx.bak, legacy_code_archive/

Commented-out blocks of code kept "just in case"

Redundant stylesheets or duplicated configs in subfolders

Obsolete .md files with stale instructions

🔄 Ongoing Clean-Up Habits:

 Regularly audit the codebase (monthly minimum).

 Assign owners for “code debt cleanup” as part of sprint cycles.

 Add clean-up steps to PR reviews (e.g., "Is anything being left behind?").

✅ Pre-Merge Checklist:

 All added files belong to the correct module or feature

 No unrelated or forgotten files included in the commit

 No leftover debugging or console.log, print() statements

 No commented-out code unless accompanied by a rationale and removal plan

🧹 Enforced By:

Git hooks (pre-commit, pre-push)

Lint rules for unused imports, variables, unreachable code

CI checks for dead files or unreferenced modules

Directory visualizations and tree audits (tree -L 2, du -sh *)

🔁 If You Add, You Clean:

“You touch it, you own it.” Additions come with the duty to remove obsolete or conflicting items nearby.
---
📌 Rule 14: Engage the Correct AI Agent for Every Task
🤖 Always route the task to the most capable and specialized AI agent—no exceptions.

✨ Do not waste compute or time using general-purpose agents for domain-specific problems (e.g., UI layout, code generation, test coverage, data extraction).

✨ Match the agent to the job—front-end, back-end, infra, security, testing, documentation, data cleanup, etc.

✨ If a better-suited tool or model exists (even partially), switch or escalate—don't improvise.

✨ Ensure the selected agent is aligned with project context, tech stack, naming conventions, and code standards.

🧠 Responsibility of the implementer:

 Know your agent options and capabilities.

 Use toolchains or orchestration layers (e.g., LangChain, AutoGen, AgentOps) if applicable.

 When in doubt, escalate or document agent limitations.

❗“The right agent, with the right prompt, in the right place—every time.”

📌 Rule 15: Keep Documentation Clean, Clear, and Deduplicated
All documentation is as critical as code—treat it with the same rigor to ensure clarity, findability, and single sourcing.

✨ Core Requirements
Single Source of Truth

Store every doc in the approved location (e.g., /docs/, wiki, Confluence).

No duplicate content across multiple folders or platforms.

Clear Structure & Naming

Use a consistent folder hierarchy (overview, setup, backend, frontend, ci-cd, changelog).

Filenames must be lowercase, hyphen-separated, and descriptive (e.g., api-versioning.md, not versionAPI.txt).

Up-to-Date & Accurate

Every PR that touches code or config must update relevant docs.

Outdated docs are worse than none—remove or archive stale content immediately.

Concise & Actionable

Keep content focused: one purpose per document.

Use tables, diagrams, or code samples only when they add direct clarity.

🔄 Ongoing Maintenance
Monthly Documentation Audit

Flag and prune duplicates, obsolete guides, or half-written drafts.

Ownership & Review

Assign a doc owner per section (e.g., backend lead owns /docs/backend/).

Include documentation review as part of your standard PR checklist.

Versioning & Changelog

When docs change significantly, add a brief entry to changelog.md with date, section, and summary.

🚫 What to Avoid
❌ Scattering README files in every subfolder—centralize references instead.

❌ Copy-pasting the same content into multiple docs. Link back to the master page.

❌ Long, unstructured “wall of text”—break content into headings, bullet points, and code blocks.

Copy this into your central policy doc or /docs/documentation-standards.md to enforce clean, clear, and unified documentation practices.

📌 Rule 16: Use Local LLMs Exclusively via Ollama, Default to TinyLlama
All on-premise language model usage must go through the Ollama framework to ensure consistency, security, and resource control.

✨ Default Model

For all initial setup, prototyping, and low-compute tasks, use TinyLlama via Ollama.

🔧 Usage Requirements

Invocation

bash
Copy
Edit
ollama run tinyllama --prompt "Your prompt here"
Configuration

Store Ollama settings in a centralized config (/config/ollama.yaml).

Pin TinyLlama version to avoid drift.

Resource Constraints

Define CPU/GPU limits in Ollama profile—no model may exceed allocated resources.

🔄 Model Upgrades & Overrides

To use any other local LLM, update the Ollama config and document the change in docs/llm-standards.md.

All overrides require a one-line justification in the PR and sign-off from the AI/Infra lead.

🚫 What’s Forbidden

Directly calling LLM binaries or APIs outside of Ollama.

Using large, unvetted models for initial setup—always start with TinyLlama.

Copy this into your /docs/llm-standards.md or the central policy file to enforce standardized, secure local LLM usage.
