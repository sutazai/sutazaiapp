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
No fictional components, fake classes, dream APIs, or concrete implementation or real example infrastructure. All code must reflect actual, working systems.

✨ No 'specific future version or roadmap item' solutions.
Avoid comments like // TODO: automatically, programmatically scale this later or // imagine this uses a future AI module. If it doesn't exist now, it doesn't go in the codebase.

✨ Be honest with the present limitations.
Code must work today, not in a hypothetical perfect setup. Assume real-world constraints like flaky hardware, latency, cold starts, and limited memory.
All code and documentation must use real, grounded constructs—no metaphors, specific implementation name (e.g., emailSender, dataProcessor) terms, or hypothetical "black-box" AI.

✨ Forbidden:

Terms like wizardService, specificHandler (e.g., authHandler, dataHandler), teleportData(), or comments such as // TODO: add telekinesis here.

Pseudo-functions that don't map to an actual library or API (e.g. superIntuitiveAI.optimize()).

✨ Mandated Practices:

Name things concretely: emailSender, not magicMailer.

Use real libraries: import from nodemailer, not from "the built-in mailer."

Link to docs in comments or README—every external API or framework must be verifiable.

✅ Pre-Commit Checks:

 Search for banned keywords (specific implementation name (e.g., emailSender, dataProcessor), wizard, black-box, etc.) in your diff.

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

[NEW MEMORY ENTRY]
📝 Captured Insights from First Deployment Cycle
• Successfully integrated comprehensive deployment script following Rule 12
• Identified key challenges in environment-specific configurations
• Developed robust error handling and rollback mechanisms
• Implemented version pinning for critical dependencies
• Created centralized logging for deployment events
