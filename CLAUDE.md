✅ Codebase Standards & Implementation Checklist

🔧 Codebase Hygiene
A clean, consistent, and organized codebase is non-negotiable. It reflects engineering discipline and enables scalability, team velocity, and fault tolerance.

Every contributor is accountable for maintaining and improving hygiene—not just avoiding harm.

🧼 Enforce Consistency Relentlessly
✅ Follow the existing structure, naming patterns, and conventions. Never introduce your own style or shortcuts.

✅ Centralize logic — do not duplicate code across files, modules, or services.

🚫 Avoid multiple versions of:

• APIs doing the same task (REST + GraphQL duplicating effort, for example)

• UI components or CSS/SCSS modules with near-identical logic or styling

• Scripts that solve the same problem in slightly different ways

• Requirements files scattered across environments with conflicting dependencies

• Documentation split across folders with different levels of accuracy

📂 Project Structure Discipline
📌 Never dump files or code in random or top-level folders.

📌 Place everything intentionally, following modular boundaries:

• components/ for reusable UI parts

• services/ or api/ for network interactions

• utils/ for pure logic or helpers

• hooks/ for reusable frontend logic

• schemas/ or types/ for data validation

If the ideal location doesn't exist, propose a clear structure and open a small RFC (Request for Comments) before proceeding.

🗑️ Dead Code is Debt
🔥 Regularly delete unused code, legacy assets, stale test files, or experimental stubs.

❌ "Just in case" or "might be useful later" is not a valid reason to keep clutter.

🧪 Temporary test code must be removed or clearly gated (e.g. with feature flags or development-only checks).

🧪 Use Tools to Automate Discipline
✅ Mandatory for all contributors:

• Linters: ESLint, Flake8, RuboCop

• Formatters: Prettier, Black, gofmt

• Static analysis: TypeScript, mypy, SonarQube, Bandit

• Dependency managers: pip-tools, Poetry, pnpm, npm lockfiles

• Schema enforcement: JSON schema, Pydantic, zod

• Test coverage tooling: Jest, pytest-cov, Istanbul

🔄 Integrate these tools in pre-commit, pre-push, and CI/CD workflows:

• No code gets into production branches without passing hygiene checks.

• Every PR should be green and self-explanatory.

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

[... existing content continues ...]

🆕 AI and Automation Principles
• Always use Ai Agents for all tasks