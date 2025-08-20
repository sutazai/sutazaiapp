---
document_id: "DOC-2025-OPS-CLN-0003"
title: "Playwright E2E Plan and Canonical Locations (Investigation-First)"
created_date: "2025-08-19 21:18:00 UTC"
created_by: "frontend-architect.md"
last_modified: "2025-08-19 21:18:00 UTC"
status: "active"
owner: "frontend.team@sutazaiapp.local"
category: "process"
---

# Playwright E2E Plan and Canonical Locations

Findings (from audit):
- Configs found:
  - `/opt/sutazaiapp/playwright.config.ts` with `testDir: './tests/e2e'`
  - `/opt/sutazaiapp/docs/playwright.config.ts` with `testDir: './e2e'`
- No first-party tests discovered under the expected directories; candidates appeared in `node_modules`.

Decision (single source of truth):
- Canonical Playwright test directory: **`/opt/sutazaiapp/tests/e2e`**
- Do not maintain duplicate Playwright configs under `/docs`. All E2E test code should reside in `/tests/e2e`.

Next Steps (non-destructive):
1. Confirm `baseURL` and project settings in root `playwright.config.ts`.
2. Discover actual app routes (backend and frontend) and define smoke tests.
3. Add initial specs under `/tests/e2e` following naming `*.spec.ts`.
4. Integrate into CI with `npx playwright test` from repository root.

Notes:
- This plan does not add tests yet; it creates structure and process.
- Any changes to configs must be recorded in directory CHANGELOGs and cross-referenced to IMPORTANT docs.
