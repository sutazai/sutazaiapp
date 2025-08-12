#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime

ROOT = Path('/opt/sutazaiapp')
AGENTS = ROOT / '.claude' / 'agents'
CHANGELOG = ROOT / 'docs' / 'CHANGELOG.md'

START_MARK = '## Role Definition (Standardized v2)'

BESPOKE = {
    'security-auditor.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use when assessing auth flows, secrets handling, headers, dependencies, and network surfaces.
- Trigger on changes to `auth/`, `security/`, JWT/OAuth code, Dockerfiles, compose, ingress, and CI secrets.

Operating Procedure
1. Read CLAUDE.md and IMPORTANT/ security docs; grep for prior patterns to reuse.
2. Run static checks (Bandit, npm audit), secret scan, and dependency audit.
3. Verify auth: token lifetime, rotation, audience, issuer, clock skew, and storage.
4. Enforce headers: CSP, HSTS, X-Content-Type-Options, Referrer-Policy, CORS.
5. Verify input validation, parameterized queries, and schema enforcement.
6. Propose the minimal fix; add tests for the exploit and the fix.
7. Update docs and CHANGELOG with risks and mitigations.

Deliverables
- Audit report with CVSS/severity, PoC steps, and fix diffs.
- Tests proving exploit is closed and no regression.

Success Metrics
- 0 criticals exploitable in automated checks; no auth breakage; headers present with correct directives.

References
- OWASP ASVS: https://owasp.org/www-project-application-security-verification-standard/
- SonarQube: https://docs.sonarsource.com/sonarqube/
''',

    'container-vulnerability-scanner-trivy.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to scan images, filesystems, and IaC for CVEs, misconfigurations, secrets, and licenses.
- Trigger on Dockerfile, base image, dependency, or IaC changes; during CI gates.

Operating Procedure
1. Select scan target (image/fs/repo) and severity policy.
2. Run Trivy image scan with `--severity CRITICAL,HIGH --exit-code 1` for production.
3. Run config scan for Dockerfile and IaC; enable secret and license checks when applicable.
4. Map findings to remediation: upgrade base images/deps, patch configs, or rotate secrets.
5. Re-scan to confirm clean; document exceptions with justification and expiry.

Deliverables
- Trivy reports (SARIF/JSON) stored as CI artifacts with remediation notes.

Success Metrics
- No CRITICAL/HIGH in production images; misconfig count trending down; zero hard‑coded secrets.

References
- Trivy docs: https://aquasecurity.github.io/trivy/latest/
''',

    'distributed-tracing-analyzer-jaeger.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to instrument and analyze end‑to‑end traces for latency, errors, and bottlenecks.

Operating Procedure
1. Confirm OpenTelemetry/Jaeger client config and service names.
2. Ensure context propagation across HTTP/queue boundaries.
3. Add spans for critical paths; include tags for status codes and DB operations.
4. Analyze traces: p95/p99 latency, span outliers, and error hot spots.
5. Propose targeted fixes; add alerts if SLOs breached.

Deliverables
- Trace analysis with screenshots/links and a prioritized fix list.

Success Metrics
- Reduced p95/p99 latency and error rate on traced paths.

References
- Jaeger docs: https://www.jaegertracing.io/docs/
''',

    'observability-dashboard-manager-grafana.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to create actionable dashboards and alerts aligned to service SLOs.

Operating Procedure
1. Inventory metrics; avoid high‑cardinality labels.
2. Define SLOs and error budgets; map to panels and alerts.
3. Build dashboards with drill‑downs; annotate deployments/incidents.
4. Validate alert noise; document runbooks.

Deliverables
- Grafana dashboard JSON, alert rules, and runbooks.

Success Metrics
- Alert precision/recall improvements; MTTR down; SLO adherence.

References
- Grafana docs: https://grafana.com/docs/
''',

    'cicd-pipeline-orchestrator.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to design/update CI pipelines ensuring lint, type, test, security, and build gates.

Operating Procedure
1. Audit current GitLab CI pipeline (.gitlab-ci.yml) and Makefile targets.
2. Add stages: lint → type → test → security → build → deploy.
3. Cache dependencies; pin tool versions; fail on warnings for protected branches.
4. Store artifacts; add Trivy/SAST; gate deploy on green checks.

Deliverables
- CI config diffs and documentation of stages and gates.

Success Metrics
- Median pipeline time down; consistent green builds; zero flaky jobs.

References
- GitLab CI: https://docs.gitlab.com/ee/ci/
''',

    'code-review-specialist.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to review PRs/patches for correctness, safety, and adherence to repo standards.

Operating Procedure
1. Read CLAUDE.md, IMPORTANT/, and affected docs; map blast radius.
2. Verify reuse (no duplication), public APIs, schema changes, and migrations.
3. Check tests, types, lint, and security findings; request missing coverage.
4. Approve only minimal, reversible changes with clear rollback.

Deliverables
- Review notes with rationale, nits vs. blockers, and acceptance criteria.

Success Metrics
- Post‑merge regressions prevented; decreased rework; consistent standards.

References
- SonarQube quality gate concepts: https://docs.sonarsource.com/sonarqube/
''',

    'backend-api-architect.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to design/adjust backend APIs, schemas, and contracts without breaking clients.

Operating Procedure
1. Inventory existing endpoints; confirm consumers and compatibility constraints.
2. Define schemas and validation; plan migrations and feature flags.
3. Design idempotent, observable endpoints; document errors and timeouts.
4. Add tests (unit/integration/contract); measure latency and throughput.

Deliverables
- API spec changes, migration plan, and tests.

Success Metrics
- Zero breaking changes; improved latency; accurate docs.

References
- FastAPI docs: https://fastapi.tiangolo.com/
''',

    'mcp-server-architect.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to design, secure, and validate MCP servers/tools and registry entries.

Operating Procedure
1. Review MCP contracts and server lifecycle; verify tool schemas.
2. Enforce least privilege and secure transport; validate auth if applicable.
3. Add integration tests for critical tools; document versioning and compat.

Deliverables
- MCP server design notes, tool specs, and integration tests.

Success Metrics
- Stable tool contracts; no auth/data‑leak issues; documented registry alignment.

References
- MCP (Model Context Protocol): https://github.com/modelcontextprotocol
''',

    'data-version-controller-dvc.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to manage data artifacts and pipelines with reproducible, versioned stages.

Operating Procedure
1. Define DVC stages/params/remote; ensure storage and permissions.
2. Track datasets/models; pin hashes; document lineage.
3. Integrate with CI for cache/push/pull and reproducibility checks.

Deliverables
- dvc.yaml updates, lockfiles, and CI integration notes.

Success Metrics
- Reproducible runs; accurate lineage; reduced drift.

References
- DVC docs: https://dvc.org/doc
''',

    'ml-experiment-tracker-mlflow.md': '''## Role Definition (Bespoke v3)

Scope and Triggers
- Use to track experiments, params, metrics, and artifacts with MLflow.

Operating Procedure
1. Configure MLflow tracking URI and artifact store.
2. Log params/metrics/artifacts; standardize metric names and tags.
3. Compare runs; select candidates; archive results to reports.

Deliverables
- Tracking configuration, logging utilities, and comparison reports.

Success Metrics
- Complete, comparable runs; faster model selection; reproducible artifacts.

References
- MLflow docs: https://mlflow.org/docs/latest/index.html
''',
}


def apply_bespoke(path: Path, content: str) -> tuple[bool, str]:
    bespoke = BESPOKE.get(path.name)
    if not bespoke:
        return False, 'no-bespoke'
    if START_MARK in content:
        new = content.split(START_MARK, 1)[0].rstrip() + '\n\n' + bespoke.strip() + '\n'
        return True, new
    else:
        # Append if not found
        new = content.rstrip() + '\n\n' + bespoke.strip() + '\n'
        return True, new


def main() -> None:
    targets = list(BESPOKE.keys())
    updated = []
    for name in targets:
        p = AGENTS / name
        if not p.exists():
            print(f"Skip (missing): {name}")
            continue
        text = p.read_text(encoding='utf-8', errors='ignore')
        changed, new = apply_bespoke(p, text)
        if changed:
            p.write_text(new, encoding='utf-8')
            print(f"Bespoke applied: {name}")
            updated.append(name)
    if updated:
        tsd = datetime.now().strftime('%Y-%m-%d')
        tsm = datetime.now().strftime('%Y-%m-%d %H:%M')
        entry = (
            f"\n## {tsd}\n\n"
            f"### Agent Role Definitions Updated (Bespoke v3)\n\n"
            f"- Time: {tsm}\n"
            f"- Component: .claude/agents (bespoke batch)\n"
            f"- Change Type: docs-improvement\n"
            f"- Description: Replaced standardized block with research-grounded, role-specific definitions for: {', '.join(updated)}.\n"
            f"- Owner: AI agent (apply_bespoke_role_definitions.py)\n\n---\n"
        )
        try:
            prev = CHANGELOG.read_text(encoding='utf-8')
            idx = prev.find('\n## ')
            new = prev[:idx if idx != -1 else len(prev)] + entry + (prev[idx:] if idx != -1 else '')
            CHANGELOG.write_text(new, encoding='utf-8')
            print('CHANGELOG updated')
        except Exception as e:
            print(f'CHANGELOG update failed: {e}')


if __name__ == '__main__':
    main()

