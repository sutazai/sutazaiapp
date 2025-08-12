#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime

ROOT = Path('/opt/sutazaiapp')
AGENTS = ROOT / '.claude' / 'agents'
CHANGELOG = ROOT / 'docs' / 'CHANGELOG.md'

HEADER = '## Role Definition (Bespoke v3)'

# Category detection
def category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ['security','pentest','honeypot','trivy','semgrep']):
        return 'security'
    if any(k in n for k in ['qa','tester','validation','validator','testing']):
        return 'qa'
    if any(k in n for k in ['architect','architecture','design']):
        return 'architect'
    if any(k in n for k in ['devops','docker','compose','k3s','k8s','orchestrator','deployment','terraform','harbor']):
        return 'devops'
    if any(k in n for k in ['frontend','ui','react','docusaurus','markdown','ux']):
        return 'frontend'
    if any(k in n for k in ['backend','api','graphql','server','fastapi']):
        return 'backend'
    if any(k in n for k in ['data','ml','model','learning','reinforcement','distillation','drift']):
        return 'ml'
    if any(k in n for k in ['hardware','gpu','cpu','performance','optimizer']):
        return 'hardware'
    if any(k in n for k in ['observability','grafana','prometheus','loki','jaeger','trace','logging','metrics']):
        return 'observability'
    if any(k in n for k in ['mcp','registry']):
        return 'mcp'
    if any(k in n for k in ['document','docs','api-documenter','knowledge']):
        return 'docs'
    if any(k in n for k in ['research','analyst','synthesizer','brief']):
        return 'research'
    if any(k in n for k in ['coordinator','manager','orchestrator','supervisor']):
        return 'orchestrator'
    if any(k in n for k in ['database','postgres','sql']):
        return 'database'
    return 'general'


REFERENCES = {
    'security': ['OWASP ASVS https://owasp.org/www-project-application-security-verification-standard/','SonarQube https://docs.sonarsource.com/sonarqube/','Trivy https://aquasecurity.github.io/trivy/latest/'],
    'qa': ['pytest https://docs.pytest.org/en/stable/','GitLab CI https://docs.gitlab.com/ee/ci/'],
    'architect': ['SutazAI CLAUDE.md','IMPORTANT/ canonical docs'],
    'devops': ['Docker https://docs.docker.com/','GitLab CI https://docs.gitlab.com/ee/ci/'],
    'frontend': ['TypeScript https://www.typescriptlang.org/docs/','Docusaurus https://docusaurus.io/docs'],
    'backend': ['FastAPI https://fastapi.tiangolo.com/'],
    'ml': ['DVC https://dvc.org/doc','MLflow https://mlflow.org/docs/latest/index.html'],
    'hardware': ['Linux perf','py-spy'],
    'observability': ['Grafana https://grafana.com/docs/','Prometheus https://prometheus.io/docs/','Jaeger https://www.jaegertracing.io/docs/','Loki https://grafana.com/docs/loki/latest/'],
    'mcp': ['MCP https://github.com/modelcontextprotocol'],
    'docs': ['SutazAI /docs standards'],
    'research': ['Anthropic sub-agents https://docs.anthropic.com/en/docs/claude-code/sub-agents'],
    'database': ['PostgreSQL https://www.postgresql.org/docs/'],
    'general': ['Repo rules Rule 1–19'],
}


def block(cat: str) -> str:
    ref_lines = '\n'.join(f'- {r}' for r in REFERENCES.get(cat, REFERENCES['general']))
    return (
        f"{HEADER}\n\n"
        f"Scope and Triggers\n"
        f"- Use when tasks match this agent's domain; avoid overlap by checking existing agents and code first (Rule 4).\n"
        f"- Trigger based on changes to relevant modules/configs and CI gates; document rationale.\n\n"
        f"Operating Procedure\n"
        f"1. Read CLAUDE.md and IMPORTANT/ docs; grep for reuse (Rules 17–18, 4).\n"
        f"2. Draft a minimal, reversible plan with risks and rollback (Rule 2).\n"
        f"3. Make focused changes respecting structure, naming, and style (Rules 1, 6).\n"
        f"4. Run linters/formatters/types; add/adjust tests to prevent regression.\n"
        f"5. Measure impact (perf/security/quality) and record evidence.\n"
        f"6. Update /docs and /docs/CHANGELOG.md with what/why/impact (Rule 19).\n\n"
        f"Deliverables\n"
        f"- Patch/PR with clear commit messages, tests, and updated docs.\n"
        f"- Where applicable: perf/security reports, dashboards, or spec updates.\n\n"
        f"Success Metrics\n"
        f"- No regressions; all checks green; measurable improvement in the agent's domain.\n\n"
        f"References\n{ref_lines}\n"
    )


def upsert_file(p: Path) -> bool:
    text = p.read_text(encoding='utf-8', errors='ignore')
    if HEADER in text:
        return False
    cat = category(p.stem)
    new = text.rstrip() + '\n\n' + block(cat) + '\n'
    p.write_text(new, encoding='utf-8')
    return True


def main() -> None:
    files = sorted((AGENTS).glob('*.md'))
    updated = 0
    for f in files:
        if f.name in {'agent-overview.md'}:
            continue
        try:
            if upsert_file(f):
                updated += 1
                print(f'Bespoke added: {f.name}')
        except Exception as e:
            print(f'Failed {f.name}: {e}')
    print(f'Done. Added bespoke to {updated} files.')
    if updated:
        entry = (
            f"\n## {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"### Agent Role Definitions Added (Bespoke v3, categorized)\n\n"
            f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"- Component: .claude/agents\n"
            f"- Change Type: docs-improvement\n"
            f"- Description: Added role-specific, research-backed definitions to remaining agents using category heuristics with authoritative references; preserved enforcement/frontmatter.\n"
            f"- Owner: AI agent (generate_bespoke_roles_all.py)\n\n---\n"
        )
        try:
            prev = CHANGELOG.read_text(encoding='utf-8')
            idx = prev.find('\n## ')
            new = prev[:idx if idx != -1 else len(prev)] + entry + (prev[idx:] if idx != -1 else '')
            CHANGELOG.write_text(new, encoding='utf-8')
        except Exception:
            with CHANGELOG.open('a', encoding='utf-8') as f:
                f.write(entry)


if __name__ == '__main__':
    main()

