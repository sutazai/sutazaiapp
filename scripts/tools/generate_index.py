#!/usr/bin/env python3
import json
import os
from pathlib import Path

ROOT = Path('/opt/sutazaiapp')

def count_files(globs):
    total = 0
    matches = []
    for pattern in globs:
        for p in ROOT.glob(pattern):
            if p.is_file():
                total += 1
                matches.append(str(p.relative_to(ROOT)))
    return total, sorted(matches)

def load_mcp_servers():
    cfg = ROOT/'.mcp.json'
    if not cfg.exists():
        return {}
    try:
        data = json.loads(cfg.read_text())
        return data.get('mcpServers', {})
    except Exception:
        return {}

def write_index(content: str):
    out = ROOT/'IMPORTANT'/'INDEX.md'
    out.write_text(content)

def main():
    docker_count, docker_list = count_files([
        'docker/**/Dockerfile*',
        'docker/**/docker-compose*.yml',
        'docker/**/docker-compose*.yaml',
        'docker/**/*.yml',
        'docker/**/*.yaml',
    ])
    mcp_servers = load_mcp_servers()
    wrappers = sorted([str(p.relative_to(ROOT)) for p in (ROOT/'scripts'/'mcp'/'wrappers').glob('*.sh') if p.is_file()])

    lines = []
    lines.append('# Comprehensive System Index')
    lines.append('')
    lines.append(f'- Root: `{ROOT}`')
    lines.append(f'- Docker config files: {docker_count}')
    lines.append('')
    lines.append('## Docker Files')
    for item in docker_list:
        lines.append(f'- `{item}`')
    lines.append('')
    lines.append('## MCP Servers (.mcp.json)')
    if mcp_servers:
        for name, spec in sorted(mcp_servers.items()):
            t = spec.get('type','?')
            cmd = spec.get('command','')
            lines.append(f'- {name}: type={t}, command=`{cmd}`')
    else:
        lines.append('- (none found)')
    lines.append('')
    lines.append('## MCP Wrapper Scripts')
    for w in wrappers:
        lines.append(f'- `{w}`')
    lines.append('')
    lines.append('## Agents Directories')
    lines.append('- Canonical roles: `.claude/agents/`')
    lines.append('- Codex/OpenAI entrypoint: `.codex/agents/`')
    lines.append('')
    lines.append('## Canonical Registries')
    lines.append('- Ports: `config/port-registry.yaml` (system-level)')
    lines.append('- Ports (observed): `config/port-registry-actual.yaml`')

    write_index('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()

