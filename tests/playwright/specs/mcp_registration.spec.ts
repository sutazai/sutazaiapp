import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';

function run(cmd: string, args: string[] = []): { stdout: string; stderr: string; code: number } {
  try {
    const stdout = execFileSync(cmd, args, { encoding: 'utf8' });
    return { stdout, stderr: '', code: 0 };
  } catch (err: any) {
    return { stdout: err?.stdout?.toString?.() ?? '', stderr: err?.stderr?.toString?.() ?? String(err), code: err?.status ?? 1 };
  }
}

test.describe('Claude MCP registrations', () => {
  const required = (process.env.PLAYWRIGHT_MCP_CONTEXTS || 'context7').split(',').map((s) => s.trim()).filter(Boolean);

  test('required contexts are registered', async () => {
    const hasClaude = run('bash', ['-lc', 'command -v claude >/dev/null 2>&1 && echo yes || echo no']).stdout.trim() === 'yes';
    test.skip(!hasClaude, 'Claude CLI not available in this environment');
    const res = run('bash', ['-lc', 'claude mcp list --json 2>/dev/null || claude mcp list']);
    const out = res.stdout || res.stderr;
    for (const name of required) {
      const found = out.includes(`"name"`) ? new RegExp(`"name"\s*:\s*"${name}"`).test(out) : new RegExp(`^${name}:`, 'm').test(out);
      expect(found, `Missing MCP context: ${name}`).toBeTruthy();
    }
  });

  test('sequentialthinking container emits startup log', async () => {
    // Requires docker socket + docker-cli inside test container
    const hasDocker = run('bash', ['-lc', 'command -v docker >/dev/null 2>&1 && echo yes || echo no']).stdout.trim() === 'yes';
    test.skip(!hasDocker, 'Docker CLI not available in test container');
    const ps = run('bash', ['-lc', "docker ps --format '{{.ID}} {{.Names}}' | grep mcp-sequentialthinking | awk '{print $1}' | head -n1"]);
    const id = ps.stdout.trim();
    expect(id.length).toBeGreaterThan(0);
    const logs = run('bash', ['-lc', `docker logs ${id} 2>&1 | tail -n 100`]);
    expect(logs.code).toBe(0);
    expect(logs.stdout + logs.stderr).toContain('Sequential Thinking MCP Server running on stdio');
  });
});
