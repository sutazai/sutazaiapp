import { test, expect } from '@playwright/test';

const MCP_URL = process.env.MCP_HTTP_URL || 'http://localhost:3030';

test.describe('MCP HTTP health', () => {
  test('GET /health returns ok', async ({ request }) => {
    const res = await request.get(`${MCP_URL}/health`, { timeout: 5000 });
    expect(res.ok()).toBeTruthy();
    const json = await res.json();
    expect(json.ok).toBeTruthy();
    expect(Array.isArray(json.tools)).toBeTruthy();
  });

  test('GET /info returns resources and tools', async ({ request }) => {
    const res = await request.get(`${MCP_URL}/info`, { timeout: 5000 });
    expect(res.ok()).toBeTruthy();
    const json = await res.json();
    expect(Array.isArray(json.resources)).toBeTruthy();
    expect(json.resources.length).toBeGreaterThan(0);
    expect(Array.isArray(json.tools)).toBeTruthy();
  });
});

