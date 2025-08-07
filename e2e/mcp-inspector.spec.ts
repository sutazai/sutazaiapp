import { test, expect } from '@playwright/test';

const INSPECTOR_URL = process.env.MCP_INSPECTOR_URL || 'http://localhost:6274';

test.describe('MCP Inspector smoke', () => {
  test('inspector UI responds', async ({ request }) => {
    // Try a simple GET on root; skip if not reachable
    let res;
    try {
      res = await request.get(INSPECTOR_URL, { timeout: 2000 });
    } catch (e) {
      test.skip(true, 'MCP Inspector not running');
      return;
    }
    expect(res.ok()).toBeTruthy();
  });
});

