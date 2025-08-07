import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/playwright/specs',
  reporter: [['list'], ['html', { outputFolder: 'reports/playwright-mcp/html' }]],
  timeout: 60_000,
  use: {
    headless: true,
  },
});

