import { defineConfig } from '@playwright/test';

export default defineConfig({
  // testDir is resolved relative to this config file directory
  testDir: './specs',
  reporter: [['list'], ['html', { outputFolder: 'reports/playwright-mcp/html' }]],
  timeout: 60_000,
  use: {
    headless: true,
  },
});
