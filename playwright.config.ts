import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:10011',
    headless: true,
    trace: 'on-first-retry',
  },
  // Opt-in retries for flakiness in CI; override via CLI if needed
  retries: process.env.CI ? 1 : 0,
});

