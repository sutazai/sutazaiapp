import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  use: {
    baseURL: 'http://localhost:10011',
    headless: true,
  },
  timeout: 30000,
  retries: 2,
  reporter: [['html'], ['json', { outputFile: 'test-results.json' }]],
});
