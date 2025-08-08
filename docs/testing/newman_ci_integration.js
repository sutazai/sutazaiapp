#!/usr/bin/env node

// Newman CLI Integration Script for CI/CD Pipeline
// File: scripts/run-newman-tests.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Configuration
const config = {
  postmanCollection: path.join(__dirname, '../docs/testing/postman_collection_jarvis_endpoints.json'),
  environment: {
    BASE_URL: process.env.BASE_URL || 'http://localhost:10010',
    JARVIS_VOICE_URL: process.env.JARVIS_VOICE_URL || 'http://localhost:11150',
    JARVIS_KNOWLEDGE_URL: process.env.JARVIS_KNOWLEDGE_URL || 'http://localhost:11101',
    JARVIS_AUTOMATION_URL: process.env.JARVIS_AUTOMATION_URL || 'http://localhost:11102',
    JARVIS_MULTIMODAL_URL: process.env.JARVIS_MULTIMODAL_URL || 'http://localhost:11103',
    JARVIS_HARDWARE_URL: process.env.JARVIS_HARDWARE_URL || 'http://localhost:11104',
  },
  reporters: ['cli', 'json', 'html', 'junit'],
  outputDir: path.join(__dirname, '../test-results'),
  timeouts: {
    request: 30000,  // 30 seconds per request
    script: 10000,   // 10 seconds for pre/test scripts
  },
  iterations: parseInt(process.env.TEST_ITERATIONS) || 1,
  bail: process.env.FAIL_FAST === 'true',
  color: process.env.NO_COLOR !== 'true',
  silent: process.env.SILENT_TESTS === 'true',
  insecure: process.env.SKIP_TLS_VERIFY === 'true',
};

// Utility functions
function ensureDirectory(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function logMessage(level, message) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}`);
}

function waitForService(url, timeout = 60000, interval = 2000) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const checkService = () => {
      const http = require('http');
      const urlParts = new URL(url);
      
      const req = http.get({
        hostname: urlParts.hostname,
        port: urlParts.port,
        path: '/health',
        timeout: 5000,
      }, (res) => {
        if (res.statusCode === 200) {
          logMessage('info', `Service ${url} is ready`);
          resolve();
        } else {
          setTimeout(checkService, interval);
        }
      });
      
      req.on('error', () => {
        if (Date.now() - startTime > timeout) {
          reject(new Error(`Timeout waiting for service ${url}`));
        } else {
          setTimeout(checkService, interval);
        }
      });
      
      req.on('timeout', () => {
        req.destroy();
        setTimeout(checkService, interval);
      });
    };
    
    checkService();
  });
}

// Health check function
async function performHealthChecks() {
  logMessage('info', 'Performing health checks...');
  
  const services = [
    config.environment.BASE_URL,
    config.environment.JARVIS_VOICE_URL,
    config.environment.JARVIS_KNOWLEDGE_URL,
    config.environment.JARVIS_AUTOMATION_URL,
    config.environment.JARVIS_MULTIMODAL_URL,
    config.environment.JARVIS_HARDWARE_URL,
  ];
  
  const healthCheckPromises = services.map(url => 
    waitForService(url, 30000, 2000).catch(err => {
      logMessage('warn', `Health check failed for ${url}: ${err.message}`);
      return { url, error: err.message };
    })
  );
  
  const results = await Promise.allSettled(healthCheckPromises);
  const failures = results.filter(result => result.status === 'rejected' || result.value?.error);
  
  if (failures.length > 0) {
    logMessage('warn', `${failures.length} services failed health checks`);
    failures.forEach(failure => {
      if (failure.value?.url) {
        logMessage('warn', `Failed: ${failure.value.url} - ${failure.value.error}`);
      }
    });
  } else {
    logMessage('info', 'All services passed health checks');
  }
  
  return failures.length === 0;
}

// Newman test execution
function runNewmanTests() {
  return new Promise((resolve, reject) => {
    logMessage('info', 'Starting Newman API tests...');
    
    // Ensure output directory exists
    ensureDirectory(config.outputDir);
    
    const newmanOptions = {
      collection: config.postmanCollection,
      environment: {
        name: 'Test Environment',
        values: Object.entries(config.environment).map(([key, value]) => ({
          key,
          value,
          enabled: true
        }))
      },
      reporters: config.reporters,
      reporter: {
        cli: {
          silent: config.silent,
          noAssertions: false,
          noSummary: false,
          noFailures: false,
          noConsole: false,
        },
        json: {
          export: path.join(config.outputDir, 'newman-results.json')
        },
        html: {
          export: path.join(config.outputDir, 'newman-report.html'),
          template: path.join(__dirname, '../docs/testing/newman-template.hbs'), // Optional custom template
        },
        junit: {
          export: path.join(config.outputDir, 'newman-junit.xml')
        }
      },
      iterationCount: config.iterations,
      bail: config.bail,
      color: config.color,
      timeout: config.timeouts,
      insecure: config.insecure,
      ignoreRedirects: false,
      delayRequest: 100, // 100ms delay between requests
    };
    
    newman.run(newmanOptions, (err, summary) => {
      if (err) {
        logMessage('error', `Newman run failed: ${err.message}`);
        reject(err);
        return;
      }
      
      // Process results
      const results = {
        total: summary.run.stats.requests.total,
        passed: summary.run.stats.assertions.total - summary.run.stats.assertions.failed,
        failed: summary.run.stats.assertions.failed,
        duration: summary.run.timings.completed,
        iterationCount: summary.run.stats.iterations.total,
      };
      
      logMessage('info', `Newman tests completed:`);
      logMessage('info', `  Total requests: ${results.total}`);
      logMessage('info', `  Passed assertions: ${results.passed}`);
      logMessage('info', `  Failed assertions: ${results.failed}`);
      logMessage('info', `  Duration: ${results.duration}ms`);
      logMessage('info', `  Iterations: ${results.iterationCount}`);
      
      // Save summary
      const summaryPath = path.join(config.outputDir, 'test-summary.json');
      fs.writeFileSync(summaryPath, JSON.stringify(results, null, 2));
      
      // Generate badge data
      const badgeData = {
        schemaVersion: 1,
        label: 'API Tests',
        message: results.failed === 0 ? 'passing' : `${results.failed} failing`,
        color: results.failed === 0 ? 'brightgreen' : 'red',
        namedLogo: 'postman',
      };
      
      const badgePath = path.join(config.outputDir, 'test-badge.json');
      fs.writeFileSync(badgePath, JSON.stringify(badgeData, null, 2));
      
      if (results.failed > 0) {
        const error = new Error(`${results.failed} test assertions failed`);
        error.results = results;
        reject(error);
      } else {
        resolve(results);
      }
    });
  });
}

// CI/CD Integration functions
function generateCIArtifacts(results) {
  logMessage('info', 'Generating CI/CD artifacts...');
  
  // Generate GitHub Actions output
  if (process.env.GITHUB_ACTIONS) {
    const githubOutput = [
      `total_tests=${results.total}`,
      `passed_tests=${results.passed}`,
      `failed_tests=${results.failed}`,
      `test_duration=${results.duration}`,
      `test_success=${results.failed === 0}`,
    ].join('\\n');
    
    console.log(`::set-output name=test_results::${githubOutput}`);
    
    if (results.failed > 0) {
      console.log(`::error::${results.failed} API tests failed`);
    } else {
      console.log(`::notice::All ${results.total} API tests passed`);
    }
  }
  
  // Generate GitLab CI output
  if (process.env.GITLAB_CI) {
    console.log(`TOTAL_TESTS=${results.total}`);
    console.log(`PASSED_TESTS=${results.passed}`);
    console.log(`FAILED_TESTS=${results.failed}`);
    console.log(`TEST_DURATION=${results.duration}`);
  }
  
  // Generate Jenkins output
  if (process.env.JENKINS_URL) {
    // Jenkins can parse JUnit XML which we already generate
    logMessage('info', 'JUnit XML report generated for Jenkins');
  }
}

// Slack notification function
function sendSlackNotification(results) {
  const webhookUrl = process.env.SLACK_WEBHOOK_URL;
  if (!webhookUrl) return;
  
  const emoji = results.failed === 0 ? ':white_check_mark:' : ':x:';
  const color = results.failed === 0 ? 'good' : 'danger';
  
  const message = {
    attachments: [
      {
        color: color,
        title: `${emoji} Jarvis API Test Results`,
        fields: [
          {
            title: 'Total Requests',
            value: results.total.toString(),
            short: true,
          },
          {
            title: 'Passed',
            value: results.passed.toString(),
            short: true,
          },
          {
            title: 'Failed',
            value: results.failed.toString(),
            short: true,
          },
          {
            title: 'Duration',
            value: `${Math.round(results.duration / 1000)}s`,
            short: true,
          },
        ],
        footer: 'Newman API Tests',
        ts: Math.floor(Date.now() / 1000),
      },
    ],
  };
  
  // Send to Slack (would need HTTP client implementation)
  logMessage('info', `Slack notification prepared: ${JSON.stringify(message)}`);
}

// Main execution function
async function main() {
  try {
    logMessage('info', 'Starting Jarvis API test suite...');
    
    // Check if collection file exists
    if (!fs.existsSync(config.postmanCollection)) {
      throw new Error(`Postman collection not found: ${config.postmanCollection}`);
    }
    
    // Perform health checks (optional, can be skipped with --skip-health-check)
    if (!process.argv.includes('--skip-health-check')) {
      const healthCheckPassed = await performHealthChecks();
      if (!healthCheckPassed && !process.argv.includes('--ignore-health-failures')) {
        logMessage('warn', 'Some services failed health checks, but continuing with tests...');
      }
    }
    
    // Run Newman tests
    const results = await runNewmanTests();
    
    // Generate CI/CD artifacts
    generateCIArtifacts(results);
    
    // Send notifications
    sendSlackNotification(results);
    
    logMessage('info', 'Test suite completed successfully');
    process.exit(0);
    
  } catch (error) {
    logMessage('error', `Test suite failed: ${error.message}`);
    
    // Generate failure artifacts
    const failureResults = error.results || {
      total: 0,
      passed: 0,
      failed: 1,
      duration: 0,
      iterationCount: 0,
    };
    
    generateCIArtifacts(failureResults);
    sendSlackNotification(failureResults);
    
    process.exit(1);
  }
}

// CLI interface
if (require.main === module) {
  // Parse command line arguments
  const args = process.argv.slice(2);
  
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: node run-newman-tests.js [options]

Options:
  --help, -h                Show help information
  --skip-health-check      Skip initial health checks
  --ignore-health-failures Continue even if health checks fail
  --iterations <n>         Number of test iterations (default: 1)
  --fail-fast              Stop on first test failure
  --silent                 Suppress output
  --no-color              Disable colored output

Environment Variables:
  BASE_URL                Backend API URL (default: http://localhost:10010)
  JARVIS_VOICE_URL       Voice service URL (default: http://localhost:11150)
  JARVIS_KNOWLEDGE_URL   Knowledge service URL (default: http://localhost:11101)
  JARVIS_AUTOMATION_URL  Automation service URL (default: http://localhost:11102)
  JARVIS_MULTIMODAL_URL  Multimodal service URL (default: http://localhost:11103)
  JARVIS_HARDWARE_URL    Hardware optimizer URL (default: http://localhost:11104)
  TEST_ITERATIONS        Number of test iterations
  FAIL_FAST              Stop on first failure (true/false)
  SILENT_TESTS           Suppress test output (true/false)
  NO_COLOR               Disable colored output (true/false)
  SKIP_TLS_VERIFY        Skip TLS certificate verification (true/false)
  SLACK_WEBHOOK_URL      Slack webhook for notifications
    `);
    process.exit(0);
  }
  
  // Override config with CLI arguments
  const iterationsIndex = args.indexOf('--iterations');
  if (iterationsIndex !== -1 && args[iterationsIndex + 1]) {
    config.iterations = parseInt(args[iterationsIndex + 1]);
  }
  
  if (args.includes('--fail-fast')) {
    config.bail = true;
  }
  
  if (args.includes('--silent')) {
    config.silent = true;
  }
  
  if (args.includes('--no-color')) {
    config.color = false;
  }
  
  // Run the main function
  main();
}

module.exports = {
  runNewmanTests,
  performHealthChecks,
  generateCIArtifacts,
  sendSlackNotification,
  config,
};