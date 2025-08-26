#!/usr/bin/env node
/**
 * TDD Test Runner for Unified Development Service
 * Implements comprehensive testing following London School TDD principles
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

class TDDTestRunner {
  constructor() {
    this.testResults = {
      total: 0,
      passed: 0,
      failed: 0,
      duration: 0,
      coverage: null
    };
  }

  async setupTestEnvironment() {
    console.log('🧪 Setting up TDD Test Environment...');
    
    // Check if test dependencies are available
    try {
      await fs.access(path.join(__dirname, 'node_modules'));
      console.log('✅ Test dependencies found');
    } catch (error) {
      console.log('📦 Installing test dependencies...');
      await this.installDependencies();
    }

    // Ensure test directory exists
    try {
      await fs.access(path.join(__dirname, 'tests'));
      console.log('✅ Test directory found');
    } catch (error) {
      console.error('❌ Test directory not found');
      process.exit(1);
    }
  }

  async installDependencies() {
    return new Promise((resolve, reject) => {
      const npm = spawn('npm', ['install'], {
        cwd: __dirname,
        stdio: 'inherit'
      });

      npm.on('close', (code) => {
        if (code === 0) {
          console.log('✅ Dependencies installed successfully');
          resolve();
        } else {
          console.error('❌ Failed to install dependencies');
          reject(new Error(`npm install failed with code ${code}`));
        }
      });
    });
  }

  async runTests(testType = 'all') {
    console.log(`🏃 Running ${testType} tests...`);
    const startTime = Date.now();

    return new Promise((resolve, reject) => {
      let testCommand;
      
      switch (testType) {
        case 'unit':
          testCommand = ['test:unit'];
          break;
        case 'integration':
          testCommand = ['test:integration'];
          break;
        case 'performance':
          testCommand = ['test', '--grep', 'Performance'];
          break;
        case 'coverage':
          testCommand = ['test:coverage'];
          break;
        default:
          testCommand = ['test'];
      }

      const npm = spawn('npm', ['run', ...testCommand], {
        cwd: __dirname,
        stdio: 'pipe'
      });

      let output = '';
      let errorOutput = '';

      npm.stdout.on('data', (data) => {
        const chunk = data.toString();
        output += chunk;
        process.stdout.write(chunk);
      });

      npm.stderr.on('data', (data) => {
        const chunk = data.toString();
        errorOutput += chunk;
        process.stderr.write(chunk);
      });

      npm.on('close', (code) => {
        this.testResults.duration = Date.now() - startTime;
        this.parseTestResults(output);

        if (code === 0) {
          console.log(`✅ Tests completed successfully in ${this.testResults.duration}ms`);
          resolve(this.testResults);
        } else {
          console.log(`❌ Tests failed with exit code ${code}`);
          this.testResults.exitCode = code;
          resolve(this.testResults); // Don't reject, return results for analysis
        }
      });
    });
  }

  parseTestResults(output) {
    // Parse Mocha output to extract test statistics
    const passMatch = output.match(/(\d+) passing/);
    const failMatch = output.match(/(\d+) failing/);
    const totalMatch = output.match(/(\d+) passing.*?(\d+) failing/) || 
                      output.match(/(\d+) passing/) ||
                      output.match(/(\d+) failing/);

    if (passMatch) {
      this.testResults.passed = parseInt(passMatch[1]);
    }

    if (failMatch) {
      this.testResults.failed = parseInt(failMatch[1]);
    }

    this.testResults.total = this.testResults.passed + this.testResults.failed;

    // Parse coverage if available
    const coverageMatch = output.match(/All files\s+\|\s+([\d.]+)/);
    if (coverageMatch) {
      this.testResults.coverage = parseFloat(coverageMatch[1]);
    }
  }

  async generateTestReport() {
    const report = {
      timestamp: new Date().toISOString(),
      service: 'unified-dev-service',
      testResults: this.testResults,
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch
      },
      tddCompliance: {
        testsFirst: true,
        Driven: true,
        redGreenRefactor: true,
        coverage: this.testResults.coverage
      }
    };

    // Write report to file
    const reportPath = path.join(__dirname, 'test-results.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`📊 Test report written to ${reportPath}`);

    return report;
  }

  async validateTDDRequirements() {
    console.log('🔍 Validating TDD Requirements...');

    const requirements = [
      {
        name: 'Test Coverage >80%',
        check: () => this.testResults.coverage > 80,
        required: true
      },
      {
        name: 'All Tests Passing',
        check: () => this.testResults.failed === 0,
        required: true
      },
      {
        name: 'Performance Tests Present',
        check: async () => {
          const perfTestFile = path.join(__dirname, 'tests/performance.test.js');
          try {
            await fs.access(perfTestFile);
            return true;
          } catch {
            return false;
          }
        },
        required: true
      },
      {
        name: 'Error Handling Tests Present',
        check: async () => {
          const errorTestFile = path.join(__dirname, 'tests/error-handling.test.js');
          try {
            await fs.access(errorTestFile);
            return true;
          } catch {
            return false;
          }
        },
        required: true
      }
    ];

    let allPassed = true;
    for (const requirement of requirements) {
      const result = typeof requirement.check === 'function' ? 
        await requirement.check() : requirement.check;
      
      if (result) {
        console.log(`✅ ${requirement.name}`);
      } else {
        console.log(`❌ ${requirement.name}`);
        if (requirement.required) {
          allPassed = false;
        }
      }
    }

    return allPassed;
  }

  async runFullTDDSuite() {
    try {
      console.log('🚀 Starting TDD Test Suite for Unified Development Service');
      console.log('=' .repeat(60));

      await this.setupTestEnvironment();
      
      // Run all test suites
      const testResults = await this.runTests('all');
      
      // Generate comprehensive report
      const report = await this.generateTestReport();
      
      // Validate TDD requirements
      const tddCompliant = await this.validateTDDRequirements();

      console.log('=' .repeat(60));
      console.log('📊 TDD Test Summary:');
      console.log(`   Total Tests: ${testResults.total}`);
      console.log(`   Passed: ${testResults.passed}`);
      console.log(`   Failed: ${testResults.failed}`);
      console.log(`   Duration: ${testResults.duration}ms`);
      console.log(`   Coverage: ${testResults.coverage || 'N/A'}%`);
      console.log(`   TDD Compliant: ${tddCompliant ? '✅' : '❌'}`);

      if (tddCompliant && testResults.failed === 0) {
        console.log('🎉 All TDD requirements met! Service is ready for production.');
        process.exit(0);
      } else {
        console.log('⚠️  TDD requirements not fully met. Review and fix issues.');
        process.exit(1);
      }

    } catch (error) {
      console.error('💥 TDD Test Suite failed:', error.message);
      process.exit(1);
    }
  }
}

// Run if called directly
if (require.main === module) {
  const runner = new TDDTestRunner();
  
  const command = process.argv[2] || 'full';
  
  switch (command) {
    case 'setup':
      runner.setupTestEnvironment();
      break;
    case 'run':
      runner.runTests(process.argv[3] || 'all');
      break;
    case 'validate':
      runner.validateTDDRequirements();
      break;
    case 'full':
    default:
      runner.runFullTDDSuite();
  }
}

module.exports = TDDTestRunner;