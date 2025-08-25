/**
 * Advanced MCP Integration Test
 * Tests concurrent operations, error handling, and real task execution
 */

const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

class AdvancedMCPTester {
    constructor() {
        this.testResults = [];
        this.memoryKeys = [];
    }

    async log(message, level = 'info') {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        const icons = {
            info: '📘',
            success: '✅',
            error: '❌',
            warning: '⚠️',
            test: '🧪',
            concurrent: '🔀'
        };
        console.log(`[${timestamp}] ${icons[level] || '📝'} ${message}`);
    }

    async testConcurrentMemoryOperations() {
        await this.log('Testing Concurrent Memory Operations', 'concurrent');
        
        const operations = [];
        const timestamp = Date.now();
        
        // Create 5 concurrent memory operations
        for (let i = 0; i < 5; i++) {
            const key = `concurrent_test_${timestamp}_${i}`;
            const value = `value_${i}_${Math.random().toString(36).substr(2, 9)}`;
            this.memoryKeys.push(key);
            
            operations.push(
                execAsync(`npx claude-flow@alpha memory store "${key}" "${value}" --namespace concurrent`)
                    .then(() => ({ operation: 'store', key, status: 'success' }))
                    .catch(err => ({ operation: 'store', key, status: 'failed', error: err.message }))
            );
        }
        
        // Execute all operations concurrently
        const results = await Promise.all(operations);
        
        // Log results
        let successCount = 0;
        for (const result of results) {
            if (result.status === 'success') {
                successCount++;
                await this.log(`Stored: ${result.key}`, 'success');
            } else {
                await this.log(`Failed: ${result.key} - ${result.error}`, 'error');
            }
        }
        
        await this.log(`Concurrent stores: ${successCount}/5 successful`, 
                      successCount === 5 ? 'success' : 'warning');
        
        // Now query all keys concurrently
        const queryOps = this.memoryKeys.map(key =>
            execAsync(`npx claude-flow@alpha memory query "${key}" --namespace concurrent`)
                .then(() => ({ key, found: true }))
                .catch(() => ({ key, found: false }))
        );
        
        const queryResults = await Promise.all(queryOps);
        const foundCount = queryResults.filter(r => r.found).length;
        
        await this.log(`Concurrent queries: ${foundCount}/${this.memoryKeys.length} found`,
                      foundCount === this.memoryKeys.length ? 'success' : 'warning');
        
        return { stores: successCount, queries: foundCount };
    }

    async testErrorHandlingAndRecovery() {
        await this.log('Testing Error Handling and Recovery', 'test');
        
        const errorTests = [
            {
                name: 'Invalid MCP command',
                command: 'npx claude-flow@alpha invalid-command',
                expectError: true
            },
            {
                name: 'Query non-existent key',
                command: 'npx claude-flow@alpha memory query "non_existent_key_xyz123"',
                expectError: false // Should handle gracefully
            },
            {
                name: 'Invalid JSON in store',
                command: 'npx claude-flow@alpha memory store "test" "{invalid json"',
                expectError: false // Should store as string
            }
        ];
        
        const results = [];
        for (const test of errorTests) {
            try {
                const { stdout, stderr } = await execAsync(test.command);
                results.push({
                    test: test.name,
                    success: !test.expectError,
                    output: stdout || stderr
                });
                await this.log(`${test.name}: Handled gracefully`, 'success');
            } catch (error) {
                results.push({
                    test: test.name,
                    success: test.expectError,
                    error: error.message
                });
                await this.log(`${test.name}: ${test.expectError ? 'Expected error caught' : 'Unexpected error'}`, 
                              test.expectError ? 'success' : 'error');
            }
        }
        
        return results;
    }

    async testRealTaskExecution() {
        await this.log('Testing Real Task Execution', 'test');
        
        // Create a real code analysis task
        const codeToAnalyze = `
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class DataProcessor {
    constructor() {
        this.data = [];
    }
    
    async process(items) {
        for (const item of items) {
            await this.validateItem(item);
            this.data.push(this.transform(item));
        }
        return this.data;
    }
    
    validateItem(item) {
        if (!item.id) throw new Error('Invalid item');
        return true;
    }
    
    transform(item) {
        return { ...item, processed: true };
    }
}
`;
        
        // Save code to file
        await fs.writeFile('test-code.js', codeToAnalyze);
        
        // Create analysis tasks
        const tasks = [
            {
                name: 'Code complexity analysis',
                command: 'npx claude-flow@alpha sparc run ask "Analyze the complexity of test-code.js" --non-interactive --quick'
            },
            {
                name: 'Performance optimization suggestions',
                command: 'npx claude-flow@alpha sparc run optimization "Optimize test-code.js" --non-interactive --quick'
            }
        ];
        
        const taskResults = [];
        for (const task of tasks) {
            await this.log(`Executing: ${task.name}`, 'test');
            try {
                const startTime = Date.now();
                const { stdout } = await execAsync(task.command, { timeout: 30000 });
                const duration = Date.now() - startTime;
                
                taskResults.push({
                    task: task.name,
                    success: true,
                    duration: `${(duration / 1000).toFixed(1)}s`,
                    outputLength: stdout.length
                });
                
                await this.log(`${task.name}: Completed in ${(duration / 1000).toFixed(1)}s`, 'success');
            } catch (error) {
                taskResults.push({
                    task: task.name,
                    success: false,
                    error: error.message
                });
                await this.log(`${task.name}: Failed - ${error.message}`, 'error');
            }
        }
        
        // Cleanup
        await fs.unlink('test-code.js').catch(() => {});
        
        return taskResults;
    }

    async testPersistenceAcrossSessions() {
        await this.log('Testing Persistence Across Sessions', 'test');
        
        const sessionKey = `session_test_${Date.now()}`;
        const sessionValue = {
            created: new Date().toISOString(),
            data: 'This should persist',
            random: Math.random()
        };
        
        // Store in first "session"
        await this.log('Session 1: Storing data', 'info');
        try {
            await execAsync(`npx claude-flow@alpha memory store "${sessionKey}" "${JSON.stringify(sessionValue).replace(/"/g, '\\"')}" --namespace persistence`);
            await this.log('Session 1: Data stored', 'success');
        } catch (error) {
            await this.log(`Session 1: Store failed - ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
        
        // Simulate session break
        await this.log('Simulating session break (2 second delay)', 'info');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Retrieve in second "session"
        await this.log('Session 2: Retrieving data', 'info');
        try {
            const { stdout } = await execAsync(`npx claude-flow@alpha memory query "${sessionKey}" --namespace persistence`);
            if (stdout.includes(sessionKey)) {
                await this.log('Session 2: Data successfully retrieved', 'success');
                return { success: true, persisted: true };
            } else {
                await this.log('Session 2: Data not found', 'warning');
                return { success: false, persisted: false };
            }
        } catch (error) {
            await this.log(`Session 2: Retrieve failed - ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    async testBatchOperations() {
        await this.log('Testing Batch Operations', 'test');
        
        // Create batch workflow
        const workflow = {
            name: 'batch-test',
            operations: [
                { type: 'store', key: 'batch_1', value: 'first' },
                { type: 'store', key: 'batch_2', value: 'second' },
                { type: 'store', key: 'batch_3', value: 'third' },
                { type: 'query', key: 'batch_1' },
                { type: 'query', key: 'batch_2' },
                { type: 'query', key: 'batch_3' }
            ]
        };
        
        await fs.writeFile('batch-workflow.json', JSON.stringify(workflow, null, 2));
        
        const batchScript = `
const workflow = require('./batch-workflow.json');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

async function runBatch() {
    console.log('Executing batch workflow: ' + workflow.name);
    const results = [];
    
    for (const op of workflow.operations) {
        try {
            if (op.type === 'store') {
                await execAsync(\`npx claude-flow@alpha memory store "\${op.key}" "\${op.value}" --namespace batch\`);
                console.log(\`✅ Stored: \${op.key}\`);
                results.push({ op: op.type, key: op.key, success: true });
            } else if (op.type === 'query') {
                const { stdout } = await execAsync(\`npx claude-flow@alpha memory query "\${op.key}" --namespace batch\`);
                console.log(\`✅ Queried: \${op.key}\`);
                results.push({ op: op.type, key: op.key, success: true });
            }
        } catch (error) {
            console.log(\`❌ Failed: \${op.key}\`);
            results.push({ op: op.type, key: op.key, success: false });
        }
    }
    
    const successCount = results.filter(r => r.success).length;
    console.log(\`\\nBatch completed: \${successCount}/\${results.length} successful\`);
}

runBatch().catch(console.error);
`;
        
        await fs.writeFile('batch-test.js', batchScript);
        
        try {
            const { stdout } = await execAsync('node batch-test.js');
            await this.log('Batch operations completed', 'success');
            
            // Parse results from output
            const successMatch = stdout.match(/(\d+)\/(\d+) successful/);
            if (successMatch) {
                const [, success, total] = successMatch;
                return { 
                    success: parseInt(success), 
                    total: parseInt(total),
                    rate: `${((parseInt(success) / parseInt(total)) * 100).toFixed(0)}%`
                };
            }
        } catch (error) {
            await this.log(`Batch operations failed: ${error.message}`, 'error');
            return { success: 0, total: 6, rate: '0%' };
        } finally {
            // Cleanup
            await fs.unlink('batch-workflow.json').catch(() => {});
            await fs.unlink('batch-test.js').catch(() => {});
        }
    }

    async generateDetailedReport() {
        await this.log('\n=== Generating Detailed Test Report ===', 'info');
        
        const report = {
            timestamp: new Date().toISOString(),
            platform: process.platform,
            nodeVersion: process.version,
            tests: this.testResults,
            summary: {
                totalTests: this.testResults.length,
                passed: this.testResults.filter(r => r.success).length,
                failed: this.testResults.filter(r => !r.success).length
            }
        };
        
        await fs.writeFile('mcp-advanced-report.json', JSON.stringify(report, null, 2));
        
        console.log('\n' + '='.repeat(60));
        console.log('📊 ADVANCED MCP TEST REPORT');
        console.log('='.repeat(60));
        console.log(`Total Tests: ${report.summary.totalTests}`);
        console.log(`✅ Passed: ${report.summary.passed}`);
        console.log(`❌ Failed: ${report.summary.failed}`);
        console.log(`Success Rate: ${((report.summary.passed / report.summary.totalTests) * 100).toFixed(1)}%`);
        console.log('='.repeat(60));
        
        return report;
    }

    async runAllTests() {
        await this.log('🚀 Starting Advanced MCP Integration Tests', 'info');
        await this.log(`Platform: ${process.platform}`, 'info');
        await this.log(`Node Version: ${process.version}`, 'info');
        
        // Run tests
        const concurrentResults = await this.testConcurrentMemoryOperations();
        this.testResults.push({ 
            test: 'Concurrent Operations', 
            success: concurrentResults.stores === 5 && concurrentResults.queries === 5,
            details: concurrentResults
        });
        
        const errorResults = await this.testErrorHandlingAndRecovery();
        this.testResults.push({ 
            test: 'Error Handling', 
            success: errorResults.every(r => r.success),
            details: errorResults
        });
        
        const persistenceResult = await this.testPersistenceAcrossSessions();
        this.testResults.push({ 
            test: 'Persistence', 
            success: persistenceResult.success && persistenceResult.persisted,
            details: persistenceResult
        });
        
        const batchResult = await this.testBatchOperations();
        this.testResults.push({ 
            test: 'Batch Operations', 
            success: batchResult && batchResult.success === batchResult.total,
            details: batchResult
        });
        
        const taskResults = await this.testRealTaskExecution();
        this.testResults.push({ 
            test: 'Real Task Execution', 
            success: taskResults.some(r => r.success),
            details: taskResults
        });
        
        // Generate report
        const report = await this.generateDetailedReport();
        
        await this.log('\n✅ Advanced testing completed!', 'success');
        await this.log('Report saved to: mcp-advanced-report.json', 'info');
        
        return report;
    }
}

// Run advanced tests
if (require.main === module) {
    const tester = new AdvancedMCPTester();
    tester.runAllTests()
        .then(() => process.exit(0))
        .catch(error => {
            console.error('❌ Test suite failed:', error);
            process.exit(1);
        });
}

module.exports = AdvancedMCPTester;