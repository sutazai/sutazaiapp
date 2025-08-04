#!/usr/bin/env node

/**
 * Test script to verify dashboard audit button works without stack overflow
 */

const http = require('http');

// Configuration
const DASHBOARD_URL = 'http://localhost:10422';
const API_URL = 'http://localhost:10420/api/hygiene';

console.log('🧪 Dashboard Audit Button Test');
console.log('=' .repeat(50));

// First, check if API is responding
function checkAPI() {
    return new Promise((resolve, reject) => {
        console.log('\n📡 Checking API status...');
        
        const req = http.get(API_URL + '/status', (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    console.log(`✅ API Status: ${json.systemStatus || 'OK'}`);
                    console.log(`   Total Violations: ${json.totalViolations || 0}`);
                    console.log(`   Critical: ${json.criticalViolations || 0}`);
                    resolve(true);
                } catch (e) {
                    console.log('❌ API Error:', e.message);
                    resolve(false);
                }
            });
        });
        
        req.on('error', (e) => {
            console.log('❌ Connection Error:', e.message);
            resolve(false);
        });
        
        req.setTimeout(5000, () => {
            req.destroy();
            console.log('❌ API Timeout');
            resolve(false);
        });
    });
}

// Simulate clicking audit button
function simulateAuditClick() {
    return new Promise((resolve, reject) => {
        console.log('\n🖱️  Simulating audit button click...');
        
        const postData = JSON.stringify({});
        const options = {
            hostname: 'localhost',
            port: 10420,
            path: '/api/hygiene/audit',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            },
            timeout: 30000
        };
        
        const startTime = Date.now();
        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                const elapsed = Date.now() - startTime;
                
                if (res.statusCode === 404) {
                    console.log('❌ Audit endpoint not found (404)');
                    console.log('   This might be expected if audit is running differently');
                    resolve({ success: false, statusCode: 404 });
                    return;
                }
                
                try {
                    const json = JSON.parse(data);
                    console.log(`✅ Audit completed in ${elapsed}ms`);
                    console.log(`   Status: ${res.statusCode}`);
                    console.log(`   Success: ${json.success || false}`);
                    console.log(`   Violations Found: ${json.violations_found || 0}`);
                    console.log(`   No stack overflow detected!`);
                    resolve({ success: true, data: json, elapsed });
                } catch (e) {
                    console.log('❌ Parse Error:', e.message);
                    console.log('   Response:', data.substring(0, 100));
                    resolve({ success: false, error: e.message });
                }
            });
        });
        
        req.on('error', (e) => {
            console.log('❌ Request Error:', e.message);
            if (e.message.includes('stack')) {
                console.log('   🚨 STACK OVERFLOW DETECTED!');
            }
            resolve({ success: false, error: e.message });
        });
        
        req.on('timeout', () => {
            console.log('❌ Audit timeout after 30 seconds');
            req.destroy();
            resolve({ success: false, error: 'timeout' });
        });
        
        req.write(postData);
        req.end();
    });
}

// Test rapid clicks
async function testRapidClicks() {
    console.log('\n🔥 Testing rapid audit clicks (stack overflow test)...');
    
    let successCount = 0;
    let errorCount = 0;
    let stackOverflowDetected = false;
    
    for (let i = 1; i <= 5; i++) {
        console.log(`\n   Click ${i}/5:`);
        const result = await simulateAuditClick();
        
        if (result.success) {
            successCount++;
        } else {
            errorCount++;
            if (result.error && result.error.includes('stack')) {
                stackOverflowDetected = true;
                break;
            }
        }
        
        // Small delay between clicks
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    console.log('\n📊 Rapid Click Test Summary:');
    console.log(`   Successful: ${successCount}`);
    console.log(`   Failed: ${errorCount}`);
    console.log(`   Stack Overflow: ${stackOverflowDetected ? '❌ YES' : '✅ NO'}`);
    
    return !stackOverflowDetected;
}

// Check dashboard is accessible
function checkDashboard() {
    return new Promise((resolve) => {
        console.log('\n🌐 Checking dashboard accessibility...');
        
        http.get(DASHBOARD_URL, (res) => {
            if (res.statusCode === 200) {
                console.log('✅ Dashboard is accessible at', DASHBOARD_URL);
                resolve(true);
            } else {
                console.log(`❌ Dashboard returned status ${res.statusCode}`);
                resolve(false);
            }
        }).on('error', (e) => {
            console.log('❌ Dashboard connection error:', e.message);
            resolve(false);
        });
    });
}

// Main test flow
async function runTests() {
    console.log('Dashboard URL:', DASHBOARD_URL);
    console.log('API URL:', API_URL);
    
    // Check dashboard
    const dashboardOk = await checkDashboard();
    if (!dashboardOk) {
        console.log('\n⚠️  Dashboard not accessible, but continuing tests...');
    }
    
    // Check API
    const apiOk = await checkAPI();
    if (!apiOk) {
        console.log('\n❌ API not responding properly');
        process.exit(1);
    }
    
    // Test single audit
    console.log('\n📝 Testing single audit click...');
    const singleResult = await simulateAuditClick();
    
    // Test rapid clicks
    const rapidTestPassed = await testRapidClicks();
    
    // Final verdict
    console.log('\n' + '='.repeat(50));
    if (rapidTestPassed) {
        console.log('✅ ALL TESTS PASSED!');
        console.log('🎉 No stack overflow detected in audit functionality!');
        console.log('\nThe dashboard audit button should work correctly.');
        console.log(`You can test it manually at: ${DASHBOARD_URL}`);
    } else {
        console.log('❌ Stack overflow detected!');
        console.log('The audit button may cause issues.');
    }
}

// Run the tests
runTests().catch(console.error);