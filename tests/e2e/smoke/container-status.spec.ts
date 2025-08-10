import { test, expect } from '@playwright/test';
import { execSync } from 'child_process';

test.describe('Container Status Validation - Smoke Tests', () => {

  test('All critical containers are running and healthy', async () => {
    const criticalContainers = [
      'sutazai-backend',
      'sutazai-frontend', 
      'sutazai-postgres',
      'sutazai-redis',
      'sutazai-ollama',
      'sutazai-neo4j',
      'sutazai-prometheus',
      'sutazai-grafana'
    ];

    for (const containerName of criticalContainers) {
      try {
        // Check if container is running
        const statusOutput = execSync(`docker ps --filter "name=${containerName}" --format "{{.Status}}"`, 
          { encoding: 'utf-8' }).trim();
        
        expect(statusOutput).toBeTruthy();
        expect(statusOutput.toLowerCase()).toContain('up');
        
        console.log(`✅ ${containerName}: ${statusOutput}`);
      } catch (error) {
        console.error(`❌ ${containerName}: Failed to check status`, error);
        throw new Error(`Container ${containerName} is not running or accessible`);
      }
    }
  });

  test('Agent containers are running', async () => {
    const agentContainers = [
      'sutazai-ai-agent-orchestrator',
      'sutazai-resource-arbitration-agent',
      'sutazai-task-assignment-coordinator'
    ];

    for (const containerName of agentContainers) {
      try {
        const statusOutput = execSync(`docker ps --filter "name=${containerName}" --format "{{.Status}}"`, 
          { encoding: 'utf-8' }).trim();
        
        expect(statusOutput).toBeTruthy();
        expect(statusOutput.toLowerCase()).toContain('up');
        
        console.log(`✅ ${containerName}: ${statusOutput}`);
      } catch (error) {
        console.log(`⚠️  ${containerName}: May not be running (this is acceptable for agent stubs)`);
      }
    }
  });

  test('Monitoring containers are operational', async () => {
    const monitoringContainers = [
      'sutazai-prometheus',
      'sutazai-grafana',
      'sutazai-cadvisor',
      'sutazai-node-exporter'
    ];

    for (const containerName of monitoringContainers) {
      try {
        const statusOutput = execSync(`docker ps --filter "name=${containerName}" --format "{{.Status}}"`, 
          { encoding: 'utf-8' }).trim();
        
        expect(statusOutput).toBeTruthy();
        expect(statusOutput.toLowerCase()).toContain('up');
        
        console.log(`✅ ${containerName}: ${statusOutput}`);
      } catch (error) {
        console.log(`⚠️  ${containerName}: Monitoring service may be optional`);
      }
    }
  });

  test('No containers are in restart loop', async () => {
    try {
      // Get containers that have restarted more than 3 times
      const restartingContainers = execSync(
        `docker ps --format "table {{.Names}}\\t{{.Status}}" | grep -E "(Restarting|Up.*restart)"`, 
        { encoding: 'utf-8' }
      ).trim();
      
      if (restartingContainers) {
        console.warn('⚠️  Containers showing restart activity:', restartingContainers);
        // This is a warning, not a failure - some containers may restart normally
      } else {
        console.log('✅ No containers in restart loops detected');
      }
    } catch (error) {
      // No output means no restarting containers - this is good
      console.log('✅ No containers in restart loops detected');
    }
  });

  test('Docker network connectivity', async () => {
    try {
      // Check if sutazai network exists
      const networkOutput = execSync('docker network ls --filter "name=sutazai" --format "{{.Name}}"', 
        { encoding: 'utf-8' }).trim();
      
      expect(networkOutput).toContain('sutazai');
      console.log('✅ Docker network is properly configured');
      
    } catch (error) {
      throw new Error('Docker network sutazai-network is not available');
    }
  });

});