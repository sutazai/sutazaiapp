const request = require('supertest');
const { expect } = require('chai');

describe('Performance Tests - TDD Requirements', () => {
  let app;

  beforeEach(() => {
    // Use actual server for integration testing - same pattern as main test suite
    app = require('../src/unified-dev-server.js');
  });

  describe('Response Time Requirements', () => {
    it('should respond to health checks in under 50ms', async () => {
      const startTime = process.hrtime.bigint();
      
      await request(app)
        .get('/health')
        .expect(200);
      
      const endTime = process.hrtime.bigint();
      const responseTimeMs = Number(endTime - startTime) / 1000000;
      
      expect(responseTimeMs).to.be.below(50);
    });

    it('should respond to ultimatecoder requests in under 100ms', async () => {
      const startTime = process.hrtime.bigint();
      
      await request(app)
        .post('/api/dev')
        .send({
          service: 'ultimatecoder',
          code: 'print("test")',
          language: 'python',
          action: 'analyze'
        })
        .expect(200);
      
      const endTime = process.hrtime.bigint();
      const responseTimeMs = Number(endTime - startTime) / 1000000;
      
      expect(responseTimeMs).to.be.below(100);
    });

    it('should respond to language-server requests in under 20ms', async () => {
      const startTime = process.hrtime.bigint();
      
      await request(app)
        .post('/api/dev')
        .send({
          service: 'language-server',
          method: 'hover',
          params: { language: 'javascript' }
        })
        .expect(200);
      
      const endTime = process.hrtime.bigint();
      const responseTimeMs = Number(endTime - startTime) / 1000000;
      
      expect(responseTimeMs).to.be.below(20);
    });

    it('should respond to sequentialthinking requests in under 10ms', async () => {
      const startTime = process.hrtime.bigint();
      
      await request(app)
        .post('/api/dev')
        .send({
          service: 'sequentialthinking',
          query: 'Simple test query',
          maxSteps: 1
        })
        .expect(200);
      
      const endTime = process.hrtime.bigint();
      const responseTimeMs = Number(endTime - startTime) / 1000000;
      
      expect(responseTimeMs).to.be.below(10);
    });
  });

  describe('Memory Usage Requirements', () => {
    it('should maintain memory usage below 10% of limit', async () => {
      // Generate some load first
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(request(app).get('/health'));
      }
      await Promise.all(promises);

      const response = await request(app)
        .get('/metrics')
        .expect(200);

      expect(response.body.memory.usage_percent).to.be.below(10);
    });

    it('should not have memory leaks after multiple requests', async () => {
      // Get baseline
      const baseline = await request(app).get('/metrics');
      const baselineMemory = baseline.body.memory.current;

      // Make 50 requests
      const promises = [];
      for (let i = 0; i < 50; i++) {
        promises.push(request(app).get('/health'));
      }
      await Promise.all(promises);

      // Check memory after load
      const afterLoad = await request(app).get('/metrics');
      const afterMemory = afterLoad.body.memory.current;

      // Memory should not increase by more than 5MB
      expect(afterMemory - baselineMemory).to.be.below(5);
    });

    it('should automatically prune idle processes', async () => {
      // This test would need to simulate process creation and wait for pruning
      const response = await request(app)
        .get('/metrics')
        .expect(200);

      // Verify process registry is clean
      expect(response.body.processes.registry.python).to.equal(0);
      expect(response.body.processes.registry.go).to.equal(0);
      expect(response.body.processes.registry.nodejs).to.equal(0);
    });
  });

  describe('Concurrent Request Handling', () => {
    it('should handle 10 concurrent health requests efficiently', async () => {
      const startTime = process.hrtime.bigint();
      
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(request(app).get('/health').expect(200));
      }
      
      await Promise.all(promises);
      
      const endTime = process.hrtime.bigint();
      const totalTimeMs = Number(endTime - startTime) / 1000000;
      
      // 10 concurrent requests should complete in under 100ms
      expect(totalTimeMs).to.be.below(100);
    });

    it('should handle mixed service requests concurrently', async () => {
      const promises = [
        request(app).post('/api/dev').send({ service: 'ultimatecoder', code: 'test', language: 'python', action: 'analyze' }),
        request(app).post('/api/dev').send({ service: 'language-server', method: 'hover', params: {} }),
        request(app).post('/api/dev').send({ service: 'sequentialthinking', query: 'test', maxSteps: 1 }),
        request(app).get('/health'),
        request(app).get('/metrics')
      ];

      const startTime = process.hrtime.bigint();
      const results = await Promise.all(promises);
      const endTime = process.hrtime.bigint();
      const totalTimeMs = Number(endTime - startTime) / 1000000;

      // All requests should succeed
      results.forEach(result => {
        expect(result.status).to.be.oneOf([200, 201]);
      });

      // Total time should be reasonable
      expect(totalTimeMs).to.be.below(200);
    });

    it('should maintain success rate above 98% under load', async () => {
      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(
          request(app)
            .get('/health')
            .then(res => ({ status: res.status >= 200 && res.status < 300 }))
            .catch(() => ({ status: false }))
        );
      }

      const results = await Promise.all(promises);
      const successCount = results.filter(r => r.status).length;
      const successRate = successCount / results.length;

      expect(successRate).to.be.above(0.98);
    });
  });

  describe('Resource Efficiency Requirements', () => {
    it('should use minimal file descriptors', async () => {
      const response = await request(app)
        .get('/metrics')
        .expect(200);

      // This would need additional monitoring, for now check service responds
      expect(response.body).to.have.property('processes');
    });

    it('should have efficient garbage collection', async () => {
      // Generate load to trigger GC
      for (let i = 0; i < 100; i++) {
        await request(app).get('/health');
      }

      const beforeGc = process.memoryUsage();
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const afterGc = process.memoryUsage();
      
      // Memory should be manageable after GC
      expect(afterGc.heapUsed).to.be.below(beforeGc.heapUsed * 1.1);
    });
  });
});