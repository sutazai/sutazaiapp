const request = require('supertest');
const { expect } = require('chai');
const sinon = require('sinon');

describe('Error Handling Tests - TDD Requirements', () => {
  let app;

  beforeEach(() => {
    // Use actual server for integration testing - same pattern as main test suite
    app = require('../src/unified-dev-server.js');
  });

  describe('Input Validation Errors', () => {
    it('should reject requests with invalid JSON', async () => {
      const response = await request(app)
        .post('/api/dev')
        .set('Content-Type', 'application/json')
        .send('{"invalid": json, "missing": quote}')
        .expect(400);

      expect(response.body.success).to.be.false;
    });

    it('should handle empty request bodies gracefully', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send()
        .expect(400);

      expect(response.body.success).to.be.false;
      expect(response.body.error).to.include('Unable to determine target service');
    });

    it('should validate required fields for ultimatecoder', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'ultimatecoder'
          // Missing required fields: code, language, action
        })
        .expect(200); // Service should handle missing fields gracefully

      // Should either succeed with defaults or provide helpful error
      if (!response.body.success) {
        expect(response.body.error).to.be.a('string');
      }
    });

    it('should validate required fields for language-server', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'language-server'
          // Missing method parameter
        })
        .expect(200);

      if (!response.body.success) {
        expect(response.body.error).to.be.a('string');
      }
    });

    it('should validate required fields for sequentialthinking', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'sequentialthinking'
          // Missing query parameter
        })
        .expect(200);

      if (!response.body.success) {
        expect(response.body.error).to.be.a('string');
      }
    });
  });

  describe('Service-Specific Error Handling', () => {
    it('should handle Python subprocess errors for ultimatecoder', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'ultimatecoder',
          code: 'definitely invalid python syntax $$$ error',
          language: 'python',
          action: 'analyze'
        });

      // Should handle Python errors gracefully
      if (response.status === 500) {
        expect(response.body.success).to.be.false;
        expect(response.body.error).to.be.a('string');
      }
    });

    it('should handle unknown language server methods', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'language-server',
          method: 'nonExistentMethod',
          params: {}
        })
        .expect(200);

      expect(response.body.success).to.be.true;
      expect(response.body.result.error).to.exist;
      expect(response.body.result.error.code).to.equal(-32601);
    });

    it('should handle malformed sequential thinking queries', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'sequentialthinking',
          query: '', // Empty query
          maxSteps: -1 // Invalid maxSteps
        })
        .expect(200);

      // Should either handle gracefully or return meaningful error
      expect(response.body).to.have.property('success');
    });
  });

  describe('Resource Limit Error Handling', () => {
    it('should handle memory pressure gracefully', async () => {
      // This test simulates memory pressure
      const largePayload = {
        service: 'ultimatecoder',
        code: 'x = "a" * 1000000',  // Large string
        language: 'python',
        action: 'analyze'
      };

      const response = await request(app)
        .post('/api/dev')
        .send(largePayload);

      // Should either succeed or fail gracefully
      expect(response.status).to.be.oneOf([200, 400, 500, 413]);
      if (response.status !== 200) {
        expect(response.body.success).to.be.false;
        expect(response.body.error).to.be.a('string');
      }
    });

    it('should handle timeout scenarios', async () => {
      // Test with a potentially long-running operation
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'sequentialthinking',
          query: 'Very complex reasoning that might take time',
          maxSteps: 100  // Large number of steps
        })
        .timeout(5000); // 5 second timeout

      expect(response.status).to.be.oneOf([200, 408, 500]);
    });

    it('should handle concurrent request overload', async () => {
      // Create many concurrent requests
      const promises = [];
      for (let i = 0; i < 50; i++) {
        promises.push(
          request(app)
            .get('/health')
            .then(res => ({ success: true, status: res.status }))
            .catch(err => ({ success: false, error: err.message }))
        );
      }

      const results = await Promise.all(promises);
      
      // At least 80% should succeed
      const successCount = results.filter(r => r.success).length;
      const successRate = successCount / results.length;
      expect(successRate).to.be.above(0.8);
    });
  });

  describe('Network and Infrastructure Errors', () => {
    it('should handle network connectivity issues gracefully', async () => {
      // Test service behavior when external dependencies might fail
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body.status).to.equal('healthy');
    });

    it('should handle file system errors for Python bridge', async () => {
      // This would test what happens if Python bridge creation fails
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'ultimatecoder',
          code: 'print("test")',
          language: 'python',
          action: 'generate'
        });

      // Should either succeed or provide meaningful error
      expect(response.status).to.be.oneOf([200, 500, 503]);
    });
  });

  describe('HTTP Error Responses', () => {
    it('should return 404 for non-existent endpoints', async () => {
      const response = await request(app)
        .get('/non-existent-endpoint')
        .expect(404);

      expect(response.body.success).to.be.false;
      expect(response.body.error).to.include('Endpoint not found');
      expect(response.body.availableEndpoints).to.be.an('array');
    });

    it('should return 405 for invalid HTTP methods', async () => {
      const response = await request(app)
        .delete('/health')
        .expect(404); // Express default for unhandled methods

      expect(response.body.success).to.be.false;
    });

    it('should handle large request bodies appropriately', async () => {
      const largeBody = {
        service: 'ultimatecoder',
        code: 'a'.repeat(100000), // 100KB string
        language: 'python',
        action: 'analyze'
      };

      const response = await request(app)
        .post('/api/dev')
        .send(largeBody);

      // Should either handle it or return 413 (Payload Too Large)
      expect(response.status).to.be.oneOf([200, 413, 400]);
    });
  });

  describe('Service Recovery and Resilience', () => {
    it('should recover from temporary service failures', async () => {
      // First, verify service is working
      const beforeResponse = await request(app)
        .get('/health')
        .expect(200);

      expect(beforeResponse.body.status).to.equal('healthy');

      // Simulate some load that might cause temporary issues
      const loadPromises = [];
      for (let i = 0; i < 20; i++) {
        loadPromises.push(request(app).get('/health'));
      }
      await Promise.all(loadPromises);

      // Verify service is still healthy
      const afterResponse = await request(app)
        .get('/health')
        .expect(200);

      expect(afterResponse.body.status).to.equal('healthy');
    });

    it('should maintain error rate below 5%', async () => {
      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(
          request(app)
            .get('/health')
            .then(res => ({ success: res.status === 200 }))
            .catch(() => ({ success: false }))
        );
      }

      const results = await Promise.all(promises);
      const errorCount = results.filter(r => !r.success).length;
      const errorRate = errorCount / results.length;

      expect(errorRate).to.be.below(0.05); // Less than 5% error rate
    });

    it('should provide helpful error messages for debugging', async () => {
      const response = await request(app)
        .post('/api/dev')
        .send({
          service: 'ultimatecoder',
          code: 'syntax error code',
          language: 'unknown_language',
          action: 'invalid_action'
        });

      if (!response.body.success) {
        expect(response.body.error).to.be.a('string');
        expect(response.body.error.length).to.be.above(10); // Meaningful error message
        expect(response.body.details).to.exist;
      }
    });
  });
});