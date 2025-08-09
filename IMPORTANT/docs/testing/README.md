# Jarvis Testing Suite Documentation

This directory contains comprehensive automated testing for all Jarvis-related endpoints and services in the SutazAI system.

## Overview

The testing suite includes:

1. **API Testing**: Postman collection with Newman CLI integration
2. **UI E2E Testing**: Cypress tests for voice/text interactions and file uploads  
3. **Load Testing**: K6 tests for 100+ concurrent users with performance metrics
4. **CI/CD Integration**: Automated test execution in deployment pipelines

## Test Structure

```
docs/testing/
├── README.md                           # This documentation
├── postman_collection_jarvis_endpoints.json  # Postman API tests
├── cypress_e2e_tests.js               # Cypress E2E tests
├── k6_load_tests.js                   # K6 load testing
├── newman_ci_integration.js           # CI/CD integration script
├── test-configs/                      # Configuration files
├── test-results/                      # Generated test reports
└── scripts/                          # Utility scripts
```

## API Testing with Postman/Newman

### Postman Collection

The collection (`postman_collection_jarvis_endpoints.json`) includes comprehensive tests for:

**Backend Jarvis Endpoints**:
- `/health` - Health check with response validation
- `/chat` - Chat interactions with Jarvis agents
- `/think` - Deep thinking/reasoning capabilities  
- `/execute` - Task execution through Jarvis

**Jarvis Service Endpoints**:
- Jarvis Voice Interface (port 11150)
- Jarvis Knowledge Management (port 11101)
- Jarvis Automation Agent (port 11102)
- Jarvis Multimodal AI (port 11103) 
- Jarvis Hardware Resource Optimizer (port 11104)

**Test Categories**:
- Positive test cases (happy path scenarios)
- Negative test cases (error handling)
- Edge cases (large payloads, invalid inputs)
- Performance validation (response times)

### Running API Tests

#### Prerequisites
```bash
npm install -g newman
npm install newman-reporter-html
```

#### Manual Execution
```bash
# Basic test run
newman run docs/testing/postman_collection_jarvis_endpoints.json

# With environment variables
newman run docs/testing/postman_collection_jarvis_endpoints.json \
  --env-var BASE_URL=http://localhost:10010 \
  --env-var JARVIS_VOICE_URL=http://localhost:11150

# Generate HTML report
newman run docs/testing/postman_collection_jarvis_endpoints.json \
  --reporters html \
  --reporter-html-export test-results/api-test-report.html
```

#### CI/CD Integration
```bash
# Run with CI integration script
node docs/testing/newman_ci_integration.js

# With options
node docs/testing/newman_ci_integration.js --iterations 3 --fail-fast
```

### Expected API Test Results

All tests should pass with these criteria:
- Health endpoints return 200 status
- Response times under 2-5 seconds depending on endpoint
- Valid JSON responses with required fields
- Error handling returns appropriate status codes (400/422 for validation errors)

## E2E Testing with Cypress

### Test Coverage

The Cypress tests (`cypress_e2e_tests.js`) cover:

**Voice Interface Tests**:
- Voice UI component visibility
- Microphone permission handling
- Voice input simulation and processing
- Voice response validation

**Text Chat Interface Tests**:
- Message input and sending
- Jarvis response display
- Streaming response handling
- Chat history management

**File Upload Tests**:
- Text file uploads and analysis
- Image file uploads and processing
- Unsupported file type rejection
- File upload security validation

**Real-time Features**:
- WebSocket connection establishment
- Real-time metric updates
- Connection interruption handling
- Reconnection logic

**Accessibility Tests**:
- Keyboard navigation
- ARIA labels and screen reader support
- Focus management

**Responsiveness Tests**:
- Mobile viewport (320x568)
- Tablet viewport (768x1024) 
- Desktop viewport (1024x768)
- Large desktop (1920x1080)

### Running E2E Tests

#### Prerequisites
```bash
npm install cypress --save-dev
```

#### Setup Configuration
Create `cypress.config.js` in project root:
```javascript
const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:10011',
    viewportWidth: 1280,
    viewportHeight: 720,
    video: true,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 10000,
    requestTimeout: 15000,
    responseTimeout: 15000,
  },
});
```

#### Execution Commands
```bash
# Interactive mode
npx cypress open

# Headless mode
npx cypress run

# Specific test file
npx cypress run --spec "cypress/e2e/jarvis_interface.cy.js"

# Different browser
npx cypress run --browser chrome

# Record video
npx cypress run --record --key=<record_key>
```

### Expected E2E Test Results

- All UI components should be visible and functional
- Voice interface should handle microphone permissions
- Chat interactions should work with proper validation
- File uploads should process successfully for supported formats
- Real-time features should establish WebSocket connections
- Accessibility standards should be met
- Interface should be responsive across all viewport sizes

## Load Testing with K6

### Test Scenarios

The K6 load tests (`k6_load_tests.js`) include:

**Baseline Load Test**:
- Ramp up to 20 users over 2 minutes
- Maintain 20 users for 5 minutes
- Scale to 50 users over 2 minutes
- Maintain 50 users for 5 minutes
- Ramp down over 2 minutes

**Stress Test**:
- Scale from 0 to 150 concurrent users
- Peak load testing at 150 users
- Target: 100+ concurrent users sustained

**Spike Test**:
- Sudden traffic bursts (10 to 200 users in 30 seconds)
- Recovery testing

**Soak Test**:
- 30 users for 30 minutes (stability testing)

### Load Test Metrics

**Performance Thresholds**:
- 95% of requests under 2000ms
- 99% of requests under 5000ms  
- Error rate under 5%
- Minimum 10 requests per second

**Jarvis-specific Metrics**:
- `jarvis_response_time`: Response times for Jarvis endpoints
- `jarvis_error_rate`: Error rate specific to Jarvis services
- `knowledge_query_latency`: Knowledge management query times
- `active_chat_sessions`: Number of concurrent chat sessions
- `concurrent_voice_requests`: Voice interface load

### Running Load Tests

#### Prerequisites
```bash
# Install K6
brew install k6  # macOS
# or
sudo apt-get install k6  # Ubuntu
# or download from https://k6.io/docs/get-started/installation/
```

#### Execution Commands
```bash
# Basic load test
k6 run docs/testing/k6_load_tests.js

# With environment variables
k6 run -e BASE_URL=http://localhost:10010 docs/testing/k6_load_tests.js

# Specific scenario
k6 run --scenario baseline_load docs/testing/k6_load_tests.js

# Generate HTML report
k6 run --out json=results.json docs/testing/k6_load_tests.js
```

#### Custom Test Scenarios
```bash
# Voice interface only
k6 run --scenario voice_load --exec voiceOnlyTest docs/testing/k6_load_tests.js

# Knowledge queries only  
k6 run --scenario knowledge_intensive --exec knowledgeOnlyTest docs/testing/k6_load_tests.js
```

### Expected Load Test Results

**Performance Targets**:
- Handle 100+ concurrent users
- Maintain sub-2-second response times for 95% of requests
- Error rate below 5% under normal load
- Error rate below 10% under peak load (150 users)
- System should recover gracefully from spike loads

**Service-specific Expectations**:
- Backend API: 200+ requests/minute sustained
- Voice Interface: 50+ concurrent voice sessions
- Knowledge Management: Query latency under 4 seconds
- All health endpoints: Sub-1-second response times

## CI/CD Pipeline Integration

### GitHub Actions Integration

Add to `.github/workflows/jarvis-tests.yml`:

```yaml
name: Jarvis Testing Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  api-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          npm install -g newman
          npm install newman-reporter-html
          
      - name: Start services
        run: |
          docker-compose up -d
          sleep 60  # Wait for services to be ready
          
      - name: Run API tests
        run: |
          node docs/testing/newman_ci_integration.js
          
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: api-test-results
          path: test-results/
          
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: npm install cypress --save-dev
        
      - name: Start services
        run: |
          docker-compose up -d
          sleep 60
          
      - name: Run E2E tests
        run: npx cypress run --spec "docs/testing/cypress_e2e_tests.js"
        
      - name: Upload screenshots
        uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: cypress-screenshots
          path: cypress/screenshots/
          
  load-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Install K6
        run: |
          sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
          
      - name: Start services
        run: |
          docker-compose up -d
          sleep 60
          
      - name: Run load tests
        run: k6 run --scenario baseline_load docs/testing/k6_load_tests.js
        
      - name: Upload load test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: load-test-results
          path: |
            load-test-report.html
            load-test-results.json
```

### GitLab CI Integration

Add to `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - load-test

api-tests:
  stage: test
  image: node:18
  services:
    - docker:dind
  variables:
    DOCKER_DRIVER: overlay2
  before_script:
    - npm install -g newman newman-reporter-html
    - docker-compose up -d
    - sleep 60
  script:
    - node docs/testing/newman_ci_integration.js
  artifacts:
    when: always
    reports:
      junit: test-results/newman-junit.xml
    paths:
      - test-results/
    expire_in: 1 week
    
e2e-tests:
  stage: test
  image: cypress/base:18
  services:
    - docker:dind
  before_script:
    - npm install cypress
    - docker-compose up -d
    - sleep 60
  script:
    - npx cypress run --spec "docs/testing/cypress_e2e_tests.js"
  artifacts:
    when: always
    paths:
      - cypress/videos/
      - cypress/screenshots/
    expire_in: 1 week
    
load-tests:
  stage: load-test
  image: ubuntu:20.04
  only:
    - main
  before_script:
    - apt-get update && apt-get install -y curl gnupg2 software-properties-common
    - apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
    - echo "deb https://dl.k6.io/deb stable main" | tee /etc/apt/sources.list.d/k6.list
    - apt-get update && apt-get install -y k6
    - docker-compose up -d
    - sleep 60
  script:
    - k6 run --scenario baseline_load docs/testing/k6_load_tests.js
  artifacts:
    when: always
    paths:
      - load-test-report.html
      - load-test-results.json
    expire_in: 1 week
```

## Test Environment Setup

### Local Development

1. **Start Services**:
   ```bash
   docker-compose up -d
   sleep 60  # Wait for services to initialize
   ```

2. **Verify Services**:
   ```bash
   curl http://localhost:10010/health  # Backend
   curl http://localhost:11150/health  # Voice Interface
   curl http://localhost:11101/health  # Knowledge Management
   ```

3. **Run Tests**:
   ```bash
   # API tests
   node docs/testing/newman_ci_integration.js
   
   # E2E tests (requires GUI for interactive mode)
   npx cypress run
   
   # Load tests
   k6 run docs/testing/k6_load_tests.js
   ```

### Staging/Production Testing

Use environment variables to target different environments:

```bash
# Staging environment
export BASE_URL=https://jarvis-staging.sutazai.com
export JARVIS_VOICE_URL=https://voice-staging.sutazai.com
export JARVIS_KNOWLEDGE_URL=https://knowledge-staging.sutazai.com

# Run tests against staging
node docs/testing/newman_ci_integration.js
```

## Test Data and Fixtures

### API Test Data

The tests use predefined datasets:
- Chat messages for various scenarios
- Reasoning queries for cognitive testing
- Task descriptions for execution testing
- Error scenarios for negative testing

### File Upload Test Data

- Sample text files for document analysis
- Test images (base64 encoded 1x1 pixel PNG)
- Invalid file types for security testing

### Load Test Data

- Randomized chat messages
- Various reasoning query types
- Different task complexity levels
- User simulation with realistic usage patterns

## Monitoring and Alerting

### Test Result Monitoring

The test suite generates multiple output formats:
- **JSON**: Machine-readable results for monitoring systems
- **HTML**: Human-readable reports with visualizations  
- **JUnit XML**: Standard format for CI/CD systems
- **Prometheus metrics**: For monitoring dashboards

### Slack Integration

Configure Slack notifications for test results:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
node docs/testing/newman_ci_integration.js
```

### Dashboard Integration

Test results can be integrated with:
- Grafana dashboards for performance metrics
- Prometheus alerting for test failures
- Custom monitoring solutions via JSON/XML outputs

## Performance Baselines and SLAs

### Response Time SLAs

- **Health Endpoints**: < 1 second (95th percentile)
- **Chat Endpoints**: < 3 seconds (95th percentile)
- **Thinking Endpoints**: < 5 seconds (95th percentile)
- **Task Execution**: < 10 seconds (95th percentile)

### Throughput SLAs

- **Backend API**: 200+ requests/minute sustained
- **Concurrent Users**: 100+ users simultaneously  
- **Voice Sessions**: 50+ concurrent voice interactions
- **Knowledge Queries**: 30+ queries/minute

### Error Rate SLAs

- **Normal Load**: < 2% error rate
- **Peak Load**: < 5% error rate
- **Stress Conditions**: < 10% error rate
- **Recovery Time**: < 30 seconds after load reduction

## Troubleshooting

### Common Issues

1. **Services Not Ready**:
   - Increase wait times in scripts
   - Check Docker logs: `docker-compose logs [service-name]`
   - Verify port accessibility

2. **Test Timeouts**:
   - Increase timeout values in test configurations
   - Check system resource availability
   - Monitor network latency

3. **Authentication Errors**:
   - Verify JWT tokens if using secured endpoints
   - Check API key configurations
   - Validate environment variables

4. **Load Test Failures**:
   - Check system resources (CPU, memory)
   - Verify database connections
   - Monitor service logs during load tests

### Debugging Tips

- Use `--verbose` flags for detailed output
- Check service health endpoints before running tests
- Monitor system resources during test execution
- Review generated reports for failure patterns

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Include both positive and negative test cases
3. Add proper assertions and validations
4. Update documentation with new test scenarios
5. Ensure tests are idempotent and can run repeatedly
6. Include performance validations where appropriate

### Test Review Checklist

- [ ] Tests cover happy path scenarios
- [ ] Error handling is tested
- [ ] Performance assertions are included
- [ ] Tests are properly documented
- [ ] CI/CD integration is configured
- [ ] Test data is realistic and varied
- [ ] Cleanup is handled properly
- [ ] Tests can run independently

## Changelog

All testing suite changes are documented in `/opt/sutazaiapp/docs/CHANGELOG.md` under the Testing section.