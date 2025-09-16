Rule 1: Real Implementation Only - No Fantasy Code
Requirement: Every line of code must work today, on current systems, with existing dependencies.
✅ Required Practices:

Use concrete, descriptive names: emailSender, userValidator, paymentProcessor
Import actual, installed libraries: import nodemailer from 'nodemailer'
Reference real APIs with documented endpoints and authentication
Write functions that have actual implementations, not placeholder stubs
Use environment variables that exist and are documented
Reference database tables, columns, and schemas that actually exist
Use file paths that are valid and accessible in target environments
Implement error handling for real, documented failure scenarios
Log to actual, configured log destinations (files, services, streams)
Pin dependencies to specific, tested version numbers
Use realistic test data that represents actual use cases
Reference configuration keys that are defined and documented
Call network endpoints that are reachable and monitored
Implement authentication mechanisms that are actually deployed
Use error codes and messages that correspond to real system responses
Reference documentation links that are valid and current
Handle timeouts and rate limits for actual service constraints
Use database connections that are configured and tested
Implement caching with actual cache stores (Redis, Memcached)
Reference monitoring and alerting systems that are operational
Use SSL certificates that are valid and not self-signed in production
Implement backup and recovery procedures that are tested
Use load balancers and reverse proxies that are configured
Reference container registries and image repositories that exist
Implement health checks that actually validate service status
Use message queues and event streams that are configured
Reference secrets management systems that are operational
Implement rate limiting using actual throttling mechanisms
Use CDN endpoints that are configured and accessible

#### Use Testing Protocol (Use Playwright MCP) for every testing aspect
```bash
npx playwright test
npx playwright test --ui
npx playwright test --project=chromium
npx playwright test example
npx playwright test --debug
npx playwright codegen

## Critical Instructions
- Start again from the beginning if you have to
- Make sure to follow these steps
- Make sure to fully test every change properly every step of the way

## Testing Requirements - Use Playwright MCP for Proper Testing
```bash
npx playwright test          # Runs the end-to-end tests
npx playwright test --ui     # Starts the interactive UI mode
npx playwright test --project=chromium  # Runs the tests only on Desktop Chrome
npx playwright test example  # Runs the tests in a specific file
npx playwright test --debug  # Runs the tests in debug mode
npx playwright codegen      # Auto generate tests with Codegen
🚫 Forbidden Practices:

Abstract service names: mailService, automationHandler, intelligentSystem
Placeholder comments: // TODO: add AI automation here, // magic happens
Fictional integrations: imports from non-existent packages or "future" APIs
Theoretical abstractions: code that assumes capabilities we don't have
Imaginary infrastructure: references to systems that don't exist
Mock implementations in production code paths
Hardcoded localhost or development URLs in production builds
References to non-existent database tables or columns
Placeholder data that doesn't represent real scenarios
Comments suggesting features that aren't implemented
Abstract interfaces without concrete implementations available
Theoretical error codes that don't exist in the system
References to undefined or missing configuration keys
Imports from local development paths not in production
Usage of experimental or unstable API versions
Assumptions about "future" system capabilities
References to non-existent environment variables
Calls to endpoints that don't exist or aren't documented
Usage of libraries not listed in dependency manifests
References to monitoring systems that aren't configured
Placeholder authentication or authorization mechanisms
File operations on paths that don't exist in target systems
Database queries against non-existent schemas
Integration with services that aren't accessible
Magic strings or numbers without defined constants
Assumptions about infinite resources or perfect networks
References to "eventual" or "planned" infrastructure
Theoretical scaling patterns not implemented
Imaginary performance characteristics
References to non-existent security policies
Placeholder encryption or hashing algorithms
Assumptions about zero-latency operations
References to ideal-world deployment scenarios
Theoretical data consistency models not implemented

Validation Criteria:

All imports resolve to actual installed packages
All API calls reference documented, accessible endpoints
All environment variables are defined in configuration
All functions contain working implementations
No TODOs referencing non-existent systems or capabilities
All database references point to existing tables and columns
All file paths are accessible in deployment environments
All network calls have proper error handling and timeouts
All configuration keys are documented in environment configs
All test scenarios use realistic, representative data
All dependencies are pinned to stable, tested versions
All external services have health checks and monitoring
All error messages reference real, documented error conditions
All logging destinations are configured and accessible
All documentation links resolve to valid, current resources
All authentication mechanisms are implemented and tested
All cache references point to configured cache stores
All monitoring and alerting integrations are operational
All rate limiting and timeout values match actual service limits
All database connections are properly configured and pooled
All SSL/TLS configurations use valid certificates
All backup procedures are documented and tested
All load balancing configurations are verified and functional
All container images exist in accessible registries
All health check endpoints return meaningful status information
All message queue configurations are tested and monitored
All secrets are stored in configured management systems
All throttling mechanisms are implemented and tested
All CDN configurations are verified and accessible
All CI/CD pipeline steps are functional and repeatable


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/
