// Cypress E2E Tests for Jarvis Interface
// File: cypress/e2e/jarvis_interface.cy.js

describe('Jarvis Interface E2E Tests', () => {
  const BASE_URL = 'http://localhost:10011'; // Frontend URL
  const API_URL = 'http://localhost:10010'; // Backend API URL
  
  beforeEach(() => {
    // Visit the main Jarvis interface
    cy.visit(BASE_URL);
    
    // Wait for the page to load and check for essential elements
    cy.get('body', { timeout: 10000 }).should('be.visible');
  });

  describe('Voice Interface Tests', () => {
    it('should display voice interface components', () => {
      // Check for voice-related UI elements
      cy.get('[data-testid="voice-input-button"]', { timeout: 5000 })
        .should('exist');
      
      cy.get('[data-testid="microphone-button"]', { timeout: 5000 })
        .should('exist');
      
      // Check for voice status indicators
      cy.get('[data-testid="voice-status"]', { timeout: 5000 })
        .should('contain.text', 'Ready');
    });

    it('should handle microphone permissions', () => {
      // Test microphone button click
      cy.get('[data-testid="microphone-button"]')
        .click();
      
      // Check for permission request or activation
      cy.get('[data-testid="voice-status"]')
        .should('contain.text', 'Listening', { timeout: 2000 });
    });

    it('should process voice input simulation', () => {
      // Simulate voice input by triggering the voice processing
      cy.window().then((win) => {
        // Trigger voice processing simulation
        win.postMessage({
          type: 'VOICE_INPUT',
          data: { transcript: 'Hello Jarvis, what can you do?' }
        }, '*');
      });

      // Check for response processing
      cy.get('[data-testid="voice-response"]', { timeout: 5000 })
        .should('be.visible');
    });
  });

  describe('Text Chat Interface Tests', () => {
    it('should allow text input and send messages', () => {
      const testMessage = 'Hello Jarvis, can you help me with system optimization?';
      
      // Find chat input field
      cy.get('[data-testid="chat-input"]', { timeout: 5000 })
        .should('be.visible')
        .type(testMessage);
      
      // Send the message
      cy.get('[data-testid="send-button"]')
        .click();
      
      // Check for message in chat history
      cy.get('[data-testid="chat-messages"]')
        .should('contain.text', testMessage);
    });

    it('should display Jarvis responses', () => {
      const testMessage = 'What are your capabilities?';
      
      cy.get('[data-testid="chat-input"]')
        .type(testMessage);
      
      cy.get('[data-testid="send-button"]')
        .click();
      
      // Wait for Jarvis response
      cy.get('[data-testid="jarvis-response"]', { timeout: 10000 })
        .should('be.visible')
        .and('contain.text', 'I can help you');
    });

    it('should handle streaming responses', () => {
      const testMessage = 'Explain artificial intelligence in detail';
      
      cy.get('[data-testid="chat-input"]')
        .type(testMessage);
      
      cy.get('[data-testid="send-button"]')
        .click();
      
      // Check for streaming indicator
      cy.get('[data-testid="typing-indicator"]', { timeout: 2000 })
        .should('be.visible');
      
      // Wait for complete response
      cy.get('[data-testid="jarvis-response"]', { timeout: 15000 })
        .should('be.visible')
        .and('have.length.greaterThan', 0);
    });
  });

  describe('File Upload Tests', () => {
    it('should display file upload interface', () => {
      // Check for file upload area
      cy.get('[data-testid="file-upload-area"]', { timeout: 5000 })
        .should('be.visible');
      
      cy.get('[data-testid="upload-button"]')
        .should('exist');
    });

    it('should handle text file uploads', () => {
      // Create a test file
      const fileName = 'test-document.txt';
      const fileContent = 'This is a test document for Jarvis to analyze.';
      
      cy.get('[data-testid="file-upload-input"]')
        .selectFile({
          contents: Cypress.Buffer.from(fileContent),
          fileName: fileName,
          mimeType: 'text/plain',
        });
      
      // Check for file upload confirmation
      cy.get('[data-testid="upload-status"]')
        .should('contain.text', fileName);
      
      // Process the uploaded file
      cy.get('[data-testid="process-file-button"]')
        .click();
      
      // Wait for processing result
      cy.get('[data-testid="file-analysis-result"]', { timeout: 10000 })
        .should('be.visible');
    });

    it('should handle image file uploads', () => {
      const fileName = 'test-image.png';
      
      // Upload a test image (base64 encoded 1x1 pixel PNG)
      const imageData = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
      
      cy.get('[data-testid="file-upload-input"]')
        .selectFile({
          contents: Cypress.Buffer.from(imageData, 'base64'),
          fileName: fileName,
          mimeType: 'image/png',
        });
      
      cy.get('[data-testid="upload-status"]')
        .should('contain.text', fileName);
      
      cy.get('[data-testid="process-file-button"]')
        .click();
      
      cy.get('[data-testid="image-analysis-result"]', { timeout: 10000 })
        .should('be.visible');
    });

    it('should reject unsupported file types', () => {
      const fileName = 'malicious-file.exe';
      const fileContent = 'This should be rejected';
      
      cy.get('[data-testid="file-upload-input"]')
        .selectFile({
          contents: Cypress.Buffer.from(fileContent),
          fileName: fileName,
          mimeType: 'application/octet-stream',
        });
      
      // Check for error message
      cy.get('[data-testid="upload-error"]', { timeout: 5000 })
        .should('be.visible')
        .and('contain.text', 'not supported');
    });
  });

  describe('Real-time Features Tests', () => {
    it('should establish WebSocket connection', () => {
      // Check for WebSocket connection status
      cy.get('[data-testid="connection-status"]', { timeout: 5000 })
        .should('contain.text', 'Connected');
    });

    it('should handle real-time updates', () => {
      // Trigger an action that should produce real-time updates
      cy.get('[data-testid="chat-input"]')
        .type('Monitor system metrics in real-time');
      
      cy.get('[data-testid="send-button"]')
        .click();
      
      // Check for real-time metric updates
      cy.get('[data-testid="realtime-metrics"]', { timeout: 10000 })
        .should('be.visible');
    });

    it('should handle connection interruption gracefully', () => {
      // Simulate connection interruption
      cy.window().then((win) => {
        if (win.websocket) {
          win.websocket.close();
        }
      });
      
      // Check for reconnection attempt
      cy.get('[data-testid="connection-status"]', { timeout: 5000 })
        .should('contain.text', 'Reconnecting');
    });
  });

  describe('Accessibility Tests', () => {
    it('should be keyboard navigable', () => {
      // Test tab navigation
      cy.get('body').tab();
      cy.focused().should('have.attr', 'data-testid', 'chat-input');
      
      cy.focused().tab();
      cy.focused().should('have.attr', 'data-testid', 'send-button');
      
      cy.focused().tab();
      cy.focused().should('have.attr', 'data-testid', 'microphone-button');
    });

    it('should have proper ARIA labels', () => {
      cy.get('[data-testid="chat-input"]')
        .should('have.attr', 'aria-label', 'Chat input field');
      
      cy.get('[data-testid="send-button"]')
        .should('have.attr', 'aria-label', 'Send message');
      
      cy.get('[data-testid="microphone-button"]')
        .should('have.attr', 'aria-label', 'Start voice input');
    });

    it('should support screen readers', () => {
      // Check for screen reader announcements
      cy.get('[aria-live="polite"]', { timeout: 5000 })
        .should('exist');
      
      // Trigger an action and check for announcement
      cy.get('[data-testid="send-button"]').click();
      
      cy.get('[aria-live="polite"]')
        .should('contain.text', 'Message sent');
    });
  });

  describe('Responsiveness Tests', () => {
    const viewports = [
      { width: 320, height: 568 },   // Mobile
      { width: 768, height: 1024 },  // Tablet
      { width: 1024, height: 768 },  // Desktop
      { width: 1920, height: 1080 }  // Large Desktop
    ];

    viewports.forEach((viewport) => {
      it(`should work correctly at ${viewport.width}x${viewport.height}`, () => {
        cy.viewport(viewport.width, viewport.height);
        
        // Check that essential elements are visible and functional
        cy.get('[data-testid="chat-input"]')
          .should('be.visible');
        
        cy.get('[data-testid="send-button"]')
          .should('be.visible');
        
        // Test basic functionality
        cy.get('[data-testid="chat-input"]')
          .type('Test responsiveness');
        
        cy.get('[data-testid="send-button"]')
          .click();
        
        cy.get('[data-testid="chat-messages"]')
          .should('contain.text', 'Test responsiveness');
      });
    });
  });

  describe('Performance Tests', () => {
    it('should load within acceptable time', () => {
      // Measure page load time
      cy.window().then((win) => {
        const loadTime = win.performance.timing.loadEventEnd - win.performance.timing.navigationStart;
        expect(loadTime).to.be.lessThan(5000); // 5 seconds
      });
    });

    it('should handle multiple rapid requests', () => {
      const messages = [
        'What is AI?',
        'How do neural networks work?',
        'Explain machine learning',
        'What are the benefits of automation?',
        'How can I optimize performance?'
      ];
      
      // Send multiple messages rapidly
      messages.forEach((message, index) => {
        cy.get('[data-testid="chat-input"]')
          .clear()
          .type(message);
        
        cy.get('[data-testid="send-button"]')
          .click();
        
        // Brief wait to avoid overwhelming the system
        cy.wait(500);
      });
      
      // Check that all messages appear in chat
      messages.forEach((message) => {
        cy.get('[data-testid="chat-messages"]')
          .should('contain.text', message);
      });
    });
  });

  describe('Error Handling Tests', () => {
    it('should handle API errors gracefully', () => {
      // Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test API failure
      cy.intercept('POST', `${API_URL}/chat`, {
        statusCode: 500,
        body: { error: 'Internal Server Error' }
      });
      
      cy.get('[data-testid="chat-input"]')
        .type('This should trigger an error');
      
      cy.get('[data-testid="send-button"]')
        .click();
      
      // Check for error message display
      cy.get('[data-testid="error-message"]', { timeout: 5000 })
        .should('be.visible')
        .and('contain.text', 'error');
    });

    it('should handle network disconnection', () => {
      // Simulate offline condition
      cy.visit(BASE_URL, {
        onBeforeLoad: (win) => {
          Object.defineProperty(win.navigator, 'onLine', {
            writable: true,
            value: false
          });
        }
      });
      
      cy.get('[data-testid="offline-indicator"]', { timeout: 5000 })
        .should('be.visible');
    });
  });

  describe('Security Tests', () => {
    it('should sanitize user input', () => {
      const maliciousInput = '<script>alert("XSS")</script>';
      
      cy.get('[data-testid="chat-input"]')
        .type(maliciousInput);
      
      cy.get('[data-testid="send-button"]')
        .click();
      
      // Check that script was sanitized
      cy.get('[data-testid="chat-messages"]')
        .should('contain.text', '&lt;script&gt;')
        .and('not.contain.html', '<script>');
    });

    it('should prevent harmful file uploads', () => {
      // Test with various potentially harmful file types
      const harmfulFiles = [
        { name: 'virus.exe', type: 'application/octet-stream' },
        { name: 'script.js', type: 'application/javascript' },
        { name: 'macro.docm', type: 'application/vnd.ms-word.document.macroEnabled.12' }
      ];
      
      harmfulFiles.forEach(file => {
        cy.get('[data-testid="file-upload-input"]')
          .selectFile({
            contents: Cypress.Buffer.from('harmful content'),
            fileName: file.name,
            mimeType: file.type,
          });
        
        cy.get('[data-testid="upload-error"]')
          .should('be.visible')
          .and('contain.text', 'not allowed');
      });
    });
  });
});

// Configuration for Cypress tests
// File: cypress.config.js (to be placed in project root)
const cypressConfig = {
  e2e: {
    baseUrl: 'http://localhost:10011',
    viewportWidth: 1280,
    viewportHeight: 720,
    video: true,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 10000,
    requestTimeout: 15000,
    responseTimeout: 15000,
    supportFile: 'cypress/support/e2e.js',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    setupNodeEvents(on, config) {
      // Test result reporting
      on('after:run', (results) => {
        console.log('Test Results Summary:');
        console.log(`Total Tests: ${results.totalTests}`);
        console.log(`Passed: ${results.totalPassed}`);
        console.log(`Failed: ${results.totalFailed}`);
        console.log(`Duration: ${results.totalDuration}ms`);
      });

      // Custom commands for Jarvis testing
      on('task', {
        log(message) {
          console.log(message);
          return null;
        },
        
        // Custom task to simulate voice input
        simulateVoiceInput(transcript) {
          // Implementation would depend on your voice interface
          console.log(`Simulating voice input: ${transcript}`);
          return { success: true, transcript };
        }
      });
      
      return config;
    }
  },
  component: {
    devServer: {
      framework: 'react', // or whatever framework you're using
      bundler: 'webpack'   // or vite
    }
  }
};

// Export the config if this was a separate file
// module.exports = cypressConfig;