/**
 * Integration Tests
 * 
 * Tests for workflows that involve multiple stores working together
 */

import { act, renderHook } from '@testing-library/react';
import { useVoiceStore } from '../voiceStore';
import { useTextInputStore } from '../textInputStore';
import { useStreamingStore } from '../streamingStore';
import { useConversationStore } from '../conversationStore';
import { useSidebarStore } from '../sidebarStore';
import { MessageType } from '../types';

// Mock WebSocket
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = WebSocket.CONNECTING;
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 100);
  }

  send(data) {
    // Simulate receiving a response
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage({
          data: JSON.stringify({
            type: 'chunk',
            content: 'Response chunk',
            timestamp: Date.now()
          })
        });
      }
    }, 50);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) this.onclose({ wasClean: true });
  }
}

global.WebSocket = MockWebSocket;
global.fetch = jest.fn();

describe('Store Integration Tests', () => {
  let voiceStore, textInputStore, streamingStore, conversationStore, sidebarStore;

  beforeEach(() => {
    const voiceHook = renderHook(() => useVoiceStore());
    const textInputHook = renderHook(() => useTextInputStore());
    const streamingHook = renderHook(() => useStreamingStore());
    const conversationHook = renderHook(() => useConversationStore());
    const sidebarHook = renderHook(() => useSidebarStore());

    voiceStore = voiceHook.result.current;
    textInputStore = textInputHook.result.current;
    streamingStore = streamingHook.result.current;
    conversationStore = conversationHook.result.current;
    sidebarStore = sidebarHook.result.current;

    // Reset Mocks
    jest.clearAllMocks();
  });

  describe('Complete Voice Workflow', () => {
    test('should handle complete voice recording and conversation flow', async () => {
      // Step 1: Create a conversation session
      let sessionId;
      act(() => {
        sessionId = conversationStore.createSession('Voice Test Session');
      });

      expect(conversationStore.currentSession).toBeTruthy();
      expect(conversationStore.currentSession.title).toBe('Voice Test Session');

      // Step 2: Simulate voice recording
      // Mock navigator.mediaDevices
      global.navigator.mediaDevices = {
        getUserMedia: jest.fn().MockResolvedValue({
          getTracks: () => [{ stop: jest.fn() }]
        })
      };

      await act(async () => {
        await voiceStore.startRecording();
      });

      expect(voiceStore.status).toBe('recording');

      act(() => {
        voiceStore.stopRecording();
      });

      expect(voiceStore.status).toBe('processing');

      // Step 3: Process voice and add to conversation
      const MockTranscript = 'Hello, this is a voice message';
      fetch.MockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          transcript: MockTranscript,
          confidence: 0.9
        })
      });

      let voiceResult;
      await act(async () => {
        // Set up audio blob for processing
        voiceStore.audioBlob = new Blob(['fake audio'], { type: 'audio/webm' });
        voiceStore.recordingDuration = 3.5;
        
        voiceResult = await voiceStore.processAudio();
      });

      expect(voiceResult).toBeTruthy();
      expect(voiceResult.transcript).toBe(MockTranscript);

      // Step 4: Add voice message to conversation
      let messageId;
      act(() => {
        messageId = conversationStore.addMessage(
          MockTranscript,
          MessageType.VOICE,
          { 
            audioBlob: voiceStore.audioBlob,
            duration: voiceStore.recordingDuration,
            confidence: voiceResult.confidence
          }
        );
      });

      expect(conversationStore.currentSession.messages).toHaveLength(1);
      expect(conversationStore.currentSession.messages[0].type).toBe(MessageType.VOICE);
      expect(conversationStore.currentSession.messages[0].content).toBe(MockTranscript);

      // Step 5: Apply filters in sidebar
      act(() => {
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      expect(sidebarStore.filteredSessions).toHaveLength(1);
      expect(sidebarStore.filteredSessions[0].id).toBe(sessionId);
    });
  });

  describe('Text Input and Streaming Workflow', () => {
    test('should handle complete text input and streaming response flow', async () => {
      // Step 1: Create session
      let sessionId;
      act(() => {
        sessionId = conversationStore.createSession();
      });

      // Step 2: Set up text input
      const userMessage = 'What is the weather like today?';
      act(() => {
        textInputStore.setText(userMessage);
      });

      expect(textInputStore.currentText).toBe(userMessage);
      expect(textInputStore.canSubmit()).toBe(true);

      // Step 3: Connect streaming
      await act(async () => {
        await streamingStore.connect('ws://localhost:8888/ws');
      });

      expect(streamingStore.isConnected()).toBe(true);

      // Step 4: Submit text and start streaming
      const MockSubmit = async (text) => {
        // Add user message to conversation
        conversationStore.addMessage(text, MessageType.TEXT);
        
        // Send via streaming
        streamingStore.sendMessage({
          type: 'text',
          content: text
        });
        
        return { success: true };
      };

      let submitResult;
      await act(async () => {
        submitResult = await textInputStore.submitText(MockSubmit);
      });

      expect(submitResult.success).toBe(true);
      expect(textInputStore.currentText).toBe(''); // Cleared after submit
      expect(textInputStore.history).toHaveLength(1);
      expect(conversationStore.currentSession.messages).toHaveLength(1);

      // Step 5: Handle streaming response
      await act(async () => {
        // Simulate receiving chunks
        streamingStore.handleMessage({
          type: 'chunk',
          content: 'The weather is ',
          timestamp: Date.now()
        });
      });

      await act(async () => {
        streamingStore.handleMessage({
          type: 'chunk', 
          content: 'sunny and warm today.',
          timestamp: Date.now()
        });
      });

      await act(async () => {
        streamingStore.handleMessage({
          type: 'complete'
        });
      });

      expect(streamingStore.currentResponse).toContain('The weather is sunny and warm today.');
      expect(streamingStore.status).toBe('complete');

      // Step 6: Add final response to conversation
      let responseMessageId;
      act(() => {
        responseMessageId = conversationStore.addMessage(
          streamingStore.currentResponse,
          MessageType.ASSISTANT,
          { streaming: true, chunks: streamingStore.chunks.length }
        );
      });

      expect(conversationStore.currentSession.messages).toHaveLength(2);
      expect(conversationStore.currentSession.messages[1].type).toBe(MessageType.ASSISTANT);
    });
  });

  describe('Search and Filter Integration', () => {
    test('should handle complex search and filtering scenarios', async () => {
      // Step 1: Create multiple sessions with different content
      const sessions = [];
      
      act(() => {
        // Session 1: Recent work conversation
        const workSessionId = conversationStore.createSession('Work Discussion');
        conversationStore.addMessage('Can you help me with the project?', MessageType.TEXT);
        conversationStore.addMessage('Sure, I can help!', MessageType.ASSISTANT);
        conversationStore.tagSession(workSessionId, ['work', 'project']);
        sessions.push(conversationStore.currentSession);

        // Session 2: Personal conversation
        const personalSessionId = conversationStore.createSession('Personal Chat');
        conversationStore.addMessage('What should I cook for dinner?', MessageType.TEXT);
        conversationStore.addMessage('How about pasta?', MessageType.ASSISTANT);
        conversationStore.tagSession(personalSessionId, ['personal', 'cooking']);
        sessions.push(conversationStore.currentSession);

        // Session 3: Technical discussion
        const techSessionId = conversationStore.createSession('API Integration');
        conversationStore.addMessage('How do I implement OAuth?', MessageType.TEXT);
        conversationStore.addMessage('Here are the steps...', MessageType.ASSISTANT);
        conversationStore.tagSession(techSessionId, ['technical', 'api']);
        sessions.push(conversationStore.currentSession);
      });

      expect(conversationStore.sessions).toHaveLength(3);

      // Step 2: Test text search
      act(() => {
        sidebarStore.setSearchQuery('project');
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      expect(sidebarStore.filteredSessions).toHaveLength(1);
      expect(sidebarStore.filteredSessions[0].title).toBe('Work Discussion');

      // Step 3: Test tag filtering
      act(() => {
        sidebarStore.clearFilters();
        sidebarStore.addSelectedTag('technical');
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      expect(sidebarStore.filteredSessions).toHaveLength(1);
      expect(sidebarStore.filteredSessions[0].title).toBe('API Integration');

      // Step 4: Test combined search and tag filter
      act(() => {
        sidebarStore.clearFilters();
        sidebarStore.setSearchQuery('help');
        sidebarStore.addSelectedTag('work');
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      expect(sidebarStore.filteredSessions).toHaveLength(1);
      expect(sidebarStore.filteredSessions[0].title).toBe('Work Discussion');

      // Step 5: Test conversation search function
      let searchResults;
      act(() => {
        searchResults = conversationStore.searchSessions('cook', {
          tags: ['personal']
        });
      });

      expect(searchResults).toHaveLength(1);
      expect(searchResults[0].title).toBe('Personal Chat');
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should handle errors gracefully across stores', async () => {
      // Test streaming connection error
      const MockWebSocketError = class extends MockWebSocket {
        constructor(url) {
          super(url);
          setTimeout(() => {
            this.readyState = WebSocket.CLOSED;
            if (this.onerror) this.onerror(new Error('Connection failed'));
            if (this.onclose) this.onclose({ wasClean: false });
          }, 100);
        }
      };

      global.WebSocket = MockWebSocketError;

      await act(async () => {
        await streamingStore.connect('ws://invalid-url');
      });

      expect(streamingStore.hasError()).toBe(true);
      expect(streamingStore.error).toContain('Connection failed');

      // Test voice processing error
      fetch.MockRejectedValueOnce(new Error('Voice API unavailable'));

      await act(async () => {
        voiceStore.audioBlob = new Blob(['test'], { type: 'audio/webm' });
        await voiceStore.processAudio();
      });

      expect(voiceStore.hasError()).toBe(true);
      expect(voiceStore.error).toContain('Voice API unavailable');

      // Test conversation persistence error
      const MockSetItem = jest.fn(() => {
        throw new Error('Storage quota exceeded');
      });
      global.localStorage.setItem = MockSetItem;

      await act(async () => {
        conversationStore.createSession('Test Session');
        await conversationStore.persistSessions();
      });

      expect(conversationStore.error).toContain('Failed to save');

      // Test error recovery
      act(() => {
        streamingStore.resetError();
        voiceStore.resetError();
        conversationStore.resetError();
      });

      expect(streamingStore.hasError()).toBe(false);
      expect(voiceStore.hasError()).toBe(false);
      expect(conversationStore.error).toBeNull();
    });
  });

  describe('State Synchronization', () => {
    test('should maintain consistency across related stores', async () => {
      // Create session and verify it appears in sidebar
      let sessionId;
      act(() => {
        sessionId = conversationStore.createSession('Sync Test');
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      expect(sidebarStore.filteredSessions).toHaveLength(1);
      expect(sidebarStore.filteredSessions[0].id).toBe(sessionId);

      // Add message and verify session is updated
      act(() => {
        conversationStore.addMessage('Test message', MessageType.TEXT);
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      const updatedSession = sidebarStore.filteredSessions.find(s => s.id === sessionId);
      expect(updatedSession.messages).toHaveLength(1);
      expect(updatedSession.metadata.messageCount).toBe(1);

      // Delete session and verify it's removed from sidebar
      act(() => {
        conversationStore.deleteSession(sessionId);
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      expect(sidebarStore.filteredSessions).toHaveLength(0);
      expect(conversationStore.sessions).toHaveLength(0);
    });
  });

  describe('Performance and Memory Management', () => {
    test('should handle large datasets efficiently', () => {
      // Create many sessions
      act(() => {
        for (let i = 0; i < 150; i++) {
          const sessionId = conversationStore.createSession(`Session ${i}`);
          
          // Add messages to some sessions
          if (i % 10 === 0) {
            for (let j = 0; j < 20; j++) {
              conversationStore.addMessage(`Message ${j}`, MessageType.TEXT);
            }
          }
        }
      });

      // Should respect max sessions limit
      expect(conversationStore.sessions.length).toBeLessThanOrEqual(
        conversationStore.maxSessions
      );

      // Test filtering performance
      const startTime = performance.now();
      
      act(() => {
        sidebarStore.setSearchQuery('Session 5');
        sidebarStore.applyFilters(conversationStore.sessions);
      });

      const endTime = performance.now();
      const filterTime = endTime - startTime;
      
      // Should complete within reasonable time (< 100ms)
      expect(filterTime).toBeLessThan(100);
      
      // Should return relevant results
      expect(sidebarStore.filteredSessions.length).toBeGreaterThan(0);
    });
  });

  describe('Cleanup and Resource Management', () => {
    test('should clean up resources properly', () => {
      // Set up connections and timers
      act(() => {
        streamingStore.connect('ws://localhost:8888');
        voiceStore.startRecording();
        textInputStore.setText('test');
      });

      // Cleanup all stores
      act(() => {
        voiceStore.clearAudio();
        streamingStore.cleanup();
        conversationStore.cleanup();
        sidebarStore.cleanup();
      });

      // Verify cleanup
      expect(voiceStore.audioBlob).toBeNull();
      expect(voiceStore.stream).toBeNull();
      expect(streamingStore.connection).toBeNull();
      expect(conversationStore.autoSaveTimer).toBeNull();
    });
  });
});