/**
 * Conversation Store Tests
 * 
 * Comprehensive tests for conversation history management
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { useConversationStore } from '../conversationStore';
import { MessageType } from '../types';

// Mock IndexedDB
const mockIndexedDB = {
  open: jest.fn(),
  result: {
    transaction: jest.fn(),
    objectStoreNames: { contains: jest.fn(() => false) }
  }
};

global.indexedDB = mockIndexedDB;

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn()
};
global.localStorage = mockLocalStorage;

describe('useConversationStore', () => {
  let store;
  const mockSession = {
    id: 'test-session-1',
    title: 'Test Conversation',
    messages: [
      {
        id: 'msg-1',
        type: MessageType.TEXT,
        content: 'Hello',
        timestamp: new Date('2024-01-01T10:00:00Z'),
        metadata: { sender: 'user' }
      },
      {
        id: 'msg-2',
        type: MessageType.ASSISTANT,
        content: 'Hi there!',
        timestamp: new Date('2024-01-01T10:01:00Z'),
        metadata: { sender: 'assistant' }
      }
    ],
    createdAt: new Date('2024-01-01T10:00:00Z'),
    updatedAt: new Date('2024-01-01T10:01:00Z'),
    tags: ['test', 'conversation'],
    metadata: {
      messageCount: 2,
      lastActivity: new Date('2024-01-01T10:01:00Z')
    }
  };

  beforeEach(() => {
    const { result } = renderHook(() => useConversationStore());
    store = result.current;
    
    // Reset mocks
    jest.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
  });

  describe('Initial State', () => {
    test('should have correct initial state', () => {
      expect(store.sessions).toEqual([]);
      expect(store.currentSessionId).toBeNull();
      expect(store.currentSession).toBeNull();
      expect(store.isLoading).toBe(false);
      expect(store.error).toBeNull();
      expect(store.maxSessions).toBe(100);
      expect(store.autoPersist).toBe(true);
    });
  });

  describe('createSession', () => {
    test('should create a new session with default title', () => {
      let sessionId;
      act(() => {
        sessionId = store.createSession();
      });

      expect(sessionId).toBeTruthy();
      expect(store.sessions).toHaveLength(1);
      expect(store.currentSessionId).toBe(sessionId);
      expect(store.currentSession).toBeTruthy();
      expect(store.currentSession.title).toContain('Conversation');
      expect(store.currentSession.messages).toEqual([]);
    });

    test('should create a new session with custom title', () => {
      const customTitle = 'My Custom Session';
      let sessionId;
      
      act(() => {
        sessionId = store.createSession(customTitle);
      });

      expect(store.sessions[0].title).toBe(customTitle);
    });

    test('should create session with metadata', () => {
      const metadata = { source: 'api', priority: 'high' };
      
      act(() => {
        store.createSession('Test Session', metadata);
      });

      expect(store.sessions[0].metadata.source).toBe('api');
      expect(store.sessions[0].metadata.priority).toBe('high');
    });

    test('should maintain max sessions limit', () => {
      // Set a low limit for testing
      act(() => {
        store.maxSessions = 3;
      });

      // Create more sessions than the limit
      act(() => {
        for (let i = 0; i < 5; i++) {
          store.createSession(`Session ${i}`);
        }
      });

      expect(store.sessions).toHaveLength(3);
    });
  });

  describe('setCurrentSession', () => {
    beforeEach(() => {
      act(() => {
        store.sessions = [mockSession];
      });
    });

    test('should set current session correctly', () => {
      act(() => {
        store.setCurrentSession(mockSession.id);
      });

      expect(store.currentSessionId).toBe(mockSession.id);
      expect(store.currentSession).toEqual(mockSession);
      expect(store.error).toBeNull();
    });

    test('should handle non-existent session', () => {
      act(() => {
        store.setCurrentSession('non-existent-id');
      });

      expect(store.currentSessionId).toBeNull();
      expect(store.error).toContain('Session not found');
    });
  });

  describe('addMessage', () => {
    beforeEach(() => {
      act(() => {
        store.createSession('Test Session');
      });
    });

    test('should add a text message', () => {
      let messageId;
      act(() => {
        messageId = store.addMessage('Hello world', MessageType.TEXT);
      });

      expect(messageId).toBeTruthy();
      expect(store.currentSession.messages).toHaveLength(1);
      expect(store.currentSession.messages[0].content).toBe('Hello world');
      expect(store.currentSession.messages[0].type).toBe(MessageType.TEXT);
    });

    test('should add message with metadata', () => {
      const metadata = { confidence: 0.95, language: 'en' };
      
      act(() => {
        store.addMessage('Hello', MessageType.TEXT, metadata);
      });

      expect(store.currentSession.messages[0].metadata.confidence).toBe(0.95);
      expect(store.currentSession.messages[0].metadata.language).toBe('en');
    });

    test('should update session metadata', () => {
      const initialUpdatedAt = store.currentSession.updatedAt;
      
      // Wait a moment to ensure timestamp difference
      setTimeout(() => {
        act(() => {
          store.addMessage('Test message', MessageType.TEXT);
        });

        expect(store.currentSession.metadata.messageCount).toBe(1);
        expect(store.currentSession.updatedAt).not.toEqual(initialUpdatedAt);
      }, 10);
    });

    test('should maintain message limit per session', () => {
      // Set a low limit for testing
      act(() => {
        store.maxMessagesPerSession = 3;
      });

      // Add more messages than the limit
      act(() => {
        for (let i = 0; i < 5; i++) {
          store.addMessage(`Message ${i}`, MessageType.TEXT);
        }
      });

      expect(store.currentSession.messages).toHaveLength(3);
      // Should keep the most recent messages
      expect(store.currentSession.messages[0].content).toBe('Message 2');
      expect(store.currentSession.messages[2].content).toBe('Message 4');
    });

    test('should auto-generate title from first message', () => {
      const longMessage = 'This is a very long message that should be truncated for the title';
      
      act(() => {
        store.addMessage(longMessage, MessageType.TEXT);
      });

      expect(store.currentSession.title).toBe('This is a very long message that should be truncat...');
    });

    test('should create session if none exists', () => {
      // Clear current session
      act(() => {
        store.currentSession = null;
        store.currentSessionId = null;
        store.sessions = [];
      });

      act(() => {
        store.addMessage('Hello', MessageType.TEXT);
      });

      expect(store.sessions).toHaveLength(1);
      expect(store.currentSession).toBeTruthy();
      expect(store.currentSession.messages).toHaveLength(1);
    });
  });

  describe('updateMessage', () => {
    let messageId;

    beforeEach(() => {
      act(() => {
        store.createSession('Test Session');
        messageId = store.addMessage('Original message', MessageType.TEXT);
      });
    });

    test('should update message content', () => {
      act(() => {
        store.updateMessage(messageId, { content: 'Updated message' });
      });

      expect(store.currentSession.messages[0].content).toBe('Updated message');
      expect(store.currentSession.messages[0].updatedAt).toBeTruthy();
    });

    test('should update message metadata', () => {
      act(() => {
        store.updateMessage(messageId, { 
          metadata: { ...store.currentSession.messages[0].metadata, edited: true }
        });
      });

      expect(store.currentSession.messages[0].metadata.edited).toBe(true);
    });
  });

  describe('deleteMessage', () => {
    let messageId;

    beforeEach(() => {
      act(() => {
        store.createSession('Test Session');
        messageId = store.addMessage('Test message', MessageType.TEXT);
        store.addMessage('Another message', MessageType.TEXT);
      });
    });

    test('should delete message correctly', () => {
      act(() => {
        store.deleteMessage(messageId);
      });

      expect(store.currentSession.messages).toHaveLength(1);
      expect(store.currentSession.messages[0].content).toBe('Another message');
      expect(store.currentSession.metadata.messageCount).toBe(1);
    });
  });

  describe('deleteSession', () => {
    beforeEach(() => {
      act(() => {
        store.sessions = [mockSession];
        store.currentSessionId = mockSession.id;
        store.currentSession = mockSession;
      });
    });

    test('should delete session and clear current if it was active', () => {
      act(() => {
        store.deleteSession(mockSession.id);
      });

      expect(store.sessions).toHaveLength(0);
      expect(store.currentSessionId).toBeNull();
      expect(store.currentSession).toBeNull();
    });

    test('should set new current session if available after deletion', () => {
      const anotherSession = { ...mockSession, id: 'session-2' };
      
      act(() => {
        store.sessions = [mockSession, anotherSession];
        store.deleteSession(mockSession.id);
      });

      expect(store.sessions).toHaveLength(1);
      expect(store.currentSessionId).toBe(anotherSession.id);
      expect(store.currentSession).toEqual(anotherSession);
    });
  });

  describe('clearAllSessions', () => {
    beforeEach(() => {
      act(() => {
        store.sessions = [mockSession];
        store.currentSessionId = mockSession.id;
        store.currentSession = mockSession;
      });
    });

    test('should clear all sessions and state', () => {
      act(() => {
        store.clearAllSessions();
      });

      expect(store.sessions).toEqual([]);
      expect(store.currentSessionId).toBeNull();
      expect(store.currentSession).toBeNull();
      expect(store.searchResults).toEqual([]);
    });
  });

  describe('searchSessions', () => {
    beforeEach(() => {
      const sessions = [
        mockSession,
        {
          ...mockSession,
          id: 'session-2',
          title: 'Another Conversation',
          messages: [
            {
              id: 'msg-3',
              content: 'Different content',
              type: MessageType.TEXT,
              timestamp: new Date(),
              metadata: {}
            }
          ],
          tags: ['different', 'tags']
        }
      ];
      
      act(() => {
        store.sessions = sessions;
      });
    });

    test('should search by title', () => {
      let results;
      act(() => {
        results = store.searchSessions('Test');
      });

      expect(results).toHaveLength(1);
      expect(results[0].title).toBe('Test Conversation');
    });

    test('should search by message content', () => {
      let results;
      act(() => {
        results = store.searchSessions('Hello');
      });

      expect(results).toHaveLength(1);
      expect(results[0].id).toBe(mockSession.id);
    });

    test('should search with filters', () => {
      let results;
      act(() => {
        results = store.searchSessions('', { 
          tags: ['test']
        });
      });

      expect(results).toHaveLength(1);
      expect(results[0].tags).toContain('test');
    });

    test('should clear search results when query is empty', () => {
      act(() => {
        store.searchSessions('');
      });

      expect(store.searchResults).toEqual([]);
      expect(store.isSearching).toBe(false);
    });
  });

  describe('tagSession', () => {
    beforeEach(() => {
      act(() => {
        store.sessions = [mockSession];
        store.currentSessionId = mockSession.id;
        store.currentSession = mockSession;
      });
    });

    test('should add tags to session', () => {
      act(() => {
        store.tagSession(mockSession.id, ['new-tag', 'another-tag']);
      });

      const session = store.sessions.find(s => s.id === mockSession.id);
      expect(session.tags).toContain('new-tag');
      expect(session.tags).toContain('another-tag');
      expect(session.tags).toContain('test'); // Original tags preserved
    });

    test('should not add duplicate tags', () => {
      act(() => {
        store.tagSession(mockSession.id, ['test', 'new-tag']);
      });

      const session = store.sessions.find(s => s.id === mockSession.id);
      const testTags = session.tags.filter(tag => tag === 'test');
      expect(testTags).toHaveLength(1);
    });
  });

  describe('Persistence', () => {
    test('should save to localStorage on persistSessions', async () => {
      act(() => {
        store.sessions = [mockSession];
      });

      await act(async () => {
        await store.persistSessions();
      });

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'sutazai_conversations',
        expect.stringContaining(mockSession.id)
      );
      expect(store.lastSavedAt).toBeTruthy();
    });

    test('should load from localStorage', async () => {
      const storedData = JSON.stringify({
        sessions: [mockSession],
        timestamp: new Date().toISOString(),
        version: '1.0'
      });
      
      mockLocalStorage.getItem.mockReturnValue(storedData);

      await act(async () => {
        await store.loadSessions();
      });

      expect(store.sessions).toHaveLength(1);
      expect(store.sessions[0].id).toBe(mockSession.id);
    });
  });

  describe('Utility Methods', () => {
    beforeEach(() => {
      act(() => {
        store.sessions = [mockSession];
      });
    });

    test('getSessionById should return correct session', () => {
      const session = store.getSessionById(mockSession.id);
      expect(session).toEqual(mockSession);
    });

    test('getSessionById should return null for non-existent session', () => {
      const session = store.getSessionById('non-existent');
      expect(session).toBeNull();
    });

    test('getAllTags should return unique tags', () => {
      const tags = store.getAllTags();
      expect(tags).toContain('test');
      expect(tags).toContain('conversation');
      expect(tags).toHaveLength(2);
    });

    test('getStats should return correct statistics', () => {
      const stats = store.getStats();
      expect(stats.totalSessions).toBe(1);
      expect(stats.totalMessages).toBe(2);
    });
  });

  describe('Import/Export', () => {
    test('should export sessions correctly', () => {
      act(() => {
        store.sessions = [mockSession];
      });

      const exported = store.exportSessions();
      const parsed = JSON.parse(exported);

      expect(parsed.sessions).toHaveLength(1);
      expect(parsed.sessions[0].id).toBe(mockSession.id);
      expect(parsed.version).toBe('1.0');
    });

    test('should import sessions correctly', () => {
      const importData = JSON.stringify({
        sessions: [mockSession],
        version: '1.0'
      });

      let result;
      act(() => {
        result = store.importSessions(importData);
      });

      expect(result).toBe(true);
      expect(store.sessions).toHaveLength(1);
      expect(store.sessions[0].id).toBe(mockSession.id);
    });

    test('should handle invalid import data', () => {
      let result;
      act(() => {
        result = store.importSessions('invalid json');
      });

      expect(result).toBe(false);
      expect(store.error).toContain('Import failed');
    });
  });
});