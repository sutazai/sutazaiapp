/**
 * Voice Store Tests
 * 
 * Comprehensive tests for voice recording state management
 */

import { act, renderHook } from '@testing-library/react';
import { useVoiceStore } from '../voiceStore';

// Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test MediaRecorder and related APIs
global.MediaRecorder = class Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestMediaRecorder {
  constructor(stream, options) {
    this.stream = stream;
    this.options = options;
    this.state = 'inactive';
    this.ondataavailable = null;
    this.onstop = null;
    this.onerror = null;
  }

  start(interval) {
    this.state = 'recording';
    setTimeout(() => {
      if (this.ondataavailable) {
        this.ondataavailable({ data: new Blob(['test audio'], { type: 'audio/webm' }) });
      }
    }, 100);
  }

  stop() {
    this.state = 'inactive';
    if (this.onstop) {
      setTimeout(this.onstop, 50);
    }
  }
};

global.navigator.mediaDevices = {
  getUserMedia: jest.fn().Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResolvedValue({
    getTracks: () => [{ stop: jest.fn() }]
  })
};

// Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test fetch for API calls
global.fetch = jest.fn();

describe('useVoiceStore', () => {
  let store;

  beforeEach(() => {
    const { result } = renderHook(() => useVoiceStore());
    store = result.current;
    
    // Reset Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Tests
    jest.clearAllRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Tests();
    navigator.mediaDevices.getUserMedia.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResolvedValue({
      getTracks: () => [{ stop: jest.fn() }]
    });
  });

  describe('Initial State', () => {
    test('should have correct initial state', () => {
      expect(store.status).toBe('idle');
      expect(store.audioBlob).toBeNull();
      expect(store.mediaRecorder).toBeNull();
      expect(store.stream).toBeNull();
      expect(store.audioChunks).toEqual([]);
      expect(store.transcript).toBeNull();
      expect(store.error).toBeNull();
      expect(store.recordingDuration).toBe(0);
      expect(store.isSupported).toBe(true); // True in test environment
    });
  });

  describe('startRecording', () => {
    test('should start recording successfully', async () => {
      await act(async () => {
        await store.startRecording();
      });

      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      expect(store.status).toBe('recording');
      expect(store.error).toBeNull();
    });

    test('should handle microphone access denied', async () => {
      navigator.mediaDevices.getUserMedia.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestRejectedValueOnce(
        new Error('NotAllowedError')
      );

      await act(async () => {
        await store.startRecording();
      });

      expect(store.status).toBe('error');
      expect(store.error).toContain('access denied');
    });

    test('should not start if already recording', async () => {
      // First start
      await act(async () => {
        await store.startRecording();
      });

      const firstCallCount = navigator.mediaDevices.getUserMedia.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.calls.length;

      // Try to start again
      await act(async () => {
        await store.startRecording();
      });

      // Should not call getUserMedia again
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(firstCallCount);
    });

    test('should handle unsupported browser', async () => {
      const originalIsSupported = store.isSupported;
      
      // Temporarily make it unsupported
      act(() => {
        store.isSupported = false;
        store.setError = jest.fn();
      });

      await act(async () => {
        await store.startRecording();
      });

      expect(store.status).toBe('error');
      
      // Restore
      act(() => {
        store.isSupported = originalIsSupported;
      });
    });
  });

  describe('stopRecording', () => {
    test('should stop recording successfully', async () => {
      // Start recording first
      await act(async () => {
        await store.startRecording();
      });

      act(() => {
        store.stopRecording();
      });

      expect(store.status).toBe('processing');
    });

    test('should handle stop when not recording', () => {
      act(() => {
        store.stopRecording();
      });

      // Should not crash or change state inappropriately
      expect(store.status).toBe('idle');
    });
  });

  describe('processAudio', () => {
    beforeEach(() => {
      // Set up audio blob
      act(() => {
        store.audioBlob = new Blob(['test'], { type: 'audio/webm' });
        store.recordingDuration = 5.5;
      });
    });

    test('should process audio successfully', async () => {
      const Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResponse = {
        transcript: 'Hello world',
        confidence: 0.95
      };

      fetch.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResponse)
      });

      let result;
      await act(async () => {
        result = await store.processAudio('http://localhost:8888');
      });

      expect(fetch).toHaveBeenCalledWith('http://localhost:8888/jarvis/voice/process', {
        method: 'POST',
        body: expect.any(FormData)
      });
      
      expect(store.transcript).toBe('Hello world');
      expect(store.status).toBe('idle');
      expect(result.transcript).toBe('Hello world');
      expect(result.confidence).toBe(0.95);
    });

    test('should handle API error', async () => {
      fetch.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResolvedValueOnce({
        ok: false,
        status: 500
      });

      let result;
      await act(async () => {
        result = await store.processAudio('http://localhost:8888');
      });

      expect(store.status).toBe('error');
      expect(store.error).toContain('processing failed');
      expect(result).toBeNull();
    });

    test('should handle network error', async () => {
      fetch.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestRejectedValueOnce(new Error('Network error'));

      let result;
      await act(async () => {
        result = await store.processAudio('http://localhost:8888');
      });

      expect(store.status).toBe('error');
      expect(store.error).toContain('Network error');
      expect(result).toBeNull();
    });

    test('should handle missing audio blob', async () => {
      act(() => {
        store.audioBlob = null;
      });

      let result;
      await act(async () => {
        result = await store.processAudio('http://localhost:8888');
      });

      expect(store.status).toBe('error');
      expect(store.error).toContain('No audio data');
      expect(result).toBeNull();
    });
  });

  describe('clearAudio', () => {
    test('should clear all audio data', async () => {
      // Set up some data first
      await act(async () => {
        await store.startRecording();
      });

      act(() => {
        store.audioBlob = new Blob(['test'], { type: 'audio/webm' });
        store.transcript = 'test transcript';
        store.recordingDuration = 10;
      });

      act(() => {
        store.clearAudio();
      });

      expect(store.audioBlob).toBeNull();
      expect(store.transcript).toBeNull();
      expect(store.recordingDuration).toBe(0);
      expect(store.status).toBe('idle');
      expect(store.audioChunks).toEqual([]);
    });
  });

  describe('Error Handling', () => {
    test('should set error correctly', () => {
      const errorMessage = 'Test error';
      
      act(() => {
        store.setError(errorMessage);
      });

      expect(store.error).toBe(errorMessage);
      expect(store.status).toBe('error');
    });

    test('should reset error correctly', () => {
      // Set error first
      act(() => {
        store.setError('Test error');
      });

      act(() => {
        store.resetError();
      });

      expect(store.error).toBeNull();
      expect(store.status).toBe('idle');
    });
  });

  describe('Utility Methods', () => {
    test('isRecording should return correct status', () => {
      expect(store.isRecording()).toBe(false);
      
      act(() => {
        store.status = 'recording';
      });
      
      expect(store.isRecording()).toBe(true);
    });

    test('isProcessing should return correct status', () => {
      expect(store.isProcessing()).toBe(false);
      
      act(() => {
        store.status = 'processing';
      });
      
      expect(store.isProcessing()).toBe(true);
    });

    test('hasAudio should return correct status', () => {
      expect(store.hasAudio()).toBe(false);
      
      act(() => {
        store.audioBlob = new Blob(['test'], { type: 'audio/webm' });
      });
      
      expect(store.hasAudio()).toBe(true);
    });

    test('hasError should return correct status', () => {
      expect(store.hasError()).toBe(false);
      
      act(() => {
        store.error = 'Test error';
      });
      
      expect(store.hasError()).toBe(true);
    });

    test('getFormattedDuration should format correctly', () => {
      act(() => {
        store.recordingDuration = 65; // 1 minute 5 seconds
      });
      
      expect(store.getFormattedDuration()).toBe('1:05');
      
      act(() => {
        store.recordingDuration = 5; // 5 seconds
      });
      
      expect(store.getFormattedDuration()).toBe('0:05');
    });
  });

  describe('Advanced Features', () => {
    test('downloadAudio should create download link', () => {
      // Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test URL.createObjectURL
      global.URL.createObjectURL = jest.fn(() => 'blob:url');
      global.URL.revokeObjectURL = jest.fn();
      
      // Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test document methods
      const Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestLink = {
        href: '',
        download: '',
        click: jest.fn()
      };
      document.createElement = jest.fn(() => Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestLink);
      document.body.appendChild = jest.fn();
      document.body.removeChild = jest.fn();

      act(() => {
        store.audioBlob = new Blob(['test'], { type: 'audio/webm' });
      });

      act(() => {
        store.downloadAudio();
      });

      expect(URL.createObjectURL).toHaveBeenCalledWith(store.audioBlob);
      expect(Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestLink.click).toHaveBeenCalled();
      expect(document.body.appendChild).toHaveBeenCalledWith(Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestLink);
      expect(document.body.removeChild).toHaveBeenCalledWith(Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestLink);
    });

    test('playAudio should create audio element', () => {
      global.URL.createObjectURL = jest.fn(() => 'blob:url');
      
      const Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestAudio = {
        play: jest.fn().Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestResolvedValue()
      };
      global.Audio = jest.fn(() => Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestAudio);

      act(() => {
        store.audioBlob = new Blob(['test'], { type: 'audio/webm' });
      });

      let audioElement;
      act(() => {
        audioElement = store.playAudio();
      });

      expect(Audio).toHaveBeenCalledWith('blob:url');
      expect(Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestAudio.play).toHaveBeenCalled();
      expect(audioElement).toBe(Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestAudio);
    });

    test('playAudio should return null without audio', () => {
      let result;
      act(() => {
        result = store.playAudio();
      });

      expect(result).toBeNull();
    });
  });
});
