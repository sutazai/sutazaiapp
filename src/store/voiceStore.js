/**
 * Voice Recording State Management
 * 
 * Handles voice input recording, processing, and audio blob storage
 * using Web Audio API and MediaRecorder.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { VoiceRecordingStatus } from './types';

// Audio constraints for optimal voice recording
const AUDIO_CONSTRAINTS = {
  audio: {
    sampleRate: 16000,
    channelCount: 1,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  }
};

// Supported audio formats with fallbacks
const AUDIO_FORMATS = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/wav'
];

export const useVoiceStore = create(
  immer((set, get) => ({
    // State
    status: VoiceRecordingStatus.IDLE,
    audioBlob: null,
    mediaRecorder: null,
    stream: null,
    audioChunks: [],
    transcript: null,
    error: null,
    isSupported: !!navigator.mediaDevices?.getUserMedia && !!window.MediaRecorder,
    recordingDuration: 0,
    maxRecordingDuration: 300, // 5 minutes in seconds
    recordingStartTime: null,
    
    // Actions
    startRecording: async () => {
      const state = get();
      
      if (!state.isSupported) {
        set((draft) => {
          draft.error = 'Voice recording is not supported in this browser';
          draft.status = VoiceRecordingStatus.ERROR;
        });
        return;
      }

      if (state.status === VoiceRecordingStatus.RECORDING) {
        return; // Already recording
      }

      try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia(AUDIO_CONSTRAINTS);
        
        // Find supported audio format
        const supportedFormat = AUDIO_FORMATS.find(format => 
          MediaRecorder.isTypeSupported(format)
        );
        
        if (!supportedFormat) {
          throw new Error('No supported audio format found');
        }

        // Create MediaRecorder
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: supportedFormat
        });

        const audioChunks = [];
        const recordingStartTime = Date.now();

        // Set up event handlers
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data);
            set((draft) => {
              draft.audioChunks = [...audioChunks];
            });
          }
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: supportedFormat });
          const duration = (Date.now() - recordingStartTime) / 1000;
          
          set((draft) => {
            draft.audioBlob = audioBlob;
            draft.status = VoiceRecordingStatus.IDLE;
            draft.recordingDuration = duration;
          });

          // Clean up stream
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.onerror = (event) => {
          console.error('MediaRecorder error:', event.error);
          set((draft) => {
            draft.error = `Recording error: ${event.error?.message || 'Unknown error'}`;
            draft.status = VoiceRecordingStatus.ERROR;
          });
          
          // Clean up
          stream.getTracks().forEach(track => track.stop());
        };

        // Start recording
        mediaRecorder.start(1000); // Collect data every second
        
        // Set up auto-stop timer
        const maxDuration = state.maxRecordingDuration * 1000;
        const autoStopTimer = setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            get().stopRecording();
          }
        }, maxDuration);

        set((draft) => {
          draft.mediaRecorder = mediaRecorder;
          draft.stream = stream;
          draft.status = VoiceRecordingStatus.RECORDING;
          draft.error = null;
          draft.audioChunks = audioChunks;
          draft.recordingStartTime = recordingStartTime;
          draft.autoStopTimer = autoStopTimer;
        });

        // Update duration every second
        const durationTimer = setInterval(() => {
          const currentState = get();
          if (currentState.status === VoiceRecordingStatus.RECORDING) {
            const elapsed = (Date.now() - recordingStartTime) / 1000;
            set((draft) => {
              draft.recordingDuration = elapsed;
            });
          } else {
            clearInterval(durationTimer);
          }
        }, 1000);

      } catch (error) {
        console.error('Error starting recording:', error);
        set((draft) => {
          draft.error = error.name === 'NotAllowedError' 
            ? 'Microphone access denied. Please allow microphone permissions.'
            : `Recording error: ${error.message}`;
          draft.status = VoiceRecordingStatus.ERROR;
        });
      }
    },

    stopRecording: () => {
      const state = get();
      
      if (state.mediaRecorder && state.status === VoiceRecordingStatus.RECORDING) {
        // Clear auto-stop timer
        if (state.autoStopTimer) {
          clearTimeout(state.autoStopTimer);
        }
        
        // Stop recording
        state.mediaRecorder.stop();
        
        set((draft) => {
          draft.status = VoiceRecordingStatus.PROCESSING;
        });
      }
    },

    processAudio: async (apiBaseUrl = 'http://localhost:10010') => {
      const state = get();
      
      if (!state.audioBlob) {
        set((draft) => {
          draft.error = 'No audio data to process';
          draft.status = VoiceRecordingStatus.ERROR;
        });
        return null;
      }

      set((draft) => {
        draft.status = VoiceRecordingStatus.PROCESSING;
        draft.error = null;
      });

      try {
        // Create FormData for API request
        const formData = new FormData();
        formData.append('audio', state.audioBlob, 'voice_input.webm');
        formData.append('duration', state.recordingDuration.toString());

        // Send to voice processing API
        const response = await fetch(`${apiBaseUrl}/jarvis/voice/process`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        set((draft) => {
          draft.transcript = result.transcript || '';
          draft.status = VoiceRecordingStatus.IDLE;
          draft.error = null;
        });

        return {
          transcript: result.transcript,
          confidence: result.confidence,
          duration: state.recordingDuration,
          audioBlob: state.audioBlob,
          ...result
        };

      } catch (error) {
        console.error('Error processing audio:', error);
        set((draft) => {
          draft.error = `Audio processing failed: ${error.message}`;
          draft.status = VoiceRecordingStatus.ERROR;
        });
        return null;
      }
    },

    clearAudio: () => {
      const state = get();
      
      // Clean up any active recording
      if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
      }
      
      if (state.autoStopTimer) {
        clearTimeout(state.autoStopTimer);
      }

      set((draft) => {
        draft.audioBlob = null;
        draft.mediaRecorder = null;
        draft.stream = null;
        draft.audioChunks = [];
        draft.transcript = null;
        draft.status = VoiceRecordingStatus.IDLE;
        draft.recordingDuration = 0;
        draft.recordingStartTime = null;
        draft.autoStopTimer = null;
      });
    },

    setError: (error) => {
      set((draft) => {
        draft.error = error;
        draft.status = VoiceRecordingStatus.ERROR;
      });
    },

    resetError: () => {
      set((draft) => {
        draft.error = null;
        if (draft.status === VoiceRecordingStatus.ERROR) {
          draft.status = VoiceRecordingStatus.IDLE;
        }
      });
    },

    // Utility actions
    isRecording: () => get().status === VoiceRecordingStatus.RECORDING,
    isProcessing: () => get().status === VoiceRecordingStatus.PROCESSING,
    hasAudio: () => !!get().audioBlob,
    hasError: () => !!get().error,
    
    getFormattedDuration: () => {
      const duration = get().recordingDuration;
      const minutes = Math.floor(duration / 60);
      const seconds = Math.floor(duration % 60);
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    },

    // Advanced features
    downloadAudio: () => {
      const state = get();
      if (!state.audioBlob) return;

      const url = URL.createObjectURL(state.audioBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `voice_recording_${new Date().toISOString().slice(0, 19)}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },

    playAudio: () => {
      const state = get();
      if (!state.audioBlob) return null;

      const audio = new Audio(URL.createObjectURL(state.audioBlob));
      audio.play().catch(console.error);
      return audio;
    }
  }))
);
