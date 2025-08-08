/**
 * Text Input State Management
 * 
 * Handles text input submission, history management, and form state
 * with validation and error handling.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

// Default configuration
const DEFAULT_CONFIG = {
  maxHistory: 50,
  maxInputLength: 10000,
  autoTrim: true,
  debounceMs: 300
};

export const useTextInputStore = create(
  immer((set, get) => ({
    // State
    currentText: '',
    isSubmitting: false,
    history: [],
    historyIndex: -1,
    error: null,
    maxHistory: DEFAULT_CONFIG.maxHistory,
    maxInputLength: DEFAULT_CONFIG.maxInputLength,
    validationErrors: [],
    isDirty: false,
    lastSubmittedAt: null,
    submissionCount: 0,
    
    // Debounce state
    debounceTimer: null,

    // Actions
    setText: (text) => {
      const state = get();
      
      // Clear previous debounce
      if (state.debounceTimer) {
        clearTimeout(state.debounceTimer);
      }

      // Validate and process text
      const processedText = state.maxInputLength > 0 
        ? text.slice(0, state.maxInputLength)
        : text;

      set((draft) => {
        draft.currentText = processedText;
        draft.isDirty = processedText !== '';
        draft.historyIndex = -1; // Reset history navigation
        draft.validationErrors = [];
        
        // Clear errors when user starts typing
        if (draft.error && processedText.trim()) {
          draft.error = null;
        }
      });

      // Set debounced validation
      const debounceTimer = setTimeout(() => {
        get().validateInput(processedText);
      }, DEFAULT_CONFIG.debounceMs);

      set((draft) => {
        draft.debounceTimer = debounceTimer;
      });
    },

    clearText: () => {
      const state = get();
      
      if (state.debounceTimer) {
        clearTimeout(state.debounceTimer);
      }

      set((draft) => {
        draft.currentText = '';
        draft.isDirty = false;
        draft.historyIndex = -1;
        draft.validationErrors = [];
        draft.error = null;
        draft.debounceTimer = null;
      });
    },

    submitText: async (onSubmit, options = {}) => {
      const state = get();
      const {
        autoTrim = DEFAULT_CONFIG.autoTrim,
        validateBeforeSubmit = true,
        addToHistory = true
      } = options;
      
      let textToSubmit = state.currentText;
      
      if (autoTrim) {
        textToSubmit = textToSubmit.trim();
      }

      // Early validation
      if (!textToSubmit) {
        set((draft) => {
          draft.error = 'Please enter a message';
        });
        return false;
      }

      if (state.isSubmitting) {
        set((draft) => {
          draft.error = 'Already submitting, please wait';
        });
        return false;
      }

      // Validate if enabled
      if (validateBeforeSubmit) {
        const isValid = get().validateInput(textToSubmit);
        if (!isValid) {
          return false;
        }
      }

      set((draft) => {
        draft.isSubmitting = true;
        draft.error = null;
      });

      try {
        // Call the submission handler
        const result = await onSubmit(textToSubmit);
        
        // Add to history if successful and enabled
        if (addToHistory) {
          get().addToHistory(textToSubmit);
        }

        // Clear input after successful submission
        get().clearText();
        
        set((draft) => {
          draft.isSubmitting = false;
          draft.lastSubmittedAt = new Date();
          draft.submissionCount += 1;
        });

        return result;

      } catch (error) {
        console.error('Text submission error:', error);
        set((draft) => {
          draft.error = error.message || 'Failed to submit message';
          draft.isSubmitting = false;
        });
        return false;
      }
    },

    addToHistory: (text) => {
      if (!text || !text.trim()) return;

      set((draft) => {
        // Remove duplicate if it exists
        const filtered = draft.history.filter(item => item.text !== text.trim());
        
        // Add new item to beginning
        draft.history = [
          {
            text: text.trim(),
            timestamp: new Date(),
            id: Date.now() + Math.random()
          },
          ...filtered
        ].slice(0, draft.maxHistory);
        
        draft.historyIndex = -1;
      });
    },

    getFromHistory: (index) => {
      const state = get();
      if (index >= 0 && index < state.history.length) {
        const historyItem = state.history[index];
        set((draft) => {
          draft.currentText = historyItem.text;
          draft.historyIndex = index;
          draft.isDirty = true;
        });
        return historyItem.text;
      }
      return null;
    },

    navigateHistory: (direction) => {
      const state = get();
      let newIndex = state.historyIndex;

      if (direction === 'up') {
        // Navigate to older entries
        newIndex = Math.min(newIndex + 1, state.history.length - 1);
      } else if (direction === 'down') {
        // Navigate to newer entries
        newIndex = Math.max(newIndex - 1, -1);
      }

      if (newIndex !== state.historyIndex) {
        if (newIndex === -1) {
          // Back to current input
          set((draft) => {
            draft.currentText = '';
            draft.historyIndex = -1;
            draft.isDirty = false;
          });
        } else {
          // Set to history item
          const historyItem = state.history[newIndex];
          set((draft) => {
            draft.currentText = historyItem.text;
            draft.historyIndex = newIndex;
            draft.isDirty = true;
          });
        }
      }
    },

    clearHistory: () => {
      set((draft) => {
        draft.history = [];
        draft.historyIndex = -1;
      });
    },

    searchHistory: (query) => {
      const state = get();
      if (!query.trim()) return state.history;

      const searchTerm = query.toLowerCase();
      return state.history.filter(item => 
        item.text.toLowerCase().includes(searchTerm)
      );
    },

    deleteHistoryItem: (id) => {
      set((draft) => {
        draft.history = draft.history.filter(item => item.id !== id);
        draft.historyIndex = -1;
      });
    },

    setError: (error) => {
      set((draft) => {
        draft.error = error;
      });
    },

    resetError: () => {
      set((draft) => {
        draft.error = null;
        draft.validationErrors = [];
      });
    },

    // Validation
    validateInput: (text = null) => {
      const textToValidate = text || get().currentText;
      const errors = [];

      // Required field validation
      if (!textToValidate.trim()) {
        errors.push('Message cannot be empty');
      }

      // Length validation
      if (textToValidate.length > get().maxInputLength) {
        errors.push(`Message exceeds maximum length of ${get().maxInputLength} characters`);
      }

      // Custom validation rules
      const state = get();
      if (state.customValidation) {
        try {
          const customErrors = state.customValidation(textToValidate);
          if (Array.isArray(customErrors)) {
            errors.push(...customErrors);
          }
        } catch (error) {
          console.error('Custom validation error:', error);
        }
      }

      set((draft) => {
        draft.validationErrors = errors;
        if (errors.length > 0) {
          draft.error = errors[0]; // Show first error
        }
      });

      return errors.length === 0;
    },

    setCustomValidation: (validationFn) => {
      set((draft) => {
        draft.customValidation = validationFn;
      });
    },

    // Utility getters
    canSubmit: () => {
      const state = get();
      return !state.isSubmitting && 
             state.currentText.trim() && 
             state.validationErrors.length === 0;
    },

    getCharacterCount: () => get().currentText.length,

    getRemainingCharacters: () => {
      const state = get();
      return state.maxInputLength - state.currentText.length;
    },

    hasHistory: () => get().history.length > 0,

    getHistoryPreview: (maxItems = 5) => {
      return get().history.slice(0, maxItems);
    },

    // Configuration
    setMaxHistory: (max) => {
      set((draft) => {
        draft.maxHistory = Math.max(1, max);
        // Trim existing history if needed
        if (draft.history.length > draft.maxHistory) {
          draft.history = draft.history.slice(0, draft.maxHistory);
        }
      });
    },

    setMaxInputLength: (max) => {
      set((draft) => {
        draft.maxInputLength = Math.max(0, max);
        // Trim current text if needed
        if (draft.currentText.length > max) {
          draft.currentText = draft.currentText.slice(0, max);
        }
      });
    },

    // Persistence
    exportHistory: () => {
      const state = get();
      return {
        history: state.history,
        exportedAt: new Date(),
        version: '1.0'
      };
    },

    importHistory: (data) => {
      if (!data || !Array.isArray(data.history)) {
        throw new Error('Invalid history data format');
      }

      set((draft) => {
        draft.history = data.history.slice(0, draft.maxHistory);
        draft.historyIndex = -1;
      });
    },

    // Keyboard shortcuts
    handleKeyDown: (event, options = {}) => {
      const { 
        enableHistoryNavigation = true,
        enableSubmitOnEnter = true,
        submitOnShiftEnter = false
      } = options;

      switch (event.key) {
        case 'ArrowUp':
          if (enableHistoryNavigation && event.ctrlKey) {
            event.preventDefault();
            get().navigateHistory('up');
          }
          break;

        case 'ArrowDown':
          if (enableHistoryNavigation && event.ctrlKey) {
            event.preventDefault();
            get().navigateHistory('down');
          }
          break;

        case 'Enter':
          if (enableSubmitOnEnter) {
            const shouldSubmit = submitOnShiftEnter ? event.shiftKey : !event.shiftKey;
            if (shouldSubmit && get().canSubmit()) {
              event.preventDefault();
              return 'submit';
            }
          }
          break;

        case 'Escape':
          get().clearText();
          break;

        default:
          break;
      }

      return null;
    }
  }))
);