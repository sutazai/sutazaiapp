/**
 * Sidebar State Management
 * 
 * Manages sidebar visibility, filters, search functionality, and session display
 * with support for multiple filter types, date ranges, and tags.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { FilterType } from './types';

// Date utility functions
const DateUtils = {
  getStartOfDay: (date) => {
    const start = new Date(date);
    start.setHours(0, 0, 0, 0);
    return start;
  },

  getEndOfDay: (date) => {
    const end = new Date(date);
    end.setHours(23, 59, 59, 999);
    return end;
  },

  getStartOfWeek: (date) => {
    const start = new Date(date);
    const day = start.getDay();
    const diff = start.getDate() - day;
    start.setDate(diff);
    start.setHours(0, 0, 0, 0);
    return start;
  },

  getStartOfMonth: (date) => {
    const start = new Date(date);
    start.setDate(1);
    start.setHours(0, 0, 0, 0);
    return start;
  },

  isToday: (date) => {
    const today = new Date();
    const checkDate = new Date(date);
    return today.toDateString() === checkDate.toDateString();
  },

  isThisWeek: (date) => {
    const now = new Date();
    const weekStart = DateUtils.getStartOfWeek(now);
    const checkDate = new Date(date);
    return checkDate >= weekStart && checkDate <= now;
  },

  isThisMonth: (date) => {
    const now = new Date();
    const monthStart = DateUtils.getStartOfMonth(now);
    const checkDate = new Date(date);
    return checkDate >= monthStart && checkDate <= now;
  }
};

export const useSidebarStore = create(
  immer((set, get) => ({
    // State
    isOpen: true,
    activeFilter: FilterType.ALL,
    searchQuery: '',
    selectedTags: [],
    dateRange: {
      start: null,
      end: null,
      preset: null // 'today', 'week', 'month', 'custom'
    },
    filteredSessions: [],
    isLoading: false,
    error: null,
    
    // UI State
    showFilters: false,
    showTagSelector: false,
    sortBy: 'updatedAt', // 'updatedAt', 'createdAt', 'title', 'messageCount'
    sortOrder: 'desc', // 'asc', 'desc'
    
    // Performance optimizations
    filterDebounceTimer: null,
    lastFilterUpdate: null,
    
    // Actions
    toggleSidebar: () => {
      set((draft) => {
        draft.isOpen = !draft.isOpen;
      });
    },

    openSidebar: () => {
      set((draft) => {
        draft.isOpen = true;
      });
    },

    closeSidebar: () => {
      set((draft) => {
        draft.isOpen = false;
      });
    },

    setFilter: (filterType) => {
      if (!Object.values(FilterType).includes(filterType)) {
        return;
      }

      set((draft) => {
        draft.activeFilter = filterType;
        
        // Auto-set date range for time-based filters
        const now = new Date();
        switch (filterType) {
          case FilterType.TODAY:
            draft.dateRange = {
              start: DateUtils.getStartOfDay(now),
              end: DateUtils.getEndOfDay(now),
              preset: 'today'
            };
            break;
          case FilterType.WEEK:
            draft.dateRange = {
              start: DateUtils.getStartOfWeek(now),
              end: now,
              preset: 'week'
            };
            break;
          case FilterType.MONTH:
            draft.dateRange = {
              start: DateUtils.getStartOfMonth(now),
              end: now,
              preset: 'month'
            };
            break;
          case FilterType.ALL:
            draft.dateRange = {
              start: null,
              end: null,
              preset: null
            };
            break;
        }
      });

      // Apply filters
      get().debouncedApplyFilters();
    },

    setSearchQuery: (query) => {
      set((draft) => {
        draft.searchQuery = query;
        draft.activeFilter = query.trim() ? FilterType.KEYWORD : FilterType.ALL;
      });

      get().debouncedApplyFilters();
    },

    addSelectedTag: (tag) => {
      if (!tag || typeof tag !== 'string') return;

      set((draft) => {
        if (!draft.selectedTags.includes(tag)) {
          draft.selectedTags.push(tag);
          draft.activeFilter = FilterType.TAG;
        }
      });

      get().debouncedApplyFilters();
    },

    removeSelectedTag: (tag) => {
      set((draft) => {
        draft.selectedTags = draft.selectedTags.filter(t => t !== tag);
        
        // Reset filter if no tags selected
        if (draft.selectedTags.length === 0 && draft.activeFilter === FilterType.TAG) {
          draft.activeFilter = FilterType.ALL;
        }
      });

      get().debouncedApplyFilters();
    },

    clearSelectedTags: () => {
      set((draft) => {
        draft.selectedTags = [];
        if (draft.activeFilter === FilterType.TAG) {
          draft.activeFilter = FilterType.ALL;
        }
      });

      get().debouncedApplyFilters();
    },

    setDateRange: (start, end, preset = 'custom') => {
      set((draft) => {
        draft.dateRange = {
          start: start ? new Date(start) : null,
          end: end ? new Date(end) : null,
          preset
        };
        
        if (start || end) {
          draft.activeFilter = FilterType.ALL; // Custom date range
        }
      });

      get().debouncedApplyFilters();
    },

    setSortBy: (sortBy, sortOrder = null) => {
      const validSortFields = ['updatedAt', 'createdAt', 'title', 'messageCount'];
      if (!validSortFields.includes(sortBy)) return;

      set((draft) => {
        draft.sortBy = sortBy;
        if (sortOrder) {
          draft.sortOrder = sortOrder;
        }
      });

      get().applySorting();
    },

    toggleSortOrder: () => {
      set((draft) => {
        draft.sortOrder = draft.sortOrder === 'asc' ? 'desc' : 'asc';
      });

      get().applySorting();
    },

    applyFilters: (sessions) => {
      const state = get();
      
      if (!sessions || !Array.isArray(sessions)) {
        set((draft) => {
          draft.filteredSessions = [];
          draft.error = 'No sessions provided to filter';
        });
        return [];
      }

      set((draft) => {
        draft.isLoading = true;
        draft.error = null;
      });

      try {
        let filtered = [...sessions];

        // Apply text search
        if (state.searchQuery.trim()) {
          const query = state.searchQuery.toLowerCase();
          filtered = filtered.filter(session => {
            return session.title.toLowerCase().includes(query) ||
                   session.messages.some(msg => 
                     msg.content.toLowerCase().includes(query)
                   ) ||
                   session.tags.some(tag => 
                     tag.toLowerCase().includes(query)
                   );
          });
        }

        // Apply tag filter
        if (state.selectedTags.length > 0) {
          filtered = filtered.filter(session => {
            return state.selectedTags.some(tag => 
              session.tags.includes(tag)
            );
          });
        }

        // Apply date range filter
        if (state.dateRange.start || state.dateRange.end) {
          filtered = filtered.filter(session => {
            const sessionDate = new Date(session.updatedAt);
            
            if (state.dateRange.start && sessionDate < state.dateRange.start) {
              return false;
            }
            
            if (state.dateRange.end && sessionDate > state.dateRange.end) {
              return false;
            }
            
            return true;
          });
        }

        // Apply time-based filters (if no custom date range)
        if (!state.dateRange.start && !state.dateRange.end) {
          switch (state.activeFilter) {
            case FilterType.TODAY:
              filtered = filtered.filter(session => 
                DateUtils.isToday(session.updatedAt)
              );
              break;
            case FilterType.WEEK:
              filtered = filtered.filter(session => 
                DateUtils.isThisWeek(session.updatedAt)
              );
              break;
            case FilterType.MONTH:
              filtered = filtered.filter(session => 
                DateUtils.isThisMonth(session.updatedAt)
              );
              break;
          }
        }

        // Apply sorting
        filtered = get().sortSessions(filtered);

        set((draft) => {
          draft.filteredSessions = filtered;
          draft.lastFilterUpdate = new Date();
          draft.isLoading = false;
        });

        return filtered;

      } catch (error) {
        console.error('Filter error:', error);
        set((draft) => {
          draft.error = `Filter error: ${error.message}`;
          draft.filteredSessions = [];
          draft.isLoading = false;
        });
        return [];
      }
    },

    sortSessions: (sessions) => {
      const state = get();
      
      return [...sessions].sort((a, b) => {
        let aValue, bValue;

        switch (state.sortBy) {
          case 'updatedAt':
            aValue = new Date(a.updatedAt);
            bValue = new Date(b.updatedAt);
            break;
          case 'createdAt':
            aValue = new Date(a.createdAt);
            bValue = new Date(b.createdAt);
            break;
          case 'title':
            aValue = a.title.toLowerCase();
            bValue = b.title.toLowerCase();
            break;
          case 'messageCount':
            aValue = a.messages ? a.messages.length : 0;
            bValue = b.messages ? b.messages.length : 0;
            break;
          default:
            aValue = new Date(a.updatedAt);
            bValue = new Date(b.updatedAt);
        }

        if (state.sortOrder === 'asc') {
          return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
        } else {
          return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
        }
      });
    },

    applySorting: () => {
      const state = get();
      const sorted = get().sortSessions(state.filteredSessions);
      
      set((draft) => {
        draft.filteredSessions = sorted;
      });
    },

    debouncedApplyFilters: () => {
      const state = get();
      
      if (state.filterDebounceTimer) {
        clearTimeout(state.filterDebounceTimer);
      }

      const timer = setTimeout(() => {
        // Note: This will need sessions from the conversation store
        // In actual usage, this would be called with sessions as parameter
        // or access the conversation store directly
        set((draft) => {
          draft.filterDebounceTimer = null;
        });
      }, 300);

      set((draft) => {
        draft.filterDebounceTimer = timer;
      });
    },

    clearFilters: () => {
      const state = get();
      
      if (state.filterDebounceTimer) {
        clearTimeout(state.filterDebounceTimer);
      }

      set((draft) => {
        draft.activeFilter = FilterType.ALL;
        draft.searchQuery = '';
        draft.selectedTags = [];
        draft.dateRange = {
          start: null,
          end: null,
          preset: null
        };
        draft.filterDebounceTimer = null;
      });

      // Reapply with cleared filters
      get().debouncedApplyFilters();
    },

    toggleFilters: () => {
      set((draft) => {
        draft.showFilters = !draft.showFilters;
      });
    },

    toggleTagSelector: () => {
      set((draft) => {
        draft.showTagSelector = !draft.showTagSelector;
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
      });
    },

    // Utility getters
    hasActiveFilters: () => {
      const state = get();
      return state.activeFilter !== FilterType.ALL ||
             state.searchQuery.trim() !== '' ||
             state.selectedTags.length > 0 ||
             state.dateRange.start !== null ||
             state.dateRange.end !== null;
    },

    getFilterSummary: () => {
      const state = get();
      const parts = [];

      if (state.searchQuery.trim()) {
        parts.push(`Search: "${state.searchQuery}"`);
      }

      if (state.selectedTags.length > 0) {
        parts.push(`Tags: ${state.selectedTags.join(', ')}`);
      }

      if (state.dateRange.preset) {
        parts.push(`Date: ${state.dateRange.preset}`);
      } else if (state.dateRange.start || state.dateRange.end) {
        const start = state.dateRange.start ? 
          state.dateRange.start.toLocaleDateString() : 'Any';
        const end = state.dateRange.end ? 
          state.dateRange.end.toLocaleDateString() : 'Any';
        parts.push(`Date: ${start} - ${end}`);
      }

      if (state.activeFilter !== FilterType.ALL && !parts.some(p => p.startsWith('Date'))) {
        parts.push(`Filter: ${state.activeFilter}`);
      }

      return parts.join(' | ') || 'All conversations';
    },

    getFilterCount: () => get().filteredSessions.length,

    // Presets for quick filtering
    showTodaysSessions: () => get().setFilter(FilterType.TODAY),
    showWeekSessions: () => get().setFilter(FilterType.WEEK),
    showMonthSessions: () => get().setFilter(FilterType.MONTH),
    showAllSessions: () => get().setFilter(FilterType.ALL),

    // Advanced search helpers
    searchByContent: (query) => {
      get().setSearchQuery(query);
      get().setFilter(FilterType.KEYWORD);
    },

    filterByTags: (tags) => {
      set((draft) => {
        draft.selectedTags = Array.isArray(tags) ? [...tags] : [tags];
        draft.activeFilter = FilterType.TAG;
      });
      get().debouncedApplyFilters();
    },

    filterByDatePreset: (preset) => {
      const now = new Date();
      let start, end;

      switch (preset) {
        case 'today':
          start = DateUtils.getStartOfDay(now);
          end = DateUtils.getEndOfDay(now);
          break;
        case 'yesterday':
          const yesterday = new Date(now);
          yesterday.setDate(yesterday.getDate() - 1);
          start = DateUtils.getStartOfDay(yesterday);
          end = DateUtils.getEndOfDay(yesterday);
          break;
        case 'week':
          start = DateUtils.getStartOfWeek(now);
          end = now;
          break;
        case 'month':
          start = DateUtils.getStartOfMonth(now);
          end = now;
          break;
        default:
          start = null;
          end = null;
      }

      get().setDateRange(start, end, preset);
    },

    // Cleanup
    cleanup: () => {
      const state = get();
      if (state.filterDebounceTimer) {
        clearTimeout(state.filterDebounceTimer);
      }
    }
  }))
);