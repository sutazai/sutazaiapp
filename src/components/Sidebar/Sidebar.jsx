/**
 * Sidebar Component
 * 
 * Main sidebar containing conversation list, search, and filter functionality
 * with responsive design and accessibility features.
 */

import React, { useEffect, useCallback, useState } from 'react';
import { Search, Filter, X, Menu, Plus, Settings, Archive } from 'lucide-react';
import { useSidebarStore } from '../../store/sidebarStore';
import { useConversationStore } from '../../store/conversationStore';
import ConversationList from './ConversationList';
import SearchBar from './SearchBar';
import FilterControls from './FilterControls';
import TagSelector from './TagSelector';
import './Sidebar.css';

const Sidebar = ({ className = '', onSessionSelect, onNewSession }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  
  // Zustand stores
  const {
    isOpen,
    toggleSidebar,
    closeSidebar,
    filteredSessions,
    hasActiveFilters,
    getFilterSummary,
    getFilterCount,
    showFilters,
    toggleFilters,
    showTagSelector,
    toggleTagSelector,
    isLoading: isFiltering,
    error: filterError,
    applyFilters
  } = useSidebarStore();

  const {
    sessions,
    currentSessionId,
    isLoading: isLoadingSessions,
    error: conversationError,
    loadSessions,
    createSession
  } = useConversationStore();

  // Load sessions on mount
  useEffect(() => {
    if (!isLoaded) {
      loadSessions().finally(() => {
        setIsLoaded(true);
      });
    }
  }, [loadSessions, isLoaded]);

  // Apply filters when sessions change
  useEffect(() => {
    if (sessions.length > 0) {
      applyFilters(sessions);
    }
  }, [sessions, applyFilters]);

  // Handle new session creation
  const handleNewSession = useCallback(() => {
    const sessionId = createSession();
    if (onNewSession) {
      onNewSession(sessionId);
    }
  }, [createSession, onNewSession]);

  // Handle session selection
  const handleSessionSelect = useCallback((sessionId) => {
    if (onSessionSelect) {
      onSessionSelect(sessionId);
    }
    
    // Close sidebar on mobile after selection
    if (window.innerWidth < 768) {
      closeSidebar();
    }
  }, [onSessionSelect, closeSidebar]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Toggle sidebar with Cmd/Ctrl + B
      if ((event.metaKey || event.ctrlKey) && event.key === 'b') {
        event.preventDefault();
        toggleSidebar();
      }

      // New session with Cmd/Ctrl + N
      if ((event.metaKey || event.ctrlKey) && event.key === 'n') {
        event.preventDefault();
        handleNewSession();
      }

      // Focus search with Cmd/Ctrl + K
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        document.querySelector('.search-input')?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [toggleSidebar, handleNewSession]);

  // Close sidebar on outside click (mobile)
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (window.innerWidth < 768 && isOpen) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar && !sidebar.contains(event.target)) {
          closeSidebar();
        }
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen, closeSidebar]);

  const renderHeader = () => (
    <div className="sidebar-header">
      <div className="header-top">
        <h2 className="sidebar-title">
          <Menu size={20} />
          Conversations
        </h2>
        <div className="header-controls">
          <button
            onClick={handleNewSession}
            className="control-btn new-session-btn"
            title="New Conversation (Ctrl+N)"
            aria-label="Start new conversation"
          >
            <Plus size={18} />
          </button>
          <button
            onClick={toggleSidebar}
            className="control-btn toggle-btn md:hidden"
            title="Close Sidebar"
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Filter Summary */}
      {hasActiveFilters() && (
        <div className="filter-summary">
          <span className="filter-text">{getFilterSummary()}</span>
          <span className="filter-count">({getFilterCount()})</span>
        </div>
      )}
    </div>
  );

  const renderToolbar = () => (
    <div className="sidebar-toolbar">
      <SearchBar />
      
      <div className="toolbar-actions">
        <button
          onClick={toggleFilters}
          className={`toolbar-btn ${showFilters ? 'active' : ''}`}
          title="Filter Options"
          aria-label="Toggle filter options"
        >
          <Filter size={18} />
          {hasActiveFilters() && <span className="active-indicator" />}
        </button>
        
        <button
          onClick={toggleTagSelector}
          className={`toolbar-btn ${showTagSelector ? 'active' : ''}`}
          title="Tag Selector"
          aria-label="Toggle tag selector"
        >
          <Archive size={18} />
        </button>
      </div>
    </div>
  );

  const renderFilterControls = () => (
    showFilters && (
      <div className="filter-controls-container">
        <FilterControls />
      </div>
    )
  );

  const renderTagSelector = () => (
    showTagSelector && (
      <div className="tag-selector-container">
        <TagSelector />
      </div>
    )
  );

  const renderContent = () => {
    // Show loading state
    if (!isLoaded || isLoadingSessions) {
      return (
        <div className="sidebar-loading">
          <div className="loading-spinner" />
          <p>Loading conversations...</p>
        </div>
      );
    }

    // Show error state
    if (conversationError || filterError) {
      return (
        <div className="sidebar-error">
          <p className="error-message">
            {conversationError || filterError}
          </p>
          <button
            onClick={() => loadSessions()}
            className="retry-btn"
          >
            Retry
          </button>
        </div>
      );
    }

    // Show empty state
    if (sessions.length === 0) {
      return (
        <div className="sidebar-empty">
          <div className="empty-content">
            <Plus size={48} className="empty-icon" />
            <h3>No conversations yet</h3>
            <p>Start your first conversation with SutazAI</p>
            <button
              onClick={handleNewSession}
              className="new-session-cta"
            >
              Start Conversation
            </button>
          </div>
        </div>
      );
    }

    // Show filtered empty state
    if (filteredSessions.length === 0 && hasActiveFilters()) {
      return (
        <div className="sidebar-empty">
          <div className="empty-content">
            <Search size={48} className="empty-icon" />
            <h3>No matching conversations</h3>
            <p>Try adjusting your search or filters</p>
          </div>
        </div>
      );
    }

    return (
      <ConversationList
        sessions={filteredSessions}
        currentSessionId={currentSessionId}
        onSessionSelect={handleSessionSelect}
        isLoading={isFiltering}
      />
    );
  };

  const renderFooter = () => (
    <div className="sidebar-footer">
      <div className="stats">
        <span className="stat">
          {filteredSessions.length} of {sessions.length} conversations
        </span>
      </div>
      
      <div className="footer-actions">
        <button
          className="footer-btn"
          title="Settings"
          aria-label="Open settings"
        >
          <Settings size={16} />
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="sidebar-overlay md:hidden" 
          onClick={closeSidebar}
          aria-hidden="true"
        />
      )}
      
      {/* Sidebar */}
      <aside 
        className={`sidebar ${isOpen ? 'open' : 'closed'} ${className}`}
        role="complementary"
        aria-label="Conversation sidebar"
      >
        <div className="sidebar-container">
          {renderHeader()}
          {renderToolbar()}
          {renderFilterControls()}
          {renderTagSelector()}
          
          <div className="sidebar-content">
            {renderContent()}
          </div>
          
          {renderFooter()}
        </div>
      </aside>
    </>
  );
};

export default Sidebar;