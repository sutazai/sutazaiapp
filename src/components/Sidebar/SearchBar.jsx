/**
 * SearchBar Component
 * 
 * Provides search functionality for conversations with autocomplete
 * and search suggestions.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Search, X, Clock, Tag } from 'lucide-react';
import { useSidebarStore } from '../../store/sidebarStore';
import { useConversationStore } from '../../store/conversationStore';
import './SearchBar.css';

const SearchBar = ({ placeholder = "Search conversations...", className = '' }) => {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestion, setSelectedSuggestion] = useState(-1);
  const [recentSearches, setRecentSearches] = useState([]);
  
  const inputRef = useRef(null);
  const suggestionsRef = useRef(null);

  const {
    searchQuery,
    setSearchQuery,
    addSelectedTag
  } = useSidebarStore();

  const {
    sessions,
    getAllTags
  } = useConversationStore();

  // Load recent searches from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('sutazai_recent_searches');
    if (stored) {
      try {
        setRecentSearches(JSON.parse(stored));
      } catch (error) {
        console.error('Error loading recent searches:', error);
      }
    }
  }, []);

  // Generate suggestions based on input
  const generateSuggestions = useCallback((query) => {
    if (!query.trim()) {
      return recentSearches.slice(0, 5).map(search => ({
        type: 'recent',
        text: search,
        icon: Clock
      }));
    }

    const suggestions = [];
    const lowerQuery = query.toLowerCase();

    // Session title suggestions
    const titleMatches = sessions
      .filter(session => 
        session.title.toLowerCase().includes(lowerQuery)
      )
      .slice(0, 3)
      .map(session => ({
        type: 'title',
        text: session.title,
        sessionId: session.id,
        icon: Search
      }));

    // Tag suggestions
    const tagMatches = getAllTags()
      .filter(tag => 
        tag.toLowerCase().includes(lowerQuery)
      )
      .slice(0, 3)
      .map(tag => ({
        type: 'tag',
        text: tag,
        icon: Tag
      }));

    // Content matches (from message content)
    const contentMatches = [];
    for (const session of sessions) {
      for (const message of session.messages || []) {
        if (message.content.toLowerCase().includes(lowerQuery) && 
            contentMatches.length < 2) {
          const preview = message.content.slice(0, 50).replace(/\n/g, ' ');
          contentMatches.push({
            type: 'content',
            text: `"${preview}${preview.length < message.content.length ? '...' : ''}"`,
            fullText: message.content,
            sessionId: session.id,
            sessionTitle: session.title,
            icon: Search
          });
          break; // Only one match per session
        }
      }
    }

    return [
      ...titleMatches,
      ...tagMatches,
      ...contentMatches
    ].slice(0, 8);
  }, [sessions, getAllTags, recentSearches]);

  // Update suggestions when input changes
  useEffect(() => {
    const newSuggestions = generateSuggestions(inputValue);
    setSuggestions(newSuggestions);
    setSelectedSuggestion(-1);
  }, [inputValue, generateSuggestions]);

  // Sync input with store
  useEffect(() => {
    if (searchQuery !== inputValue) {
      setInputValue(searchQuery);
    }
  }, [searchQuery]);

  // Handle input change
  const handleInputChange = (event) => {
    const value = event.target.value;
    setInputValue(value);
    setShowSuggestions(true);
    
    // Debounce the actual search
    const timeoutId = setTimeout(() => {
      setSearchQuery(value);
    }, 300);

    return () => clearTimeout(timeoutId);
  };

  // Handle suggestion selection
  const handleSuggestionSelect = (suggestion) => {
    let searchText = '';
    
    switch (suggestion.type) {
      case 'tag':
        addSelectedTag(suggestion.text);
        searchText = '';
        break;
      case 'recent':
      case 'title':
      case 'content':
        searchText = suggestion.type === 'content' ? 
          suggestion.fullText || suggestion.text : 
          suggestion.text;
        break;
      default:
        searchText = suggestion.text;
    }

    setInputValue(searchText);
    setSearchQuery(searchText);
    setShowSuggestions(false);
    
    // Save to recent searches if it's a new search
    if (searchText.trim() && suggestion.type !== 'recent') {
      saveRecentSearch(searchText);
    }

    inputRef.current?.blur();
  };

  // Save recent search
  const saveRecentSearch = (search) => {
    const trimmedSearch = search.trim();
    if (!trimmedSearch) return;

    const updated = [
      trimmedSearch,
      ...recentSearches.filter(s => s !== trimmedSearch)
    ].slice(0, 10); // Keep only 10 recent searches

    setRecentSearches(updated);
    localStorage.setItem('sutazai_recent_searches', JSON.stringify(updated));
  };

  // Handle keyboard navigation
  const handleKeyDown = (event) => {
    if (!showSuggestions || suggestions.length === 0) {
      if (event.key === 'Enter') {
        const trimmedValue = inputValue.trim();
        if (trimmedValue) {
          setSearchQuery(trimmedValue);
          saveRecentSearch(trimmedValue);
          setShowSuggestions(false);
          inputRef.current?.blur();
        }
      }
      return;
    }

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setSelectedSuggestion(prev => 
          prev < suggestions.length - 1 ? prev + 1 : prev
        );
        break;

      case 'ArrowUp':
        event.preventDefault();
        setSelectedSuggestion(prev => prev > -1 ? prev - 1 : -1);
        break;

      case 'Enter':
        event.preventDefault();
        if (selectedSuggestion >= 0) {
          handleSuggestionSelect(suggestions[selectedSuggestion]);
        } else {
          const trimmedValue = inputValue.trim();
          if (trimmedValue) {
            setSearchQuery(trimmedValue);
            saveRecentSearch(trimmedValue);
            setShowSuggestions(false);
            inputRef.current?.blur();
          }
        }
        break;

      case 'Escape':
        setShowSuggestions(false);
        inputRef.current?.blur();
        break;

      default:
        break;
    }
  };

  // Handle clear search
  const handleClear = () => {
    setInputValue('');
    setSearchQuery('');
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  // Handle focus/blur
  const handleFocus = () => {
    setShowSuggestions(true);
  };

  const handleBlur = (event) => {
    // Don't close suggestions if clicking on a suggestion
    if (suggestionsRef.current?.contains(event.relatedTarget)) {
      return;
    }
    setTimeout(() => setShowSuggestions(false), 150);
  };

  // Clear recent searches
  const clearRecentSearches = () => {
    setRecentSearches([]);
    localStorage.removeItem('sutazai_recent_searches');
  };

  const renderSuggestion = (suggestion, index) => {
    const Icon = suggestion.icon;
    const isSelected = index === selectedSuggestion;

    return (
      <div
        key={index}
        className={`suggestion-item ${isSelected ? 'selected' : ''} ${suggestion.type}`}
        onClick={() => handleSuggestionSelect(suggestion)}
        onMouseEnter={() => setSelectedSuggestion(index)}
        role="option"
        aria-selected={isSelected}
      >
        <div className="suggestion-content">
          <Icon size={14} className="suggestion-icon" />
          <span className="suggestion-text">{suggestion.text}</span>
          
          {suggestion.type === 'content' && suggestion.sessionTitle && (
            <span className="suggestion-context">
              in "{suggestion.sessionTitle}"
            </span>
          )}
          
          {suggestion.type === 'tag' && (
            <span className="suggestion-label">Tag</span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={`search-bar ${className}`}>
      <div className="search-input-container">
        <Search size={18} className="search-icon" />
        
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          className="search-input"
          autoComplete="off"
          aria-label="Search conversations"
          aria-autocomplete="list"
          aria-expanded={showSuggestions}
          aria-haspopup="listbox"
        />
        
        {inputValue && (
          <button
            onClick={handleClear}
            className="clear-button"
            aria-label="Clear search"
            tabIndex={-1}
          >
            <X size={16} />
          </button>
        )}
      </div>

      {showSuggestions && suggestions.length > 0 && (
        <div 
          ref={suggestionsRef}
          className="suggestions-dropdown"
          role="listbox"
          aria-label="Search suggestions"
        >
          {suggestions.map(renderSuggestion)}
          
          {recentSearches.length > 0 && !inputValue.trim() && (
            <div className="suggestions-footer">
              <button
                onClick={clearRecentSearches}
                className="clear-recent-btn"
              >
                Clear recent searches
              </button>
            </div>
          )}
        </div>
      )}

      {/* Search shortcuts hint */}
      {showSuggestions && (
        <div className="search-hints">
          <small>
            Use ↑↓ to navigate, Enter to search, Esc to close
          </small>
        </div>
      )}
    </div>
  );
};

export default SearchBar;