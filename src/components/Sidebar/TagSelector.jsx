/**
 * TagSelector Component
 * 
 * Provides tag-based filtering with tag creation, selection, and management
 * including popular tags and recent tags.
 */

import React, { useState, useRef, useEffect } from 'react';
import { Tag, Plus, X, Hash, TrendingUp } from 'lucide-react';
import { useSidebarStore } from '../../store/sidebarStore';
import { useConversationStore } from '../../store/conversationStore';
import './TagSelector.css';

const TagSelector = ({ className = '' }) => {
  const [newTag, setNewTag] = useState('');
  const [showCreateTag, setShowCreateTag] = useState(false);
  const inputRef = useRef(null);

  const {
    selectedTags,
    addSelectedTag,
    removeSelectedTag,
    clearSelectedTags
  } = useSidebarStore();

  const {
    getAllTags,
    sessions,
    currentSession,
    tagSession
  } = useConversationStore();

  const allTags = getAllTags();

  // Get tag usage statistics
  const getTagStats = () => {
    const tagCounts = {};
    
    sessions.forEach(session => {
      session.tags?.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    });

    return tagCounts;
  };

  const tagStats = getTagStats();

  // Get popular tags (most used)
  const getPopularTags = () => {
    return Object.entries(tagStats)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([tag, count]) => ({ tag, count }));
  };

  // Get recent tags (from recent sessions)
  const getRecentTags = () => {
    const recentTags = new Set();
    const recentSessions = sessions
      .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
      .slice(0, 20);

    recentSessions.forEach(session => {
      session.tags?.forEach(tag => recentTags.add(tag));
    });

    return Array.from(recentTags).slice(0, 10);
  };

  // Handle tag creation
  const handleCreateTag = () => {
    const trimmedTag = newTag.trim().toLowerCase();
    
    if (!trimmedTag) return;
    
    if (trimmedTag.length > 50) {
      alert('Tag must be 50 characters or less');
      return;
    }

    if (allTags.includes(trimmedTag)) {
      addSelectedTag(trimmedTag);
    } else {
      // Add tag to current session if exists
      if (currentSession) {
        tagSession(currentSession.id, [trimmedTag]);
      }
      addSelectedTag(trimmedTag);
    }

    setNewTag('');
    setShowCreateTag(false);
  };

  // Handle tag input keydown
  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleCreateTag();
    } else if (event.key === 'Escape') {
      setNewTag('');
      setShowCreateTag(false);
    }
  };

  // Focus input when shown
  useEffect(() => {
    if (showCreateTag && inputRef.current) {
      inputRef.current.focus();
    }
  }, [showCreateTag]);

  // Validate tag name
  const isValidTag = (tag) => {
    return tag.trim().length > 0 && 
           tag.trim().length <= 50 && 
           /^[a-zA-Z0-9\s\-_]+$/.test(tag.trim());
  };

  const renderTagChip = (tag, options = {}) => {
    const { 
      count, 
      isSelected = false, 
      showCount = false, 
      showRemove = false,
      onClick,
      onRemove
    } = options;

    return (
      <div
        key={tag}
        className={`tag-chip ${isSelected ? 'selected' : ''}`}
        onClick={() => onClick?.(tag)}
      >
        <Hash size={12} className="tag-icon" />
        <span className="tag-name">{tag}</span>
        
        {showCount && count && (
          <span className="tag-count">({count})</span>
        )}
        
        {showRemove && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemove?.(tag);
            }}
            className="remove-tag-btn"
            aria-label={`Remove ${tag} tag`}
          >
            <X size={12} />
          </button>
        )}
      </div>
    );
  };

  const popularTags = getPopularTags();
  const recentTags = getRecentTags();
  const availableTags = allTags.filter(tag => !selectedTags.includes(tag));

  return (
    <div className={`tag-selector ${className}`}>
      <div className="tag-selector-header">
        <h4>Filter by Tags</h4>
        {selectedTags.length > 0 && (
          <button
            onClick={clearSelectedTags}
            className="clear-tags-btn"
            title="Clear selected tags"
          >
            <X size={14} />
            Clear
          </button>
        )}
      </div>

      {/* Selected tags */}
      {selectedTags.length > 0 && (
        <div className="selected-tags-section">
          <h5>Selected Tags</h5>
          <div className="selected-tags">
            {selectedTags.map(tag => 
              renderTagChip(tag, {
                isSelected: true,
                showRemove: true,
                onRemove: removeSelectedTag
              })
            )}
          </div>
        </div>
      )}

      {/* Create new tag */}
      <div className="create-tag-section">
        {showCreateTag ? (
          <div className="create-tag-input">
            <input
              ref={inputRef}
              type="text"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={() => {
                if (!newTag.trim()) {
                  setShowCreateTag(false);
                }
              }}
              placeholder="Enter tag name..."
              className="tag-input"
              maxLength={50}
            />
            <div className="create-tag-actions">
              <button
                onClick={handleCreateTag}
                disabled={!isValidTag(newTag)}
                className="btn-primary btn-sm"
              >
                Add
              </button>
              <button
                onClick={() => {
                  setNewTag('');
                  setShowCreateTag(false);
                }}
                className="btn-secondary btn-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowCreateTag(true)}
            className="create-tag-btn"
          >
            <Plus size={14} />
            Create Tag
          </button>
        )}
      </div>

      {/* Popular tags */}
      {popularTags.length > 0 && (
        <div className="tag-section">
          <h5>
            <TrendingUp size={14} />
            Popular Tags
          </h5>
          <div className="tag-list">
            {popularTags.map(({ tag, count }) => 
              renderTagChip(tag, {
                count,
                showCount: true,
                isSelected: selectedTags.includes(tag),
                onClick: selectedTags.includes(tag) ? removeSelectedTag : addSelectedTag
              })
            )}
          </div>
        </div>
      )}

      {/* Recent tags */}
      {recentTags.length > 0 && recentTags.some(tag => !popularTags.find(p => p.tag === tag)) && (
        <div className="tag-section">
          <h5>Recent Tags</h5>
          <div className="tag-list">
            {recentTags
              .filter(tag => !popularTags.find(p => p.tag === tag))
              .slice(0, 8)
              .map(tag => 
                renderTagChip(tag, {
                  isSelected: selectedTags.includes(tag),
                  onClick: selectedTags.includes(tag) ? removeSelectedTag : addSelectedTag
                })
              )}
          </div>
        </div>
      )}

      {/* All available tags */}
      {availableTags.length > 0 && (
        <div className="tag-section">
          <h5>All Tags ({availableTags.length})</h5>
          <div className="tag-list scrollable">
            {availableTags
              .sort((a, b) => a.localeCompare(b))
              .map(tag => 
                renderTagChip(tag, {
                  count: tagStats[tag],
                  showCount: true,
                  onClick: addSelectedTag
                })
              )}
          </div>
        </div>
      )}

      {/* Empty state */}
      {allTags.length === 0 && (
        <div className="empty-tags">
          <Tag size={32} className="empty-icon" />
          <p>No tags yet</p>
          <p className="empty-subtitle">
            Tags help organize and find conversations quickly
          </p>
        </div>
      )}

      {/* Tag help */}
      <div className="tag-help">
        <details>
          <summary>How to use tags</summary>
          <div className="help-content">
            <ul>
              <li>Tags help categorize conversations</li>
              <li>Click a tag to filter conversations</li>
              <li>You can select multiple tags</li>
              <li>Tags are automatically saved</li>
              <li>Use alphanumeric characters, spaces, hyphens, and underscores</li>
            </ul>
          </div>
        </details>
      </div>
    </div>
  );
};

export default TagSelector;