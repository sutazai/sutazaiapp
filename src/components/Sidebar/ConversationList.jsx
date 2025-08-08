/**
 * ConversationList Component
 * 
 * Displays the list of conversation sessions with selection, preview,
 * and action capabilities.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { MessageCircle, Clock, Tag, MoreVertical, Edit3, Trash2, Archive } from 'lucide-react';
import { useConversationStore } from '../../store/conversationStore';
import { format, formatDistanceToNow, isToday, isYesterday } from 'date-fns';
import './ConversationList.css';

const ConversationItem = ({ 
  session, 
  isActive, 
  onSelect, 
  onDelete, 
  onEdit, 
  onToggleArchive 
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(session.title);
  const menuRef = useRef(null);
  const editRef = useRef(null);

  // Format date for display
  const formatDate = (date) => {
    const sessionDate = new Date(date);
    
    if (isToday(sessionDate)) {
      return format(sessionDate, 'HH:mm');
    } else if (isYesterday(sessionDate)) {
      return 'Yesterday';
    } else {
      const distance = formatDistanceToNow(sessionDate, { addSuffix: false });
      if (distance.includes('day')) {
        return format(sessionDate, 'MMM d');
      }
      return distance;
    }
  };

  // Get preview text from last message
  const getPreviewText = () => {
    if (!session.messages || session.messages.length === 0) {
      return 'No messages yet';
    }

    const lastMessage = session.messages[session.messages.length - 1];
    const preview = lastMessage.content.replace(/\n/g, ' ').trim();
    return preview.length > 60 ? `${preview.slice(0, 60)}...` : preview;
  };

  // Handle menu click outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowMenu(false);
      }
    };

    if (showMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showMenu]);

  // Handle edit mode
  const handleEditStart = () => {
    setIsEditing(true);
    setShowMenu(false);
    setTimeout(() => {
      editRef.current?.focus();
      editRef.current?.select();
    }, 50);
  };

  const handleEditSubmit = () => {
    if (editTitle.trim() && editTitle !== session.title) {
      onEdit(session.id, editTitle.trim());
    }
    setIsEditing(false);
    setEditTitle(session.title);
  };

  const handleEditCancel = () => {
    setIsEditing(false);
    setEditTitle(session.title);
  };

  const handleEditKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleEditSubmit();
    } else if (event.key === 'Escape') {
      handleEditCancel();
    }
  };

  const handleMenuAction = (action, event) => {
    event.stopPropagation();
    setShowMenu(false);
    
    switch (action) {
      case 'edit':
        handleEditStart();
        break;
      case 'delete':
        onDelete(session.id);
        break;
      case 'archive':
        onToggleArchive(session.id);
        break;
    }
  };

  return (
    <div 
      className={`conversation-item ${isActive ? 'active' : ''}`}
      onClick={() => onSelect(session.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(session.id);
        }
      }}
      aria-label={`Conversation: ${session.title}`}
    >
      <div className="conversation-main">
        <div className="conversation-header">
          <div className="conversation-icon">
            <MessageCircle size={16} />
          </div>
          
          <div className="conversation-info">
            {isEditing ? (
              <input
                ref={editRef}
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={handleEditSubmit}
                onKeyDown={handleEditKeyDown}
                className="edit-title-input"
                maxLength={100}
              />
            ) : (
              <h3 className="conversation-title">{session.title}</h3>
            )}
            
            <div className="conversation-meta">
              <span className="message-count">
                {session.messages?.length || 0} messages
              </span>
              <span className="conversation-date">
                <Clock size={12} />
                {formatDate(session.updatedAt)}
              </span>
            </div>
          </div>

          <div className="conversation-actions">
            <button
              className="menu-trigger"
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              aria-label="Conversation actions"
            >
              <MoreVertical size={16} />
            </button>
            
            {showMenu && (
              <div ref={menuRef} className="conversation-menu">
                <button
                  onClick={(e) => handleMenuAction('edit', e)}
                  className="menu-item"
                >
                  <Edit3 size={14} />
                  Rename
                </button>
                <button
                  onClick={(e) => handleMenuAction('archive', e)}
                  className="menu-item"
                >
                  <Archive size={14} />
                  {session.metadata?.archived ? 'Unarchive' : 'Archive'}
                </button>
                <button
                  onClick={(e) => handleMenuAction('delete', e)}
                  className="menu-item danger"
                >
                  <Trash2 size={14} />
                  Delete
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="conversation-preview">
          <p className="preview-text">{getPreviewText()}</p>
        </div>

        {session.tags && session.tags.length > 0 && (
          <div className="conversation-tags">
            {session.tags.slice(0, 3).map((tag, index) => (
              <span key={index} className="tag">
                <Tag size={10} />
                {tag}
              </span>
            ))}
            {session.tags.length > 3 && (
              <span className="tag-overflow">
                +{session.tags.length - 3}
              </span>
            )}
          </div>
        )}
      </div>
      
      {isActive && (
        <div className="active-indicator" aria-hidden="true" />
      )}
    </div>
  );
};

const ConversationList = ({ 
  sessions = [], 
  currentSessionId, 
  onSessionSelect,
  isLoading = false
}) => {
  const {
    setCurrentSession,
    deleteSession,
    tagSession,
    updateSession
  } = useConversationStore();

  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const handleSessionSelect = useCallback((sessionId) => {
    setCurrentSession(sessionId);
    if (onSessionSelect) {
      onSessionSelect(sessionId);
    }
  }, [setCurrentSession, onSessionSelect]);

  const handleSessionEdit = useCallback((sessionId, newTitle) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      updateSession(sessionId, { title: newTitle });
    }
  }, [sessions, updateSession]);

  const handleSessionDelete = useCallback((sessionId) => {
    setDeleteConfirm(sessionId);
  }, []);

  const confirmDelete = useCallback((sessionId) => {
    deleteSession(sessionId);
    setDeleteConfirm(null);
  }, [deleteSession]);

  const cancelDelete = useCallback(() => {
    setDeleteConfirm(null);
  }, []);

  const handleToggleArchive = useCallback((sessionId) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      const isArchived = session.metadata?.archived || false;
      updateSession(sessionId, {
        metadata: {
          ...session.metadata,
          archived: !isArchived
        }
      });
      
      // Add/remove archive tag
      if (!isArchived) {
        tagSession(sessionId, ['archived']);
      }
    }
  }, [sessions, updateSession, tagSession]);

  // Group sessions by date
  const groupedSessions = sessions.reduce((groups, session) => {
    const date = new Date(session.updatedAt);
    let groupKey;

    if (isToday(date)) {
      groupKey = 'Today';
    } else if (isYesterday(date)) {
      groupKey = 'Yesterday';
    } else {
      const daysAgo = Math.floor((Date.now() - date.getTime()) / (1000 * 60 * 60 * 24));
      if (daysAgo < 7) {
        groupKey = 'This Week';
      } else if (daysAgo < 30) {
        groupKey = 'This Month';
      } else {
        groupKey = format(date, 'MMMM yyyy');
      }
    }

    if (!groups[groupKey]) {
      groups[groupKey] = [];
    }
    groups[groupKey].push(session);
    return groups;
  }, {});

  const groupOrder = ['Today', 'Yesterday', 'This Week', 'This Month'];
  const sortedGroups = Object.keys(groupedSessions).sort((a, b) => {
    const aIndex = groupOrder.indexOf(a);
    const bIndex = groupOrder.indexOf(b);
    
    if (aIndex !== -1 && bIndex !== -1) {
      return aIndex - bIndex;
    } else if (aIndex !== -1) {
      return -1;
    } else if (bIndex !== -1) {
      return 1;
    } else {
      return b.localeCompare(a); // Reverse alphabetical for months
    }
  });

  if (isLoading) {
    return (
      <div className="conversation-list-loading">
        <div className="loading-skeleton">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="skeleton-item">
              <div className="skeleton-header">
                <div className="skeleton-circle" />
                <div className="skeleton-text" />
              </div>
              <div className="skeleton-preview" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="conversation-list" role="list">
      {sortedGroups.map((groupKey) => (
        <div key={groupKey} className="conversation-group">
          <h4 className="group-header">{groupKey}</h4>
          <div className="group-items">
            {groupedSessions[groupKey].map((session) => (
              <ConversationItem
                key={session.id}
                session={session}
                isActive={session.id === currentSessionId}
                onSelect={handleSessionSelect}
                onEdit={handleSessionEdit}
                onDelete={handleSessionDelete}
                onToggleArchive={handleToggleArchive}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Delete confirmation modal */}
      {deleteConfirm && (
        <div className="delete-modal-overlay">
          <div className="delete-modal">
            <h3>Delete Conversation</h3>
            <p>Are you sure you want to delete this conversation? This action cannot be undone.</p>
            <div className="modal-actions">
              <button
                onClick={cancelDelete}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => confirmDelete(deleteConfirm)}
                className="btn-danger"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConversationList;