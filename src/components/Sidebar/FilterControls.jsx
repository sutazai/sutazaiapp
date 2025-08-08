/**
 * FilterControls Component
 * 
 * Provides date/time filtering controls for conversations
 * with preset options and custom date range selection.
 */

import React, { useState } from 'react';
import { Calendar, Clock, Filter, X, ChevronDown } from 'lucide-react';
import { useSidebarStore } from '../../store/sidebarStore';
import { FilterType } from '../../store/types';
import { format } from 'date-fns';
import './FilterControls.css';

const FilterControls = ({ className = '' }) => {
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');

  const {
    activeFilter,
    dateRange,
    setFilter,
    setDateRange,
    clearFilters,
    hasActiveFilters,
    filterByDatePreset
  } = useSidebarStore();

  // Filter options
  const filterOptions = [
    { value: FilterType.ALL, label: 'All Time', icon: Filter },
    { value: FilterType.TODAY, label: 'Today', icon: Clock },
    { value: FilterType.WEEK, label: 'This Week', icon: Calendar },
    { value: FilterType.MONTH, label: 'This Month', icon: Calendar }
  ];

  const datePresets = [
    { value: 'today', label: 'Today' },
    { value: 'yesterday', label: 'Yesterday' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'custom', label: 'Custom Range' }
  ];

  // Handle filter selection
  const handleFilterChange = (filterType) => {
    setFilter(filterType);
    setShowDatePicker(false);
  };

  // Handle date preset selection
  const handleDatePreset = (preset) => {
    if (preset === 'custom') {
      setShowDatePicker(true);
    } else {
      filterByDatePreset(preset);
      setShowDatePicker(false);
    }
  };

  // Handle custom date range
  const handleCustomDateRange = () => {
    const start = customStart ? new Date(customStart) : null;
    const end = customEnd ? new Date(customEnd) : null;
    
    // Validate dates
    if (start && end && start > end) {
      alert('Start date must be before end date');
      return;
    }
    
    setDateRange(start, end, 'custom');
    setShowDatePicker(false);
  };

  // Format date range for display
  const formatDateRange = () => {
    if (!dateRange.start && !dateRange.end) return null;
    
    if (dateRange.preset && dateRange.preset !== 'custom') {
      return dateRange.preset.charAt(0).toUpperCase() + dateRange.preset.slice(1);
    }
    
    const start = dateRange.start ? format(dateRange.start, 'MMM d') : 'Any';
    const end = dateRange.end ? format(dateRange.end, 'MMM d') : 'Any';
    
    if (start === end) return start;
    return `${start} - ${end}`;
  };

  return (
    <div className={`filter-controls ${className}`}>
      <div className="filter-header">
        <h4>Filter Conversations</h4>
        {hasActiveFilters() && (
          <button
            onClick={clearFilters}
            className="clear-filters-btn"
            title="Clear all filters"
          >
            <X size={14} />
            Clear
          </button>
        )}
      </div>

      {/* Time-based filters */}
      <div className="filter-section">
        <label className="filter-label">Time Period</label>
        <div className="filter-options">
          {filterOptions.map((option) => {
            const Icon = option.icon;
            const isActive = activeFilter === option.value;
            
            return (
              <button
                key={option.value}
                onClick={() => handleFilterChange(option.value)}
                className={`filter-option ${isActive ? 'active' : ''}`}
                aria-pressed={isActive}
              >
                <Icon size={14} />
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Date range section */}
      <div className="filter-section">
        <div className="date-range-header">
          <label className="filter-label">Date Range</label>
          {dateRange.start || dateRange.end ? (
            <span className="date-range-display">
              {formatDateRange()}
            </span>
          ) : null}
        </div>
        
        <div className="date-presets">
          {datePresets.map((preset) => (
            <button
              key={preset.value}
              onClick={() => handleDatePreset(preset.value)}
              className={`date-preset ${
                dateRange.preset === preset.value ? 'active' : ''
              }`}
            >
              {preset.label}
              {preset.value === 'custom' && (
                <ChevronDown 
                  size={14} 
                  className={`chevron ${showDatePicker ? 'rotated' : ''}`}
                />
              )}
            </button>
          ))}
        </div>

        {/* Custom date picker */}
        {showDatePicker && (
          <div className="custom-date-picker">
            <div className="date-inputs">
              <div className="date-input-group">
                <label htmlFor="start-date">Start Date</label>
                <input
                  id="start-date"
                  type="date"
                  value={customStart}
                  onChange={(e) => setCustomStart(e.target.value)}
                  max={customEnd || format(new Date(), 'yyyy-MM-dd')}
                  className="date-input"
                />
              </div>
              
              <div className="date-input-group">
                <label htmlFor="end-date">End Date</label>
                <input
                  id="end-date"
                  type="date"
                  value={customEnd}
                  onChange={(e) => setCustomEnd(e.target.value)}
                  min={customStart}
                  max={format(new Date(), 'yyyy-MM-dd')}
                  className="date-input"
                />
              </div>
            </div>
            
            <div className="date-picker-actions">
              <button
                onClick={() => setShowDatePicker(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleCustomDateRange}
                className="btn-primary"
                disabled={!customStart && !customEnd}
              >
                Apply
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Quick filters */}
      <div className="filter-section">
        <label className="filter-label">Quick Filters</label>
        <div className="quick-filters">
          <button
            className="quick-filter"
            onClick={() => {
              // Filter for conversations with attachments
              // This would need additional logic in the store
            }}
          >
            📎 With Files
          </button>
          <button
            className="quick-filter"
            onClick={() => {
              // Filter for conversations with voice messages
              // This would need additional logic in the store
            }}
          >
            🎤 Voice Messages
          </button>
          <button
            className="quick-filter"
            onClick={() => {
              // Filter for long conversations
              // This would need additional logic in the store
            }}
          >
            💬 Long Conversations
          </button>
        </div>
      </div>

      {/* Active filters summary */}
      {hasActiveFilters() && (
        <div className="active-filters-summary">
          <h5>Active Filters:</h5>
          <div className="active-filter-tags">
            {activeFilter !== FilterType.ALL && (
              <span className="filter-tag">
                {filterOptions.find(f => f.value === activeFilter)?.label}
                <button
                  onClick={() => setFilter(FilterType.ALL)}
                  className="remove-filter"
                >
                  <X size={12} />
                </button>
              </span>
            )}
            
            {(dateRange.start || dateRange.end) && (
              <span className="filter-tag">
                {formatDateRange()}
                <button
                  onClick={() => setDateRange(null, null)}
                  className="remove-filter"
                >
                  <X size={12} />
                </button>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FilterControls;