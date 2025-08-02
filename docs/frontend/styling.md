# Frontend Styling Guide

## Overview

The SutazAI frontend uses a modern, responsive design system built on top of Streamlit with custom CSS enhancements. The styling focuses on clarity, accessibility, and professional appearance while maintaining consistency across all components.

## Design System

### Color Palette

#### Primary Colors
```css
:root {
  --primary-blue: #007bff;
  --primary-dark: #0056b3;
  --primary-light: #66b3ff;
  
  --secondary-gray: #6c757d;
  --secondary-light: #e9ecef;
  
  --success-green: #28a745;
  --warning-yellow: #ffc107;
  --danger-red: #dc3545;
  --info-blue: #17a2b8;
}
```

#### Status Colors
```css
/* Agent and Task Status Colors */
.status-active { color: var(--success-green); }
.status-inactive { color: var(--secondary-gray); }
.status-error { color: var(--danger-red); }
.status-warning { color: var(--warning-yellow); }
.status-processing { color: var(--info-blue); }

/* Background variants */
.bg-status-active { background-color: #d4edda; border-color: var(--success-green); }
.bg-status-inactive { background-color: #f8f9fa; border-color: var(--secondary-gray); }
.bg-status-error { background-color: #f8d7da; border-color: var(--danger-red); }
```

#### Semantic Colors
```css
/* Semantic color usage */
.text-primary { color: var(--primary-blue); }
.text-secondary { color: var(--secondary-gray); }
.text-success { color: var(--success-green); }
.text-warning { color: var(--warning-yellow); }
.text-danger { color: var(--danger-red); }
.text-info { color: var(--info-blue); }
```

### Typography

#### Font Families
```css
:root {
  --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'Fira Code', 'Monaco', 'Cascadia Code', monospace;
  --font-heading: 'Inter', sans-serif;
}

body {
  font-family: var(--font-primary);
  line-height: 1.6;
  color: #333;
}

/* Code and monospace text */
code, pre, .code-block {
  font-family: var(--font-mono);
  font-size: 0.9em;
}
```

#### Typography Scale
```css
/* Heading styles */
h1, .h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; }
h2, .h2 { font-size: 2rem; font-weight: 600; margin-bottom: 0.8rem; }
h3, .h3 { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.6rem; }
h4, .h4 { font-size: 1.25rem; font-weight: 500; margin-bottom: 0.5rem; }
h5, .h5 { font-size: 1rem; font-weight: 500; margin-bottom: 0.4rem; }

/* Body text styles */
.text-large { font-size: 1.125rem; }
.text-normal { font-size: 1rem; }
.text-small { font-size: 0.875rem; }
.text-xs { font-size: 0.75rem; }

/* Font weights */
.font-light { font-weight: 300; }
.font-normal { font-weight: 400; }
.font-medium { font-weight: 500; }
.font-semibold { font-weight: 600; }
.font-bold { font-weight: 700; }
```

### Spacing System

#### Margin and Padding Scale
```css
:root {
  --spacing-xs: 0.25rem;   /* 4px */
  --spacing-sm: 0.5rem;    /* 8px */
  --spacing-md: 1rem;      /* 16px */
  --spacing-lg: 1.5rem;    /* 24px */
  --spacing-xl: 2rem;      /* 32px */
  --spacing-2xl: 3rem;     /* 48px */
  --spacing-3xl: 4rem;     /* 64px */
}

/* Utility classes */
.m-0 { margin: 0; }
.m-xs { margin: var(--spacing-xs); }
.m-sm { margin: var(--spacing-sm); }
.m-md { margin: var(--spacing-md); }
.m-lg { margin: var(--spacing-lg); }

.p-0 { padding: 0; }
.p-xs { padding: var(--spacing-xs); }
.p-sm { padding: var(--spacing-sm); }
.p-md { padding: var(--spacing-md); }
.p-lg { padding: var(--spacing-lg); }
```

## Component Styles

### Agent Cards
```css
.agent-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1rem 0;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
  border: 1px solid #e2e8f0;
  border-left: 4px solid var(--primary-blue);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.agent-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  border-left-color: var(--primary-dark);
}

.agent-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--primary-blue), var(--primary-light));
}

.agent-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.agent-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0;
}

.agent-status {
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.agent-description {
  color: #4a5568;
  margin-bottom: 1rem;
  line-height: 1.5;
}

.agent-capabilities {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.capability-tag {
  background: #f7fafc;
  color: #2d3748;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-size: 0.75rem;
  border: 1px solid #e2e8f0;
}

.agent-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}
```

### Task Components
```css
.task-container {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1rem 0;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #e2e8f0;
}

.task-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.task-status-badge {
  padding: 0.375rem 0.75rem;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.task-progress {
  margin: 1rem 0;
}

.progress-bar {
  background: #e2e8f0;
  border-radius: 4px;
  height: 8px;
  overflow: hidden;
  position: relative;
}

.progress-fill {
  background: linear-gradient(90deg, var(--primary-blue), var(--primary-light));
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 4px;
}

.task-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e2e8f0;
}

.meta-item {
  text-align: center;
}

.meta-label {
  font-size: 0.75rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.25rem;
}

.meta-value {
  font-size: 1rem;
  font-weight: 600;
  color: #2d3748;
}
```

### Form Styles
```css
.form-container {
  background: white;
  border-radius: 8px;
  padding: 2rem;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-label {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.5rem;
}

.form-input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 1rem;
  transition: border-color 0.2s ease;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-blue);
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.form-textarea {
  resize: vertical;
  min-height: 120px;
  font-family: var(--font-mono);
}

.form-select {
  background-image: url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3E%3Cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3E%3C/svg%3E");
  background-position: right 0.5rem center;
  background-repeat: no-repeat;
  background-size: 1.5em 1.5em;
  padding-right: 2.5rem;
}

.form-help {
  font-size: 0.875rem;
  color: #6b7280;
  margin-top: 0.25rem;
}
```

### Button Styles
```css
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.btn-primary {
  background: var(--primary-blue);
  color: white;
}

.btn-primary:hover {
  background: var(--primary-dark);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4);
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background: #5a6268;
}

.btn-success {
  background: var(--success-green);
  color: white;
}

.btn-danger {
  background: var(--danger-red);
  color: white;
}

.btn-outline {
  background: transparent;
  border: 1px solid var(--primary-blue);
  color: var(--primary-blue);
}

.btn-outline:hover {
  background: var(--primary-blue);
  color: white;
}

.btn-small {
  padding: 0.5rem 1rem;
  font-size: 0.75rem;
}

.btn-large {
  padding: 1rem 2rem;
  font-size: 1rem;
}

.btn-icon {
  padding: 0.75rem;
  border-radius: 50%;
}
```

### Dashboard Styles
```css
.dashboard-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 1rem 0;
}

.dashboard-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  position: relative;
  overflow: hidden;
}

.dashboard-card-header {
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 1rem;
}

.dashboard-card-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
}

.dashboard-card-icon {
  font-size: 1.5rem;
  opacity: 0.7;
}

.metric-container {
  text-align: center;
  padding: 1rem;
}

.metric-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--primary-blue);
  margin-bottom: 0.5rem;
}

.metric-label {
  font-size: 0.875rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-change {
  font-size: 0.875rem;
  font-weight: 500;
  margin-top: 0.25rem;
}

.metric-change.positive {
  color: var(--success-green);
}

.metric-change.negative {
  color: var(--danger-red);
}
```

### Navigation Styles
```css
.sidebar-nav {
  background: #f8fafc;
  border-right: 1px solid #e2e8f0;
  padding: 1rem;
}

.nav-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--primary-blue);
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.nav-menu {
  list-style: none;
  padding: 0;
  margin: 0;
}

.nav-item {
  margin-bottom: 0.5rem;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  text-decoration: none;
  color: #4a5568;
  font-weight: 500;
  transition: all 0.2s ease;
}

.nav-link:hover {
  background: #e2e8f0;
  color: var(--primary-blue);
}

.nav-link.active {
  background: var(--primary-blue);
  color: white;
  box-shadow: 0 2px 4px rgba(0, 123, 255, 0.3);
}

.nav-icon {
  font-size: 1.25rem;
}

.breadcrumb {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid #e2e8f0;
}

.breadcrumb-item {
  color: #718096;
  font-size: 0.875rem;
}

.breadcrumb-separator {
  color: #cbd5e0;
}

.breadcrumb-current {
  color: var(--primary-blue);
  font-weight: 500;
}
```

## Responsive Design

### Mobile-First Approach
```css
/* Base styles (mobile) */
.container {
  padding: 1rem;
  margin: 0 auto;
  max-width: 100%;
}

/* Tablet styles */
@media (min-width: 768px) {
  .container {
    padding: 1.5rem;
    max-width: 1200px;
  }
  
  .dashboard-container {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .agent-card {
    padding: 1.5rem;
  }
}

/* Desktop styles */
@media (min-width: 1024px) {
  .container {
    padding: 2rem;
  }
  
  .dashboard-container {
    grid-template-columns: repeat(3, 1fr);
  }
  
  .form-container {
    max-width: 600px;
    margin: 0 auto;
  }
}

/* Large desktop styles */
@media (min-width: 1280px) {
  .dashboard-container {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

### Mobile-Specific Adjustments
```css
@media (max-width: 767px) {
  .agent-card {
    margin: 0.5rem 0;
    padding: 1rem;
  }
  
  .agent-actions {
    flex-direction: column;
  }
  
  .btn {
    width: 100%;
    justify-content: center;
  }
  
  .task-meta {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  
  .nav-link {
    padding: 1rem;
    font-size: 1rem;
  }
  
  .metric-value {
    font-size: 2rem;
  }
}
```

## Dark Mode Support

### CSS Variables for Theming
```css
:root {
  /* Light theme variables */
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --text-primary: #1a202c;
  --text-secondary: #4a5568;
  --border-color: #e2e8f0;
  --shadow: rgba(0, 0, 0, 0.1);
}

[data-theme="dark"] {
  /* Dark theme variables */
  --bg-primary: #1a202c;
  --bg-secondary: #2d3748;
  --text-primary: #f7fafc;
  --text-secondary: #e2e8f0;
  --border-color: #4a5568;
  --shadow: rgba(255, 255, 255, 0.1);
}

/* Apply theme variables */
body {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
}

.agent-card,
.task-container,
.dashboard-card {
  background-color: var(--bg-primary);
  border-color: var(--border-color);
  box-shadow: 0 2px 4px var(--shadow);
}
```

## Animation and Transitions

### Loading States
```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.loading-skeleton {
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.spinner {
  border: 2px solid #f3f3f3;
  border-top: 2px solid var(--primary-blue);
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
}
```

### Hover Effects
```css
.hover-lift {
  transition: transform 0.2s ease;
}

.hover-lift:hover {
  transform: translateY(-2px);
}

.hover-shadow {
  transition: box-shadow 0.3s ease;
}

.hover-shadow:hover {
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}
```

## Accessibility Features

### Focus States
```css
.focus-ring {
  outline: 2px solid transparent;
  outline-offset: 2px;
}

.focus-ring:focus {
  outline-color: var(--primary-blue);
  outline-offset: 0;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .agent-card,
  .task-container {
    border-width: 2px;
  }
  
  .btn {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### Screen Reader Support
```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}
```

## Custom CSS Integration with Streamlit

### CSS Injection Method
```python
def load_css():
    """Load custom CSS styles into Streamlit"""
    
    css_file = Path(__file__).parent / "static" / "css" / "styles.css"
    
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Additional inline styles
    st.markdown("""
    <style>
    /* Streamlit specific overrides */
    .stSelectbox > div > div > div {
        border-radius: 6px;
    }
    
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stProgress .stProgress-bar {
        background: linear-gradient(90deg, var(--primary-blue), var(--primary-light));
    }
    </style>
    """, unsafe_allow_html=True)
```

This comprehensive styling guide ensures a consistent, professional, and accessible user interface for the SutazAI frontend application.