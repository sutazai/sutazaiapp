"""
Enhanced Enter Key Handler Component for Streamlit
Provides comprehensive Enter-to-submit functionality across all inputs
"""

import streamlit as st
from streamlit.components.v1 import html

def add_enter_key_handler():
    """Add comprehensive Enter key handling to all inputs in the UI"""
    
    enter_key_js = """
    <script>
    (function() {
        'use strict';
        
        // Configuration
        const SELECTORS = {
            chatInput: '[data-testid="stChatInput"] input',
            textInputs: 'input[type="text"]:not([readonly]):not([disabled])',
            textareas: 'textarea:not([readonly]):not([disabled])',
            buttons: {
                primary: 'button[kind="primary"]',
                submit: 'button[type="submit"]',
                executeTask: 'button:contains("🚀 Execute Task")',
                solveProblem: 'button:contains("🧠 Solve Problem")',
                addKnowledge: 'button:contains("📚 Add Knowledge")',
                search: 'button:contains("🔍 Search")',
                send: 'button:contains("Send")',
                generate: 'button:contains("Generate")'
            }
        };
        
        // Helper function to find submit button for an input
        function findSubmitButton(inputElement) {
            // Try to find button in the same container
            let container = inputElement.closest('.stForm') || 
                           inputElement.closest('[data-testid="column"]') || 
                           inputElement.closest('.element-container');
            
            if (container) {
                // Look for specific buttons based on context
                let button = container.querySelector(SELECTORS.buttons.primary) ||
                           container.querySelector(SELECTORS.buttons.submit) ||
                           container.querySelector(SELECTORS.buttons.executeTask) ||
                           container.querySelector(SELECTORS.buttons.solveProblem) ||
                           container.querySelector(SELECTORS.buttons.addKnowledge) ||
                           container.querySelector(SELECTORS.buttons.search) ||
                           container.querySelector(SELECTORS.buttons.send) ||
                           container.querySelector(SELECTORS.buttons.generate);
                
                if (button && !button.disabled) {
                    return button;
                }
            }
            
            // Fallback: look for any submit button in the entire document
            const buttons = document.querySelectorAll('button');
            for (let btn of buttons) {
                if (!btn.disabled && (
                    btn.textContent.includes('Execute') ||
                    btn.textContent.includes('Send') ||
                    btn.textContent.includes('Submit') ||
                    btn.textContent.includes('Search') ||
                    btn.textContent.includes('Add') ||
                    btn.textContent.includes('Generate') ||
                    btn.getAttribute('kind') === 'primary'
                )) {
                    return btn;
                }
            }
            
            return null;
        }
        
        // Enhanced Enter key handler
        function addEnterKeyListener(element) {
            if (element.hasEnterKeyHandler) return;
            element.hasEnterKeyHandler = true;
            
            element.addEventListener('keydown', function(event) {
                if (event.key === 'Enter') {
                    // Allow Shift+Enter for multi-line in textareas
                    if (event.shiftKey && this.tagName.toLowerCase() === 'textarea') {
                        return; // Let normal behavior continue
                    }
                    
                    event.preventDefault();
                    
                    // Special handling for chat input
                    if (this.closest('[data-testid="stChatInput"]')) {
                        // Streamlit chat input has built-in Enter handling
                        const submitEvent = new KeyboardEvent('keydown', {
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13,
                            which: 13,
                            bubbles: true,
                            cancelable: true
                        });
                        this.dispatchEvent(submitEvent);
                        return;
                    }
                    
                    // Find and click the appropriate submit button
                    const submitButton = findSubmitButton(this);
                    if (submitButton) {
                        console.log('Enter key: Clicking button:', submitButton.textContent);
                        submitButton.click();
                    } else {
                        console.log('Enter key: No submit button found for input:', this);
                    }
                }
            });
        }
        
        // Function to attach listeners to all inputs
        function attachEnterListeners() {
            try {
                // Handle all text inputs
                const textInputs = document.querySelectorAll(SELECTORS.textInputs);
                textInputs.forEach(addEnterKeyListener);
                
                // Handle all textareas
                const textareas = document.querySelectorAll(SELECTORS.textareas);
                textareas.forEach(addEnterKeyListener);
                
                // Handle chat input specifically
                const chatInput = document.querySelector(SELECTORS.chatInput);
                if (chatInput) {
                    addEnterKeyListener(chatInput);
                }
                
            } catch (error) {
                console.error('Error attaching enter listeners:', error);
            }
        }
        
        // Initialize when DOM is ready
        function initialize() {
            attachEnterListeners();
            
            // Set up MutationObserver for dynamic content
            const observer = new MutationObserver(function(mutations) {
                let shouldReattach = false;
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length > 0) {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === 1) { // Element node
                                const hasInputs = node.querySelectorAll && (
                                    node.querySelectorAll('input[type="text"], textarea').length > 0 ||
                                    node.querySelector('[data-testid="stChatInput"]')
                                );
                                if (hasInputs) {
                                    shouldReattach = true;
                                }
                            }
                        });
                    }
                });
                
                if (shouldReattach) {
                    setTimeout(attachEnterListeners, 100);
                }
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            // Periodic recheck for reliability
            setInterval(attachEnterListeners, 3000);
        }
        
        // Start when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initialize);
        } else {
            initialize();
        }
        
        // Also initialize after a short delay to catch late-loading elements
        setTimeout(initialize, 1000);
        
    })();
    </script>
    """
    
    # Inject the JavaScript
    html(enter_key_js, height=0)

def show_enter_key_hint(message="💡 Tip: Press Enter to submit"):
    """Show a helpful hint about Enter key functionality"""
    st.markdown(f'<div class="enter-hint">{message}</div>', unsafe_allow_html=True)