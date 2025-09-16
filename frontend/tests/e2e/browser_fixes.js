/**
 * Browser Compatibility Fixes and Polyfills for JARVIS Frontend
 * 
 * This file contains fixes for cross-browser compatibility issues
 * detected during testing. Include this before other scripts.
 */

(function() {
    'use strict';

    // ============================================
    // 1. CSS Custom Properties (Variables) Polyfill
    // ============================================
    if (!window.CSS || !CSS.supports('color', 'var(--test)')) {
        console.warn('CSS custom properties not fully supported, applying polyfill');
        
        // Basic polyfill for CSS variables
        const cssVars = {
            '--jarvis-primary': '#00D4FF',
            '--jarvis-secondary': '#0099CC',
            '--jarvis-accent': '#FF6B6B',
            '--jarvis-dark': '#0A0E27',
            '--jarvis-light': '#E6F3FF'
        };
        
        // Apply variables to root
        const root = document.documentElement;
        Object.keys(cssVars).forEach(key => {
            root.style.setProperty(key, cssVars[key]);
        });
    }

    // ============================================
    // 2. Optional Chaining Polyfill
    // ============================================
    if (!eval("try { ({})?.a } catch { false }")) {
        console.warn('Optional chaining not supported, using polyfill');
        // Note: Full polyfill requires Babel transformation
        // This is a simplified helper function
        window.optionalChain = function(obj, ...keys) {
            return keys.reduce((o, k) => o?.[k], obj);
        };
    }

    // ============================================
    // 3. Nullish Coalescing Polyfill
    // ============================================
    if (!eval("try { null ?? true } catch { false }")) {
        console.warn('Nullish coalescing not supported, using polyfill');
        window.nullishCoalesce = function(value, defaultValue) {
            return value !== null && value !== undefined ? value : defaultValue;
        };
    }

    // ============================================
    // 4. WebSocket Fallback with Long Polling
    // ============================================
    if (typeof WebSocket === 'undefined') {
        console.warn('WebSocket not supported, using long polling fallback');
        
        window.WebSocket = function(url) {
            this.url = url;
            this.readyState = 0;
            this.CONNECTING = 0;
            this.OPEN = 1;
            this.CLOSING = 2;
            this.CLOSED = 3;
            
            this.onopen = null;
            this.onmessage = null;
            this.onerror = null;
            this.onclose = null;
            
            // Simulate connection
            setTimeout(() => {
                this.readyState = 1;
                if (this.onopen) this.onopen();
                this._startPolling();
            }, 100);
            
            this._startPolling = function() {
                const poll = () => {
                    if (this.readyState !== 1) return;
                    
                    fetch(this.url.replace('ws://', 'http://').replace('wss://', 'https://'))
                        .then(response => response.json())
                        .then(data => {
                            if (this.onmessage) {
                                this.onmessage({ data: JSON.stringify(data) });
                            }
                            setTimeout(poll, 1000);
                        })
                        .catch(error => {
                            if (this.onerror) this.onerror(error);
                        });
                };
                poll();
            };
            
            this.send = function(data) {
                fetch(this.url.replace('ws://', 'http://').replace('wss://', 'https://'), {
                    method: 'POST',
                    body: data,
                    headers: { 'Content-Type': 'application/json' }
                });
            };
            
            this.close = function() {
                this.readyState = 3;
                if (this.onclose) this.onclose();
            };
        };
    }

    // ============================================
    // 5. getUserMedia Polyfill
    // ============================================
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('getUserMedia not supported, adding polyfill');
        
        navigator.mediaDevices = navigator.mediaDevices || {};
        
        navigator.mediaDevices.getUserMedia = navigator.mediaDevices.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia ||
            navigator.msGetUserMedia;
        
        if (!navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia = function(constraints) {
                return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
            };
        }
    }

    // ============================================
    // 6. MediaRecorder Polyfill
    // ============================================
    if (typeof MediaRecorder === 'undefined') {
        console.warn('MediaRecorder not supported, using basic polyfill');
        
        window.MediaRecorder = function(stream) {
            this.stream = stream;
            this.state = 'inactive';
            this.ondataavailable = null;
            this.onstop = null;
            
            const chunks = [];
            
            this.start = function() {
                this.state = 'recording';
                // Simulate recording
                setTimeout(() => {
                    if (this.ondataavailable) {
                        this.ondataavailable({
                            data: new Blob([], { type: 'audio/wav' })
                        });
                    }
                }, 1000);
            };
            
            this.stop = function() {
                this.state = 'inactive';
                if (this.onstop) this.onstop();
            };
        };
        
        MediaRecorder.isTypeSupported = function() { return false; };
    }

    // ============================================
    // 7. RequestAnimationFrame Polyfill
    // ============================================
    (function() {
        var lastTime = 0;
        var vendors = ['ms', 'moz', 'webkit', 'o'];
        
        for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
            window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
            window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame'] ||
                                        window[vendors[x]+'CancelRequestAnimationFrame'];
        }
        
        if (!window.requestAnimationFrame) {
            window.requestAnimationFrame = function(callback) {
                var currTime = new Date().getTime();
                var timeToCall = Math.max(0, 16 - (currTime - lastTime));
                var id = window.setTimeout(function() {
                    callback(currTime + timeToCall);
                }, timeToCall);
                lastTime = currTime + timeToCall;
                return id;
            };
        }
        
        if (!window.cancelAnimationFrame) {
            window.cancelAnimationFrame = function(id) {
                clearTimeout(id);
            };
        }
    }());

    // ============================================
    // 8. CSS Animation Vendor Prefixes
    // ============================================
    function addVendorPrefixes() {
        const styles = document.styleSheets;
        
        for (let i = 0; i < styles.length; i++) {
            try {
                const rules = styles[i].cssRules || styles[i].rules;
                
                for (let j = 0; j < rules.length; j++) {
                    const rule = rules[j];
                    
                    if (rule.style) {
                        // Add vendor prefixes for animations
                        if (rule.style.animation) {
                            rule.style.webkitAnimation = rule.style.animation;
                            rule.style.mozAnimation = rule.style.animation;
                        }
                        
                        // Add vendor prefixes for transitions
                        if (rule.style.transition) {
                            rule.style.webkitTransition = rule.style.transition;
                            rule.style.mozTransition = rule.style.transition;
                        }
                        
                        // Add vendor prefixes for transforms
                        if (rule.style.transform) {
                            rule.style.webkitTransform = rule.style.transform;
                            rule.style.mozTransform = rule.style.transform;
                            rule.style.msTransform = rule.style.transform;
                        }
                        
                        // Add vendor prefixes for gradients
                        if (rule.style.background && rule.style.background.includes('gradient')) {
                            const gradient = rule.style.background;
                            rule.style.background = gradient.replace('linear-gradient', '-webkit-linear-gradient');
                        }
                    }
                }
            } catch (e) {
                // Cross-origin stylesheets will throw errors
                console.warn('Cannot access stylesheet:', e);
            }
        }
    }
    
    // Apply prefixes after DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addVendorPrefixes);
    } else {
        addVendorPrefixes();
    }

    // ============================================
    // 9. Touch Event Support for Mobile
    // ============================================
    if (!('ontouchstart' in window)) {
        console.info('Touch events not supported, adding mouse event mapping');
        
        // Map mouse events to touch events for testing
        document.addEventListener('mousedown', function(e) {
            const touchEvent = new CustomEvent('touchstart', {
                bubbles: true,
                cancelable: true,
                detail: { touches: [{ clientX: e.clientX, clientY: e.clientY }] }
            });
            e.target.dispatchEvent(touchEvent);
        });
        
        document.addEventListener('mousemove', function(e) {
            if (e.buttons === 1) {
                const touchEvent = new CustomEvent('touchmove', {
                    bubbles: true,
                    cancelable: true,
                    detail: { touches: [{ clientX: e.clientX, clientY: e.clientY }] }
                });
                e.target.dispatchEvent(touchEvent);
            }
        });
        
        document.addEventListener('mouseup', function(e) {
            const touchEvent = new CustomEvent('touchend', {
                bubbles: true,
                cancelable: true,
                detail: { touches: [] }
            });
            e.target.dispatchEvent(touchEvent);
        });
    }

    // ============================================
    // 10. Performance API Polyfill
    // ============================================
    if (!window.performance || !window.performance.now) {
        console.warn('Performance API not fully supported, adding polyfill');
        
        window.performance = window.performance || {};
        
        window.performance.now = window.performance.now || function() {
            return Date.now();
        };
        
        window.performance.timing = window.performance.timing || {
            navigationStart: Date.now(),
            domContentLoadedEventEnd: Date.now() + 1000,
            loadEventEnd: Date.now() + 2000
        };
    }

    // ============================================
    // 11. Fetch API Polyfill (basic)
    // ============================================
    if (!window.fetch) {
        console.warn('Fetch API not supported, adding XMLHttpRequest polyfill');
        
        window.fetch = function(url, options) {
            options = options || {};
            
            return new Promise(function(resolve, reject) {
                const xhr = new XMLHttpRequest();
                
                xhr.open(options.method || 'GET', url, true);
                
                // Set headers
                if (options.headers) {
                    Object.keys(options.headers).forEach(key => {
                        xhr.setRequestHeader(key, options.headers[key]);
                    });
                }
                
                xhr.onload = function() {
                    const response = {
                        ok: xhr.status >= 200 && xhr.status < 300,
                        status: xhr.status,
                        statusText: xhr.statusText,
                        text: function() { return Promise.resolve(xhr.responseText); },
                        json: function() { return Promise.resolve(JSON.parse(xhr.responseText)); }
                    };
                    resolve(response);
                };
                
                xhr.onerror = function() {
                    reject(new Error('Network request failed'));
                };
                
                xhr.send(options.body || null);
            });
        };
    }

    // ============================================
    // 12. Promise Polyfill (basic)
    // ============================================
    if (typeof Promise === 'undefined') {
        console.warn('Promise not supported, adding basic polyfill');
        
        window.Promise = function(executor) {
            var self = this;
            this.state = 'pending';
            this.value = undefined;
            this.callbacks = [];
            
            function resolve(value) {
                if (self.state === 'pending') {
                    self.state = 'fulfilled';
                    self.value = value;
                    self.callbacks.forEach(cb => cb.onFulfilled(value));
                }
            }
            
            function reject(reason) {
                if (self.state === 'pending') {
                    self.state = 'rejected';
                    self.value = reason;
                    self.callbacks.forEach(cb => cb.onRejected(reason));
                }
            }
            
            this.then = function(onFulfilled, onRejected) {
                return new Promise(function(resolve, reject) {
                    function handle() {
                        try {
                            if (self.state === 'fulfilled') {
                                const result = onFulfilled ? onFulfilled(self.value) : self.value;
                                resolve(result);
                            } else if (self.state === 'rejected') {
                                if (onRejected) {
                                    const result = onRejected(self.value);
                                    resolve(result);
                                } else {
                                    reject(self.value);
                                }
                            }
                        } catch (e) {
                            reject(e);
                        }
                    }
                    
                    if (self.state === 'pending') {
                        self.callbacks.push({
                            onFulfilled: function(value) {
                                handle();
                            },
                            onRejected: function(reason) {
                                handle();
                            }
                        });
                    } else {
                        handle();
                    }
                });
            };
            
            this.catch = function(onRejected) {
                return this.then(null, onRejected);
            };
            
            executor(resolve, reject);
        };
        
        Promise.resolve = function(value) {
            return new Promise(function(resolve) { resolve(value); });
        };
        
        Promise.reject = function(reason) {
            return new Promise(function(_, reject) { reject(reason); });
        };
    }

    // ============================================
    // 13. Safari-specific Fixes
    // ============================================
    if (/^((?!chrome|android).)*safari/i.test(navigator.userAgent)) {
        console.info('Safari browser detected, applying specific fixes');
        
        // Fix for Safari date parsing
        if (!Date.prototype.toISOString) {
            Date.prototype.toISOString = function() {
                return this.getUTCFullYear() + '-' +
                    ('0' + (this.getUTCMonth() + 1)).slice(-2) + '-' +
                    ('0' + this.getUTCDate()).slice(-2) + 'T' +
                    ('0' + this.getUTCHours()).slice(-2) + ':' +
                    ('0' + this.getUTCMinutes()).slice(-2) + ':' +
                    ('0' + this.getUTCSeconds()).slice(-2) + 'Z';
            };
        }
        
        // Fix for Safari flexbox bugs
        document.addEventListener('DOMContentLoaded', function() {
            const flexContainers = document.querySelectorAll('[style*="display: flex"]');
            flexContainers.forEach(container => {
                container.style.webkitBoxOrient = 'horizontal';
                container.style.webkitBoxDirection = 'normal';
            });
        });
    }

    // ============================================
    // 14. IE11 Specific Fixes (if needed)
    // ============================================
    if (navigator.userAgent.indexOf('Trident/') > -1) {
        console.warn('Internet Explorer detected, applying compatibility fixes');
        
        // Array.from polyfill
        if (!Array.from) {
            Array.from = function(object) {
                return [].slice.call(object);
            };
        }
        
        // Array.includes polyfill
        if (!Array.prototype.includes) {
            Array.prototype.includes = function(searchElement) {
                return this.indexOf(searchElement) !== -1;
            };
        }
        
        // Object.assign polyfill
        if (!Object.assign) {
            Object.assign = function(target) {
                for (var i = 1; i < arguments.length; i++) {
                    var source = arguments[i];
                    for (var key in source) {
                        if (Object.prototype.hasOwnProperty.call(source, key)) {
                            target[key] = source[key];
                        }
                    }
                }
                return target;
            };
        }
    }

    // ============================================
    // 15. Mobile Viewport Fix
    // ============================================
    function fixMobileViewport() {
        const viewport = document.querySelector('meta[name="viewport"]');
        
        if (!viewport) {
            const meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes';
            document.head.appendChild(meta);
        }
        
        // Prevent iOS bounce scrolling
        if (/iPad|iPhone|iPod/.test(navigator.userAgent)) {
            document.body.style.webkitOverflowScrolling = 'touch';
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fixMobileViewport);
    } else {
        fixMobileViewport();
    }

    // ============================================
    // 16. Console Polyfill for Old Browsers
    // ============================================
    if (!window.console) {
        window.console = {
            log: function() {},
            error: function() {},
            warn: function() {},
            info: function() {},
            debug: function() {}
        };
    }

    // ============================================
    // Browser Compatibility Report
    // ============================================
    console.info('Browser Compatibility Fixes Loaded');
    console.info('User Agent:', navigator.userAgent);
    console.info('Browser Features:', {
        cssVariables: window.CSS && CSS.supports('color', 'var(--test)'),
        webSocket: typeof WebSocket !== 'undefined',
        getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        mediaRecorder: typeof MediaRecorder !== 'undefined',
        fetch: typeof fetch !== 'undefined',
        promises: typeof Promise !== 'undefined',
        touchEvents: 'ontouchstart' in window,
        serviceWorker: 'serviceWorker' in navigator
    });

})();