#!/usr/bin/env python3
"""
mitmproxy script to intercept Claude API requests and set temperature to 0.2
Usage: mitmproxy -s claude-temp-interceptor.py
"""

import json
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    # Only intercept Claude API requests
    if "api.anthropic.com" in flow.request.pretty_host:
        if flow.request.method == "POST" and "/messages" in flow.request.path:
            try:
                # Parse the JSON body
                body = json.loads(flow.request.get_text())
                
                # Force temperature to 0.2 for accuracy
                body["temperature"] = 0.2
                body["top_p"] = 0.6
                
                # Update the request
                flow.request.set_text(json.dumps(body))
                
                print(f"[INTERCEPTED] Set temperature to 0.2 for Claude API request")
            except Exception as e:
                print(f"Error modifying request: {e}")

def response(flow: http.HTTPFlow) -> None:
    # Log successful interceptions
    if "api.anthropic.com" in flow.request.pretty_host:
        print(f"[RESPONSE] Status: {flow.response.status_code}")