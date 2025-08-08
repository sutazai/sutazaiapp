#!/usr/bin/env python3
"""
Ollama + TinyLlama Server Health Verification Script

Verifies Ollama service health and TinyLlama model availability based on CLAUDE.md truth document.
This script follows Rule 16: Use Local LLMs Exclusively via Ollama, Default to TinyLlama

Usage:
    python scripts/devops/health_check_ollama.py --host localhost --port 10104
    python scripts/devops/health_check_ollama.py --url http://localhost:10104
    python scripts/devops/health_check_ollama.py --timeout 10 --verbose

Created: December 19, 2024
Author: infrastructure-devops-manager agent
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import urllib.request
import urllib.parse
import urllib.error
import socket


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration with timestamp."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_tcp_connection(host: str, port: int, timeout: float) -> bool:
    """Check TCP connectivity to Ollama server."""
    try:
        start_time = time.time()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            latency = int((time.time() - start_time) * 1000)
            logging.info(f"TCP connection to {host}:{port} successful (~{latency}ms)")
            return True
    except Exception as e:
        logging.error(f"TCP connection to {host}:{port} failed: {e}")
        return False


def make_http_request(url: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Make HTTP request and return JSON response."""
    try:
        start_time = time.time()
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = int((time.time() - start_time) * 1000)
            content = response.read().decode('utf-8')
            
            logging.debug(f"HTTP {response.getcode()} from {url} in {latency}ms")
            
            if content:
                return json.loads(content)
            return {"status": "ok", "latency_ms": latency}
            
    except urllib.error.HTTPError as e:
        logging.error(f"HTTP error {e.code} from {url}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        logging.error(f"URL error from {url}: {e.reason}")
        return None
    except json.JSONDecodeError as e:
        logging.warning(f"Non-JSON response from {url}: {e}")
        return {"status": "non_json_response"}
    except Exception as e:
        logging.error(f"Request to {url} failed: {e}")
        return None


def check_ollama_api(base_url: str, timeout: float) -> bool:
    """Check Ollama API health endpoint."""
    url = f"{base_url.rstrip('/')}/api/tags"
    response = make_http_request(url, timeout)
    
    if response is None:
        return False
    
    if 'models' in response:
        logging.info(f"Ollama API responding with {len(response['models'])} models")
        return True
    elif response.get('status') == 'ok':
        logging.info("Ollama API responding (basic health check)")
        return True
    else:
        logging.warning("Ollama API responding but with unexpected format")
        return False


def check_tinyllama_model(base_url: str, timeout: float) -> bool:
    """Verify TinyLlama model is loaded (per CLAUDE.md truth document)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    response = make_http_request(url, timeout)
    
    if response is None:
        return False
    
    if 'models' not in response:
        logging.warning("No models list in Ollama API response")
        return False
    
    models = response['models']
    tinyllama_found = False
    
    for model in models:
        model_name = model.get('name', '').lower()
        if 'tinyllama' in model_name:
            tinyllama_found = True
            model_size = model.get('size', 0)
            modified = model.get('modified_at', 'unknown')
            logging.info(f"TinyLlama model found: {model['name']} (size: {model_size}, modified: {modified})")
            break
    
    if not tinyllama_found:
        logging.error("TinyLlama model not found in loaded models")
        logging.info(f"Available models: {[m.get('name') for m in models]}")
        return False
    
    return True


def test_text_generation(base_url: str, timeout: float) -> bool:
    """Test text generation with TinyLlama model."""
    url = f"{base_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": "tinyllama",
        "prompt": "What is Docker?",
        "stream": False
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        req.add_header('Content-Type', 'application/json')
        
        start_time = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as response:
            generation_time = int((time.time() - start_time) * 1000)
            content = response.read().decode('utf-8')
            
            result = json.loads(content)
            if 'response' in result:
                response_text = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
                logging.info(f"Text generation successful in {generation_time}ms: {response_text}")
                return True
            else:
                logging.warning(f"Generation response missing 'response' field: {result}")
                return False
                
    except Exception as e:
        logging.error(f"Text generation test failed: {e}")
        return False


def main():
    """Main function with comprehensive Ollama health verification."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Ollama + TinyLlama health verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/devops/health_check_ollama.py
    python scripts/devops/health_check_ollama.py --host localhost --port 10104
    python scripts/devops/health_check_ollama.py --url http://127.0.0.1:10104
    python scripts/devops/health_check_ollama.py --timeout 10 --verbose
        """
    )
    
    parser.add_argument('--host', default='localhost',
                       help='Ollama server host (default: localhost)')
    parser.add_argument('--port', type=int, default=10104,
                       help='Ollama server port (default: 10104)')
    parser.add_argument('--url', 
                       help='Full Ollama URL (overrides host:port)')
    parser.add_argument('--timeout', type=float, default=10.0,
                       help='Request timeout in seconds (default: 10.0)')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip text generation test')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Determine base URL
    if args.url:
        base_url = args.url
        # Extract host and port for TCP check
        parsed = urllib.parse.urlparse(args.url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 10104
    else:
        host = args.host
        port = args.port
        base_url = f"http://{host}:{port}"
    
    logging.info(f"Starting Ollama health check for {base_url}")
    
    # Health check results
    results = {
        'timestamp': datetime.now().isoformat(),
        'service': 'ollama',
        'base_url': base_url,
        'checks': {}
    }
    
    all_passed = True
    
    # 1. TCP connectivity check
    logging.info("Step 1: Checking TCP connectivity...")
    tcp_ok = check_tcp_connection(host, port, args.timeout)
    results['checks']['tcp_connectivity'] = tcp_ok
    if not tcp_ok:
        all_passed = False
    
    # 2. Ollama API health check
    logging.info("Step 2: Checking Ollama API health...")
    api_ok = check_ollama_api(base_url, args.timeout)
    results['checks']['api_health'] = api_ok
    if not api_ok:
        all_passed = False
    
    # 3. TinyLlama model verification
    logging.info("Step 3: Verifying TinyLlama model...")
    model_ok = check_tinyllama_model(base_url, args.timeout)
    results['checks']['tinyllama_model'] = model_ok
    if not model_ok:
        all_passed = False
    
    # 4. Text generation test (optional)
    if not args.skip_generation and model_ok:
        logging.info("Step 4: Testing text generation...")
        generation_ok = test_text_generation(base_url, args.timeout)
        results['checks']['text_generation'] = generation_ok
        if not generation_ok:
            all_passed = False
    else:
        results['checks']['text_generation'] = 'skipped'
    
    # Summary
    results['overall_status'] = 'healthy' if all_passed else 'unhealthy'
    results['checks_passed'] = sum(1 for v in results['checks'].values() if v is True)
    results['checks_total'] = sum(1 for v in results['checks'].values() if v is not 'skipped')
    
    if all_passed:
        logging.info("✅ All Ollama health checks passed")
        logging.info(f"Service: {base_url} is healthy and ready")
    else:
        logging.error("❌ One or more Ollama health checks failed")
        logging.error(f"Service: {base_url} requires attention")
    
    # Output results as JSON for CI/CD integration
    if args.verbose:
        print(json.dumps(results, indent=2))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())