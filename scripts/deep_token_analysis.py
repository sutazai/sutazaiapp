#!/usr/bin/env python3
"""
Deep Token Usage Analysis Tool
Performs comprehensive analysis of Claude Code token consumption
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

class TokenAnalyzer:
    def __init__(self):
        self.results = {}
        self.CHARS_PER_TOKEN = 4  # Approximate
        
    def analyze_agent_registry(self):
        """Analyze agent registry token usage"""
        registry_path = Path('/root/.claude/agents/agent_registry.json')
        backup_path = Path('/root/.claude/agents/agent_registry.json.backup')
        
        current_stats = {}
        if registry_path.exists():
            size = registry_path.stat().st_size
            current_stats['size_bytes'] = size
            current_stats['tokens'] = size // self.CHARS_PER_TOKEN
            
            with open(registry_path) as f:
                data = json.load(f)
                agents = data.get('agents', {})
                current_stats['agent_count'] = len(agents)
                
                # Analyze compression quality
                if agents:
                    sample = next(iter(agents.values()))
                    current_stats['keys_per_agent'] = len(sample.keys())
                    
        backup_stats = {}
        if backup_path.exists():
            size = backup_path.stat().st_size
            backup_stats['size_bytes'] = size
            backup_stats['tokens'] = size // self.CHARS_PER_TOKEN
            
        reduction = 0
        if backup_stats and current_stats:
            reduction = (backup_stats['size_bytes'] - current_stats['size_bytes']) / backup_stats['size_bytes'] * 100
            
        return {
            'current': current_stats,
            'backup': backup_stats,
            'reduction_percent': reduction,
            'tokens_saved': backup_stats.get('tokens', 0) - current_stats.get('tokens', 0)
        }
    
    def analyze_memory_usage(self):
        """Analyze memory MCP accumulation"""
        memory_dirs = [
            '/opt/sutazaiapp/backend/memory-bank',
            '/tmp/memory',
            os.path.expanduser('~/.memory')
        ]
        
        total_size = 0
        file_count = 0
        
        for dir_path in memory_dirs:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith(('.json', '.db')):
                            file_path = os.path.join(root, file)
                            size = os.path.getsize(file_path)
                            total_size += size
                            file_count += 1
                            
        return {
            'total_bytes': total_size,
            'estimated_tokens': total_size // self.CHARS_PER_TOKEN,
            'file_count': file_count,
            'directories': memory_dirs
        }
    
    def analyze_mcp_servers(self):
        """Analyze MCP server configurations"""
        mcp_configs = []
        
        # Check original config
        original_path = Path('/opt/sutazaiapp/.mcp.json')
        if original_path.exists():
            with open(original_path) as f:
                data = json.load(f)
                mcp_configs.append({
                    'type': 'original',
                    'server_count': len(data.get('mcpServers', {})),
                    'servers': list(data.get('mcpServers', {}).keys())
                })
        
        # Check optimized config
        optimized_path = Path('/opt/sutazaiapp/.mcp-optimized.json')
        if optimized_path.exists():
            with open(optimized_path) as f:
                data = json.load(f)
                mcp_configs.append({
                    'type': 'optimized',
                    'server_count': len(data.get('mcpServers', {})),
                    'servers': list(data.get('mcpServers', {}).keys()),
                    'has_limits': any('settings' in v for v in data.get('mcpServers', {}).values())
                })
                
        return mcp_configs
    
    def estimate_total_tokens(self):
        """Calculate total token usage"""
        # Claude Code baseline (system prompt + tools)
        CLAUDE_BASELINE = 14000
        
        # Get current measurements
        agent_data = self.analyze_agent_registry()
        memory_data = self.analyze_memory_usage()
        
        # Current token usage
        current_tokens = CLAUDE_BASELINE
        current_tokens += agent_data['current'].get('tokens', 0)
        current_tokens += memory_data['estimated_tokens']
        
        # Original token usage (if backup exists)
        original_tokens = CLAUDE_BASELINE
        original_tokens += agent_data['backup'].get('tokens', 0)
        original_tokens += 2000  # Estimated unoptimized memory
        
        return {
            'current_total': current_tokens,
            'original_total': original_tokens if agent_data['backup'] else None,
            'breakdown': {
                'claude_baseline': CLAUDE_BASELINE,
                'agent_registry': agent_data['current'].get('tokens', 0),
                'memory_mcp': memory_data['estimated_tokens']
            },
            'savings': original_tokens - current_tokens if agent_data['backup'] else 0,
            'reduction_percent': ((original_tokens - current_tokens) / original_tokens * 100) if agent_data['backup'] else 0
        }
    
    def run_performance_test(self):
        """Test actual performance impact"""
        import time
        
        results = {}
        
        # Test agent registry parse time
        registry_path = '/root/.claude/agents/agent_registry.json'
        if os.path.exists(registry_path):
            start = time.time()
            with open(registry_path) as f:
                json.load(f)
            results['registry_parse_ms'] = (time.time() - start) * 1000
            
        # Test backup parse time
        backup_path = '/root/.claude/agents/agent_registry.json.backup'
        if os.path.exists(backup_path):
            start = time.time()
            with open(backup_path) as f:
                json.load(f)
            results['backup_parse_ms'] = (time.time() - start) * 1000
            
        return results
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("="*60)
        print("     CLAUDE CODE TOKEN OPTIMIZATION - DEEP ANALYSIS")
        print("="*60)
        
        # Agent Registry Analysis
        agent_data = self.analyze_agent_registry()
        print("\n📊 AGENT REGISTRY ANALYSIS")
        print("-"*40)
        if agent_data['current']:
            print(f"Current Size: {agent_data['current']['size_bytes']:,} bytes")
            print(f"Current Tokens: ~{agent_data['current']['tokens']:,} tokens")
            print(f"Agent Count: {agent_data['current']['agent_count']} agents")
            print(f"Keys per Agent: {agent_data['current'].get('keys_per_agent', 'N/A')}")
        
        if agent_data['backup']:
            print(f"\nOriginal Size: {agent_data['backup']['size_bytes']:,} bytes")
            print(f"Original Tokens: ~{agent_data['backup']['tokens']:,} tokens")
            print(f"Reduction: {agent_data['reduction_percent']:.1f}%")
            print(f"Tokens Saved: ~{agent_data['tokens_saved']:,} tokens")
        
        # Memory Usage Analysis
        memory_data = self.analyze_memory_usage()
        print("\n💾 MEMORY MCP ANALYSIS")
        print("-"*40)
        print(f"Total Size: {memory_data['total_bytes']:,} bytes")
        print(f"Estimated Tokens: ~{memory_data['estimated_tokens']:,} tokens")
        print(f"File Count: {memory_data['file_count']} files")
        
        # MCP Server Analysis
        mcp_data = self.analyze_mcp_servers()
        print("\n🔧 MCP SERVER CONFIGURATIONS")
        print("-"*40)
        for config in mcp_data:
            print(f"{config['type'].upper()}:")
            print(f"  Server Count: {config['server_count']}")
            if config.get('has_limits'):
                print(f"  Has Token Limits: ✓")
            if config['server_count'] <= 5:
                print(f"  Servers: {', '.join(config['servers'])}")
        
        # Total Token Estimation
        total_data = self.estimate_total_tokens()
        print("\n🎯 TOTAL TOKEN USAGE ESTIMATION")
        print("-"*40)
        print(f"Current Total: ~{total_data['current_total']:,} tokens")
        if total_data['original_total']:
            print(f"Original Total: ~{total_data['original_total']:,} tokens")
            print(f"Total Savings: ~{total_data['savings']:,} tokens")
            print(f"Reduction: {total_data['reduction_percent']:.1f}%")
        
        print("\nBREAKDOWN:")
        for key, value in total_data['breakdown'].items():
            print(f"  {key.replace('_', ' ').title()}: ~{value:,} tokens")
        
        # Performance Impact
        perf_data = self.run_performance_test()
        if perf_data:
            print("\n⚡ PERFORMANCE IMPACT")
            print("-"*40)
            if 'registry_parse_ms' in perf_data:
                print(f"Current Parse Time: {perf_data['registry_parse_ms']:.2f}ms")
            if 'backup_parse_ms' in perf_data:
                print(f"Original Parse Time: {perf_data['backup_parse_ms']:.2f}ms")
                if 'registry_parse_ms' in perf_data:
                    speedup = (perf_data['backup_parse_ms'] - perf_data['registry_parse_ms']) / perf_data['backup_parse_ms'] * 100
                    print(f"Speedup: {speedup:.1f}%")
        
        # Final Verdict
        print("\n" + "="*60)
        print("                    FINAL VERDICT")
        print("="*60)
        
        if total_data['current_total'] < 15000:
            print("✅ SUCCESS: Token usage is under 15,000 target!")
            print(f"   Current: ~{total_data['current_total']:,} tokens")
        elif total_data['current_total'] < 20000:
            print("⚠️  PARTIAL: Token usage reduced but above target")
            print(f"   Current: ~{total_data['current_total']:,} tokens (Target: <15,000)")
        else:
            print("❌ NEEDS WORK: Token usage still too high")
            print(f"   Current: ~{total_data['current_total']:,} tokens (Target: <15,000)")
        
        print("\n📋 RECOMMENDATIONS:")
        if memory_data['estimated_tokens'] > 500:
            print("• Run memory cleanup more frequently")
        if total_data['current_total'] > 15000:
            print("• Consider disabling non-essential MCP servers")
            print("• Further compress agent descriptions")
        print("• Restart Claude Code to apply all optimizations")
        
        print("\n" + "="*60)
        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    analyzer.generate_report()
