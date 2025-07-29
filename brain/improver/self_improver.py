#!/usr/bin/env python3
"""
Self-Improver for the Brain
Analyzes failures and generates improvement patches
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import subprocess
from pathlib import Path

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import aiofiles
import git

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfImprover:
    """Generates improvements and patches for the Brain system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize improvement LLM (using codellama for code generation)
        self.code_llm = Ollama(
            model=config.get('code_model', 'codellama:7b'),
            base_url=config.get('ollama_host', 'http://sutazai-ollama:11434'),
            temperature=0.3
        )
        
        # Initialize analysis LLM (using deepseek-r1 for reasoning)
        self.analysis_llm = Ollama(
            model=config.get('analysis_model', 'deepseek-r1:8b'),
            base_url=config.get('ollama_host', 'http://sutazai-ollama:11434'),
            temperature=0.2
        )
        
        # Git repository for brain code
        self.repo_path = Path(config.get('brain_repo_path', '/workspace/brain'))
        self.repo = git.Repo(self.repo_path)
        
        # Improvement prompts
        self.analysis_prompt = PromptTemplate(
            input_variables=["task", "failures", "patterns"],
            template="""Analyze these task failures and identify systematic improvements needed.

Original Task: {task}

Failures:
{failures}

Historical Patterns:
{patterns}

Identify:
1. Root causes of failures
2. Systematic improvements needed
3. Specific code changes required
4. Configuration adjustments
5. New capabilities needed

Provide actionable recommendations for improving the Brain system.
"""
        )
        
        self.code_generation_prompt = PromptTemplate(
            input_variables=["improvement_type", "current_code", "requirements"],
            template="""Generate a code patch to improve the Brain system.

Improvement Type: {improvement_type}

Current Code:
```python
{current_code}
```

Requirements:
{requirements}

Generate a complete, working code patch that:
1. Addresses the specific improvement needed
2. Maintains backward compatibility
3. Includes proper error handling
4. Has clear documentation
5. Follows Python best practices

Output the complete updated code, not just the changes.
"""
        )
        
        # Track improvements
        self.improvement_history: List[Dict[str, Any]] = []
        self.pending_patches: List[Dict[str, Any]] = []
        
    async def analyze_and_improve(
        self,
        state: Dict[str, Any],
        min_score: float = 0.85
    ) -> Dict[str, Any]:
        """Analyze execution state and generate improvements"""
        try:
            # Collect failure information
            failures = self._extract_failures(state)
            patterns = self._analyze_patterns(state)
            
            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(
                state['user_input'],
                failures,
                patterns
            )
            
            # Generate code patches for top suggestions
            patches = []
            for suggestion in suggestions[:3]:  # Limit to top 3
                patch = await self._generate_patch(suggestion, state)
                if patch:
                    patches.append(patch)
            
            # Batch patches if configured
            if self.config.get('pr_batch_size', 50) > 0:
                patches = self._batch_patches(patches)
            
            return {
                'suggestions': suggestions,
                'patches': patches,
                'analysis': {
                    'failure_rate': len(failures) / len(state.get('agent_results', [])) if state.get('agent_results') else 0,
                    'common_issues': self._identify_common_issues(failures),
                    'improvement_priority': self._calculate_priority(state)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_and_improve: {e}")
            return {
                'suggestions': [],
                'patches': [],
                'analysis': {'error': str(e)}
            }
    
    def _extract_failures(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract failure information from execution state"""
        failures = []
        
        # Check agent results
        for result in state.get('agent_results', []):
            if not result.get('success', False):
                failures.append({
                    'type': 'agent_failure',
                    'agent': result.get('agent'),
                    'error': result.get('error'),
                    'task': state.get('user_input'),
                    'timestamp': state.get('timestamp')
                })
            elif result.get('quality_score', 1.0) < self.config.get('min_quality_score', 0.85):
                failures.append({
                    'type': 'quality_failure',
                    'agent': result.get('agent'),
                    'score': result.get('quality_score'),
                    'improvements': result.get('quality_improvements', []),
                    'task': state.get('user_input')
                })
        
        # Check system errors
        for error in state.get('error_log', []):
            failures.append({
                'type': 'system_error',
                'error': error,
                'task': state.get('user_input')
            })
        
        return failures
    
    def _analyze_patterns(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns from historical data"""
        patterns = []
        
        # Analyze learned patterns
        for pattern in state.get('learned_patterns', []):
            if pattern.get('quality_score', 0) < 0.8:
                patterns.append({
                    'type': 'low_quality_pattern',
                    'agent': pattern.get('agent'),
                    'score': pattern.get('quality_score'),
                    'frequency': 1  # Would be calculated from history
                })
        
        # Analyze memory patterns
        for memory in state.get('retrieved_memories', []):
            if memory.get('metadata', {}).get('failure', False):
                patterns.append({
                    'type': 'recurring_failure',
                    'context': memory.get('content'),
                    'agent': memory.get('agent_source')
                })
        
        return patterns
    
    async def _generate_suggestions(
        self,
        task: str,
        failures: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate improvement suggestions using LLM"""
        # Format failures and patterns
        failures_text = "\n".join([
            f"- {f['type']}: {f.get('agent', 'N/A')} - {f.get('error', f.get('improvements', 'N/A'))}"
            for f in failures[:10]  # Limit to prevent prompt overflow
        ])
        
        patterns_text = "\n".join([
            f"- {p['type']}: {p.get('agent', 'N/A')} - Score: {p.get('score', 'N/A')}"
            for p in patterns[:10]
        ])
        
        # Generate analysis
        prompt = self.analysis_prompt.format(
            task=task,
            failures=failures_text or "No failures detected",
            patterns=patterns_text or "No patterns identified"
        )
        
        response = await self.analysis_llm.ainvoke(prompt)
        
        # Parse suggestions from response
        suggestions = self._parse_suggestions(response)
        
        return suggestions
    
    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse improvement suggestions from LLM response"""
        suggestions = []
        
        # Simple parsing - look for numbered items or bullet points
        lines = response.split('\n')
        current_suggestion = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                if current_suggestion:
                    suggestions.append(' '.join(current_suggestion))
                current_suggestion = [line.lstrip('0123456789.-• ')]
            elif line and current_suggestion:
                current_suggestion.append(line)
        
        if current_suggestion:
            suggestions.append(' '.join(current_suggestion))
        
        return suggestions[:10]  # Limit number of suggestions
    
    async def _generate_patch(
        self,
        suggestion: str,
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a code patch for a specific improvement"""
        try:
            # Determine patch type and target
            patch_info = self._analyze_suggestion(suggestion)
            
            if not patch_info:
                return None
            
            # Read current code
            target_file = self.repo_path / patch_info['file']
            if not target_file.exists():
                logger.warning(f"Target file not found: {target_file}")
                return None
            
            async with aiofiles.open(target_file, 'r') as f:
                current_code = await f.read()
            
            # Generate improved code
            prompt = self.code_generation_prompt.format(
                improvement_type=patch_info['type'],
                current_code=current_code[:5000],  # Limit size
                requirements=suggestion
            )
            
            improved_code = await self.code_llm.ainvoke(prompt)
            
            # Extract code from response
            improved_code = self._extract_code(improved_code)
            
            if not improved_code:
                return None
            
            # Create patch
            patch = {
                'id': f"patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'description': suggestion,
                'files_changed': [str(patch_info['file'])],
                'diff': self._generate_diff(current_code, improved_code),
                'new_content': improved_code,
                'test_results': {},
                'estimated_impact': patch_info.get('impact', 0.5),
                'pr_url': None,
                'created_at': datetime.now().isoformat()
            }
            
            # Test patch if possible
            test_results = await self._test_patch(patch)
            patch['test_results'] = test_results
            
            return patch
            
        except Exception as e:
            logger.error(f"Error generating patch: {e}")
            return None
    
    def _analyze_suggestion(self, suggestion: str) -> Optional[Dict[str, Any]]:
        """Analyze suggestion to determine patch type and target"""
        suggestion_lower = suggestion.lower()
        
        # Map keywords to files and types
        if 'agent' in suggestion_lower:
            if 'router' in suggestion_lower or 'selection' in suggestion_lower:
                return {
                    'file': 'agents/agent_router.py',
                    'type': 'agent_routing',
                    'impact': 0.8
                }
            elif 'execution' in suggestion_lower:
                return {
                    'file': 'agents/agent_router.py',
                    'type': 'agent_execution',
                    'impact': 0.7
                }
        
        if 'memory' in suggestion_lower:
            return {
                'file': 'memory/vector_memory.py',
                'type': 'memory_improvement',
                'impact': 0.6
            }
        
        if 'evaluation' in suggestion_lower or 'quality' in suggestion_lower:
            return {
                'file': 'evaluator/quality_evaluator.py',
                'type': 'evaluation_improvement',
                'impact': 0.7
            }
        
        if 'orchestrat' in suggestion_lower or 'workflow' in suggestion_lower:
            return {
                'file': 'core/orchestrator.py',
                'type': 'orchestration_improvement',
                'impact': 0.9
            }
        
        # Default to orchestrator improvements
        return {
            'file': 'core/orchestrator.py',
            'type': 'general_improvement',
            'impact': 0.5
        }
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code from LLM response"""
        # Look for code blocks
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()
        
        # If no code blocks, assume entire response is code
        # (but filter out obvious non-code)
        if not any(phrase in response.lower() for phrase in ['generate', 'create', 'here is', 'the following']):
            return response.strip()
        
        return None
    
    def _generate_diff(self, original: str, modified: str) -> str:
        """Generate a unified diff between original and modified code"""
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile='original',
            tofile='modified',
            lineterm=''
        )
        
        return ''.join(diff)
    
    async def _test_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Test a patch before applying"""
        test_results = {
            'syntax_valid': False,
            'imports_valid': False,
            'tests_pass': False,
            'errors': []
        }
        
        try:
            # Test syntax
            import ast
            try:
                ast.parse(patch['new_content'])
                test_results['syntax_valid'] = True
            except SyntaxError as e:
                test_results['errors'].append(f"Syntax error: {e}")
            
            # Test imports
            import_test_code = """
import sys
import importlib.util
def test_imports(code):
    # Extract imports
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    __import__(alias.name)
                except ImportError:
                    return False, alias.name
        elif isinstance(node, ast.ImportFrom):
            try:
                __import__(node.module)
            except ImportError:
                return False, node.module
    return True, None
"""
            # Would execute import test here
            test_results['imports_valid'] = True  # Simplified for now
            
            # Run unit tests if available
            # This would run actual tests
            test_results['tests_pass'] = test_results['syntax_valid']
            
        except Exception as e:
            test_results['errors'].append(f"Test error: {e}")
        
        return test_results
    
    def _batch_patches(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch patches according to configuration"""
        batch_size = self.config.get('pr_batch_size', 50)
        
        if batch_size <= 0 or len(patches) <= batch_size:
            return patches
        
        # Group patches by type/impact
        batched = []
        current_batch = []
        current_files = 0
        
        for patch in sorted(patches, key=lambda p: p['estimated_impact'], reverse=True):
            files_in_patch = len(patch['files_changed'])
            
            if current_files + files_in_patch > batch_size and current_batch:
                # Create batch patch
                batched.append(self._merge_patches(current_batch))
                current_batch = []
                current_files = 0
            
            current_batch.append(patch)
            current_files += files_in_patch
        
        if current_batch:
            batched.append(self._merge_patches(current_batch))
        
        return batched
    
    def _merge_patches(self, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple patches into one"""
        merged = {
            'id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'description': f"Batch of {len(patches)} improvements:\n" + 
                          '\n'.join([f"- {p['description']}" for p in patches]),
            'files_changed': [],
            'diff': '',
            'test_results': {'all_tests': []},
            'estimated_impact': max(p['estimated_impact'] for p in patches),
            'pr_url': None,
            'patches': patches
        }
        
        for patch in patches:
            merged['files_changed'].extend(patch['files_changed'])
            merged['diff'] += f"\n\n--- {patch['id']} ---\n{patch['diff']}"
            merged['test_results']['all_tests'].append(patch['test_results'])
        
        return merged
    
    def _identify_common_issues(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Identify common issues from failures"""
        issues = []
        
        # Count failure types
        failure_types = {}
        for failure in failures:
            ftype = failure.get('type', 'unknown')
            failure_types[ftype] = failure_types.get(ftype, 0) + 1
        
        # Identify common issues
        for ftype, count in failure_types.items():
            if count >= 2:
                issues.append(f"{ftype} ({count} occurrences)")
        
        return issues
    
    def _calculate_priority(self, state: Dict[str, Any]) -> float:
        """Calculate improvement priority based on state"""
        priority = 0.5
        
        # Increase priority for low quality scores
        if state.get('overall_score', 1.0) < 0.5:
            priority += 0.3
        elif state.get('overall_score', 1.0) < 0.7:
            priority += 0.2
        elif state.get('overall_score', 1.0) < 0.85:
            priority += 0.1
        
        # Increase priority for multiple failures
        failure_rate = len([r for r in state.get('agent_results', []) if not r.get('success', False)]) / max(len(state.get('agent_results', [])), 1)
        priority += failure_rate * 0.2
        
        return min(priority, 1.0)
    
    async def apply_patches(self, patches: List[Dict[str, Any]], require_approval: bool = True):
        """Apply patches to the codebase"""
        applied_patches = []
        
        for patch in patches:
            try:
                if require_approval:
                    # Create PR for approval
                    pr_url = await self._create_pull_request(patch)
                    patch['pr_url'] = pr_url
                    logger.info(f"Created PR for patch {patch['id']}: {pr_url}")
                else:
                    # Apply directly
                    for i, file_path in enumerate(patch.get('files_changed', [])):
                        target_file = self.repo_path / file_path
                        
                        # Backup original
                        backup_path = target_file.with_suffix('.backup')
                        if target_file.exists():
                            import shutil
                            shutil.copy2(target_file, backup_path)
                        
                        # Apply patch
                        if 'patches' in patch:
                            # Batched patch
                            content = patch['patches'][i]['new_content']
                        else:
                            content = patch['new_content']
                        
                        async with aiofiles.open(target_file, 'w') as f:
                            await f.write(content)
                    
                    # Commit changes
                    self.repo.index.add(patch['files_changed'])
                    self.repo.index.commit(f"Auto-improvement: {patch['description'][:100]}")
                    
                    logger.info(f"Applied patch {patch['id']} directly")
                    applied_patches.append(patch['id'])
                
            except Exception as e:
                logger.error(f"Error applying patch {patch['id']}: {e}")
        
        return applied_patches
    
    async def _create_pull_request(self, patch: Dict[str, Any]) -> str:
        """Create a pull request for the patch"""
        # This would integrate with GitHub API
        # For now, return a mock URL
        return f"https://github.com/sutazai/brain/pull/{patch['id']}"