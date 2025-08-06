#!/usr/bin/env python3
"""
Purpose: Optimize prompts for efficient token usage across different models
Usage: from agents.core.prompt_optimizer import PromptOptimizer
Requirements: tiktoken, nltk, asyncio
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization"""
    model_type: str
    max_tokens: int
    compression_level: float  # 0.0 to 1.0
    preserve_keywords: List[str]
    abbreviations: Dict[str, str]


class PromptOptimizer:
    """Optimize prompts for different model tiers to minimize token usage"""
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'tinyllama': OptimizationConfig(
            model_type='tinyllama',
            max_tokens=500,
            compression_level=0.8,  # Aggressive compression
            preserve_keywords=['error', 'bug', 'fix', 'security', 'critical'],
            abbreviations={
                'implementation': 'impl',
                'configuration': 'config',
                'optimization': 'opt',
                'function': 'fn',
                'variable': 'var',
                'parameter': 'param',
                'database': 'db',
                'application': 'app',
                'development': 'dev',
                'production': 'prod',
                'repository': 'repo',
                'environment': 'env',
                'authentication': 'auth',
                'authorization': 'authz',
                'administrator': 'admin'
            }
        ),
        'tinyllama2.5-coder:7b': OptimizationConfig(
            model_type='tinyllama2.5-coder:7b',
            max_tokens=1500,
            compression_level=0.5,  # Moderate compression
            preserve_keywords=['error', 'bug', 'fix', 'security', 'critical', 'test', 'deploy'],
            abbreviations={
                'implementation': 'implement',
                'configuration': 'config',
                'optimization': 'optimize',
                'function': 'func',
                'variable': 'var',
                'parameter': 'param',
                'database': 'database',
                'application': 'app',
                'development': 'dev',
                'production': 'prod'
            }
        ),
        'tinyllama': OptimizationConfig(
            model_type='tinyllama',
            max_tokens=3000,
            compression_level=0.2,  # Minimal compression
            preserve_keywords=['error', 'bug', 'fix', 'security', 'critical', 'architecture', 'design'],
            abbreviations={}  # No abbreviations for complex reasoning
        )
    }
    
    def __init__(self):
        self.optimization_cache = {}
        self.token_counter = TokenCounter()
        
    def optimize_prompt(self, 
                       prompt: str, 
                       model: str,
                       task_type: str = 'general',
                       context: Optional[Dict[str, Any]] = None) -> str:
        """
        Optimize a prompt for a specific model and task type
        
        Args:
            prompt: Original prompt text
            model: Model name (tinyllama, tinyllama2.5-coder:7b, tinyllama)
            task_type: Type of task (code_generation, analysis, etc.)
            context: Additional context for optimization
            
        Returns:
            Optimized prompt string
        """
        # Get model configuration
        config = self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS['tinyllama'])
        
        # Check cache
        cache_key = self._get_cache_key(prompt, model, task_type)
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Apply optimization pipeline
        optimized = prompt
        
        # Step 1: Remove redundancy
        optimized = self._remove_redundancy(optimized)
        
        # Step 2: Apply abbreviations based on compression level
        if config.compression_level > 0.3:
            optimized = self._apply_abbreviations(optimized, config.abbreviations)
        
        # Step 3: Compress whitespace and formatting
        optimized = self._compress_whitespace(optimized)
        
        # Step 4: Structure optimization based on task type
        optimized = self._optimize_structure(optimized, task_type, config)
        
        # Step 5: Trim to token limit
        optimized = self._trim_to_limit(optimized, config.max_tokens, config.preserve_keywords)
        
        # Cache result
        self.optimization_cache[cache_key] = optimized
        
        # Log optimization stats
        original_tokens = self.token_counter.count(prompt)
        optimized_tokens = self.token_counter.count(optimized)
        reduction = (1 - optimized_tokens / original_tokens) * 100 if original_tokens > 0 else 0
        
        logger.debug(f"Prompt optimization: {original_tokens} -> {optimized_tokens} tokens ({reduction:.1f}% reduction)")
        
        return optimized
    
    def _get_cache_key(self, prompt: str, model: str, task_type: str) -> str:
        """Generate cache key for optimized prompt"""
        content = f"{prompt[:100]}{model}{task_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases and repeated information"""
        # Remove repeated words
        words = text.split()
        cleaned_words = []
        prev_word = ""
        
        for word in words:
            if word.lower() != prev_word.lower() or word.lower() in ['the', 'a', 'an', 'is', 'are']:
                cleaned_words.append(word)
                prev_word = word
        
        text = ' '.join(cleaned_words)
        
        # Remove redundant phrases
        redundant_patterns = [
            r'\b(in order to)\b',  # -> to
            r'\b(due to the fact that)\b',  # -> because
            r'\b(at this point in time)\b',  # -> now
            r'\b(in the event that)\b',  # -> if
            r'\b(despite the fact that)\b',  # -> although
        ]
        
        replacements = ['to', 'because', 'now', 'if', 'although']
        
        for pattern, replacement in zip(redundant_patterns, replacements):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_abbreviations(self, text: str, abbreviations: Dict[str, str]) -> str:
        """Apply abbreviations to common terms"""
        for full_term, abbrev in abbreviations.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + full_term + r'\b'
            text = re.sub(pattern, abbrev, text, flags=re.IGNORECASE)
        
        return text
    
    def _compress_whitespace(self, text: str) -> str:
        """Compress excessive whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _optimize_structure(self, text: str, task_type: str, config: OptimizationConfig) -> str:
        """Optimize prompt structure based on task type"""
        if task_type == 'code_generation':
            # Use concise code-focused structure
            if config.compression_level > 0.5:
                text = self._convert_to_bullet_points(text)
                text = self._add_code_markers(text)
        
        elif task_type == 'analysis':
            # Keep structured but remove fluff
            text = self._extract_key_points(text)
        
        elif task_type == 'question_answering':
            # Direct question format
            text = self._simplify_question(text)
        
        return text
    
    def _convert_to_bullet_points(self, text: str) -> str:
        """Convert verbose descriptions to bullet points"""
        sentences = text.split('. ')
        if len(sentences) > 3:
            # Convert to bullet format
            bullets = []
            for sent in sentences:
                if sent.strip():
                    bullets.append(f"- {sent.strip()}")
            return '\n'.join(bullets)
        return text
    
    def _add_code_markers(self, text: str) -> str:
        """Add code-specific markers for clarity"""
        # Add language hints if detected
        if any(keyword in text.lower() for keyword in ['python', 'javascript', 'java', 'c++']):
            text = f"[CODE] {text}"
        return text
    
    def _extract_key_points(self, text: str) -> str:
        """Extract key points from verbose text"""
        # Simple extraction based on importance markers
        important_phrases = ['must', 'should', 'need', 'require', 'important', 'critical', 'key']
        
        sentences = text.split('. ')
        key_sentences = []
        
        for sent in sentences:
            if any(phrase in sent.lower() for phrase in important_phrases):
                key_sentences.append(sent)
        
        # If we extracted key points, use them; otherwise, keep original
        if key_sentences and len(key_sentences) < len(sentences) * 0.7:
            return '. '.join(key_sentences) + '.'
        
        return text
    
    def _simplify_question(self, text: str) -> str:
        """Simplify question format"""
        # Remove unnecessary question words
        text = re.sub(r'^(Could you please|Would you mind|I would like to know)', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        # Ensure it ends with a question mark
        if not text.endswith('?'):
            text += '?'
        
        return text
    
    def _trim_to_limit(self, text: str, max_tokens: int, preserve_keywords: List[str]) -> str:
        """Trim text to token limit while preserving important keywords"""
        current_tokens = self.token_counter.count(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Preserve sentences with keywords
        sentences = text.split('. ')
        important_sentences = []
        other_sentences = []
        
        for sent in sentences:
            if any(keyword in sent.lower() for keyword in preserve_keywords):
                important_sentences.append(sent)
            else:
                other_sentences.append(sent)
        
        # Build result prioritizing important sentences
        result = important_sentences
        tokens_used = self.token_counter.count('. '.join(result))
        
        # Add other sentences until limit reached
        for sent in other_sentences:
            sent_tokens = self.token_counter.count(sent)
            if tokens_used + sent_tokens < max_tokens:
                result.append(sent)
                tokens_used += sent_tokens
            else:
                break
        
        return '. '.join(result) + '.'
    
    def create_system_prompt(self, 
                           agent_type: str,
                           model: str,
                           capabilities: List[str]) -> str:
        """Create an optimized system prompt for an agent"""
        config = self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS['tinyllama'])
        
        if config.compression_level > 0.6:
            # Highly compressed format
            prompt = f"You are {agent_type}. Skills: {', '.join(capabilities[:3])}. Be concise."
        elif config.compression_level > 0.3:
            # Moderately compressed
            prompt = f"You are a {agent_type} with expertise in {', '.join(capabilities[:5])}. Provide clear, efficient solutions."
        else:
            # Detailed format for complex models
            prompt = f"""You are an expert {agent_type} with the following capabilities:
{chr(10).join(f'- {cap}' for cap in capabilities)}

Analyze problems thoroughly and provide detailed, well-reasoned solutions."""
        
        return prompt
    
    def batch_optimize(self, prompts: List[Tuple[str, str, str]]) -> List[str]:
        """Optimize multiple prompts in batch"""
        optimized = []
        
        for prompt, model, task_type in prompts:
            optimized.append(self.optimize_prompt(prompt, model, task_type))
        
        return optimized


class TokenCounter:
    """Estimate token count for prompts"""
    
    def __init__(self):
        # Simple word-based estimation
        # In production, use tiktoken or model-specific tokenizer
        self.avg_chars_per_token = 4
    
    def count(self, text: str) -> int:
        """Estimate token count"""
        # Simple estimation: chars / 4
        # Special handling for code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        code_chars = sum(len(block) for block in code_blocks)
        
        # Code typically has more tokens per character
        code_tokens = code_chars / 3
        
        # Regular text
        non_code_text = text
        for block in code_blocks:
            non_code_text = non_code_text.replace(block, '')
        
        text_tokens = len(non_code_text) / self.avg_chars_per_token
        
        return int(code_tokens + text_tokens)


class PromptTemplate:
    """Pre-optimized prompt templates for common tasks"""
    
    TEMPLATES = {
        'code_generation': {
            'tinyllama': "Write {language} code: {task}. Requirements: {requirements}",
            'tinyllama2.5-coder:7b': "Create {language} code for: {task}\nRequirements:\n{requirements}\nOutput clean, efficient code.",
            'tinyllama': """Develop a {language} solution for the following task:
{task}

Requirements:
{requirements}

Consider best practices, error handling, and performance optimization."""
        },
        'analysis': {
            'tinyllama': "Analyze: {subject}. Focus: {aspects}. Output: {format}",
            'tinyllama2.5-coder:7b': "Analyze {subject}\nKey aspects: {aspects}\nProvide {format} format analysis.",
            'tinyllama': """Conduct a comprehensive analysis of {subject}.

Focus on these aspects:
{aspects}

Provide your analysis in {format} format with detailed reasoning."""
        },
        'debugging': {
            'tinyllama': "Debug: {code_snippet}. Error: {error}. Fix it.",
            'tinyllama2.5-coder:7b': "Debug this code:\n{code_snippet}\nError: {error}\nProvide fixed code and explanation.",
            'tinyllama': """Debug the following code that produces this error:

Code:
{code_snippet}

Error:
{error}

Provide:
1. Root cause analysis
2. Fixed code
3. Prevention strategies"""
        }
    }
    
    @classmethod
    def get_template(cls, task_type: str, model: str, **kwargs) -> str:
        """Get and fill a template"""
        templates = cls.TEMPLATES.get(task_type, {})
        template = templates.get(model, templates.get('tinyllama', ''))
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template parameter: {e}")
            return template


# Example usage
if __name__ == "__main__":
    optimizer = PromptOptimizer()
    
    # Test optimization
    original_prompt = """
    I need you to implement a comprehensive user authentication system for a web application.
    The implementation should include user registration functionality, login functionality, 
    password reset functionality, and session management. Please ensure that the implementation
    follows security best practices including password hashing, protection against SQL injection,
    and proper session timeout handling. The system should be implemented in Python using FastAPI.
    """
    
    # Optimize for different models
    for model in ['tinyllama', 'tinyllama2.5-coder:7b', 'tinyllama']:
        optimized = optimizer.optimize_prompt(
            original_prompt, 
            model, 
            task_type='code_generation'
        )
        print(f"\n{model}:")
        print(optimized)
        print(f"Tokens: ~{optimizer.token_counter.count(optimized)}")