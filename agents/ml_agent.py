import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_agent import BaseAgent
from tools.ml_frameworks import ml_framework_manager, process_text, analyze_code
from memory import vector_memory
from models import model_manager

logger = logging.getLogger(__name__)

class MLAnalysisAgent(BaseAgent):
    """Advanced ML agent for code analysis, text processing, and AI-powered insights."""
    
    def __init__(self, agent_id: str = "ml_analysis_agent"):
        super().__init__(agent_id, "ml_analysis")
        self.capabilities = [
            "code_analysis",
            "text_processing", 
            "sentiment_analysis",
            "entity_extraction",
            "code_quality_assessment",
            "documentation_generation",
            "vulnerability_detection",
            "performance_optimization"
        ]
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML analysis tasks."""
        try:
            task_type = task.get("type", "")
            
            if task_type == "analyze_code":
                return await self._analyze_code_task(task)
            elif task_type == "process_text":
                return await self._process_text_task(task)
            elif task_type == "generate_documentation":
                return await self._generate_documentation_task(task)
            elif task_type == "assess_security":
                return await self._assess_security_task(task)
            elif task_type == "optimize_performance":
                return await self._optimize_performance_task(task)
            else:
                return await self._general_analysis_task(task)
                
        except Exception as e:
            logger.error(f"Error executing ML task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_code_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code using ML frameworks."""
        code = task.get("code", "")
        language = task.get("language", "python")
        
        if not code:
            return {"success": False, "error": "No code provided"}
        
        # Use ML frameworks to analyze code
        analysis_result = await analyze_code(code, language)
        
        # Additional NLP analysis of code comments and structure
        nlp_result = await process_text(code)
        
        # Store analysis in vector memory for future reference
        await vector_memory.store(
            content=f"Code analysis for {language}: {code[:200]}...",
            metadata={
                "type": "code_analysis",
                "language": language,
                "complexity": analysis_result.get("complexity_score", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "success": True,
            "analysis": analysis_result,
            "nlp_insights": {
                "entities": nlp_result.entities,
                "sentiment": nlp_result.sentiment,
                "keywords": nlp_result.keywords
            },
            "recommendations": await self._generate_code_recommendations(analysis_result),
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_text_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process text using comprehensive NLP."""
        text = task.get("text", "")
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        # Comprehensive text processing
        result = await process_text(text)
        
        # Store insights in vector memory
        await vector_memory.store(
            content=text,
            metadata={
                "type": "text_analysis",
                "entities_count": len(result.entities),
                "sentiment_score": result.sentiment.get("nltk_compound", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "success": True,
            "processing_result": {
                "tokens": result.tokens,
                "entities": result.entities,
                "sentiment": result.sentiment,
                "keywords": result.keywords,
                "language": result.language,
                "summary": await self._generate_summary(text),
                "embeddings_shape": result.embeddings.shape if result.embeddings is not None else None
            },
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_documentation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation using ML insights."""
        code = task.get("code", "")
        doc_type = task.get("doc_type", "api")
        
        if not code:
            return {"success": False, "error": "No code provided"}
        
        # Analyze code structure
        analysis = await analyze_code(code)
        nlp_result = await process_text(code)
        
        # Generate documentation using AI
        documentation = await self._create_ai_documentation(code, analysis, doc_type)
        
        return {
            "success": True,
            "documentation": documentation,
            "analysis_used": analysis,
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _assess_security_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security using ML techniques."""
        code = task.get("code", "")
        
        if not code:
            return {"success": False, "error": "No code provided"}
        
        # ML-based security assessment
        security_analysis = await self._ml_security_analysis(code)
        
        # Pattern-based vulnerability detection
        vulnerabilities = await self._detect_vulnerabilities(code)
        
        return {
            "success": True,
            "security_score": security_analysis.get("security_score", 0),
            "vulnerabilities": vulnerabilities,
            "recommendations": security_analysis.get("recommendations", []),
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _optimize_performance_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code performance using ML insights."""
        code = task.get("code", "")
        language = task.get("language", "python")
        
        if not code:
            return {"success": False, "error": "No code provided"}
        
        # Analyze current performance characteristics
        analysis = await analyze_code(code, language)
        
        # Generate optimization suggestions
        optimizations = await self._generate_optimizations(code, analysis)
        
        return {
            "success": True,
            "current_analysis": analysis,
            "optimizations": optimizations,
            "estimated_improvement": await self._estimate_performance_gain(optimizations),
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _general_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General analysis task for any content."""
        content = task.get("content", "")
        analysis_type = task.get("analysis_type", "comprehensive")
        
        if not content:
            return {"success": False, "error": "No content provided"}
        
        # Determine if content is code or text
        is_code = self._detect_if_code(content)
        
        if is_code:
            result = await analyze_code(content)
        else:
            result = await process_text(content)
        
        return {
            "success": True,
            "content_type": "code" if is_code else "text",
            "analysis": result,
            "insights": await self._generate_insights(content, result),
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_code_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on code analysis."""
        recommendations = []
        
        complexity_score = analysis.get("complexity_score", 0)
        if complexity_score > 50:
            recommendations.append("Consider breaking down complex functions into smaller, more manageable pieces")
        
        readability_score = analysis.get("readability_score", 0)
        if readability_score < 70:
            recommendations.append("Improve code readability with better variable names and comments")
        
        if analysis.get("suggestions"):
            recommendations.extend(analysis["suggestions"])
        
        return recommendations
    
    async def _generate_summary(self, text: str) -> str:
        """Generate text summary using ML."""
        try:
            # Use extractive summarization (simple approach)
            sentences = text.split('. ')
            if len(sentences) <= 3:
                return text
            
            # Take first and last sentences as summary
            summary = f"{sentences[0]}. {sentences[-1]}"
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    async def _create_ai_documentation(self, code: str, analysis: Dict[str, Any], doc_type: str) -> str:
        """Create AI-generated documentation."""
        try:
            # Basic documentation template
            doc_template = f"""
# Code Documentation

## Overview
This code has a complexity score of {analysis.get('complexity_score', 0):.1f} and readability score of {analysis.get('readability_score', 0):.1f}.

## Analysis Results
- **Complexity**: {analysis.get('complexity_score', 0):.1f}/100
- **Readability**: {analysis.get('readability_score', 0):.1f}/100
- **Security Issues**: {len(analysis.get('security_issues', []))}

## Recommendations
"""
            
            for suggestion in analysis.get('suggestions', []):
                doc_template += f"- {suggestion}\n"
            
            return doc_template
            
        except Exception as e:
            logger.error(f"Error creating documentation: {e}")
            return "Documentation generation failed"
    
    async def _ml_security_analysis(self, code: str) -> Dict[str, Any]:
        """ML-based security analysis."""
        try:
            # Process code with NLP
            nlp_result = await process_text(code)
            
            # Simple security scoring based on patterns
            security_score = 100.0
            recommendations = []
            
            # Check for potential security issues in text
            dangerous_patterns = ['eval(', 'exec(', 'subprocess', 'os.system', 'shell=True']
            for pattern in dangerous_patterns:
                if pattern in code:
                    security_score -= 20
                    recommendations.append(f"Avoid using {pattern} - potential security risk")
            
            # Sentiment analysis for comments (negative sentiment might indicate rushed/unsafe code)
            if nlp_result.sentiment.get('nltk_compound', 0) < -0.5:
                security_score -= 10
                recommendations.append("Code comments suggest potential issues - review carefully")
            
            return {
                "security_score": max(0, security_score),
                "recommendations": recommendations,
                "nlp_insights": nlp_result.sentiment
            }
            
        except Exception as e:
            logger.error(f"Error in security analysis: {e}")
            return {"security_score": 50, "recommendations": ["Security analysis failed"]}
    
    async def _detect_vulnerabilities(self, code: str) -> List[Dict[str, Any]]:
        """Detect potential vulnerabilities."""
        vulnerabilities = []
        
        # SQL Injection patterns
        if 'cursor.execute(' in code and '%' in code:
            vulnerabilities.append({
                "type": "SQL Injection",
                "severity": "High",
                "description": "Potential SQL injection via string formatting"
            })
        
        # Command injection patterns
        if 'subprocess' in code and 'shell=True' in code:
            vulnerabilities.append({
                "type": "Command Injection",
                "severity": "High", 
                "description": "Command execution with shell=True is dangerous"
            })
        
        return vulnerabilities
    
    async def _generate_optimizations(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance optimization suggestions."""
        optimizations = []
        
        # Basic optimization suggestions based on patterns
        if 'for ' in code and 'append(' in code:
            optimizations.append({
                "type": "List Comprehension",
                "description": "Consider using list comprehensions instead of loops with append",
                "impact": "Medium"
            })
        
        if analysis.get('complexity_score', 0) > 70:
            optimizations.append({
                "type": "Function Decomposition", 
                "description": "Break down complex functions for better performance",
                "impact": "High"
            })
        
        return optimizations
    
    async def _estimate_performance_gain(self, optimizations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Estimate performance improvement from optimizations."""
        if not optimizations:
            return {"estimated_gain": "0%", "confidence": "N/A"}
        
        # Simple estimation based on optimization count and types
        high_impact = sum(1 for opt in optimizations if opt.get("impact") == "High")
        medium_impact = sum(1 for opt in optimizations if opt.get("impact") == "Medium")
        
        estimated_gain = (high_impact * 20) + (medium_impact * 10)
        
        return {
            "estimated_gain": f"{min(estimated_gain, 50)}%",
            "confidence": "Medium" if estimated_gain > 0 else "Low"
        }
    
    def _detect_if_code(self, content: str) -> bool:
        """Simple heuristic to detect if content is code."""
        code_indicators = ['def ', 'class ', 'import ', 'function ', '{', '}', 'return ', 'if (']
        return any(indicator in content for indicator in code_indicators)
    
    async def _generate_insights(self, content: str, analysis: Any) -> List[str]:
        """Generate general insights from analysis."""
        insights = []
        
        if hasattr(analysis, 'entities') and analysis.entities:
            insights.append(f"Found {len(analysis.entities)} named entities")
        
        if hasattr(analysis, 'sentiment') and analysis.sentiment:
            compound_score = analysis.sentiment.get('nltk_compound', 0)
            if compound_score > 0.1:
                insights.append("Overall positive sentiment detected")
            elif compound_score < -0.1:
                insights.append("Overall negative sentiment detected")
        
        if hasattr(analysis, 'complexity_score'):
            if analysis.complexity_score > 70:
                insights.append("High complexity detected - consider simplification")
        
        return insights

# Global instance
ml_analysis_agent = MLAnalysisAgent()