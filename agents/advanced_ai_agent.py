import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import tempfile
import os
from pathlib import Path

from .base_agent import BaseAgent
from tools.advanced_frameworks import (
    advanced_framework_manager,
    process_image,
    analyze_text_advanced,
    create_fast_nn
)
from tools.ml_frameworks import ml_framework_manager, process_text
from memory import vector_memory
from models import model_manager

logger = logging.getLogger(__name__)

class AdvancedAIAgent(BaseAgent):
    """Advanced AI agent with comprehensive ML/AI capabilities including computer vision, 
    specialized neural networks, and advanced NLP processing."""
    
    def __init__(self, agent_id: str = "advanced_ai_agent"):
        super().__init__(agent_id, "advanced_ai")
        self.capabilities = [
            "computer_vision",
            "image_analysis",
            "face_detection",
            "advanced_nlp",
            "multilingual_processing", 
            "language_detection",
            "entity_extraction",
            "sentiment_analysis",
            "neural_network_creation",
            "fast_neural_networks",
            "dynamic_computation",
            "model_optimization",
            "framework_integration",
            "multimodal_analysis",
            "pattern_recognition",
            "feature_extraction"
        ]
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced AI tasks across multiple frameworks."""
        try:
            task_type = task.get("type", "")
            
            if task_type == "analyze_image":
                return await self._analyze_image_task(task)
            elif task_type == "detect_faces":
                return await self._detect_faces_task(task)
            elif task_type == "multilingual_analysis":
                return await self._multilingual_analysis_task(task)
            elif task_type == "create_neural_network":
                return await self._create_neural_network_task(task)
            elif task_type == "advanced_sentiment":
                return await self._advanced_sentiment_task(task)
            elif task_type == "multimodal_analysis":
                return await self._multimodal_analysis_task(task)
            elif task_type == "pattern_recognition":
                return await self._pattern_recognition_task(task)
            elif task_type == "framework_benchmark":
                return await self._framework_benchmark_task(task)
            else:
                return await self._general_advanced_task(task)
                
        except Exception as e:
            logger.error(f"Error executing advanced AI task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_image_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive image analysis using computer vision frameworks."""
        image_path = task.get("image_path", "")
        operations = task.get("operations", ["detect_faces", "extract_features"])
        
        if not image_path or not os.path.exists(image_path):
            return {"success": False, "error": "Invalid or missing image path"}
        
        # Process image with advanced CV
        cv_result = await process_image(image_path, operations)
        
        # Store analysis in vector memory
        await vector_memory.store(
            content=f"Image analysis: {os.path.basename(image_path)}",
            metadata={
                "type": "image_analysis",
                "operations": operations,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_id
            }
        )
        
        return {
            "success": True,
            "image_path": image_path,
            "operations": operations,
            "analysis": cv_result,
            "capabilities_used": ["computer_vision", "image_analysis"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _detect_faces_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Face detection using advanced computer vision."""
        image_path = task.get("image_path", "")
        
        if not image_path or not os.path.exists(image_path):
            return {"success": False, "error": "Invalid or missing image path"}
        
        # Use computer vision processor directly
        face_result = await advanced_framework_manager.cv_processor.detect_faces(image_path)
        
        return {
            "success": True,
            "image_path": image_path,
            "face_detection": face_result,
            "capabilities_used": ["face_detection", "computer_vision"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _multilingual_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced multilingual text analysis."""
        text = task.get("text", "")
        target_languages = task.get("languages", [])
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        # Advanced multilingual analysis
        advanced_result = await analyze_text_advanced(text)
        
        # Also use standard ML frameworks for comparison
        standard_result = await process_text(text)
        
        # Combine results
        combined_analysis = {
            "advanced_analysis": advanced_result,
            "standard_analysis": {
                "entities": standard_result.entities,
                "sentiment": standard_result.sentiment,
                "keywords": standard_result.keywords,
                "language": standard_result.language
            }
        }
        
        # Store in vector memory
        await vector_memory.store(
            content=text[:500] + "..." if len(text) > 500 else text,
            metadata={
                "type": "multilingual_analysis",
                "detected_language": advanced_result.get("analysis", {}).get("language_detection", {}).get("language"),
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_id
            }
        )
        
        return {
            "success": True,
            "text_length": len(text),
            "target_languages": target_languages,
            "analysis": combined_analysis,
            "capabilities_used": ["multilingual_processing", "advanced_nlp", "language_detection"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_neural_network_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create neural networks using specialized frameworks."""
        network_config = task.get("config", {})
        network_type = task.get("network_type", "fast_nn")
        
        if network_type == "fast_nn":
            # Use FANN for fast neural networks
            result = await create_fast_nn(network_config)
        elif network_type == "chainer":
            # Use Chainer for dynamic networks
            result = await self._create_chainer_network(network_config)
        else:
            return {"success": False, "error": f"Unsupported network type: {network_type}"}
        
        return {
            "success": True,
            "network_type": network_type,
            "configuration": network_config,
            "creation_result": result,
            "capabilities_used": ["neural_network_creation", "fast_neural_networks"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _create_chainer_network(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Chainer dynamic network."""
        try:
            input_size = config.get("input_size", 784)
            hidden_size = config.get("hidden_size", 100)
            output_size = config.get("output_size", 10)
            model_name = config.get("name", "chainer_model")
            
            success = await advanced_framework_manager.chainer_processor.create_mlp_model(
                input_size, hidden_size, output_size, model_name
            )
            
            return {
                "success": success,
                "model_name": model_name,
                "architecture": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "output_size": output_size,
                    "framework": "chainer"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _advanced_sentiment_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced sentiment analysis with word-level insights."""
        text = task.get("text", "")
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        # Get advanced sentiment analysis
        advanced_result = await analyze_text_advanced(text)
        sentiment_data = advanced_result.get("analysis", {}).get("sentiment", {})
        
        # Also get standard sentiment for comparison
        standard_result = await process_text(text)
        
        return {
            "success": True,
            "text_length": len(text),
            "sentiment_analysis": {
                "advanced": sentiment_data,
                "standard": standard_result.sentiment,
                "comparison": self._compare_sentiment_results(sentiment_data, standard_result.sentiment)
            },
            "capabilities_used": ["sentiment_analysis", "advanced_nlp"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _multimodal_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Multimodal analysis combining text, image, and other data types."""
        text = task.get("text", "")
        image_path = task.get("image_path", "")
        
        results = {}
        
        # Text analysis if provided
        if text:
            results["text_analysis"] = await analyze_text_advanced(text)
        
        # Image analysis if provided
        if image_path and os.path.exists(image_path):
            results["image_analysis"] = await process_image(image_path)
        
        # Cross-modal insights
        insights = []
        if text and image_path:
            insights.append("Multimodal content detected - text and image analysis combined")
            
            # Extract entities from text
            text_entities = results.get("text_analysis", {}).get("analysis", {}).get("entities", {}).get("entities", [])
            
            # Check if image has faces
            faces = results.get("image_analysis", {}).get("results", {}).get("face_detection", {}).get("faces_detected", 0)
            
            if text_entities and faces > 0:
                insights.append(f"Found {len(text_entities)} text entities and {faces} faces - potential person-image correlation")
        
        return {
            "success": True,
            "modalities": {
                "text": bool(text),
                "image": bool(image_path and os.path.exists(image_path))
            },
            "analysis": results,
            "cross_modal_insights": insights,
            "capabilities_used": ["multimodal_analysis", "pattern_recognition"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _pattern_recognition_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern recognition across different data types."""
        data = task.get("data", "")
        data_type = task.get("data_type", "text")
        pattern_types = task.get("pattern_types", ["entities", "sentiment", "keywords"])
        
        patterns = {}
        
        if data_type == "text":
            # Text pattern recognition
            analysis = await analyze_text_advanced(data)
            
            if "entities" in pattern_types:
                entities = analysis.get("analysis", {}).get("entities", {}).get("entities", [])
                patterns["entities"] = {
                    "count": len(entities),
                    "types": list(set([e.get("tag", "unknown") for e in entities])),
                    "patterns": entities
                }
            
            if "sentiment" in pattern_types:
                sentiment = analysis.get("analysis", {}).get("sentiment", {})
                patterns["sentiment"] = {
                    "overall": sentiment.get("sentiment_label", "neutral"),
                    "polarity": sentiment.get("overall_polarity", 0.0),
                    "word_level": sentiment.get("word_sentiments", [])
                }
            
            if "keywords" in pattern_types:
                # Extract keyword patterns
                text_result = await process_text(data)
                patterns["keywords"] = {
                    "extracted": text_result.keywords,
                    "frequency_patterns": self._analyze_keyword_patterns(text_result.keywords)
                }
        
        return {
            "success": True,
            "data_type": data_type,
            "pattern_types": pattern_types,
            "patterns": patterns,
            "capabilities_used": ["pattern_recognition", "advanced_nlp"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _framework_benchmark_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark different AI frameworks."""
        test_data = task.get("test_data", {})
        frameworks = task.get("frameworks", ["advanced", "standard"])
        
        benchmarks = {}
        
        # Benchmark advanced frameworks
        if "advanced" in frameworks:
            benchmarks["advanced"] = await advanced_framework_manager.benchmark_advanced_frameworks(test_data)
        
        # Benchmark standard ML frameworks
        if "standard" in frameworks and test_data.get("text"):
            start_time = datetime.utcnow()
            ml_result = await ml_framework_manager.benchmark_frameworks(test_data["text"])
            end_time = datetime.utcnow()
            
            benchmarks["standard"] = {
                "benchmarks": ml_result,
                "total_time": (end_time - start_time).total_seconds()
            }
        
        return {
            "success": True,
            "frameworks_tested": frameworks,
            "test_data_summary": {
                "has_text": bool(test_data.get("text")),
                "has_image": bool(test_data.get("image_path")),
                "text_length": len(test_data.get("text", ""))
            },
            "benchmarks": benchmarks,
            "capabilities_used": ["framework_integration"],
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _general_advanced_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general advanced AI tasks."""
        content = task.get("content", "")
        analysis_type = task.get("analysis_type", "comprehensive")
        
        if not content:
            return {"success": False, "error": "No content provided"}
        
        # Determine content type and apply appropriate analysis
        if self._is_image_path(content):
            result = await process_image(content)
            used_capabilities = ["computer_vision", "image_analysis"]
        else:
            # Treat as text
            result = await analyze_text_advanced(content)
            used_capabilities = ["advanced_nlp", "multilingual_processing"]
        
        return {
            "success": True,
            "content_type": "image" if self._is_image_path(content) else "text",
            "analysis_type": analysis_type,
            "analysis": result,
            "capabilities_used": used_capabilities,
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _is_image_path(self, content: str) -> bool:
        """Check if content is an image file path."""
        if not isinstance(content, str):
            return False
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        path = Path(content.lower())
        return path.suffix in image_extensions and os.path.exists(content)
    
    def _compare_sentiment_results(self, advanced: Dict[str, Any], standard: Dict[str, Any]) -> Dict[str, Any]:
        """Compare sentiment results from different frameworks."""
        comparison = {
            "agreement": "unknown",
            "confidence_difference": 0.0,
            "notes": []
        }
        
        try:
            # Get sentiment labels
            advanced_label = advanced.get("sentiment_label", "neutral")
            standard_compound = standard.get("nltk_compound", 0.0)
            
            # Convert standard to label
            if standard_compound > 0.1:
                standard_label = "positive"
            elif standard_compound < -0.1:
                standard_label = "negative"
            else:
                standard_label = "neutral"
            
            # Check agreement
            comparison["agreement"] = "agree" if advanced_label == standard_label else "disagree"
            comparison["advanced_sentiment"] = advanced_label
            comparison["standard_sentiment"] = standard_label
            
            if comparison["agreement"] == "disagree":
                comparison["notes"].append("Different frameworks show different sentiment - manual review recommended")
            
        except Exception as e:
            comparison["notes"].append(f"Error comparing sentiments: {str(e)}")
        
        return comparison
    
    def _analyze_keyword_patterns(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze patterns in extracted keywords."""
        if not keywords:
            return {"patterns": [], "frequency": {}}
        
        # Simple frequency analysis
        frequency = {}
        for keyword in keywords:
            frequency[keyword] = frequency.get(keyword, 0) + 1
        
        # Identify patterns
        patterns = []
        if len(keywords) > 10:
            patterns.append("High keyword density")
        
        # Check for repeated patterns
        if max(frequency.values()) > 2:
            patterns.append("Repeated keywords detected")
        
        return {
            "patterns": patterns,
            "frequency": frequency,
            "unique_keywords": len(set(keywords)),
            "total_keywords": len(keywords)
        }

# Global instance
advanced_ai_agent = AdvancedAIAgent()