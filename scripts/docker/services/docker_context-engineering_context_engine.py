from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging

app = FastAPI(title="Context Engineering Framework", version="1.0.0")

class ContextRequest(BaseModel):
    text: str
    context_type: str = "general"
    max_length: int = 2048
    compression_ratio: float = 0.5

class ContextResponse(BaseModel):
    original_text: str
    compressed_text: str
    context_vectors: List[float]
    compression_ratio: float
    relevance_score: float

class ContextEngine:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_templates = {
            "code": "Focus on: functions, classes, imports, logic flow, variables",
            "documentation": "Focus on: key concepts, procedures, examples, requirements",
            "general": "Focus on: main topics, important details, relationships, conclusions",
            "technical": "Focus on: specifications, requirements, architecture, implementation"
        }
        
    async def load_model(self):
        if self.model is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            
    def extract_key_sentences(self, text: str, compression_ratio: float) -> str:
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
            
        # Simple extractive summarization based on sentence length and position
        target_sentences = max(1, int(len(sentences) * compression_ratio))
        
        # Score sentences (longer sentences and those with keywords get higher scores)
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Base score on word count
            if i == 0 or i == len(sentences) - 1:  # First and last sentences are important
                score *= 1.5
            if any(keyword in sentence.lower() for keyword in ['important', 'key', 'main', 'critical', 'essential']):
                score *= 1.3
            sentence_scores.append((score, sentence))
            
        # Select top sentences
        sentence_scores.sort(reverse=True)
        selected_sentences = [sentence for _, sentence in sentence_scores[:target_sentences]]
        
        return '. '.join(selected_sentences)
        
    def create_context_vectors(self, text: str) -> List[float]:
        if self.model is None:
            return [0.0] * 384  # Default dimension for MiniLM
            
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
        return embeddings.cpu().numpy().tolist()
        
    def calculate_relevance_score(self, original: str, compressed: str) -> float:
        # Simple relevance score based on keyword preservation
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        if len(original_words) == 0:
            return 0.0
            
        preserved_ratio = len(original_words.intersection(compressed_words)) / len(original_words)
        return min(1.0, preserved_ratio * 1.2)  # Boost score slightly
        
    async def process_context(self, request: ContextRequest) -> ContextResponse:
        await self.load_model()
        
        # Add context template guidance
        context_guidance = self.context_templates.get(request.context_type, self.context_templates["general"])
        guided_text = f"{context_guidance}\n\n{request.text}"
        
        # Extract key information
        compressed_text = self.extract_key_sentences(guided_text, request.compression_ratio)
        
        # Create context vectors
        context_vectors = self.create_context_vectors(compressed_text)
        
        # Calculate relevance score
        relevance_score = self.calculate_relevance_score(request.text, compressed_text)
        
        return ContextResponse(
            original_text=request.text,
            compressed_text=compressed_text,
            context_vectors=context_vectors,
            compression_ratio=len(compressed_text) / len(request.text),
            relevance_score=relevance_score
        )

context_engine = ContextEngine()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "context-engineering"}

@app.post("/process", response_model=ContextResponse)
async def process_context(request: ContextRequest):
    try:
        return await context_engine.process_context(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates")
def get_context_templates():
    return context_engine.context_templates

@app.post("/update-template")
def update_context_template(context_type: str, template: str):
    context_engine.context_templates[context_type] = template
    return {"status": "updated", "context_type": context_type}

@app.get("/stats")
def get_stats():
    return {
        "model_loaded": context_engine.model is not None,
        "device": str(context_engine.device),
        "supported_types": list(context_engine.context_templates.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)