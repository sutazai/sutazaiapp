from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from typing import List, Dict, Any, Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModel, AutoTokenizer
import logging

app = FastAPI(title="Foundation Model Stack FSDP", version="1.0.0")

class FSDPRequest(BaseModel):
    model_name: str
    task: str
    input_text: str
    max_length: int = 512
    temperature: float = 0.7

class FSDPResponse(BaseModel):
    model_name: str
    output: str
    task: str
    execution_time: float
    memory_usage: Dict[str, float]

class FSDPManager:
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.environ.get('FSDP_MODEL_PATH', '/data/models')
        
    def wrap_model_with_fsdp(self, model):
        """Wrap model with FSDP for distributed training/inference"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Configure FSDP wrapping policy
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=100_000_000  # 100M parameters threshold
            )
            
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=None,  # Can be configured for fp16/bf16
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
            )
        return model
        
    def load_model(self, model_name: str):
        """Load and wrap model with FSDP if not already loaded"""
        if model_name not in self.models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
                # Move to device first
                model = model.to(self.device)
                
                # Wrap with FSDP
                model = self.wrap_model_with_fsdp(model)
                
                self.models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer
                }
                
                logging.info(f"Successfully loaded model {model_name} with FSDP")
                
            except Exception as e:
                logging.error(f"Error loading model {model_name}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
                
    def get_memory_usage(self):
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            }
        else:
            import psutil
            return {
                "cpu_memory_percent": psutil.virtual_memory().percent,
                "cpu_memory_available": psutil.virtual_memory().available / 1024**3,  # GB
                "cpu_count": psutil.cpu_count()
            }
            
    def process_text(self, request: FSDPRequest):
        """Process text using FSDP-wrapped model"""
        import time
        start_time = time.time()
        
        # Load model if needed
        self.load_model(request.model_name)
        
        model_info = self.models[request.model_name]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Tokenize input
        inputs = tokenizer(
            request.input_text, 
            return_tensors="pt", 
            max_length=request.max_length, 
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate/process with model
        with torch.no_grad():
            if request.task == "embedding":
                outputs = model(**inputs)
                # Get last hidden state mean as embedding
                embeddings = outputs.last_hidden_state.mean(dim=1)
                result = f"Generated embedding with shape: {embeddings.shape}"
                
            elif request.task == "classification":
                outputs = model(**inputs)
                # Simple classification based on embedding
                logits = outputs.last_hidden_state.mean(dim=1)
                predictions = torch.softmax(logits, dim=-1)
                result = f"Classification scores: {predictions.cpu().numpy().tolist()}"
                
            else:  # Default to feature extraction
                outputs = model(**inputs)
                result = f"Processed text with model output shape: {outputs.last_hidden_state.shape}"
        
        execution_time = time.time() - start_time
        memory_usage = self.get_memory_usage()
        
        return FSDPResponse(
            model_name=request.model_name,
            output=result,
            task=request.task,
            execution_time=execution_time,
            memory_usage=memory_usage
        )

fsdp_manager = FSDPManager()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "fms-fsdp"}

@app.post("/process", response_model=FSDPResponse)
def process_text(request: FSDPRequest):
    try:
        return fsdp_manager.process_text(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_loaded_models():
    return {
        "loaded_models": list(fsdp_manager.models.keys()),
        "device": str(fsdp_manager.device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
    }

@app.get("/memory")
def get_memory_stats():
    return fsdp_manager.get_memory_usage()

@app.delete("/models/{model_name}")
def unload_model(model_name: str):
    if model_name in fsdp_manager.models:
        del fsdp_manager.models[model_name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "unloaded", "model": model_name}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get("/stats")
def get_stats():
    return {
        "loaded_models": len(fsdp_manager.models),
        "device": str(fsdp_manager.device),
        "memory_usage": fsdp_manager.get_memory_usage(),
        "torch_version": torch.__version__,
        "fsdp_available": hasattr(torch.distributed, 'fsdp')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)