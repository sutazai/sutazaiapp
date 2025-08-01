#!/usr/bin/env python3
"""
FSDP (Fully Sharded Data Parallel) Service
Handles distributed model training and inference with memory optimization
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI FSDP Service", version="1.0.0")

class FSDPTrainingRequest(BaseModel):
    model_name: str
    num_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_length: int = 512

class FSDPInferenceRequest(BaseModel):
    model_name: str
    text: str
    max_length: int = 100

class FSDPService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.is_distributed = False
        
        # Initialize distributed training if available
        if torch.cuda.device_count() > 1:
            self.setup_distributed()
    
    def setup_distributed(self):
        """Initialize distributed training environment"""
        try:
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='env://',
                    world_size=int(os.environ.get('WORLD_SIZE', 1)),
                    rank=int(os.environ.get('RANK', 0))
                )
            
            self.is_distributed = True
            logger.info(f"Distributed training initialized. Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")
            self.is_distributed = False
    
    def wrap_model_with_fsdp(self, model):
        """Wrap model with FSDP for memory optimization"""
        if not self.is_distributed:
            return model
        
        # FSDP configuration
        fsdp_config = {
            "auto_wrap_policy": size_based_auto_wrap_policy,
            "cpu_offload": CPUOffload(offload_params=True),
            "mixed_precision": None,  # Can be configured for FP16/BF16
        }
        
        wrapped_model = FSDP(model, **fsdp_config)
        return wrapped_model
    
    async def load_model(self, model_name: str):
        """Load and wrap model with FSDP"""
        try:
            if model_name in self.models:
                return self.models[model_name]
            
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if not self.is_distributed else None
            )
            
            # Wrap with FSDP if distributed
            if self.is_distributed:
                model = self.wrap_model_with_fsdp(model)
            else:
                model = model.to(self.device)
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    async def train_model(self, request: FSDPTrainingRequest):
        """Train model using FSDP"""
        try:
            model = await self.load_model(request.model_name)
            tokenizer = self.tokenizers[request.model_name]
            
            # Training configuration
            optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
            
            # Dummy training data (in real scenario, load from dataset)
            dummy_texts = [
                "This is a sample training text.",
                "FSDP enables efficient distributed training.",
                "SutazAI implements advanced AI capabilities."
            ] * request.batch_size
            
            model.train()
            total_loss = 0.0
            
            for epoch in range(request.num_epochs):
                epoch_loss = 0.0
                
                # Tokenize batch
                inputs = tokenizer(
                    dummy_texts,
                    max_length=request.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(**inputs)
                
                # Simple loss (reconstruction)
                if hasattr(outputs, 'last_hidden_state'):
                    loss = torch.nn.functional.mse_loss(
                        outputs.last_hidden_state,
                        torch.zeros_like(outputs.last_hidden_state)
                    )
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                
                logger.info(f"Epoch {epoch + 1}/{request.num_epochs}, Loss: {epoch_loss:.4f}")
            
            return {
                "status": "success",
                "model_name": request.model_name,
                "epochs_completed": request.num_epochs,
                "final_loss": total_loss / request.num_epochs,
                "training_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")
    
    async def inference(self, request: FSDPInferenceRequest):
        """Perform inference using FSDP model"""
        try:
            model = await self.load_model(request.model_name)
            tokenizer = self.tokenizers[request.model_name]
            
            # Tokenize input
            inputs = tokenizer(
                request.text,
                max_length=request.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract embeddings or logits
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                return {
                    "status": "success",
                    "model_name": request.model_name,
                    "input_text": request.text,
                    "embeddings": embeddings,
                    "embedding_dim": len(embeddings[0]) if embeddings else 0
                }
            else:
                return {
                    "status": "success",
                    "model_name": request.model_name,
                    "input_text": request.text,
                    "output": "Model output processed"
                }
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# Initialize service
fsdp_service = FSDPService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FSDP Service",
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "distributed": fsdp_service.is_distributed,
        "loaded_models": list(fsdp_service.models.keys())
    }

@app.post("/train")
async def train_model(request: FSDPTrainingRequest):
    """Train model using FSDP"""
    return await fsdp_service.train_model(request)

@app.post("/inference")
async def inference(request: FSDPInferenceRequest):
    """Perform inference using FSDP model"""
    return await fsdp_service.inference(request)

@app.get("/models")
async def list_models():
    """List loaded models"""
    return {
        "loaded_models": list(fsdp_service.models.keys()),
        "available_tokenizers": list(fsdp_service.tokenizers.keys())
    }

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model"""
    if model_name in fsdp_service.models:
        del fsdp_service.models[model_name]
        del fsdp_service.tokenizers[model_name]
        torch.cuda.empty_cache()  # Clear GPU memory
        return {"status": "success", "message": f"Model {model_name} unloaded"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

if __name__ == "__main__":
    # Configure port and host
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info("Starting FSDP Service...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    logger.info(f"Distributed training: {fsdp_service.is_distributed}")
    
    uvicorn.run(app, host=host, port=port) 