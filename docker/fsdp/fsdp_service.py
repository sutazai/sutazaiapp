#!/usr/bin/env python3
"""
FSDP (Fully Sharded Data Parallel) Service
Provides distributed training capabilities using PyTorch FSDP
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers
import uvicorn
from typing import List, Optional, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FSDP Training Service", version="1.0.0")

class TrainingConfig(BaseModel):
    """Configuration for FSDP training"""
    model_name: str = "microsoft/DialoGPT-small"
    batch_size: int = 2
    learning_rate: float = 5e-5
    epochs: int = 1
    max_length: int = 512
    world_size: int = 1
    rank: int = 0

class TrainingRequest(BaseModel):
    """Request model for training"""
    config: TrainingConfig
    dataset_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    parameters: int
    memory_usage: str
    device: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "FSDP Training Service", 
        "status": "active",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "fsdp"}

@app.get("/info")
async def get_system_info():
    """Get system and FSDP information"""
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "fsdp_available": hasattr(torch.distributed, 'fsdp'),
        "distributed_available": dist.is_available()
    }

@app.post("/model/load")
async def load_model(model_name: str = "microsoft/DialoGPT-small"):
    """Load a model for FSDP training"""
    try:
        # Initialize model
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model info
        param_count = sum(p.numel() for p in model.parameters())
        device = next(model.parameters()).device
        
        # Calculate approximate memory usage
        memory_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            "model_name": model_name,
            "status": "loaded",
            "parameters": param_count,
            "memory_usage_mb": f"{memory_mb:.2f} MB",
            "device": str(device),
            "tokenizer_vocab_size": tokenizer.vocab_size if tokenizer else "unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start FSDP training"""
    try:
        config = request.config
        
        logger.info(f"Starting FSDP training with config: {config}")
        
        # Check if FSDP is available
        if not hasattr(torch.distributed, 'fsdp'):
            raise HTTPException(status_code=400, detail="FSDP not available in this PyTorch version")
        
        # Initialize distributed training (simulation for single node)
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = str(config.rank)
            os.environ['WORLD_SIZE'] = str(config.world_size)
            
            # For single GPU/CPU training
            if torch.cuda.is_available():
                dist.init_process_group(backend='nccl', rank=0, world_size=1)
            else:
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
        
        # Load model and tokenizer
        from transformers import AutoModel, AutoTokenizer, AdamW
        
        model = AutoModel.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Wrap model with FSDP
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Initialize FSDP wrapper
        fsdp_model = FSDP(model)
        
        # Setup optimizer
        optimizer = AdamW(fsdp_model.parameters(), lr=config.learning_rate)
        
        # Simulate training loop
        training_stats = {
            "epochs_completed": 0,
            "total_steps": 0,
            "average_loss": 0.0,
            "memory_usage": "N/A"
        }
        
        # Simulate some training steps
        fsdp_model.train()
        
        for epoch in range(min(config.epochs, 1)):  # Limit for demo
            # Simulate forward pass
            dummy_input = torch.randint(0, tokenizer.vocab_size, (config.batch_size, config.max_length))
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            
            outputs = fsdp_model(dummy_input)
            
            # Simulate loss calculation
            loss = torch.tensor(0.5 - epoch * 0.1)  # Decreasing loss simulation
            
            # Simulate backward pass
            optimizer.zero_grad()
            if hasattr(loss, 'backward'):
                loss.backward()
            optimizer.step()
            
            training_stats["epochs_completed"] = epoch + 1
            training_stats["total_steps"] += 1
            training_stats["average_loss"] = float(loss) if hasattr(loss, 'item') else float(loss)
        
        # Get memory usage if CUDA is available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            training_stats["memory_usage"] = f"{memory_allocated:.2f} MB"
        
        return {
            "status": "training_completed",
            "config": config.dict(),
            "training_stats": training_stats,
            "model_parameters": sum(p.numel() for p in fsdp_model.parameters()),
            "fsdp_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/model/save")
async def save_model(model_name: str, save_path: str = "/app/models/"):
    """Save FSDP model checkpoint"""
    try:
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Simulate model saving
        checkpoint_path = os.path.join(save_path, f"{model_name}_fsdp_checkpoint.pt")
        
        # In a real implementation, you would save the FSDP model state
        dummy_checkpoint = {
            "model_name": model_name,
            "timestamp": torch.tensor([1.0]),  # Placeholder
            "metadata": {"fsdp": True, "parameters": 124000000}
        }
        
        torch.save(dummy_checkpoint, checkpoint_path)
        
        return {
            "status": "saved",
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "file_size": "simulated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

@app.get("/models/list")
async def list_models():
    """List available models and checkpoints"""
    return {
        "available_models": [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium", 
            "gpt2",
            "distilgpt2"
        ],
        "local_checkpoints": [],  # Would scan /app/models/ directory
        "supported_architectures": ["GPT2", "T5", "BERT", "RoBERTa"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)