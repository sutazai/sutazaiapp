from fastapi import FastAPI, HTTPException
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize code generation pipeline
try:
    code_gen = pipeline("text-generation", model="DeepSeek-Coder-33B")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    code_gen = None

@app.post("/generate")
async def generate_code(prompt: str, max_tokens: int = 200):
    """
    Generate code using the DeepSeek-Coder-33B model.
    
    Args:
        prompt (str): Input prompt for code generation
        max_tokens (int, optional): Maximum number of tokens to generate
    
    Returns:
        dict: Generated code
    """
    if not code_gen:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = code_gen(prompt, max_tokens=max_tokens)
        return {
            "generated_code": result[0]['generated_text'],
            "model": "DeepSeek-Coder-33B"
        }
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Code generation failed")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify model status.
    
    Returns:
        dict: Health status of the code generation service
    """
    return {
        "status": "ok", 
        "model_loaded": code_gen is not None,
        "model_name": "DeepSeek-Coder-33B"
    }