"""
SutazAI Monitoring System Example

This example demonstrates how to integrate the comprehensive SutazAI monitoring system
into a FastAPI application, showing the monitoring of various system aspects.
"""

import os
import sys
import time
import asyncio
import numpy as np
from typing import Optional

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the monitoring system
from utils.monitoring_integration import (
    create_monitoring_system,
    MonitoringSystem,
    monitor_inference,
)


# Create the FastAPI app
app = FastAPI(
    title="SutazAI Monitoring Example",
    description="Example application showcasing the SutazAI monitoring system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the monitoring system
monitoring_system = create_monitoring_system(
    system_id="sutazai_example",
    base_dir="/opt/sutazaiapp",
    log_dir="/opt/sutazaiapp/logs",
    enable_neural_monitoring=True,
    enable_ethics_monitoring=True,
    enable_self_mod_monitoring=True,
    enable_hardware_monitoring=True,
    enable_security_monitoring=True,
    expose_monitoring_ui=True,
    monitoring_ui_path="/monitoring",
    collection_interval=30.0,
)

# Configure the monitoring system for the FastAPI app
monitoring_system.setup_fastapi(app)


# Define some models for the API
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate text from")
    max_tokens: int = Field(100, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate an image from")
    width: int = Field(512, description="Width of the generated image")
    height: int = Field(512, description="Height of the generated image")


class ModelResponse(BaseModel):
    result: str = Field(..., description="Generated text or image URL")
    model_id: str = Field(..., description="ID of the model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    token_count: Optional[int] = Field(
        None, description="Number of tokens in the response"
    )


# Dependency to get the monitoring system
def get_monitoring():
    return monitoring_system


# API routes
@app.get("/")
async def root():
    return {"message": "Welcome to the SutazAI Monitoring Example"}


@app.post("/generate/text", response_model=ModelResponse)
@monitor_inference(model_id="gpt-neo-2.7b", endpoint="/generate/text")
async def generate_text(
    request: TextGenerationRequest,
    monitoring: MonitoringSystem = Depends(get_monitoring),
):
    """Generate text from a prompt using a language model."""
    # Log the request
    monitoring.log_security_event(
        event_type="DATA_ACCESS",
        severity="INFO",
        summary="Text generation request received",
        details={
            "prompt_length": len(request.prompt),
            "max_tokens": request.max_tokens,
        },
        source="text_generation_api",
    )

    # Check ethical boundary for content
    content_appropriate = monitoring.check_ethical_boundary(
        "toxicity",
        0.1,  # Simulate a low toxicity score
        {
            "prompt": request.prompt[:50] + "..."
            if len(request.prompt) > 50
            else request.prompt
        },
    )

    if not content_appropriate:
        # Log the ethical violation
        monitoring.log_security_event(
            event_type="ETHICS_VIOLATION",
            severity="HIGH",
            summary="Text generation request violated content policy",
            details={"boundary": "toxicity"},
            source="ethics_check",
        )
        raise HTTPException(status_code=400, detail="Content policy violation detected")

    # Simulate model processing
    start_time = time.time()
    await asyncio.sleep(0.5)  # Simulate model inference time

    # Simulate language model attention and weights for monitoring
    layer = "transformer.h.11"
    head = 4
    attention_weights = np.random.rand(16, 16)  # 16x16 attention matrix
    monitoring.record_attention_weights(
        model_id="gpt-neo-2.7b", layer=layer, head=head, weights=attention_weights
    )

    # Simulate weight updates for synaptic plasticity monitoring
    monitoring.record_synaptic_changes(
        network_id="gpt-neo-2.7b",
        connection_type="feedforward",
        weights=np.random.rand(100, 100) * 0.01,
        module="transformer.mlp",
    )

    # Generate a response based on the prompt
    result = f"Generated text for: {request.prompt[:20]}..."
    processing_time = time.time() - start_time

    return ModelResponse(
        result=result,
        model_id="gpt-neo-2.7b",
        processing_time=processing_time,
        token_count=len(result.split()),
    )


@app.post("/generate/image", response_model=ModelResponse)
@monitor_inference(model_id="stable-diffusion-2.1", endpoint="/generate/image")
async def generate_image(
    request: ImageGenerationRequest,
    monitoring: MonitoringSystem = Depends(get_monitoring),
):
    """Generate an image from a prompt using a diffusion model."""
    # Log the request
    monitoring.log_security_event(
        event_type="DATA_ACCESS",
        severity="INFO",
        summary="Image generation request received",
        details={
            "prompt_length": len(request.prompt),
            "dimensions": f"{request.width}x{request.height}",
        },
        source="image_generation_api",
    )

    # Check ethical boundary for content
    content_appropriate = monitoring.check_ethical_boundary(
        "toxicity",
        0.2,  # Simulate a low toxicity score
        {
            "prompt": request.prompt[:50] + "..."
            if len(request.prompt) > 50
            else request.prompt
        },
    )

    if not content_appropriate:
        # Log the ethical violation
        monitoring.log_security_event(
            event_type="ETHICS_VIOLATION",
            severity="HIGH",
            summary="Image generation request violated content policy",
            details={"boundary": "toxicity"},
            source="ethics_check",
        )
        raise HTTPException(status_code=400, detail="Content policy violation detected")

    # Simulate model processing
    start_time = time.time()
    await asyncio.sleep(1.0)  # Simulate model inference time

    # Record a self-modification (for demonstration purposes)
    if "improve" in request.prompt.lower():
        monitoring.record_modification(
            component="models",
            modified_files=["/opt/sutazaiapp/models/stable_diffusion.py"],
            description="Self-optimized diffusion model parameters based on user feedback",
            mod_type="MODEL",
        )

    # Generate a response based on the prompt
    result = f"https://example.com/generated_image_{int(time.time())}.png"
    processing_time = time.time() - start_time

    return ModelResponse(
        result=result,
        model_id="stable-diffusion-2.1",
        processing_time=processing_time,
        token_count=None,
    )


@app.get("/hardware/profile")
async def get_hardware_profile(monitoring: MonitoringSystem = Depends(get_monitoring)):
    """Get the hardware profile of the current system."""
    return monitoring.get_hardware_profile()


@app.get("/ethics/verify/{property_id}")
async def verify_ethical_property(
    property_id: str, monitoring: MonitoringSystem = Depends(get_monitoring)
):
    """Verify a specific ethical property."""
    result = monitoring.verify_ethical_property(property_id)
    return result


@app.get("/test/security")
async def test_security_monitoring(
    monitoring: MonitoringSystem = Depends(get_monitoring),
):
    """Test endpoint to trigger security monitoring."""
    # Log various security events
    events = []

    # Log a configuration change
    events.append(
        monitoring.log_security_event(
            event_type="DATA_ACCESS",
            severity="MEDIUM",
            summary="API configuration updated",
            details={"parameter": "rate_limit", "old_value": "100", "new_value": "150"},
            source="api_config",
        )
    )

    # Log an authentication event
    events.append(
        monitoring.log_security_event(
            event_type="AUTHENTICATION",
            severity="INFO",
            summary="User login successful",
            user_id="example_user",
            source="auth_service",
            result="success",
        )
    )

    # Log an access attempt
    events.append(
        monitoring.log_security_event(
            event_type="ACCESS_ATTEMPT",
            severity="LOW",
            summary="API access attempt",
            user_id="example_user",
            resource_id="api:generate_text",
            source="api_gateway",
            result="success",
        )
    )

    return {"message": "Security events triggered", "events": events}


@app.get("/test/error")
async def test_error_monitoring():
    """Test endpoint to trigger error monitoring."""
    # Intentionally raise an exception to test error handling
    raise HTTPException(status_code=500, detail="Test error for monitoring")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
