#!/usr/bin/env python3
"""Demo server for testing the Streamlit UI without a real LLM model."""

import time
import random
import uuid
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling")
    top_k: int = Field(50, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    stop_sequences: list = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming")

class GenerateResponse(BaseModel):
    text: str
    request_id: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    finish_reason: str
    metadata: Dict[str, Any] = None

class HealthResponse(BaseModel):
    status: str
    is_loaded: bool
    is_ready: bool
    backend_type: str
    model_loaded: bool
    tokenizer_loaded: bool
    gpu_available: bool
    gpu_count: int = None
    gpu_memory: Dict[str, Any] = None

class StatsResponse(BaseModel):
    is_loaded: bool
    is_ready: bool
    backend_type: str
    request_count: int
    total_tokens_generated: int
    total_inference_time: float
    avg_tokens_per_request: float
    avg_time_per_request: float
    avg_tokens_per_second: float

# Demo responses for different prompt types
DEMO_RESPONSES = {
    "code": [
        "Here's a Python function that solves your problem:\n\n```python\ndef solution():\n    return 'Hello, World!'\n```\n\nThis function demonstrates the basic structure you requested.",
        "I can help you with that! Here's a code example:\n\n```python\nimport numpy as np\n\ndef process_data(data):\n    return np.array(data).mean()\n```\n\nThis should handle your data processing needs.",
        "Let me provide a solution:\n\n```python\nclass DataProcessor:\n    def __init__(self):\n        self.data = []\n    \n    def add_item(self, item):\n        self.data.append(item)\n```\n\nThis class-based approach gives you more flexibility."
    ],
    "creative": [
        "Once upon a time, in a digital realm where code and creativity intertwined, there lived a curious developer who discovered the magic of AI-powered storytelling...",
        "The morning sun cast long shadows across the keyboard as the programmer began their quest to build something extraordinary. Little did they know, this would be the day everything changed...",
        "In the bustling world of technology, where algorithms dance and data flows like rivers, a new story was about to unfold—one that would bridge the gap between human imagination and artificial intelligence..."
    ],
    "question": [
        "That's an excellent question! Based on current research and best practices, I can provide you with a comprehensive answer that covers the key aspects you're interested in.",
        "Great question! Let me break this down for you step by step to give you a clear and thorough understanding of the topic.",
        "I'm happy to help explain this concept. The answer involves several important factors that I'll outline for you in a structured way."
    ],
    "general": [
        "Hello! I'm here to help you with any questions or tasks you might have. Feel free to ask me about coding, creative writing, analysis, or general conversation.",
        "Hi there! I'm an AI assistant powered by this custom LLM system. I can help with a wide variety of tasks including programming, writing, problem-solving, and more.",
        "Greetings! I'm ready to assist you with whatever you need. Whether it's technical questions, creative projects, or just a friendly chat, I'm here to help.",
        "Thank you for your message! I understand what you're asking about, and I'll do my best to provide you with a helpful and informative response."
    ]
}

# Create FastAPI app
app = FastAPI(
    title="Demo LLM Server",
    description="Demo server for testing Streamlit UI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server statistics
stats = {
    "request_count": 0,
    "total_tokens_generated": 0,
    "total_inference_time": 0.0,
    "start_time": time.time()
}

def classify_prompt(prompt: str) -> str:
    """Classify the prompt to select appropriate demo response."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['code', 'python', 'function', 'class', 'programming', 'script']):
        return "code"
    elif any(word in prompt_lower for word in ['story', 'creative', 'write', 'imagine', 'tale', 'narrative']):
        return "creative"
    elif any(word in prompt_lower for word in ['what', 'how', 'why', 'explain', 'question', '?']):
        return "question"
    else:
        return "general"

def generate_demo_response(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate a demo response based on the prompt."""
    prompt_type = classify_prompt(prompt)
    responses = DEMO_RESPONSES[prompt_type]
    
    # Select response based on temperature (higher temp = more random)
    if temperature > 0.8:
        response = random.choice(responses)
    elif temperature > 0.5:
        response = responses[len(responses) // 2]
    else:
        response = responses[0]
    
    # Simulate token limit
    words = response.split()
    if len(words) > max_tokens // 4:  # Rough approximation: 4 chars per token
        words = words[:max_tokens // 4]
        response = ' '.join(words) + "..."
    
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        is_loaded=True,
        is_ready=True,
        backend_type="demo",
        model_loaded=True,
        tokenizer_loaded=True,
        gpu_available=True,
        gpu_count=1,
        gpu_memory={"total": "8GB", "used": "2GB", "free": "6GB"}
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get server statistics."""
    uptime = time.time() - stats["start_time"]
    
    avg_tokens_per_request = (
        stats["total_tokens_generated"] / stats["request_count"]
        if stats["request_count"] > 0 else 0
    )
    
    avg_time_per_request = (
        stats["total_inference_time"] / stats["request_count"]
        if stats["request_count"] > 0 else 0
    )
    
    avg_tokens_per_second = (
        stats["total_tokens_generated"] / stats["total_inference_time"]
        if stats["total_inference_time"] > 0 else 0
    )
    
    return StatsResponse(
        is_loaded=True,
        is_ready=True,
        backend_type="demo",
        request_count=stats["request_count"],
        total_tokens_generated=stats["total_tokens_generated"],
        total_inference_time=stats["total_inference_time"],
        avg_tokens_per_request=avg_tokens_per_request,
        avg_time_per_request=avg_time_per_request,
        avg_tokens_per_second=avg_tokens_per_second
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text endpoint."""
    start_time = time.time()
    
    # Simulate processing time based on max_tokens
    processing_time = min(0.5 + (request.max_tokens / 1000), 3.0)
    await asyncio.sleep(processing_time)
    
    # Generate demo response
    response_text = generate_demo_response(
        request.prompt, 
        request.max_tokens, 
        request.temperature
    )
    
    # Calculate metrics
    generation_time = time.time() - start_time
    tokens_generated = len(response_text.split())  # Rough token count
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Update stats
    stats["request_count"] += 1
    stats["total_tokens_generated"] += tokens_generated
    stats["total_inference_time"] += generation_time
    
    return GenerateResponse(
        text=response_text,
        request_id=str(uuid.uuid4()),
        tokens_generated=tokens_generated,
        generation_time=generation_time,
        tokens_per_second=tokens_per_second,
        finish_reason="completed",
        metadata={
            "prompt_type": classify_prompt(request.prompt),
            "demo_mode": True
        }
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Demo LLM Server",
        "status": "running",
        "endpoints": ["/health", "/stats", "/generate"],
        "note": "This is a demo server for testing the Streamlit UI"
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo LLM Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Demo LLM Server on http://{args.host}:{args.port}")
    print("📝 This is a demo server that provides realistic responses for testing")
    print("🔗 Use this with the Streamlit UI to test the interface")
    print("⚠️  Note: This is not a real LLM - responses are pre-generated demos")
    
    uvicorn.run(
        "demo_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )