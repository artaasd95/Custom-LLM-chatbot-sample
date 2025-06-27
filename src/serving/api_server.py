"""FastAPI server for LLM inference API."""

import logging
import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from ..core.config import ConfigManager
from .model_server import ModelServer, InferenceRequest, InferenceResponse


# Pydantic models for API
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str = Field(..., description="Generated text")
    request_id: str = Field(..., description="Request identifier")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_per_second: float = Field(..., description="Generation speed")
    finish_reason: str = Field(..., description="Reason for completion")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    is_loaded: bool = Field(..., description="Model loaded status")
    is_ready: bool = Field(..., description="Model ready status")
    backend_type: Optional[str] = Field(None, description="Backend type")
    model_loaded: bool = Field(..., description="Model component loaded")
    tokenizer_loaded: bool = Field(..., description="Tokenizer component loaded")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_count: Optional[int] = Field(None, description="Number of GPUs")
    gpu_memory: Optional[Dict[str, Any]] = Field(None, description="GPU memory usage")


class StatsResponse(BaseModel):
    """Response model for server statistics."""
    is_loaded: bool = Field(..., description="Model loaded status")
    is_ready: bool = Field(..., description="Model ready status")
    backend_type: Optional[str] = Field(None, description="Backend type")
    request_count: int = Field(..., description="Total requests processed")
    total_tokens_generated: int = Field(..., description="Total tokens generated")
    total_inference_time: float = Field(..., description="Total inference time")
    avg_tokens_per_request: float = Field(..., description="Average tokens per request")
    avg_time_per_request: float = Field(..., description="Average time per request")
    avg_tokens_per_second: float = Field(..., description="Average generation speed")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: str = Field(..., description="Error timestamp")


class APIServer:
    """FastAPI server for LLM inference."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize API server.
        
        Args:
            config_manager: Configuration manager instance.
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize model server
        self.model_server = ModelServer(config_manager)
        
        # Request tracking
        self.active_requests = {}
        self.request_history = []
        
        # Create FastAPI app
        self.app = self._create_app()
        
        self.logger.info("APIServer initialized")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application.
        
        Returns:
            FastAPI application instance.
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Starting API server")
            yield
            # Shutdown
            self.logger.info("Shutting down API server")
            await self.model_server.unload_model()
        
        app = FastAPI(
            title="LLM Inference API",
            description="High-performance API for Large Language Model inference",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.serving.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Add request logging middleware
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            self.logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes to the FastAPI app.
        
        Args:
            app: FastAPI application instance.
        """
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "LLM Inference API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                health_data = self.model_server.health_check()
                return HealthResponse(**health_data)
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats", response_model=StatsResponse)
        async def get_stats():
            """Get server statistics."""
            try:
                stats_data = self.model_server.get_stats()
                return StatsResponse(**stats_data)
            except Exception as e:
                self.logger.error(f"Failed to get stats: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/load_model")
        async def load_model(
            model_path: str,
            backend: str = "pytorch",
            background_tasks: BackgroundTasks = None
        ):
            """Load model for inference."""
            try:
                if self.model_server.is_loaded:
                    return {"message": "Model already loaded", "status": "success"}
                
                success = await self.model_server.load_model(model_path, backend)
                
                if success:
                    return {
                        "message": f"Model loaded successfully with {backend} backend",
                        "status": "success",
                        "model_path": model_path,
                        "backend": backend
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to load model"
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/unload_model")
        async def unload_model():
            """Unload current model."""
            try:
                await self.model_server.unload_model()
                return {"message": "Model unloaded successfully", "status": "success"}
            except Exception as e:
                self.logger.error(f"Failed to unload model: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/generate", response_model=GenerateResponse)
        async def generate_text(request: GenerateRequest):
            """Generate text from the model."""
            if not self.model_server.is_ready:
                raise HTTPException(
                    status_code=503,
                    detail="Model not ready. Please load a model first."
                )
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            try:
                # Track request
                self.active_requests[request_id] = {
                    "start_time": time.time(),
                    "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
                }
                
                # Create inference request
                inference_request = InferenceRequest(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    stop_sequences=request.stop_sequences,
                    stream=False,  # Non-streaming for this endpoint
                    request_id=request_id,
                    metadata=request.metadata
                )
                
                # Generate response
                response = await self.model_server.generate(inference_request)
                
                # Clean up tracking
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                
                # Add to history
                self.request_history.append({
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "tokens_generated": response.tokens_generated,
                    "generation_time": response.generation_time
                })
                
                # Keep only last 1000 requests in history
                if len(self.request_history) > 1000:
                    self.request_history = self.request_history[-1000:]
                
                return GenerateResponse(
                    text=response.text,
                    request_id=response.request_id,
                    tokens_generated=response.tokens_generated,
                    generation_time=response.generation_time,
                    tokens_per_second=response.tokens_per_second,
                    finish_reason=response.finish_reason,
                    metadata=response.metadata
                )
                
            except Exception as e:
                # Clean up tracking
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                
                self.logger.error(f"Generation failed for request {request_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/generate_stream")
        async def generate_stream(request: GenerateRequest):
            """Generate text with streaming response."""
            if not self.model_server.is_ready:
                raise HTTPException(
                    status_code=503,
                    detail="Model not ready. Please load a model first."
                )
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            try:
                # Track request
                self.active_requests[request_id] = {
                    "start_time": time.time(),
                    "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
                }
                
                # Create inference request
                inference_request = InferenceRequest(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    stop_sequences=request.stop_sequences,
                    stream=True,
                    request_id=request_id,
                    metadata=request.metadata
                )
                
                # Generate streaming response
                async def generate_chunks():
                    try:
                        async for chunk in await self.model_server.generate(inference_request):
                            yield f"data: {chunk}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        self.logger.error(f"Streaming generation failed: {str(e)}")
                        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                    finally:
                        # Clean up tracking
                        if request_id in self.active_requests:
                            del self.active_requests[request_id]
                
                return StreamingResponse(
                    generate_chunks(),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Request-ID": request_id
                    }
                )
                
            except Exception as e:
                # Clean up tracking
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                
                self.logger.error(f"Streaming setup failed for request {request_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/active_requests")
        async def get_active_requests():
            """Get currently active requests."""
            return {
                "active_count": len(self.active_requests),
                "requests": [
                    {
                        "request_id": req_id,
                        "duration": time.time() - req_data["start_time"],
                        "prompt_preview": req_data["prompt"]
                    }
                    for req_id, req_data in self.active_requests.items()
                ]
            }
        
        @app.get("/request_history")
        async def get_request_history(limit: int = 100):
            """Get request history."""
            return {
                "total_requests": len(self.request_history),
                "recent_requests": self.request_history[-limit:] if self.request_history else []
            }
        
        # Error handlers
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=exc.detail,
                    error_type="HTTPException",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                ).dict()
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Unhandled exception: {str(exc)}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal server error",
                    error_type=type(exc).__name__,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                ).dict()
            )
    
    async def start_server(self, 
                          host: str = "0.0.0.0", 
                          port: int = 8000,
                          reload: bool = False) -> None:
        """Start the API server.
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
            reload: Enable auto-reload for development.
        """
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        self.logger.info(f"Starting API server on {host}:{port}")
        await server.serve()
    
    def run_server(self, 
                   host: str = "0.0.0.0", 
                   port: int = 8000,
                   reload: bool = False) -> None:
        """Run the API server (blocking).
        
        Args:
            host: Host to bind to.
            port: Port to bind to.
            reload: Enable auto-reload for development.
        """
        uvicorn.run(
            app=self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance.
        
        Returns:
            FastAPI application.
        """
        return self.app


# Standalone server runner
def create_api_server(config_path: str = "config.yaml") -> APIServer:
    """Create API server instance.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        APIServer instance.
    """
    from ..core.config import ConfigManager
    
    config_manager = ConfigManager()
    config_manager.load_config(config_path)
    
    return APIServer(config_manager)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Inference API Server")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run server
    api_server = create_api_server(args.config)
    api_server.run_server(
        host=args.host,
        port=args.port,
        reload=args.reload
    )