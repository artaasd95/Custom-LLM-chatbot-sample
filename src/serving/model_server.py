"""Model server for LLM inference."""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        GenerationConfig, TextStreamer
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from ..core.config import ConfigManager
from ..core.model_manager import ModelManager


@dataclass
class InferenceRequest:
    """Request for model inference."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResponse:
    """Response from model inference."""
    text: str
    request_id: Optional[str] = None
    tokens_generated: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "completed"
    metadata: Optional[Dict[str, Any]] = None


class ModelServer:
    """High-performance model server for LLM inference."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize model server.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model_manager = None
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # ONNX components
        self.onnx_session = None
        self.onnx_tokenizer = None
        
        # Server state
        self.is_loaded = False
        self.is_ready = False
        self.backend_type = None  # 'pytorch', 'onnx'
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.serving.max_concurrent_requests)
        self.request_lock = threading.Lock()
        
        self.logger.info("ModelServer initialized")
    
    async def load_model(self, model_path: str, backend: str = "pytorch") -> bool:
        """Load model for inference.
        
        Args:
            model_path: Path to the model.
            backend: Backend type ('pytorch' or 'onnx').
            
        Returns:
            True if model loaded successfully.
        """
        try:
            self.logger.info(f"Loading model from {model_path} with {backend} backend")
            
            if backend == "pytorch":
                success = await self._load_pytorch_model(model_path)
            elif backend == "onnx":
                success = await self._load_onnx_model(model_path)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            if success:
                self.backend_type = backend
                self.is_loaded = True
                self.is_ready = True
                self.logger.info(f"Model loaded successfully with {backend} backend")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    async def _load_pytorch_model(self, model_path: str) -> bool:
        """Load PyTorch model.
        
        Args:
            model_path: Path to the model.
            
        Returns:
            True if successful.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        try:
            # Initialize model manager
            self.model_manager = ModelManager(self.config)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.config.get_torch_dtype(),
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.serving.max_tokens,
                temperature=self.config.serving.temperature,
                top_p=self.config.serving.top_p,
                top_k=self.config.serving.top_k,
                repetition_penalty=self.config.serving.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {str(e)}")
            return False
    
    async def _load_onnx_model(self, model_path: str) -> bool:
        """Load ONNX model.
        
        Args:
            model_path: Path to the ONNX model.
            
        Returns:
            True if successful.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        try:
            # Setup ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            
            self.onnx_session = ort.InferenceSession(
                str(Path(model_path) / "model.onnx"),
                providers=providers
            )
            
            # Load tokenizer
            tokenizer_path = Path(model_path) / "tokenizer"
            if tokenizer_path.exists():
                self.onnx_tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    trust_remote_code=True
                )
            else:
                # Fallback to model path
                self.onnx_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            
            if self.onnx_tokenizer.pad_token is None:
                self.onnx_tokenizer.pad_token = self.onnx_tokenizer.eos_token
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {str(e)}")
            return False
    
    async def generate(self, request: InferenceRequest) -> Union[InferenceResponse, AsyncGenerator[str, None]]:
        """Generate text from the model.
        
        Args:
            request: Inference request.
            
        Returns:
            Inference response or async generator for streaming.
        """
        if not self.is_ready:
            raise RuntimeError("Model not ready for inference")
        
        start_time = time.time()
        
        try:
            if request.stream:
                return self._generate_stream(request, start_time)
            else:
                return await self._generate_sync(request, start_time)
                
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return InferenceResponse(
                text="",
                request_id=request.request_id,
                finish_reason="error",
                metadata={"error": str(e)}
            )
    
    async def _generate_sync(self, request: InferenceRequest, start_time: float) -> InferenceResponse:
        """Generate text synchronously.
        
        Args:
            request: Inference request.
            start_time: Request start time.
            
        Returns:
            Inference response.
        """
        if self.backend_type == "pytorch":
            return await self._generate_pytorch_sync(request, start_time)
        elif self.backend_type == "onnx":
            return await self._generate_onnx_sync(request, start_time)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")
    
    async def _generate_pytorch_sync(self, request: InferenceRequest, start_time: float) -> InferenceResponse:
        """Generate text with PyTorch backend.
        
        Args:
            request: Inference request.
            start_time: Request start time.
            
        Returns:
            Inference response.
        """
        # Tokenize input
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.serving.max_input_length
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Update generation config
        gen_config = GenerationConfig(
            max_new_tokens=min(request.max_tokens, self.config.serving.max_tokens),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_generated = len(generated_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Update statistics
        with self.request_lock:
            self.request_count += 1
            self.total_tokens_generated += tokens_generated
            self.total_inference_time += generation_time
        
        return InferenceResponse(
            text=generated_text,
            request_id=request.request_id,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            finish_reason="completed"
        )
    
    async def _generate_onnx_sync(self, request: InferenceRequest, start_time: float) -> InferenceResponse:
        """Generate text with ONNX backend.
        
        Args:
            request: Inference request.
            start_time: Request start time.
            
        Returns:
            Inference response.
        """
        # Tokenize input
        inputs = self.onnx_tokenizer(
            request.prompt,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.config.serving.max_input_length
        )
        
        # Prepare ONNX inputs
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
        # Run inference
        outputs = self.onnx_session.run(None, onnx_inputs)
        
        # Process outputs (simplified - actual implementation depends on model structure)
        # This is a placeholder - ONNX text generation requires more complex handling
        generated_text = "ONNX generation not fully implemented"
        tokens_generated = 0
        
        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Update statistics
        with self.request_lock:
            self.request_count += 1
            self.total_tokens_generated += tokens_generated
            self.total_inference_time += generation_time
        
        return InferenceResponse(
            text=generated_text,
            request_id=request.request_id,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            finish_reason="completed"
        )
    
    async def _generate_stream(self, request: InferenceRequest, start_time: float) -> AsyncGenerator[str, None]:
        """Generate text with streaming.
        
        Args:
            request: Inference request.
            start_time: Request start time.
            
        Yields:
            Generated text chunks.
        """
        if self.backend_type != "pytorch":
            raise NotImplementedError("Streaming only supported for PyTorch backend")
        
        # Tokenize input
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.serving.max_input_length
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Setup streaming
        class AsyncStreamer:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.tokens = []
                self.queue = asyncio.Queue()
                self.finished = False
            
            def put(self, value):
                if value is None:
                    self.finished = True
                    asyncio.create_task(self.queue.put(None))
                else:
                    self.tokens.append(value)
                    text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
                    asyncio.create_task(self.queue.put(text))
            
            def end(self):
                self.finished = True
                asyncio.create_task(self.queue.put(None))
            
            async def __aiter__(self):
                while not self.finished:
                    item = await self.queue.get()
                    if item is None:
                        break
                    yield item
        
        streamer = AsyncStreamer(self.tokenizer)
        
        # Update generation config
        gen_config = GenerationConfig(
            max_new_tokens=min(request.max_tokens, self.config.serving.max_tokens),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Start generation in background
        def generate_background():
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        streamer=streamer
                    )
            except Exception as e:
                self.logger.error(f"Streaming generation failed: {str(e)}")
            finally:
                streamer.end()
        
        # Start generation thread
        generation_thread = threading.Thread(target=generate_background)
        generation_thread.start()
        
        # Stream results
        async for text_chunk in streamer:
            yield text_chunk
        
        # Wait for generation to complete
        generation_thread.join()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.
        
        Returns:
            Dictionary with server stats.
        """
        with self.request_lock:
            avg_tokens_per_request = self.total_tokens_generated / self.request_count if self.request_count > 0 else 0
            avg_time_per_request = self.total_inference_time / self.request_count if self.request_count > 0 else 0
            avg_tokens_per_second = self.total_tokens_generated / self.total_inference_time if self.total_inference_time > 0 else 0
            
            return {
                "is_loaded": self.is_loaded,
                "is_ready": self.is_ready,
                "backend_type": self.backend_type,
                "request_count": self.request_count,
                "total_tokens_generated": self.total_tokens_generated,
                "total_inference_time": self.total_inference_time,
                "avg_tokens_per_request": avg_tokens_per_request,
                "avg_time_per_request": avg_time_per_request,
                "avg_tokens_per_second": avg_tokens_per_second
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status information.
        """
        status = {
            "status": "healthy" if self.is_ready else "unhealthy",
            "is_loaded": self.is_loaded,
            "is_ready": self.is_ready,
            "backend_type": self.backend_type,
            "model_loaded": self.model is not None or self.onnx_session is not None,
            "tokenizer_loaded": self.tokenizer is not None or self.onnx_tokenizer is not None
        }
        
        # Check GPU availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            status["gpu_available"] = True
            status["gpu_count"] = torch.cuda.device_count()
            status["gpu_memory"] = {
                f"gpu_{i}": {
                    "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
                }
                for i in range(torch.cuda.device_count())
            }
        else:
            status["gpu_available"] = False
        
        return status
    
    async def unload_model(self) -> None:
        """Unload the model and free resources."""
        self.logger.info("Unloading model")
        
        self.is_ready = False
        self.is_loaded = False
        
        # Clear PyTorch components
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear ONNX components
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None
        
        if self.onnx_tokenizer is not None:
            del self.onnx_tokenizer
            self.onnx_tokenizer = None
        
        # Clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.backend_type = None
        self.logger.info("Model unloaded successfully")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)