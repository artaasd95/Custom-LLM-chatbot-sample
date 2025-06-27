"""vLLM-based high-performance model server."""

import logging
import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
import json

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None

from ..core.config import ConfigManager


@dataclass
class VLLMRequest:
    """Request for vLLM inference."""
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
class VLLMResponse:
    """Response from vLLM inference."""
    text: str
    request_id: Optional[str] = None
    tokens_generated: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "completed"
    metadata: Optional[Dict[str, Any]] = None


class VLLMServer:
    """High-performance vLLM-based model server."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize vLLM server.
        
        Args:
            config_manager: Configuration manager instance.
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available. Install with: pip install vllm")
        
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # vLLM components
        self.engine = None
        self.llm = None
        
        # Server state
        self.is_loaded = False
        self.is_ready = False
        self.model_path = None
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.active_requests = {}
        
        self.logger.info("VLLMServer initialized")
    
    async def load_model(self, 
                        model_path: str,
                        tensor_parallel_size: Optional[int] = None,
                        gpu_memory_utilization: float = 0.9,
                        max_model_len: Optional[int] = None,
                        enable_streaming: bool = True) -> bool:
        """Load model with vLLM.
        
        Args:
            model_path: Path to the model.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: GPU memory utilization ratio.
            max_model_len: Maximum model sequence length.
            enable_streaming: Enable streaming support.
            
        Returns:
            True if model loaded successfully.
        """
        try:
            self.logger.info(f"Loading model with vLLM: {model_path}")
            
            # Determine tensor parallel size
            if tensor_parallel_size is None:
                try:
                    import torch
                    tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
                except ImportError:
                    tensor_parallel_size = 1
            
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len or self.config.serving.vllm.max_model_len,
                disable_log_stats=False,
                trust_remote_code=True,
                enforce_eager=False,
                max_context_len_to_capture=self.config.serving.vllm.max_context_len_to_capture,
                block_size=self.config.serving.vllm.block_size,
                swap_space=self.config.serving.vllm.swap_space,
                enable_prefix_caching=self.config.serving.vllm.enable_prefix_caching,
                disable_sliding_window=not self.config.serving.vllm.enable_sliding_window
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Also create synchronous LLM for batch processing
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len or self.config.serving.vllm.max_model_len,
                trust_remote_code=True,
                enforce_eager=False,
                disable_log_stats=False,
                enable_prefix_caching=self.config.serving.vllm.enable_prefix_caching,
                disable_sliding_window=not self.config.serving.vllm.enable_sliding_window
            )
            
            self.model_path = model_path
            self.is_loaded = True
            self.is_ready = True
            
            self.logger.info(f"vLLM model loaded successfully: {model_path}")
            self.logger.info(f"Tensor parallel size: {tensor_parallel_size}")
            self.logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vLLM model: {str(e)}")
            return False
    
    def _create_sampling_params(self, request: VLLMRequest) -> SamplingParams:
        """Create vLLM sampling parameters from request.
        
        Args:
            request: vLLM request.
            
        Returns:
            SamplingParams instance.
        """
        return SamplingParams(
            n=1,
            best_of=1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=0.0,
            use_beam_search=False,
            length_penalty=1.0,
            early_stopping=False,
            stop=request.stop_sequences,
            stop_token_ids=None,
            include_stop_str_in_output=False,
            ignore_eos=False,
            max_tokens=request.max_tokens,
            logprobs=None,
            prompt_logprobs=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True
        )
    
    async def generate(self, request: VLLMRequest) -> Union[VLLMResponse, AsyncGenerator[str, None]]:
        """Generate text using vLLM.
        
        Args:
            request: vLLM request.
            
        Returns:
            vLLM response or async generator for streaming.
        """
        if not self.is_ready:
            raise RuntimeError("vLLM model not ready for inference")
        
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Track request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
            }
            
            if request.stream:
                return self._generate_stream(request, request_id, start_time)
            else:
                return await self._generate_sync(request, request_id, start_time)
                
        except Exception as e:
            self.logger.error(f"vLLM generation failed: {str(e)}")
            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            return VLLMResponse(
                text="",
                request_id=request_id,
                finish_reason="error",
                metadata={"error": str(e)}
            )
    
    async def _generate_sync(self, request: VLLMRequest, request_id: str, start_time: float) -> VLLMResponse:
        """Generate text synchronously with vLLM.
        
        Args:
            request: vLLM request.
            request_id: Request identifier.
            start_time: Request start time.
            
        Returns:
            vLLM response.
        """
        try:
            # Create sampling parameters
            sampling_params = self._create_sampling_params(request)
            
            # Generate using async engine
            results_generator = self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # Get final result
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output is None:
                raise RuntimeError("No output generated")
            
            # Extract generated text
            generated_text = final_output.outputs[0].text
            tokens_generated = len(final_output.outputs[0].token_ids)
            finish_reason = final_output.outputs[0].finish_reason
            
            # Calculate metrics
            generation_time = time.time() - start_time
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            # Update statistics
            self.request_count += 1
            self.total_tokens_generated += tokens_generated
            self.total_inference_time += generation_time
            
            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            return VLLMResponse(
                text=generated_text,
                request_id=request_id,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
                finish_reason=finish_reason or "completed",
                metadata=request.metadata
            )
            
        except Exception as e:
            self.logger.error(f"vLLM sync generation failed: {str(e)}")
            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            raise
    
    async def _generate_stream(self, request: VLLMRequest, request_id: str, start_time: float) -> AsyncGenerator[str, None]:
        """Generate text with streaming using vLLM.
        
        Args:
            request: vLLM request.
            request_id: Request identifier.
            start_time: Request start time.
            
        Yields:
            Generated text chunks.
        """
        try:
            # Create sampling parameters
            sampling_params = self._create_sampling_params(request)
            
            # Generate using async engine
            results_generator = self.engine.generate(
                prompt=request.prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # Stream results
            previous_text = ""
            tokens_generated = 0
            
            async for request_output in results_generator:
                if request_output.outputs:
                    current_text = request_output.outputs[0].text
                    new_text = current_text[len(previous_text):]
                    
                    if new_text:
                        yield new_text
                        previous_text = current_text
                        tokens_generated = len(request_output.outputs[0].token_ids)
            
            # Update statistics
            generation_time = time.time() - start_time
            self.request_count += 1
            self.total_tokens_generated += tokens_generated
            self.total_inference_time += generation_time
            
            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
                
        except Exception as e:
            self.logger.error(f"vLLM streaming generation failed: {str(e)}")
            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            raise
    
    async def generate_batch(self, requests: List[VLLMRequest]) -> List[VLLMResponse]:
        """Generate text for multiple requests in batch.
        
        Args:
            requests: List of vLLM requests.
            
        Returns:
            List of vLLM responses.
        """
        if not self.is_ready:
            raise RuntimeError("vLLM model not ready for inference")
        
        if not requests:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare prompts and sampling parameters
            prompts = [req.prompt for req in requests]
            sampling_params_list = [self._create_sampling_params(req) for req in requests]
            
            # Use synchronous LLM for batch processing
            outputs = self.llm.generate(prompts, sampling_params_list[0])  # vLLM uses same params for all
            
            # Process outputs
            responses = []
            total_tokens = 0
            
            for i, (request, output) in enumerate(zip(requests, outputs)):
                generated_text = output.outputs[0].text
                tokens_generated = len(output.outputs[0].token_ids)
                finish_reason = output.outputs[0].finish_reason
                
                total_tokens += tokens_generated
                
                responses.append(VLLMResponse(
                    text=generated_text,
                    request_id=request.request_id or f"batch_{i}",
                    tokens_generated=tokens_generated,
                    generation_time=0.0,  # Individual timing not available in batch
                    tokens_per_second=0.0,
                    finish_reason=finish_reason or "completed",
                    metadata=request.metadata
                ))
            
            # Update statistics
            batch_time = time.time() - start_time
            self.request_count += len(requests)
            self.total_tokens_generated += total_tokens
            self.total_inference_time += batch_time
            
            # Update individual response metrics
            avg_tokens_per_second = total_tokens / batch_time if batch_time > 0 else 0
            for response in responses:
                response.generation_time = batch_time / len(requests)  # Approximate
                response.tokens_per_second = avg_tokens_per_second
            
            self.logger.info(f"Batch generation completed: {len(requests)} requests, {total_tokens} tokens, {batch_time:.2f}s")
            
            return responses
            
        except Exception as e:
            self.logger.error(f"vLLM batch generation failed: {str(e)}")
            # Return error responses
            return [
                VLLMResponse(
                    text="",
                    request_id=req.request_id or f"batch_{i}",
                    finish_reason="error",
                    metadata={"error": str(e)}
                )
                for i, req in enumerate(requests)
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vLLM server statistics.
        
        Returns:
            Dictionary with server stats.
        """
        avg_tokens_per_request = self.total_tokens_generated / self.request_count if self.request_count > 0 else 0
        avg_time_per_request = self.total_inference_time / self.request_count if self.request_count > 0 else 0
        avg_tokens_per_second = self.total_tokens_generated / self.total_inference_time if self.total_inference_time > 0 else 0
        
        stats = {
            "is_loaded": self.is_loaded,
            "is_ready": self.is_ready,
            "model_path": self.model_path,
            "backend_type": "vllm",
            "request_count": self.request_count,
            "total_tokens_generated": self.total_tokens_generated,
            "total_inference_time": self.total_inference_time,
            "avg_tokens_per_request": avg_tokens_per_request,
            "avg_time_per_request": avg_time_per_request,
            "avg_tokens_per_second": avg_tokens_per_second,
            "active_requests": len(self.active_requests)
        }
        
        # Add engine stats if available
        if self.engine:
            try:
                engine_stats = self.engine.get_model_config()
                stats["model_config"] = {
                    "max_model_len": engine_stats.max_model_len,
                    "vocab_size": engine_stats.get_vocab_size(),
                    "hidden_size": getattr(engine_stats.hf_config, 'hidden_size', None),
                    "num_attention_heads": getattr(engine_stats.hf_config, 'num_attention_heads', None),
                    "num_hidden_layers": getattr(engine_stats.hf_config, 'num_hidden_layers', None)
                }
            except Exception as e:
                self.logger.debug(f"Could not get engine stats: {str(e)}")
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check for vLLM server.
        
        Returns:
            Health status information.
        """
        status = {
            "status": "healthy" if self.is_ready else "unhealthy",
            "is_loaded": self.is_loaded,
            "is_ready": self.is_ready,
            "backend_type": "vllm",
            "model_path": self.model_path,
            "engine_loaded": self.engine is not None,
            "llm_loaded": self.llm is not None,
            "active_requests": len(self.active_requests)
        }
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
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
        except ImportError:
            status["gpu_available"] = False
        
        return status
    
    async def unload_model(self) -> None:
        """Unload the vLLM model and free resources."""
        self.logger.info("Unloading vLLM model")
        
        self.is_ready = False
        self.is_loaded = False
        
        # Clear engine
        if self.engine:
            try:
                # vLLM doesn't have explicit cleanup, but we can delete the reference
                del self.engine
                self.engine = None
            except Exception as e:
                self.logger.warning(f"Error cleaning up vLLM engine: {str(e)}")
        
        # Clear LLM
        if self.llm:
            try:
                del self.llm
                self.llm = None
            except Exception as e:
                self.logger.warning(f"Error cleaning up vLLM LLM: {str(e)}")
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.model_path = None
        self.active_requests.clear()
        
        self.logger.info("vLLM model unloaded successfully")
    
    async def abort_request(self, request_id: str) -> bool:
        """Abort a specific request.
        
        Args:
            request_id: Request identifier to abort.
            
        Returns:
            True if request was aborted successfully.
        """
        try:
            if self.engine and request_id in self.active_requests:
                await self.engine.abort(request_id)
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                self.logger.info(f"Request {request_id} aborted successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to abort request {request_id}: {str(e)}")
            return False
    
    def get_active_requests(self) -> Dict[str, Any]:
        """Get information about active requests.
        
        Returns:
            Dictionary with active request information.
        """
        current_time = time.time()
        return {
            "active_count": len(self.active_requests),
            "requests": [
                {
                    "request_id": req_id,
                    "duration": current_time - req_data["start_time"],
                    "prompt_preview": req_data["prompt"]
                }
                for req_id, req_data in self.active_requests.items()
            ]
        }
    
    def __del__(self):
        """Cleanup resources."""
        try:
            asyncio.create_task(self.unload_model())
        except Exception:
            pass


# Utility functions
def create_vllm_server(config_path: str = "config.yaml") -> VLLMServer:
    """Create vLLM server instance.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        VLLMServer instance.
    """
    from ..core.config import ConfigManager
    
    config_manager = ConfigManager()
    config_manager.load_config(config_path)
    
    return VLLMServer(config_manager)


async def benchmark_vllm_server(server: VLLMServer, 
                               prompts: List[str],
                               num_iterations: int = 10) -> Dict[str, Any]:
    """Benchmark vLLM server performance.
    
    Args:
        server: vLLM server instance.
        prompts: List of prompts for benchmarking.
        num_iterations: Number of benchmark iterations.
        
    Returns:
        Benchmark results.
    """
    if not server.is_ready:
        raise RuntimeError("vLLM server not ready for benchmarking")
    
    results = {
        "total_requests": 0,
        "total_tokens": 0,
        "total_time": 0.0,
        "avg_tokens_per_second": 0.0,
        "avg_time_per_request": 0.0,
        "iterations": num_iterations,
        "prompt_count": len(prompts)
    }
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Create requests
        requests = [
            VLLMRequest(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                request_id=f"bench_{iteration}_{i}"
            )
            for i, prompt in enumerate(prompts)
        ]
        
        # Generate batch
        responses = await server.generate_batch(requests)
        
        # Collect metrics
        for response in responses:
            results["total_requests"] += 1
            results["total_tokens"] += response.tokens_generated
    
    results["total_time"] = time.time() - start_time
    results["avg_tokens_per_second"] = results["total_tokens"] / results["total_time"]
    results["avg_time_per_request"] = results["total_time"] / results["total_requests"]
    
    return results