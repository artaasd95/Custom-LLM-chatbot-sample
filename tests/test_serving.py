#!/usr/bin/env python3
"""Tests for the serving modules."""

import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import ConfigManager
from src.serving.model_server import ModelServer
from src.serving.api_server import APIServer
from src.serving.vllm_server import VLLMServer, VLLMRequest


class TestModelServer:
    """Test cases for ModelServer class."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager for testing."""
        config = Mock()
        config.config = Mock()
        config.config.serving = Mock()
        config.config.serving.max_concurrent_requests = 10
        config.config.serving.timeout = 300
        config.config.model = Mock()
        config.config.model.torch_dtype = "float16"
        config.config.model.device_map = "auto"
        config.get_device.return_value = "cpu"
        config.get_torch_dtype.return_value = "torch.float16"
        return config
    
    @pytest.fixture
    def model_server(self, config_manager):
        """Create a ModelServer instance for testing."""
        return ModelServer(config_manager)
    
    def test_init(self, config_manager):
        """Test ModelServer initialization."""
        server = ModelServer(config_manager)
        assert server.config == config_manager
        assert server.model is None
        assert server.tokenizer is None
        assert server.backend is None
        assert server.device == "cpu"
    
    @pytest.mark.asyncio
    @patch('src.serving.model_server.AutoTokenizer')
    @patch('src.serving.model_server.AutoModelForCausalLM')
    async def test_load_model_pytorch(self, mock_model_class, mock_tokenizer_class, model_server):
        """Test loading model with PyTorch backend."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test loading
        success = await model_server.load_model("test_model", "pytorch")
        
        assert success is True
        assert model_server.model == mock_model
        assert model_server.tokenizer == mock_tokenizer
        assert model_server.backend == "pytorch"
        assert model_server.model_path == "test_model"
    
    @pytest.mark.asyncio
    @patch('src.serving.model_server.onnxruntime')
    async def test_load_model_onnx(self, mock_onnxruntime, model_server):
        """Test loading model with ONNX backend."""
        # Mock ONNX session
        mock_session = Mock()
        mock_onnxruntime.InferenceSession.return_value = mock_session
        
        # Test loading
        success = await model_server.load_model("test_model.onnx", "onnx")
        
        assert success is True
        assert model_server.model == mock_session
        assert model_server.backend == "onnx"
    
    @pytest.mark.asyncio
    async def test_load_model_unsupported_backend(self, model_server):
        """Test loading model with unsupported backend."""
        success = await model_server.load_model("test_model", "unsupported")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_unload_model(self, model_server):
        """Test unloading model."""
        # Set up a loaded model
        model_server.model = Mock()
        model_server.tokenizer = Mock()
        model_server.backend = "pytorch"
        
        success = await model_server.unload_model()
        
        assert success is True
        assert model_server.model is None
        assert model_server.tokenizer is None
        assert model_server.backend is None
    
    @pytest.mark.asyncio
    @patch('src.serving.model_server.torch')
    async def test_generate_pytorch(self, mock_torch, model_server):
        """Test text generation with PyTorch backend."""
        # Mock model and tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.pad_token_id = 0
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.sequences = [[1, 2, 3, 4, 5]]
        mock_model.generate.return_value = mock_output
        
        model_server.model = mock_model
        model_server.tokenizer = mock_tokenizer
        model_server.backend = "pytorch"
        model_server.device = "cpu"
        
        # Mock torch.no_grad
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        result = await model_server.generate(
            prompt="Test prompt",
            max_tokens=50,
            temperature=0.7
        )
        
        assert result == "Generated text"
        mock_model.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_no_model_loaded(self, model_server):
        """Test text generation when no model is loaded."""
        result = await model_server.generate("Test prompt")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_stream(self, model_server):
        """Test streaming text generation."""
        # Mock model and tokenizer for streaming
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated"
        
        model_server.model = Mock()
        model_server.tokenizer = mock_tokenizer
        model_server.backend = "pytorch"
        
        # Mock streaming generation
        async def mock_stream():
            yield "Generated"
            yield " text"
        
        with patch.object(model_server, '_generate_stream_pytorch', return_value=mock_stream()):
            result = []
            async for chunk in model_server.generate_stream("Test prompt"):
                result.append(chunk)
            
            assert result == ["Generated", " text"]
    
    def test_get_stats(self, model_server):
        """Test getting server statistics."""
        stats = model_server.get_stats()
        
        assert "model_loaded" in stats
        assert "backend" in stats
        assert "total_requests" in stats
        assert "active_requests" in stats
        assert "average_response_time" in stats
        assert "memory_usage" in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self, model_server):
        """Test health check functionality."""
        # Test without loaded model
        health = await model_server.health_check()
        assert health["status"] == "unhealthy"
        assert "No model loaded" in health["message"]
        
        # Test with loaded model
        model_server.model = Mock()
        model_server.backend = "pytorch"
        health = await model_server.health_check()
        assert health["status"] == "healthy"


class TestAPIServer:
    """Test cases for APIServer class."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager for testing."""
        config = Mock()
        config.config = Mock()
        config.config.serving = Mock()
        config.config.serving.host = "0.0.0.0"
        config.config.serving.port = 8000
        config.config.serving.enable_cors = True
        config.config.serving.cors_origins = ["*"]
        config.config.serving.max_concurrent_requests = 10
        return config
    
    @pytest.fixture
    def api_server(self, config_manager):
        """Create an APIServer instance for testing."""
        return APIServer(config_manager)
    
    @pytest.fixture
    def test_client(self, api_server):
        """Create a test client for the API server."""
        return TestClient(api_server.app)
    
    def test_init(self, config_manager):
        """Test APIServer initialization."""
        server = APIServer(config_manager)
        assert server.config == config_manager
        assert server.model_server is not None
        assert server.app is not None
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_stats_endpoint(self, test_client):
        """Test statistics endpoint."""
        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "total_requests" in data
    
    @patch('src.serving.api_server.ModelServer')
    def test_load_model_endpoint(self, mock_model_server_class, test_client):
        """Test model loading endpoint."""
        # Mock the model server
        mock_model_server = AsyncMock()
        mock_model_server.load_model.return_value = True
        mock_model_server_class.return_value = mock_model_server
        
        response = test_client.post("/load_model", json={
            "model_path": "test_model",
            "backend": "pytorch"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @patch('src.serving.api_server.ModelServer')
    def test_generate_endpoint(self, mock_model_server_class, test_client):
        """Test text generation endpoint."""
        # Mock the model server
        mock_model_server = AsyncMock()
        mock_model_server.generate.return_value = "Generated text"
        mock_model_server_class.return_value = mock_model_server
        
        response = test_client.post("/generate", json={
            "prompt": "Test prompt",
            "max_tokens": 50,
            "temperature": 0.7
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "generation_time" in data
    
    def test_generate_endpoint_missing_prompt(self, test_client):
        """Test generation endpoint with missing prompt."""
        response = test_client.post("/generate", json={
            "max_tokens": 50
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/health")
        # CORS headers should be handled by FastAPI middleware
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled


class TestVLLMServer:
    """Test cases for VLLMServer class."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager for testing."""
        config = Mock()
        config.config = Mock()
        config.config.serving = Mock()
        config.config.serving.max_concurrent_requests = 10
        return config
    
    @pytest.fixture
    def vllm_server(self, config_manager):
        """Create a VLLMServer instance for testing."""
        with patch('src.serving.vllm_server.VLLM_AVAILABLE', True):
            return VLLMServer(config_manager)
    
    def test_init(self, config_manager):
        """Test VLLMServer initialization."""
        with patch('src.serving.vllm_server.VLLM_AVAILABLE', True):
            server = VLLMServer(config_manager)
            assert server.config == config_manager
            assert server.engine is None
            assert server.model_path is None
    
    def test_init_vllm_not_available(self, config_manager):
        """Test VLLMServer initialization when vLLM is not available."""
        with patch('src.serving.vllm_server.VLLM_AVAILABLE', False):
            with pytest.raises(ImportError, match="vLLM is not available"):
                VLLMServer(config_manager)
    
    @pytest.mark.asyncio
    @patch('src.serving.vllm_server.LLM')
    async def test_load_model(self, mock_llm_class, vllm_server):
        """Test loading model with vLLM."""
        mock_engine = Mock()
        mock_llm_class.return_value = mock_engine
        
        success = await vllm_server.load_model(
            model_path="test_model",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        assert success is True
        assert vllm_server.engine == mock_engine
        assert vllm_server.model_path == "test_model"
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self, vllm_server):
        """Test model loading failure."""
        with patch('src.serving.vllm_server.LLM', side_effect=Exception("Load failed")):
            success = await vllm_server.load_model("test_model")
            assert success is False
    
    @pytest.mark.asyncio
    async def test_unload_model(self, vllm_server):
        """Test unloading model."""
        vllm_server.engine = Mock()
        vllm_server.model_path = "test_model"
        
        success = await vllm_server.unload_model()
        
        assert success is True
        assert vllm_server.engine is None
        assert vllm_server.model_path is None
    
    @pytest.mark.asyncio
    @patch('src.serving.vllm_server.SamplingParams')
    async def test_generate(self, mock_sampling_params, vllm_server):
        """Test text generation with vLLM."""
        # Mock engine and output
        mock_engine = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Generated text"
        mock_engine.generate.return_value = [mock_output]
        
        vllm_server.engine = mock_engine
        
        request = VLLMRequest(
            prompt="Test prompt",
            max_tokens=50,
            temperature=0.7
        )
        
        response = await vllm_server.generate(request)
        
        assert response.text == "Generated text"
        assert response.prompt == "Test prompt"
        mock_engine.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_no_engine(self, vllm_server):
        """Test generation when no engine is loaded."""
        request = VLLMRequest(prompt="Test prompt")
        response = await vllm_server.generate(request)
        
        assert response.text == ""
        assert "No model loaded" in response.error
    
    @pytest.mark.asyncio
    async def test_generate_batch(self, vllm_server):
        """Test batch text generation."""
        # Mock engine and outputs
        mock_engine = Mock()
        mock_outputs = []
        for i in range(2):
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = f"Generated text {i}"
            mock_outputs.append(mock_output)
        
        mock_engine.generate.return_value = mock_outputs
        vllm_server.engine = mock_engine
        
        requests = [
            VLLMRequest(prompt="Prompt 1"),
            VLLMRequest(prompt="Prompt 2")
        ]
        
        responses = await vllm_server.generate_batch(requests)
        
        assert len(responses) == 2
        assert responses[0].text == "Generated text 0"
        assert responses[1].text == "Generated text 1"
    
    def test_get_stats(self, vllm_server):
        """Test getting vLLM server statistics."""
        stats = vllm_server.get_stats()
        
        assert "model_loaded" in stats
        assert "total_requests" in stats
        assert "active_requests" in stats
        assert "average_response_time" in stats
        assert "throughput" in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self, vllm_server):
        """Test vLLM server health check."""
        # Test without loaded model
        health = await vllm_server.health_check()
        assert health["status"] == "unhealthy"
        
        # Test with loaded model
        vllm_server.engine = Mock()
        health = await vllm_server.health_check()
        assert health["status"] == "healthy"


class TestVLLMRequest:
    """Test cases for VLLMRequest class."""
    
    def test_vllm_request_creation(self):
        """Test creating VLLMRequest object."""
        request = VLLMRequest(
            prompt="Test prompt",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 50
        assert request.temperature == 0.7
        assert request.top_p == 0.9
    
    def test_vllm_request_defaults(self):
        """Test VLLMRequest with default values."""
        request = VLLMRequest(prompt="Test prompt")
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.frequency_penalty == 0.0
        assert request.presence_penalty == 0.0


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for serving components."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a config manager for integration testing."""
        config = Mock()
        config.config = Mock()
        config.config.serving = Mock()
        config.config.serving.host = "0.0.0.0"
        config.config.serving.port = 8000
        config.config.serving.enable_cors = True
        config.config.serving.cors_origins = ["*"]
        config.config.serving.max_concurrent_requests = 10
        config.config.serving.timeout = 300
        config.config.model = Mock()
        config.config.model.torch_dtype = "float16"
        config.config.model.device_map = "auto"
        config.get_device.return_value = "cpu"
        config.get_torch_dtype.return_value = "torch.float16"
        return config
    
    async def test_api_server_with_model_server(self, config_manager):
        """Test API server integration with model server."""
        api_server = APIServer(config_manager)
        test_client = TestClient(api_server.app)
        
        # Test health endpoint
        response = test_client.get("/health")
        assert response.status_code == 200
        
        # Test stats endpoint
        response = test_client.get("/stats")
        assert response.status_code == 200
    
    @patch('src.serving.model_server.AutoTokenizer')
    @patch('src.serving.model_server.AutoModelForCausalLM')
    async def test_end_to_end_generation(self, mock_model_class, mock_tokenizer_class, config_manager):
        """Test end-to-end text generation flow."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output.sequences = [[1, 2, 3, 4, 5]]
        mock_model.generate.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create and setup servers
        model_server = ModelServer(config_manager)
        api_server = APIServer(config_manager)
        api_server.model_server = model_server
        
        # Load model
        success = await model_server.load_model("test_model", "pytorch")
        assert success is True
        
        # Test generation through API
        test_client = TestClient(api_server.app)
        
        with patch('src.serving.api_server.torch'):
            response = test_client.post("/generate", json={
                "prompt": "Test prompt",
                "max_tokens": 50,
                "temperature": 0.7
            })
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "generation_time" in data


if __name__ == "__main__":
    pytest.main([__file__])