#!/usr/bin/env python3
"""Pytest configuration and shared fixtures."""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the session."""
    temp_dir = tempfile.mkdtemp(prefix="llm_chatbot_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for individual tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "model": {
            "name": "gpt2",
            "torch_dtype": "float16",
            "device_map": "auto",
            "trust_remote_code": False,
            "use_auth_token": None
        },
        "training": {
            "method": "standard",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "warmup_steps": 100,
            "save_steps": 500,
            "eval_steps": 250,
            "logging_steps": 50,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 3,
            "resume_from_checkpoint": None
        },
        "data": {
            "train_file": "train.json",
            "validation_file": "val.json",
            "test_file": "test.json",
            "max_length": 512,
            "format": "instruction",
            "text_column": "text",
            "instruction_column": "instruction",
            "output_column": "output",
            "input_column": "input",
            "chosen_column": "chosen",
            "rejected_column": "rejected",
            "conversation_column": "conversation",
            "preprocessing": {
                "clean_text": True,
                "remove_duplicates": True,
                "min_length": 10,
                "max_length": 2048,
                "chunk_size": 512,
                "chunk_overlap": 50
            }
        },
        "serving": {
            "host": "0.0.0.0",
            "port": 8000,
            "max_concurrent_requests": 10,
            "timeout": 300,
            "enable_cors": True,
            "cors_origins": ["*"],
            "backend": "pytorch"
        },
        "output": {
            "output_dir": "./outputs",
            "logging_dir": "./logs",
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3
        },
        "experiment": {
            "name": "test_experiment",
            "tracking_enabled": True,
            "platforms": ["local"],
            "tags": ["test"],
            "notes": "Test experiment"
        },
        "monitoring": {
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "max_log_files": 5,
            "max_log_size_mb": 10
        },
        "wandb": {
            "project": "test_project",
            "entity": "test_entity",
            "api_key": "test_key"
        },
        "comet": {
            "project_name": "test_project",
            "workspace": "test_workspace",
            "api_key": "test_key"
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test_experiment"
        }
    }


@pytest.fixture
def sample_instruction_data():
    """Create sample instruction-following data for testing."""
    return [
        {
            "instruction": "What is artificial intelligence?",
            "output": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior."
        },
        {
            "instruction": "Explain machine learning in simple terms.",
            "output": "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed."
        },
        {
            "instruction": "What are the benefits of deep learning?",
            "output": "Deep learning can automatically discover patterns in data, handle complex tasks like image recognition, and achieve state-of-the-art results in many domains."
        }
    ]


@pytest.fixture
def sample_conversation_data():
    """Create sample conversation data for testing."""
    return [
        {
            "conversation": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data, but I'd be happy to help you find weather information."}
            ]
        }
    ]


@pytest.fixture
def sample_dpo_data():
    """Create sample DPO (Direct Preference Optimization) data for testing."""
    return [
        {
            "prompt": "Explain quantum computing",
            "chosen": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
            "rejected": "Quantum computing is just really fast regular computing."
        },
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "The capital of France is London."
        }
    ]


@pytest.fixture
def sample_text_data():
    """Create sample text data for testing."""
    return [
        {"text": "This is a sample text for language model training."},
        {"text": "Another example of text data that can be used for training."},
        {"text": "Language models learn patterns from large amounts of text data."}
    ]


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.model_type = "gpt2"
    model.config.vocab_size = 50257
    model.num_parameters.return_value = 124000000
    
    # Mock generation
    mock_output = Mock()
    mock_output.sequences = [[1, 2, 3, 4, 5]]
    model.generate.return_value = mock_output
    
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 50257
    
    # Mock encoding/decoding
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "Generated text"
    
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(return_value={"input_ids": [1, 2, 3], "labels": [1, 2, 3]})
    return dataset


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and patches."""
    # Set test environment variables
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    
    # Mock external dependencies that might not be available in test environment
    with patch.dict('sys.modules', {
        'wandb': Mock(),
        'comet_ml': Mock(),
        'mlflow': Mock(),
        'vllm': Mock(),
        'onnxruntime': Mock(),
        'flash_attn': Mock(),
        'xformers': Mock(),
        'deepspeed': Mock()
    }):
        yield


@pytest.fixture
def disable_gpu():
    """Disable GPU for testing to ensure tests run on CPU."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.cuda.device_count', return_value=0):
            yield


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing without actual file I/O."""
    with patch('builtins.open', create=True) as mock_open:
        with patch('os.path.exists', return_value=True):
            with patch('os.makedirs'):
                yield mock_open


@pytest.fixture
def capture_logs(caplog):
    """Capture logs for testing."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "external: marks tests that require external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "TestIntegration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Mark external service tests
        if any(service in item.nodeid for service in ["wandb", "comet", "mlflow", "vllm"]):
            item.add_marker(pytest.mark.external)


# Custom assertions
def assert_file_exists(file_path):
    """Assert that a file exists."""
    assert os.path.exists(file_path), f"File {file_path} does not exist"


def assert_file_contains(file_path, content):
    """Assert that a file contains specific content."""
    assert_file_exists(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read()
        assert content in file_content, f"File {file_path} does not contain '{content}'"


def assert_json_file_valid(file_path):
    """Assert that a JSON file is valid."""
    import json
    assert_file_exists(file_path)
    try:
        with open(file_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"File {file_path} is not valid JSON: {e}")


def assert_config_valid(config):
    """Assert that a configuration dictionary is valid."""
    required_sections = ['model', 'training', 'data', 'output']
    for section in required_sections:
        assert section in config, f"Configuration missing required section: {section}"


# Test utilities
class TestUtils:
    """Utility class for common test operations."""
    
    @staticmethod
    def create_test_file(file_path, content):
        """Create a test file with given content."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            if isinstance(content, (dict, list)):
                import json
                json.dump(content, f)
            else:
                f.write(content)
    
    @staticmethod
    def create_test_json_file(file_path, data):
        """Create a test JSON file."""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def create_test_jsonl_file(file_path, data_list):
        """Create a test JSONL file."""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')
    
    @staticmethod
    def create_test_csv_file(file_path, data, headers=None):
        """Create a test CSV file."""
        import csv
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', newline='') as f:
            if headers:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(f)
                writer.writerows(data)
    
    @staticmethod
    def create_test_txt_file(file_path, lines):
        """Create a test text file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils