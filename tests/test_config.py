#!/usr/bin/env python3
"""Tests for the configuration management module."""

import os
import sys
import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import ConfigManager, Config


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary for testing."""
        return {
            "model": {
                "model_name": "microsoft/DialoGPT-medium",
                "model_type": "causal_lm",
                "cache_dir": "./models",
                "torch_dtype": "float16",
                "device_map": "auto",
                "trust_remote_code": False,
                "use_auth_token": False
            },
            "training": {
                "training_type": "sft",
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "max_seq_length": 512,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 100,
                "output_dir": "./outputs",
                "logging_dir": "./logs",
                "seed": 42,
                "fp16": True,
                "dataloader_num_workers": 4,
                "remove_unused_columns": False,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            },
            "data": {
                "data_path": "./data",
                "max_seq_length": 512,
                "chunk_size": 1000,
                "overlap_size": 100,
                "train_split": 0.8,
                "eval_split": 0.1,
                "test_split": 0.1,
                "cleaning_enabled": True,
                "min_text_length": 10,
                "max_text_length": 2048
            },
            "serving": {
                "host": "0.0.0.0",
                "port": 8000,
                "backend": "pytorch",
                "max_concurrent_requests": 10,
                "timeout": 300,
                "enable_cors": True,
                "cors_origins": ["*"]
            },
            "monitoring": {
                "experiment_tracking": {
                    "enabled": True,
                    "backend": "wandb",
                    "project_name": "custom-llm-chatbot",
                    "run_name": None,
                    "tags": [],
                    "notes": ""
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file": "./logs/app.log",
                    "max_bytes": 10485760,
                    "backup_count": 5
                }
            }
        }
    
    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance for testing."""
        return ConfigManager()
    
    def test_init_default(self, config_manager):
        """Test ConfigManager initialization with defaults."""
        assert config_manager.config is not None
        assert isinstance(config_manager.config, Config)
        assert config_manager.config_path is None
    
    def test_init_with_config_path(self):
        """Test ConfigManager initialization with config path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"model": {"model_name": "test"}}, f)
            temp_path = f.name
        
        try:
            config_manager = ConfigManager(temp_path)
            assert config_manager.config_path == temp_path
        finally:
            os.unlink(temp_path)
    
    def test_load_config_from_file(self, config_manager, sample_config_dict):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_dict, f)
            temp_path = f.name
        
        try:
            config_manager.load_config(temp_path)
            assert config_manager.config.model.model_name == "microsoft/DialoGPT-medium"
            assert config_manager.config.training.num_epochs == 3
            assert config_manager.config.data.chunk_size == 1000
        finally:
            os.unlink(temp_path)
    
    def test_load_config_from_dict(self, config_manager, sample_config_dict):
        """Test loading configuration from dictionary."""
        config_manager.load_config(sample_config_dict)
        assert config_manager.config.model.model_name == "microsoft/DialoGPT-medium"
        assert config_manager.config.training.num_epochs == 3
        assert config_manager.config.data.chunk_size == 1000
    
    def test_load_config_invalid_file(self, config_manager):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            config_manager.load_config("non_existent_file.yaml")
    
    def test_load_config_invalid_yaml(self, config_manager):
        """Test loading configuration from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                config_manager.load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_config(self, config_manager, sample_config_dict):
        """Test saving configuration to file."""
        config_manager.load_config(sample_config_dict)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config_manager.save_config(temp_path)
            
            # Load the saved config and verify
            with open(temp_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config["model"]["model_name"] == "microsoft/DialoGPT-medium"
            assert saved_config["training"]["num_epochs"] == 3
        finally:
            os.unlink(temp_path)
    
    def test_update_config(self, config_manager, sample_config_dict):
        """Test updating configuration with new values."""
        config_manager.load_config(sample_config_dict)
        
        updates = {
            "model": {"model_name": "updated_model"},
            "training": {"num_epochs": 5, "batch_size": 8}
        }
        
        config_manager.update_config(updates)
        
        assert config_manager.config.model.model_name == "updated_model"
        assert config_manager.config.training.num_epochs == 5
        assert config_manager.config.training.batch_size == 8
        # Other values should remain unchanged
        assert config_manager.config.training.learning_rate == 5e-5
    
    def test_validate_config_valid(self, config_manager, sample_config_dict):
        """Test validation of valid configuration."""
        config_manager.load_config(sample_config_dict)
        assert config_manager.validate_config() is True
    
    def test_validate_config_missing_required(self, config_manager):
        """Test validation of configuration with missing required fields."""
        incomplete_config = {
            "model": {"model_name": "test"},
            # Missing training section
        }
        config_manager.load_config(incomplete_config)
        assert config_manager.validate_config() is False
    
    def test_validate_config_invalid_values(self, config_manager, sample_config_dict):
        """Test validation of configuration with invalid values."""
        sample_config_dict["training"]["num_epochs"] = -1  # Invalid negative value
        config_manager.load_config(sample_config_dict)
        assert config_manager.validate_config() is False
    
    def test_get_device_auto(self, config_manager, sample_config_dict):
        """Test device detection with auto setting."""
        config_manager.load_config(sample_config_dict)
        device = config_manager.get_device()
        assert device in ["cuda", "cpu", "mps"]
    
    def test_get_device_specific(self, config_manager, sample_config_dict):
        """Test device detection with specific setting."""
        sample_config_dict["model"]["device_map"] = "cpu"
        config_manager.load_config(sample_config_dict)
        device = config_manager.get_device()
        assert device == "cpu"
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_cuda_available(self, mock_cuda, config_manager, sample_config_dict):
        """Test device detection when CUDA is available."""
        sample_config_dict["model"]["device_map"] = "auto"
        config_manager.load_config(sample_config_dict)
        device = config_manager.get_device()
        assert device == "cuda"
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_get_device_mps_available(self, mock_mps, mock_cuda, config_manager, sample_config_dict):
        """Test device detection when MPS is available."""
        sample_config_dict["model"]["device_map"] = "auto"
        config_manager.load_config(sample_config_dict)
        device = config_manager.get_device()
        assert device == "mps"
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_get_device_cpu_fallback(self, mock_mps, mock_cuda, config_manager, sample_config_dict):
        """Test device detection fallback to CPU."""
        sample_config_dict["model"]["device_map"] = "auto"
        config_manager.load_config(sample_config_dict)
        device = config_manager.get_device()
        assert device == "cpu"
    
    def test_get_torch_dtype(self, config_manager, sample_config_dict):
        """Test torch dtype conversion."""
        config_manager.load_config(sample_config_dict)
        dtype = config_manager.get_torch_dtype()
        # Should return a torch dtype object
        assert hasattr(dtype, '__module__')
        assert 'torch' in str(dtype)
    
    def test_get_torch_dtype_invalid(self, config_manager, sample_config_dict):
        """Test torch dtype conversion with invalid type."""
        sample_config_dict["model"]["torch_dtype"] = "invalid_dtype"
        config_manager.load_config(sample_config_dict)
        dtype = config_manager.get_torch_dtype()
        # Should fallback to float32
        assert "float32" in str(dtype)
    
    def test_to_dict(self, config_manager, sample_config_dict):
        """Test conversion of config to dictionary."""
        config_manager.load_config(sample_config_dict)
        config_dict = config_manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        assert config_dict["model"]["model_name"] == "microsoft/DialoGPT-medium"
    
    def test_from_dict(self, config_manager, sample_config_dict):
        """Test creation of config from dictionary."""
        config_manager.from_dict(sample_config_dict)
        
        assert config_manager.config.model.model_name == "microsoft/DialoGPT-medium"
        assert config_manager.config.training.num_epochs == 3
    
    def test_get_nested_value(self, config_manager, sample_config_dict):
        """Test getting nested configuration values."""
        config_manager.load_config(sample_config_dict)
        
        # Test existing nested value
        value = config_manager.get("model.model_name")
        assert value == "microsoft/DialoGPT-medium"
        
        # Test non-existent nested value with default
        value = config_manager.get("model.non_existent", "default_value")
        assert value == "default_value"
    
    def test_set_nested_value(self, config_manager, sample_config_dict):
        """Test setting nested configuration values."""
        config_manager.load_config(sample_config_dict)
        
        # Set existing nested value
        config_manager.set("model.model_name", "new_model")
        assert config_manager.config.model.model_name == "new_model"
        
        # Set new nested value
        config_manager.set("model.new_field", "new_value")
        assert hasattr(config_manager.config.model, "new_field")
        assert config_manager.config.model.new_field == "new_value"
    
    def test_merge_configs(self, config_manager):
        """Test merging multiple configurations."""
        base_config = {
            "model": {"model_name": "base_model", "cache_dir": "./models"},
            "training": {"num_epochs": 3}
        }
        
        override_config = {
            "model": {"model_name": "override_model"},
            "training": {"batch_size": 8},
            "new_section": {"new_field": "new_value"}
        }
        
        config_manager.load_config(base_config)
        config_manager.merge_config(override_config)
        
        # Check overridden values
        assert config_manager.config.model.model_name == "override_model"
        # Check preserved values
        assert config_manager.config.model.cache_dir == "./models"
        assert config_manager.config.training.num_epochs == 3
        # Check new values
        assert config_manager.config.training.batch_size == 8
        assert hasattr(config_manager.config, "new_section")
    
    def test_environment_variable_substitution(self, config_manager):
        """Test environment variable substitution in config."""
        os.environ["TEST_MODEL_NAME"] = "env_model"
        os.environ["TEST_EPOCHS"] = "5"
        
        try:
            config_with_env = {
                "model": {"model_name": "${TEST_MODEL_NAME}"},
                "training": {"num_epochs": "${TEST_EPOCHS}"}
            }
            
            config_manager.load_config(config_with_env)
            
            assert config_manager.config.model.model_name == "env_model"
            assert config_manager.config.training.num_epochs == "5"  # Note: still string
        finally:
            del os.environ["TEST_MODEL_NAME"]
            del os.environ["TEST_EPOCHS"]
    
    def test_config_inheritance(self, config_manager):
        """Test configuration inheritance from base configs."""
        base_config = {
            "model": {"model_name": "base_model", "cache_dir": "./models"},
            "training": {"num_epochs": 3, "batch_size": 4}
        }
        
        child_config = {
            "_inherit": "base",
            "model": {"model_name": "child_model"},
            "training": {"num_epochs": 5}
        }
        
        # This would require implementing inheritance logic
        # For now, just test basic merging behavior
        config_manager.load_config(base_config)
        config_manager.merge_config(child_config)
        
        assert config_manager.config.model.model_name == "child_model"
        assert config_manager.config.model.cache_dir == "./models"
        assert config_manager.config.training.num_epochs == 5
        assert config_manager.config.training.batch_size == 4


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_creation_from_dict(self):
        """Test creating Config object from dictionary."""
        config_dict = {
            "model": {"model_name": "test_model"},
            "training": {"num_epochs": 3}
        }
        
        config = Config(config_dict)
        
        assert hasattr(config, "model")
        assert hasattr(config, "training")
        assert config.model.model_name == "test_model"
        assert config.training.num_epochs == 3
    
    def test_config_nested_access(self):
        """Test nested attribute access in Config."""
        config_dict = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }
        
        config = Config(config_dict)
        
        assert config.level1.level2.level3 == "deep_value"
    
    def test_config_attribute_error(self):
        """Test AttributeError for non-existent attributes."""
        config = Config({})
        
        with pytest.raises(AttributeError):
            _ = config.non_existent_attribute
    
    def test_config_to_dict(self):
        """Test converting Config back to dictionary."""
        config_dict = {
            "model": {"model_name": "test_model"},
            "training": {"num_epochs": 3}
        }
        
        config = Config(config_dict)
        result_dict = config.to_dict()
        
        assert result_dict == config_dict


if __name__ == "__main__":
    pytest.main([__file__])