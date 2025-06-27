#!/usr/bin/env python3
"""Tests for the training modules."""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import ConfigManager
from src.training.training_orchestrator import TrainingOrchestrator
from src.training.model_manager import ModelManager
from src.training.trainer import CustomTrainer
from src.data.data_processor import DataProcessor
from src.monitoring.experiment_tracker import ExperimentTracker


class TestTrainingOrchestrator:
    """Test cases for TrainingOrchestrator class."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager for testing."""
        config = Mock()
        config.config = Mock()
        
        # Training config
        config.config.training = Mock()
        config.config.training.method = "standard"
        config.config.training.num_epochs = 3
        config.config.training.batch_size = 4
        config.config.training.learning_rate = 5e-5
        config.config.training.warmup_steps = 100
        config.config.training.save_steps = 500
        config.config.training.eval_steps = 250
        config.config.training.logging_steps = 50
        config.config.training.gradient_accumulation_steps = 1
        config.config.training.max_grad_norm = 1.0
        config.config.training.weight_decay = 0.01
        config.config.training.dataloader_num_workers = 4
        config.config.training.remove_unused_columns = False
        config.config.training.load_best_model_at_end = True
        config.config.training.metric_for_best_model = "eval_loss"
        config.config.training.greater_is_better = False
        config.config.training.save_total_limit = 3
        config.config.training.resume_from_checkpoint = None
        
        # Model config
        config.config.model = Mock()
        config.config.model.name = "gpt2"
        config.config.model.torch_dtype = "float16"
        config.config.model.device_map = "auto"
        
        # Data config
        config.config.data = Mock()
        config.config.data.train_file = "train.json"
        config.config.data.validation_file = "val.json"
        config.config.data.test_file = "test.json"
        config.config.data.max_length = 512
        config.config.data.format = "instruction"
        
        # Output config
        config.config.output = Mock()
        config.config.output.output_dir = "./outputs"
        config.config.output.logging_dir = "./logs"
        
        # Experiment config
        config.config.experiment = Mock()
        config.config.experiment.name = "test_experiment"
        config.config.experiment.tracking_enabled = True
        config.config.experiment.platforms = ["local"]
        
        # Methods
        config.get_device.return_value = "cpu"
        config.get_torch_dtype.return_value = "torch.float16"
        config.to_dict.return_value = {"test": "config"}
        
        return config
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def orchestrator(self, config_manager, temp_dir):
        """Create a TrainingOrchestrator instance for testing."""
        # Update output directory to temp directory
        config_manager.config.output.output_dir = temp_dir
        config_manager.config.output.logging_dir = os.path.join(temp_dir, "logs")
        return TrainingOrchestrator(config_manager)
    
    def test_init(self, config_manager, temp_dir):
        """Test TrainingOrchestrator initialization."""
        config_manager.config.output.output_dir = temp_dir
        orchestrator = TrainingOrchestrator(config_manager)
        
        assert orchestrator.config == config_manager
        assert orchestrator.model_manager is not None
        assert orchestrator.data_processor is not None
        assert orchestrator.experiment_tracker is not None
        assert orchestrator.trainer is None
    
    @pytest.mark.asyncio
    @patch('src.training.training_orchestrator.os.makedirs')
    async def test_setup_directories(self, mock_makedirs, orchestrator):
        """Test directory setup."""
        await orchestrator._setup_directories()
        
        # Should create output and logging directories
        assert mock_makedirs.call_count >= 2
    
    @pytest.mark.asyncio
    @patch('src.training.model_manager.AutoTokenizer')
    @patch('src.training.model_manager.AutoModelForCausalLM')
    async def test_setup_model_and_tokenizer(self, mock_model_class, mock_tokenizer_class, orchestrator):
        """Test model and tokenizer setup."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, tokenizer = await orchestrator._setup_model_and_tokenizer()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert tokenizer.pad_token == tokenizer.eos_token
    
    @pytest.mark.asyncio
    @patch('src.data.data_processor.DataProcessor.process_data')
    async def test_setup_datasets(self, mock_process_data, orchestrator, temp_dir):
        """Test dataset setup."""
        # Create mock data files
        train_file = os.path.join(temp_dir, "train.json")
        val_file = os.path.join(temp_dir, "val.json")
        
        with open(train_file, 'w') as f:
            f.write('[{"instruction": "test", "output": "response"}]')
        with open(val_file, 'w') as f:
            f.write('[{"instruction": "test", "output": "response"}]')
        
        # Update config to use temp files
        orchestrator.config.config.data.train_file = train_file
        orchestrator.config.config.data.validation_file = val_file
        
        # Mock processed data
        mock_process_data.return_value = {
            'train': [{'text': 'processed train data'}],
            'validation': [{'text': 'processed val data'}]
        }
        
        datasets = await orchestrator._setup_datasets()
        
        assert 'train' in datasets
        assert 'validation' in datasets
        mock_process_data.assert_called()
    
    @pytest.mark.asyncio
    @patch('src.training.training_orchestrator.TrainingArguments')
    @patch('src.training.training_orchestrator.CustomTrainer')
    async def test_setup_trainer(self, mock_trainer_class, mock_training_args, orchestrator):
        """Test trainer setup."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_datasets = {'train': Mock(), 'validation': Mock()}
        
        mock_training_args_instance = Mock()
        mock_training_args.return_value = mock_training_args_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        trainer = await orchestrator._setup_trainer(mock_model, mock_tokenizer, mock_datasets)
        
        assert trainer == mock_trainer_instance
        mock_training_args.assert_called_once()
        mock_trainer_class.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.object(TrainingOrchestrator, '_setup_directories')
    @patch.object(TrainingOrchestrator, '_setup_model_and_tokenizer')
    @patch.object(TrainingOrchestrator, '_setup_datasets')
    @patch.object(TrainingOrchestrator, '_setup_trainer')
    async def test_train(self, mock_setup_trainer, mock_setup_datasets, 
                        mock_setup_model, mock_setup_dirs, orchestrator):
        """Test training process."""
        # Mock setup methods
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_datasets = {'train': Mock(), 'validation': Mock()}
        mock_trainer = Mock()
        
        mock_setup_model.return_value = (mock_model, mock_tokenizer)
        mock_setup_datasets.return_value = mock_datasets
        mock_setup_trainer.return_value = mock_trainer
        
        # Mock trainer methods
        mock_trainer.train.return_value = Mock()
        mock_trainer.save_model = Mock()
        mock_trainer.save_state = Mock()
        
        # Mock experiment tracker
        orchestrator.experiment_tracker.start_experiment = Mock()
        orchestrator.experiment_tracker.end_experiment = Mock()
        
        result = await orchestrator.train()
        
        # Verify setup methods were called
        mock_setup_dirs.assert_called_once()
        mock_setup_model.assert_called_once()
        mock_setup_datasets.assert_called_once()
        mock_setup_trainer.assert_called_once()
        
        # Verify training was executed
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()
        
        # Verify experiment tracking
        orchestrator.experiment_tracker.start_experiment.assert_called_once()
        orchestrator.experiment_tracker.end_experiment.assert_called_once()
        
        assert result is not None
    
    @pytest.mark.asyncio
    @patch.object(TrainingOrchestrator, '_setup_directories')
    @patch.object(TrainingOrchestrator, '_setup_model_and_tokenizer')
    @patch.object(TrainingOrchestrator, '_setup_datasets')
    async def test_evaluate(self, mock_setup_datasets, mock_setup_model, 
                           mock_setup_dirs, orchestrator, temp_dir):
        """Test evaluation process."""
        # Create test file
        test_file = os.path.join(temp_dir, "test.json")
        with open(test_file, 'w') as f:
            f.write('[{"instruction": "test", "output": "response"}]')
        
        orchestrator.config.config.data.test_file = test_file
        
        # Mock setup methods
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_datasets = {'test': Mock()}
        
        mock_setup_model.return_value = (mock_model, mock_tokenizer)
        mock_setup_datasets.return_value = mock_datasets
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.evaluate.return_value = {'eval_loss': 0.5, 'eval_accuracy': 0.8}
        
        with patch.object(orchestrator, '_setup_trainer', return_value=mock_trainer):
            result = await orchestrator.evaluate()
        
        # Verify evaluation was executed
        mock_trainer.evaluate.assert_called_once()
        assert result is not None
        assert 'eval_loss' in result


class TestModelManager:
    """Test cases for ModelManager class."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager for testing."""
        config = Mock()
        config.config = Mock()
        config.config.model = Mock()
        config.config.model.name = "gpt2"
        config.config.model.torch_dtype = "float16"
        config.config.model.device_map = "auto"
        config.config.model.trust_remote_code = False
        config.config.model.use_auth_token = None
        config.get_device.return_value = "cpu"
        config.get_torch_dtype.return_value = "torch.float16"
        return config
    
    @pytest.fixture
    def model_manager(self, config_manager):
        """Create a ModelManager instance for testing."""
        return ModelManager(config_manager)
    
    def test_init(self, config_manager):
        """Test ModelManager initialization."""
        manager = ModelManager(config_manager)
        assert manager.config == config_manager
        assert manager.model is None
        assert manager.tokenizer is None
    
    @pytest.mark.asyncio
    @patch('src.training.model_manager.AutoTokenizer')
    @patch('src.training.model_manager.AutoModelForCausalLM')
    async def test_load_model_and_tokenizer(self, mock_model_class, mock_tokenizer_class, model_manager):
        """Test loading model and tokenizer."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, tokenizer = await model_manager.load_model_and_tokenizer()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert model_manager.model == mock_model
        assert model_manager.tokenizer == mock_tokenizer
        
        # Check that pad_token was set
        assert tokenizer.pad_token == tokenizer.eos_token
    
    @pytest.mark.asyncio
    @patch('src.training.model_manager.PeftModel')
    @patch('src.training.model_manager.LoraConfig')
    async def test_setup_peft(self, mock_lora_config, mock_peft_model, model_manager):
        """Test PEFT setup."""
        # Mock base model
        mock_base_model = Mock()
        model_manager.model = mock_base_model
        
        # Mock PEFT components
        mock_config = Mock()
        mock_lora_config.return_value = mock_config
        
        mock_peft_model_instance = Mock()
        mock_peft_model.get_peft_model.return_value = mock_peft_model_instance
        
        # Setup PEFT
        peft_model = await model_manager.setup_peft(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        assert peft_model == mock_peft_model_instance
        mock_lora_config.assert_called_once()
        mock_peft_model.get_peft_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_peft_no_model(self, model_manager):
        """Test PEFT setup when no model is loaded."""
        with pytest.raises(ValueError, match="Model must be loaded"):
            await model_manager.setup_peft()
    
    @pytest.mark.asyncio
    @patch('src.training.model_manager.BitsAndBytesConfig')
    async def test_setup_quantization(self, mock_bnb_config, model_manager):
        """Test quantization setup."""
        mock_config = Mock()
        mock_bnb_config.return_value = mock_config
        
        config = await model_manager.setup_quantization(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
        
        assert config == mock_config
        mock_bnb_config.assert_called_once()
    
    def test_get_model_info(self, model_manager):
        """Test getting model information."""
        # Mock model with parameters
        mock_model = Mock()
        mock_model.num_parameters.return_value = 1000000
        mock_model.config = Mock()
        mock_model.config.model_type = "gpt2"
        mock_model.config.vocab_size = 50257
        
        model_manager.model = mock_model
        
        info = model_manager.get_model_info()
        
        assert "num_parameters" in info
        assert "model_type" in info
        assert "vocab_size" in info
        assert info["num_parameters"] == 1000000
    
    def test_get_model_info_no_model(self, model_manager):
        """Test getting model info when no model is loaded."""
        info = model_manager.get_model_info()
        assert info == {}


class TestCustomTrainer:
    """Test cases for CustomTrainer class."""
    
    @pytest.fixture
    def mock_training_args(self):
        """Create mock training arguments."""
        args = Mock()
        args.output_dir = "./outputs"
        args.logging_dir = "./logs"
        args.num_train_epochs = 3
        args.per_device_train_batch_size = 4
        args.learning_rate = 5e-5
        return args
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.config = Mock()
        model.config.model_type = "gpt2"
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer
    
    @pytest.fixture
    def mock_datasets(self):
        """Create mock datasets."""
        train_dataset = Mock()
        eval_dataset = Mock()
        return {
            'train': train_dataset,
            'validation': eval_dataset
        }
    
    @pytest.fixture
    def custom_trainer(self, mock_model, mock_training_args, mock_tokenizer, mock_datasets):
        """Create a CustomTrainer instance for testing."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = CustomTrainer(
                model=mock_model,
                args=mock_training_args,
                train_dataset=mock_datasets['train'],
                eval_dataset=mock_datasets['validation'],
                tokenizer=mock_tokenizer
            )
            return trainer
    
    def test_init(self, mock_model, mock_training_args, mock_tokenizer, mock_datasets):
        """Test CustomTrainer initialization."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = CustomTrainer(
                model=mock_model,
                args=mock_training_args,
                train_dataset=mock_datasets['train'],
                eval_dataset=mock_datasets['validation'],
                tokenizer=mock_tokenizer
            )
            
            assert trainer.model == mock_model
            assert trainer.args == mock_training_args
            assert trainer.tokenizer == mock_tokenizer
    
    def test_compute_loss(self, custom_trainer):
        """Test custom loss computation."""
        # Mock model output
        mock_outputs = Mock()
        mock_outputs.loss = 0.5
        
        # Mock model forward pass
        custom_trainer.model = Mock()
        custom_trainer.model.return_value = mock_outputs
        
        # Mock inputs
        mock_inputs = {
            'input_ids': Mock(),
            'attention_mask': Mock(),
            'labels': Mock()
        }
        
        loss = custom_trainer.compute_loss(custom_trainer.model, mock_inputs)
        
        assert loss == 0.5
        custom_trainer.model.assert_called_once()
    
    def test_log_metrics(self, custom_trainer):
        """Test metrics logging."""
        metrics = {
            'train_loss': 0.5,
            'eval_loss': 0.3,
            'learning_rate': 5e-5
        }
        
        # Mock experiment tracker
        custom_trainer.experiment_tracker = Mock()
        
        custom_trainer.log_metrics(metrics, step=100)
        
        # Should log metrics to experiment tracker if available
        if hasattr(custom_trainer, 'experiment_tracker'):
            custom_trainer.experiment_tracker.log_metrics.assert_called_once_with(metrics, step=100)
    
    @patch('src.training.trainer.Trainer.save_model')
    def test_save_model(self, mock_super_save, custom_trainer, temp_dir):
        """Test model saving."""
        custom_trainer.save_model(temp_dir)
        
        # Should call parent save_model
        mock_super_save.assert_called_once_with(temp_dir)
    
    @patch('src.training.trainer.Trainer.evaluate')
    def test_evaluate(self, mock_super_evaluate, custom_trainer):
        """Test model evaluation."""
        mock_super_evaluate.return_value = {
            'eval_loss': 0.3,
            'eval_accuracy': 0.8
        }
        
        result = custom_trainer.evaluate()
        
        assert result['eval_loss'] == 0.3
        assert result['eval_accuracy'] == 0.8
        mock_super_evaluate.assert_called_once()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for training components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_dir):
        """Create a config manager for integration testing."""
        config = Mock()
        config.config = Mock()
        
        # Training config
        config.config.training = Mock()
        config.config.training.method = "standard"
        config.config.training.num_epochs = 1  # Short for testing
        config.config.training.batch_size = 2
        config.config.training.learning_rate = 5e-5
        config.config.training.warmup_steps = 10
        config.config.training.save_steps = 50
        config.config.training.eval_steps = 25
        config.config.training.logging_steps = 10
        
        # Model config
        config.config.model = Mock()
        config.config.model.name = "gpt2"
        config.config.model.torch_dtype = "float16"
        
        # Data config
        config.config.data = Mock()
        config.config.data.max_length = 128
        config.config.data.format = "instruction"
        
        # Output config
        config.config.output = Mock()
        config.config.output.output_dir = temp_dir
        config.config.output.logging_dir = os.path.join(temp_dir, "logs")
        
        # Experiment config
        config.config.experiment = Mock()
        config.config.experiment.name = "test_experiment"
        config.config.experiment.tracking_enabled = False  # Disable for testing
        
        # Methods
        config.get_device.return_value = "cpu"
        config.get_torch_dtype.return_value = "torch.float16"
        config.to_dict.return_value = {"test": "config"}
        
        return config
    
    async def test_training_orchestrator_integration(self, config_manager, temp_dir):
        """Test integration between training components."""
        # Create test data files
        train_file = os.path.join(temp_dir, "train.json")
        val_file = os.path.join(temp_dir, "val.json")
        
        train_data = [
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "What is ML?", "output": "ML is machine learning."}
        ]
        val_data = [
            {"instruction": "What is DL?", "output": "DL is deep learning."}
        ]
        
        import json
        with open(train_file, 'w') as f:
            json.dump(train_data, f)
        with open(val_file, 'w') as f:
            json.dump(val_data, f)
        
        # Update config with data files
        config_manager.config.data.train_file = train_file
        config_manager.config.data.validation_file = val_file
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(config_manager)
        
        # Test that components are properly initialized
        assert orchestrator.model_manager is not None
        assert orchestrator.data_processor is not None
        assert orchestrator.experiment_tracker is not None
        
        # Test directory setup
        await orchestrator._setup_directories()
        assert os.path.exists(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "logs"))
    
    @patch('src.training.model_manager.AutoTokenizer')
    @patch('src.training.model_manager.AutoModelForCausalLM')
    async def test_model_manager_integration(self, mock_model_class, mock_tokenizer_class, config_manager):
        """Test ModelManager integration with mocked transformers."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.num_parameters.return_value = 1000000
        mock_model.config = Mock()
        mock_model.config.model_type = "gpt2"
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test model manager
        model_manager = ModelManager(config_manager)
        model, tokenizer = await model_manager.load_model_and_tokenizer()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        
        # Test model info
        info = model_manager.get_model_info()
        assert "num_parameters" in info
        assert info["num_parameters"] == 1000000


if __name__ == "__main__":
    pytest.main([__file__])