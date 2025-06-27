#!/usr/bin/env python3
"""Tests for the monitoring modules."""

import os
import sys
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.monitoring.experiment_tracker import ExperimentTracker
from src.monitoring.metrics_logger import MetricsLogger


class TestExperimentTracker:
    """Test cases for ExperimentTracker class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create a test configuration."""
        return {
            'experiment': {
                'name': 'test_experiment',
                'tracking_enabled': True,
                'platforms': ['local'],
                'tags': ['test', 'unit_test'],
                'notes': 'Test experiment for unit testing'
            },
            'output': {
                'output_dir': temp_dir,
                'logging_dir': os.path.join(temp_dir, 'logs')
            },
            'wandb': {
                'project': 'test_project',
                'entity': 'test_entity',
                'api_key': 'test_key'
            },
            'comet': {
                'project_name': 'test_project',
                'workspace': 'test_workspace',
                'api_key': 'test_key'
            },
            'mlflow': {
                'tracking_uri': 'http://localhost:5000',
                'experiment_name': 'test_experiment'
            }
        }
    
    @pytest.fixture
    def experiment_tracker(self, config):
        """Create an ExperimentTracker instance for testing."""
        return ExperimentTracker(config)
    
    def test_init(self, config):
        """Test ExperimentTracker initialization."""
        tracker = ExperimentTracker(config)
        
        assert tracker.config == config
        assert tracker.experiment_name == 'test_experiment'
        assert tracker.tracking_enabled is True
        assert 'local' in tracker.platforms
        assert tracker.active_experiments == {}
    
    def test_init_tracking_disabled(self, config):
        """Test initialization with tracking disabled."""
        config['experiment']['tracking_enabled'] = False
        tracker = ExperimentTracker(config)
        
        assert tracker.tracking_enabled is False
    
    @pytest.mark.asyncio
    async def test_start_experiment_local_only(self, experiment_tracker, temp_dir):
        """Test starting experiment with local tracking only."""
        experiment_id = await experiment_tracker.start_experiment()
        
        assert experiment_id is not None
        assert experiment_id in experiment_tracker.active_experiments
        
        # Check that local experiment file was created
        experiment_file = os.path.join(temp_dir, 'logs', f'experiment_{experiment_id}.json')
        assert os.path.exists(experiment_file)
        
        # Check experiment data
        with open(experiment_file, 'r') as f:
            data = json.load(f)
            assert data['experiment_name'] == 'test_experiment'
            assert data['experiment_id'] == experiment_id
            assert 'start_time' in data
    
    @pytest.mark.asyncio
    @patch('src.monitoring.experiment_tracker.wandb')
    async def test_start_experiment_wandb(self, mock_wandb, experiment_tracker):
        """Test starting experiment with Weights & Biases."""
        # Add wandb to platforms
        experiment_tracker.platforms.append('wandb')
        
        # Mock wandb
        mock_run = Mock()
        mock_run.id = 'wandb_run_id'
        mock_wandb.init.return_value = mock_run
        
        experiment_id = await experiment_tracker.start_experiment()
        
        assert experiment_id is not None
        mock_wandb.init.assert_called_once()
        assert 'wandb' in experiment_tracker.active_experiments[experiment_id]
    
    @pytest.mark.asyncio
    @patch('src.monitoring.experiment_tracker.comet_ml')
    async def test_start_experiment_comet(self, mock_comet, experiment_tracker):
        """Test starting experiment with Comet ML."""
        # Add comet to platforms
        experiment_tracker.platforms.append('comet')
        
        # Mock comet
        mock_experiment = Mock()
        mock_experiment.get_key.return_value = 'comet_experiment_key'
        mock_comet.Experiment.return_value = mock_experiment
        
        experiment_id = await experiment_tracker.start_experiment()
        
        assert experiment_id is not None
        mock_comet.Experiment.assert_called_once()
        assert 'comet' in experiment_tracker.active_experiments[experiment_id]
    
    @pytest.mark.asyncio
    @patch('src.monitoring.experiment_tracker.mlflow')
    async def test_start_experiment_mlflow(self, mock_mlflow, experiment_tracker):
        """Test starting experiment with MLflow."""
        # Add mlflow to platforms
        experiment_tracker.platforms.append('mlflow')
        
        # Mock mlflow
        mock_run = Mock()
        mock_run.info.run_id = 'mlflow_run_id'
        mock_mlflow.start_run.return_value = mock_run
        
        experiment_id = await experiment_tracker.start_experiment()
        
        assert experiment_id is not None
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        assert 'mlflow' in experiment_tracker.active_experiments[experiment_id]
    
    @pytest.mark.asyncio
    async def test_start_experiment_tracking_disabled(self, experiment_tracker):
        """Test starting experiment when tracking is disabled."""
        experiment_tracker.tracking_enabled = False
        
        experiment_id = await experiment_tracker.start_experiment()
        
        assert experiment_id is None
    
    @pytest.mark.asyncio
    async def test_log_metrics(self, experiment_tracker, temp_dir):
        """Test logging metrics."""
        # Start experiment first
        experiment_id = await experiment_tracker.start_experiment()
        
        metrics = {
            'train_loss': 0.5,
            'eval_loss': 0.3,
            'learning_rate': 5e-5,
            'epoch': 1
        }
        
        await experiment_tracker.log_metrics(metrics, step=100, experiment_id=experiment_id)
        
        # Check that metrics were logged locally
        metrics_file = os.path.join(temp_dir, 'logs', f'metrics_{experiment_id}.jsonl')
        assert os.path.exists(metrics_file)
        
        # Check metrics data
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data['step'] == 100
            assert data['metrics']['train_loss'] == 0.5
    
    @pytest.mark.asyncio
    async def test_log_parameters(self, experiment_tracker, temp_dir):
        """Test logging parameters."""
        # Start experiment first
        experiment_id = await experiment_tracker.start_experiment()
        
        parameters = {
            'learning_rate': 5e-5,
            'batch_size': 32,
            'model_name': 'gpt2',
            'num_epochs': 3
        }
        
        await experiment_tracker.log_parameters(parameters, experiment_id=experiment_id)
        
        # Check that parameters were logged locally
        experiment_file = os.path.join(temp_dir, 'logs', f'experiment_{experiment_id}.json')
        with open(experiment_file, 'r') as f:
            data = json.load(f)
            assert 'parameters' in data
            assert data['parameters']['learning_rate'] == 5e-5
            assert data['parameters']['batch_size'] == 32
    
    @pytest.mark.asyncio
    async def test_log_artifact(self, experiment_tracker, temp_dir):
        """Test logging artifacts."""
        # Start experiment first
        experiment_id = await experiment_tracker.start_experiment()
        
        # Create a test artifact file
        artifact_path = os.path.join(temp_dir, 'test_artifact.txt')
        with open(artifact_path, 'w') as f:
            f.write('This is a test artifact')
        
        await experiment_tracker.log_artifact(artifact_path, experiment_id=experiment_id)
        
        # Check that artifact was logged locally
        experiment_file = os.path.join(temp_dir, 'logs', f'experiment_{experiment_id}.json')
        with open(experiment_file, 'r') as f:
            data = json.load(f)
            assert 'artifacts' in data
            assert any('test_artifact.txt' in artifact for artifact in data['artifacts'])
    
    @pytest.mark.asyncio
    async def test_log_model(self, experiment_tracker, temp_dir):
        """Test logging model."""
        # Start experiment first
        experiment_id = await experiment_tracker.start_experiment()
        
        # Create a test model directory
        model_path = os.path.join(temp_dir, 'test_model')
        os.makedirs(model_path)
        
        # Create some model files
        with open(os.path.join(model_path, 'config.json'), 'w') as f:
            json.dump({'model_type': 'gpt2'}, f)
        
        await experiment_tracker.log_model(model_path, experiment_id=experiment_id)
        
        # Check that model was logged locally
        experiment_file = os.path.join(temp_dir, 'logs', f'experiment_{experiment_id}.json')
        with open(experiment_file, 'r') as f:
            data = json.load(f)
            assert 'models' in data
            assert any('test_model' in model for model in data['models'])
    
    @pytest.mark.asyncio
    async def test_log_text(self, experiment_tracker, temp_dir):
        """Test logging text."""
        # Start experiment first
        experiment_id = await experiment_tracker.start_experiment()
        
        text_data = {
            'prompt': 'What is AI?',
            'response': 'AI is artificial intelligence.',
            'score': 0.95
        }
        
        await experiment_tracker.log_text('generation_example', text_data, experiment_id=experiment_id)
        
        # Check that text was logged locally
        text_file = os.path.join(temp_dir, 'logs', f'text_{experiment_id}.jsonl')
        assert os.path.exists(text_file)
        
        with open(text_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data['key'] == 'generation_example'
            assert data['data']['prompt'] == 'What is AI?'
    
    @pytest.mark.asyncio
    async def test_end_experiment(self, experiment_tracker, temp_dir):
        """Test ending experiment."""
        # Start experiment first
        experiment_id = await experiment_tracker.start_experiment()
        
        # Log some final metrics
        final_metrics = {'final_loss': 0.2, 'final_accuracy': 0.9}
        
        await experiment_tracker.end_experiment(final_metrics, experiment_id=experiment_id)
        
        # Check that experiment was ended
        assert experiment_id not in experiment_tracker.active_experiments
        
        # Check that final metrics were logged
        experiment_file = os.path.join(temp_dir, 'logs', f'experiment_{experiment_id}.json')
        with open(experiment_file, 'r') as f:
            data = json.load(f)
            assert 'end_time' in data
            assert 'final_metrics' in data
            assert data['final_metrics']['final_loss'] == 0.2
    
    def test_get_experiment_url_local(self, experiment_tracker, temp_dir):
        """Test getting experiment URL for local tracking."""
        experiment_id = 'test_experiment_id'
        experiment_tracker.active_experiments[experiment_id] = {
            'local': {'experiment_file': os.path.join(temp_dir, 'logs', f'experiment_{experiment_id}.json')}
        }
        
        url = experiment_tracker.get_experiment_url(experiment_id)
        
        assert url is not None
        assert 'file://' in url
    
    @patch('src.monitoring.experiment_tracker.wandb')
    def test_get_experiment_url_wandb(self, mock_wandb, experiment_tracker):
        """Test getting experiment URL for Weights & Biases."""
        experiment_id = 'test_experiment_id'
        mock_run = Mock()
        mock_run.url = 'https://wandb.ai/test/test_project/runs/test_run'
        
        experiment_tracker.active_experiments[experiment_id] = {
            'wandb': {'run': mock_run}
        }
        
        url = experiment_tracker.get_experiment_url(experiment_id)
        
        assert url == 'https://wandb.ai/test/test_project/runs/test_run'
    
    def test_get_experiment_url_invalid(self, experiment_tracker):
        """Test getting experiment URL for invalid experiment."""
        url = experiment_tracker.get_experiment_url('invalid_experiment_id')
        assert url is None


class TestMetricsLogger:
    """Test cases for MetricsLogger class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create a test configuration."""
        return {
            'output': {
                'logging_dir': temp_dir
            },
            'monitoring': {
                'log_level': 'INFO',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'max_log_files': 5,
                'max_log_size_mb': 10
            }
        }
    
    @pytest.fixture
    def metrics_logger(self, config):
        """Create a MetricsLogger instance for testing."""
        return MetricsLogger(config)
    
    def test_init(self, config, temp_dir):
        """Test MetricsLogger initialization."""
        logger = MetricsLogger(config)
        
        assert logger.config == config
        assert logger.log_dir == temp_dir
        assert logger.logger is not None
    
    def test_log_metrics(self, metrics_logger, temp_dir):
        """Test logging metrics to file."""
        metrics = {
            'train_loss': 0.5,
            'eval_loss': 0.3,
            'learning_rate': 5e-5
        }
        
        metrics_logger.log_metrics(metrics, step=100, epoch=1)
        
        # Check that metrics file was created
        metrics_files = [f for f in os.listdir(temp_dir) if f.startswith('metrics_')]
        assert len(metrics_files) > 0
        
        # Check metrics content
        metrics_file = os.path.join(temp_dir, metrics_files[0])
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Parse the last line (most recent metric)
            last_line = lines[-1]
            data = json.loads(last_line)
            assert data['step'] == 100
            assert data['epoch'] == 1
            assert data['metrics']['train_loss'] == 0.5
    
    def test_log_training_progress(self, metrics_logger):
        """Test logging training progress."""
        progress_data = {
            'epoch': 2,
            'step': 500,
            'total_steps': 1000,
            'progress_percent': 50.0,
            'eta_seconds': 300
        }
        
        metrics_logger.log_training_progress(progress_data)
        
        # Should not raise any exceptions
        assert True
    
    def test_log_model_info(self, metrics_logger):
        """Test logging model information."""
        model_info = {
            'model_name': 'gpt2',
            'num_parameters': 124000000,
            'model_size_mb': 500,
            'architecture': 'transformer'
        }
        
        metrics_logger.log_model_info(model_info)
        
        # Should not raise any exceptions
        assert True
    
    def test_log_system_info(self, metrics_logger):
        """Test logging system information."""
        system_info = {
            'gpu_count': 1,
            'gpu_memory_gb': 8,
            'cpu_count': 8,
            'ram_gb': 32,
            'python_version': '3.8.10'
        }
        
        metrics_logger.log_system_info(system_info)
        
        # Should not raise any exceptions
        assert True
    
    def test_log_error(self, metrics_logger, temp_dir):
        """Test logging errors."""
        error_info = {
            'error_type': 'ValueError',
            'error_message': 'Invalid input data',
            'traceback': 'Traceback (most recent call last)...'
        }
        
        metrics_logger.log_error(error_info)
        
        # Check that error was logged
        log_files = [f for f in os.listdir(temp_dir) if f.endswith('.log')]
        assert len(log_files) > 0
    
    def test_get_metrics_summary(self, metrics_logger, temp_dir):
        """Test getting metrics summary."""
        # Log some metrics first
        for i in range(5):
            metrics = {
                'train_loss': 0.5 - i * 0.1,
                'eval_loss': 0.4 - i * 0.08
            }
            metrics_logger.log_metrics(metrics, step=i * 100, epoch=i + 1)
        
        summary = metrics_logger.get_metrics_summary()
        
        assert 'total_logged_metrics' in summary
        assert 'latest_metrics' in summary
        assert 'metrics_files' in summary
        assert summary['total_logged_metrics'] == 5
    
    def test_export_metrics(self, metrics_logger, temp_dir):
        """Test exporting metrics to different formats."""
        # Log some metrics first
        for i in range(3):
            metrics = {
                'train_loss': 0.5 - i * 0.1,
                'eval_loss': 0.4 - i * 0.08
            }
            metrics_logger.log_metrics(metrics, step=i * 100, epoch=i + 1)
        
        # Export to CSV
        csv_file = os.path.join(temp_dir, 'exported_metrics.csv')
        success = metrics_logger.export_metrics(csv_file, format='csv')
        
        assert success is True
        assert os.path.exists(csv_file)
        
        # Check CSV content
        with open(csv_file, 'r') as f:
            content = f.read()
            assert 'train_loss' in content
            assert 'eval_loss' in content
    
    def test_cleanup_old_logs(self, metrics_logger, temp_dir):
        """Test cleaning up old log files."""
        # Create some old log files
        for i in range(10):
            log_file = os.path.join(temp_dir, f'old_metrics_{i}.jsonl')
            with open(log_file, 'w') as f:
                f.write('{"test": "data"}\n')
        
        # Set max files to 5
        metrics_logger.config['monitoring']['max_log_files'] = 5
        
        metrics_logger.cleanup_old_logs()
        
        # Should have at most 5 log files remaining
        log_files = [f for f in os.listdir(temp_dir) if f.endswith('.jsonl')]
        assert len(log_files) <= 5


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for monitoring components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create a config for integration testing."""
        return {
            'experiment': {
                'name': 'integration_test',
                'tracking_enabled': True,
                'platforms': ['local']
            },
            'output': {
                'output_dir': temp_dir,
                'logging_dir': os.path.join(temp_dir, 'logs')
            },
            'monitoring': {
                'log_level': 'INFO',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    async def test_experiment_tracker_and_metrics_logger_integration(self, config, temp_dir):
        """Test integration between ExperimentTracker and MetricsLogger."""
        # Create both components
        experiment_tracker = ExperimentTracker(config)
        metrics_logger = MetricsLogger(config)
        
        # Start experiment
        experiment_id = await experiment_tracker.start_experiment()
        assert experiment_id is not None
        
        # Log metrics through both systems
        metrics = {
            'train_loss': 0.5,
            'eval_loss': 0.3,
            'learning_rate': 5e-5
        }
        
        # Log through experiment tracker
        await experiment_tracker.log_metrics(metrics, step=100, experiment_id=experiment_id)
        
        # Log through metrics logger
        metrics_logger.log_metrics(metrics, step=100, epoch=1)
        
        # End experiment
        final_metrics = {'final_loss': 0.2}
        await experiment_tracker.end_experiment(final_metrics, experiment_id=experiment_id)
        
        # Verify files were created
        logs_dir = os.path.join(temp_dir, 'logs')
        assert os.path.exists(logs_dir)
        
        # Check experiment files
        experiment_files = [f for f in os.listdir(logs_dir) if f.startswith('experiment_')]
        assert len(experiment_files) > 0
        
        # Check metrics files
        metrics_files = [f for f in os.listdir(logs_dir) if f.startswith('metrics_')]
        assert len(metrics_files) > 0
    
    async def test_full_training_monitoring_simulation(self, config, temp_dir):
        """Test a full training monitoring simulation."""
        experiment_tracker = ExperimentTracker(config)
        metrics_logger = MetricsLogger(config)
        
        # Start experiment
        experiment_id = await experiment_tracker.start_experiment()
        
        # Log parameters
        parameters = {
            'learning_rate': 5e-5,
            'batch_size': 32,
            'model_name': 'gpt2',
            'num_epochs': 3
        }
        await experiment_tracker.log_parameters(parameters, experiment_id=experiment_id)
        
        # Simulate training epochs
        for epoch in range(3):
            for step in range(10):
                # Simulate decreasing loss
                train_loss = 1.0 - (epoch * 10 + step) * 0.01
                eval_loss = 0.8 - (epoch * 10 + step) * 0.008
                
                metrics = {
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'learning_rate': 5e-5 * (0.9 ** epoch)
                }
                
                # Log to both systems
                await experiment_tracker.log_metrics(metrics, step=epoch * 10 + step, experiment_id=experiment_id)
                metrics_logger.log_metrics(metrics, step=epoch * 10 + step, epoch=epoch + 1)
            
            # Log epoch progress
            progress_data = {
                'epoch': epoch + 1,
                'step': (epoch + 1) * 10,
                'total_steps': 30,
                'progress_percent': ((epoch + 1) / 3) * 100
            }
            metrics_logger.log_training_progress(progress_data)
        
        # Log final results
        final_metrics = {
            'final_train_loss': 0.7,
            'final_eval_loss': 0.56,
            'best_eval_loss': 0.56
        }
        
        await experiment_tracker.end_experiment(final_metrics, experiment_id=experiment_id)
        
        # Verify comprehensive logging
        logs_dir = os.path.join(temp_dir, 'logs')
        
        # Check all expected files exist
        experiment_file = os.path.join(logs_dir, f'experiment_{experiment_id}.json')
        assert os.path.exists(experiment_file)
        
        metrics_file = os.path.join(logs_dir, f'metrics_{experiment_id}.jsonl')
        assert os.path.exists(metrics_file)
        
        # Verify experiment data
        with open(experiment_file, 'r') as f:
            exp_data = json.load(f)
            assert exp_data['experiment_name'] == 'integration_test'
            assert 'parameters' in exp_data
            assert 'final_metrics' in exp_data
            assert 'start_time' in exp_data
            assert 'end_time' in exp_data
        
        # Verify metrics data
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 30  # 3 epochs * 10 steps
            
            # Check first and last metrics
            first_metric = json.loads(lines[0])
            last_metric = json.loads(lines[-1])
            
            assert first_metric['step'] == 0
            assert last_metric['step'] == 29
            assert first_metric['metrics']['train_loss'] > last_metric['metrics']['train_loss']


if __name__ == "__main__":
    pytest.main([__file__])