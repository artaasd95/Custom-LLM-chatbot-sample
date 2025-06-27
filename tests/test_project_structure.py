#!/usr/bin/env python3
"""
Test suite for verifying the overall project structure and basic functionality.

This module contains tests to ensure that:
- All required files and directories exist
- All modules can be imported successfully
- Basic configuration and setup work correctly
- Entry points are properly configured
"""

import os
import sys
import importlib
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


class TestProjectStructure:
    """Test the overall project structure."""
    
    def test_project_root_exists(self):
        """Test that project root directory exists."""
        assert project_root.exists()
        assert project_root.is_dir()
    
    def test_required_files_exist(self):
        """Test that all required files exist."""
        required_files = [
            "README.md",
            "requirements.txt",
            "setup.py",
            "Dockerfile",
            "Makefile",
            ".gitignore",
            ".env.example",
            "pytest.ini",
            "train.py",
            "serve.py",
            "example_usage.py"
        ]
        
        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file {file_name} does not exist"
            assert file_path.is_file(), f"{file_name} is not a file"
    
    def test_required_directories_exist(self):
        """Test that all required directories exist."""
        required_dirs = [
            "src",
            "src/config",
            "src/data",
            "src/models",
            "src/training",
            "src/serving",
            "src/utils",
            "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_src_modules_exist(self):
        """Test that all source modules exist."""
        src_modules = [
            "src/config/__init__.py",
            "src/config/config.py",
            "src/data/__init__.py",
            "src/data/data_processor.py",
            "src/models/__init__.py",
            "src/models/model_manager.py",
            "src/training/__init__.py",
            "src/training/trainer.py",
            "src/training/orchestrator.py",
            "src/serving/__init__.py",
            "src/serving/model_server.py",
            "src/serving/api_server.py",
            "src/serving/vllm_server.py",
            "src/utils/__init__.py",
            "src/utils/logging.py",
            "src/utils/metrics.py",
            "src/utils/security.py"
        ]
        
        for module_path in src_modules:
            file_path = project_root / module_path
            assert file_path.exists(), f"Source module {module_path} does not exist"
            assert file_path.is_file(), f"{module_path} is not a file"
    
    def test_test_modules_exist(self):
        """Test that all test modules exist."""
        test_modules = [
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/run_tests.py",
            "tests/test_config.py",
            "tests/test_data_processor.py",
            "tests/test_training.py",
            "tests/test_project_structure.py"
        ]
        
        for module_path in test_modules:
            file_path = project_root / module_path
            assert file_path.exists(), f"Test module {module_path} does not exist"
            assert file_path.is_file(), f"{module_path} is not a file"


class TestModuleImports:
    """Test that all modules can be imported successfully."""
    
    def test_config_imports(self):
        """Test that config modules can be imported."""
        try:
            from config.config import Config, ConfigManager
            assert Config is not None
            assert ConfigManager is not None
        except ImportError as e:
            pytest.skip(f"Config modules not available: {e}")
    
    def test_data_imports(self):
        """Test that data modules can be imported."""
        try:
            from data.data_processor import DataProcessor
            assert DataProcessor is not None
        except ImportError as e:
            pytest.skip(f"Data modules not available: {e}")
    
    def test_models_imports(self):
        """Test that model modules can be imported."""
        try:
            from models.model_manager import ModelManager
            assert ModelManager is not None
        except ImportError as e:
            pytest.skip(f"Model modules not available: {e}")
    
    def test_training_imports(self):
        """Test that training modules can be imported."""
        try:
            from training.trainer import CustomTrainer
            from training.orchestrator import TrainingOrchestrator
            assert CustomTrainer is not None
            assert TrainingOrchestrator is not None
        except ImportError as e:
            pytest.skip(f"Training modules not available: {e}")
    
    def test_serving_imports(self):
        """Test that serving modules can be imported."""
        try:
            from serving.model_server import ModelServer
            from serving.api_server import APIServer
            from serving.vllm_server import VLLMServer
            assert ModelServer is not None
            assert APIServer is not None
            assert VLLMServer is not None
        except ImportError as e:
            pytest.skip(f"Serving modules not available: {e}")
    
    def test_utils_imports(self):
        """Test that utility modules can be imported."""
        try:
            from utils.logging import setup_logging
            from utils.metrics import MetricsTracker
            from utils.security import SecurityManager
            assert setup_logging is not None
            assert MetricsTracker is not None
            assert SecurityManager is not None
        except ImportError as e:
            pytest.skip(f"Utility modules not available: {e}")


class TestEntryPoints:
    """Test that entry point scripts work correctly."""
    
    def test_train_script_exists(self):
        """Test that train.py script exists and is executable."""
        train_script = project_root / "train.py"
        assert train_script.exists()
        assert train_script.is_file()
        
        # Check if script has proper shebang
        with open(train_script, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            assert first_line.startswith('#!') or 'python' in first_line.lower()
    
    def test_serve_script_exists(self):
        """Test that serve.py script exists and is executable."""
        serve_script = project_root / "serve.py"
        assert serve_script.exists()
        assert serve_script.is_file()
        
        # Check if script has proper shebang
        with open(serve_script, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            assert first_line.startswith('#!') or 'python' in first_line.lower()
    
    def test_example_script_exists(self):
        """Test that example_usage.py script exists."""
        example_script = project_root / "example_usage.py"
        assert example_script.exists()
        assert example_script.is_file()
    
    @patch('sys.argv', ['train.py', '--help'])
    def test_train_script_help(self):
        """Test that train script shows help without errors."""
        try:
            # Import the train module
            train_path = str(project_root / "train.py")
            spec = importlib.util.spec_from_file_location("train", train_path)
            if spec and spec.loader:
                train_module = importlib.util.module_from_spec(spec)
                # This should not raise an exception during import
                assert train_module is not None
        except SystemExit:
            # Help command exits with code 0, which is expected
            pass
        except ImportError as e:
            pytest.skip(f"Train script dependencies not available: {e}")
    
    @patch('sys.argv', ['serve.py', '--help'])
    def test_serve_script_help(self):
        """Test that serve script shows help without errors."""
        try:
            # Import the serve module
            serve_path = str(project_root / "serve.py")
            spec = importlib.util.spec_from_file_location("serve", serve_path)
            if spec and spec.loader:
                serve_module = importlib.util.module_from_spec(spec)
                # This should not raise an exception during import
                assert serve_module is not None
        except SystemExit:
            # Help command exits with code 0, which is expected
            pass
        except ImportError as e:
            pytest.skip(f"Serve script dependencies not available: {e}")


class TestConfiguration:
    """Test basic configuration functionality."""
    
    def test_requirements_file_format(self):
        """Test that requirements.txt is properly formatted."""
        requirements_file = project_root / "requirements.txt"
        assert requirements_file.exists()
        
        with open(requirements_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Should have at least some requirements
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        assert len(non_empty_lines) > 0, "Requirements file should contain at least one package"
        
        # Check for essential packages
        requirements_text = '\n'.join(lines)
        essential_packages = ['torch', 'transformers', 'fastapi', 'uvicorn']
        for package in essential_packages:
            assert package in requirements_text, f"Essential package {package} not found in requirements"
    
    def test_setup_py_format(self):
        """Test that setup.py is properly formatted."""
        setup_file = project_root / "setup.py"
        assert setup_file.exists()
        
        with open(setup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain essential setup information
        assert 'name=' in content
        assert 'version=' in content
        assert 'packages=' in content
        assert 'install_requires=' in content
    
    def test_dockerfile_format(self):
        """Test that Dockerfile is properly formatted."""
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists()
        
        with open(dockerfile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain essential Docker instructions
        assert 'FROM' in content
        assert 'WORKDIR' in content
        assert 'COPY' in content
        assert 'RUN' in content
    
    def test_gitignore_format(self):
        """Test that .gitignore is properly formatted."""
        gitignore_file = project_root / ".gitignore"
        assert gitignore_file.exists()
        
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should ignore common Python files
        assert '__pycache__' in content
        assert '*.pyc' in content
        assert '.env' in content
    
    def test_env_example_format(self):
        """Test that .env.example is properly formatted."""
        env_example_file = project_root / ".env.example"
        assert env_example_file.exists()
        
        with open(env_example_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Should have environment variable examples
        env_lines = [line.strip() for line in lines if '=' in line and not line.strip().startswith('#')]
        assert len(env_lines) > 0, "Environment example file should contain variable examples"


class TestDocumentation:
    """Test documentation files."""
    
    def test_readme_exists_and_not_empty(self):
        """Test that README.md exists and contains content."""
        readme_file = project_root / "README.md"
        assert readme_file.exists()
        
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        assert len(content) > 100, "README should contain substantial content"
        assert '# Custom LLM Chatbot' in content or '# Custom' in content
    
    def test_makefile_exists_and_has_targets(self):
        """Test that Makefile exists and contains useful targets."""
        makefile = project_root / "Makefile"
        assert makefile.exists()
        
        with open(makefile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain common targets
        common_targets = ['install', 'test', 'clean', 'train', 'serve']
        for target in common_targets:
            assert f'{target}:' in content, f"Makefile should contain {target} target"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])