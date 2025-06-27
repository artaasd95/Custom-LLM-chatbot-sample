#!/usr/bin/env python3
"""Example usage of the Custom LLM Chatbot system."""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import ConfigManager
from src.core.model_manager import ModelManager
from src.data.data_processor import DataProcessor
from src.training.trainer import TrainingOrchestrator
from src.serving.model_server import ModelServer
from src.serving.vllm_server import VLLMServer, VLLMRequest, VLLM_AVAILABLE
from src.monitoring.experiment_tracker import ExperimentTracker


def example_config_usage():
    """Example of configuration management."""
    print("=== Configuration Management Example ===")
    
    # Initialize config manager
    config = ConfigManager()
    
    # Load configuration
    config.load_config("config.yaml")
    
    # Print some configuration values
    print(f"Model name: {config.config.model.model_name}")
    print(f"Training type: {config.config.training.training_type}")
    print(f"Batch size: {config.config.training.batch_size}")
    print(f"Learning rate: {config.config.training.learning_rate}")
    
    # Validate configuration
    is_valid = config.validate_config()
    print(f"Configuration valid: {is_valid}")
    
    # Get device and dtype
    device = config.get_device()
    dtype = config.get_torch_dtype()
    print(f"Device: {device}")
    print(f"Torch dtype: {dtype}")
    
    print()


def example_data_processing():
    """Example of data processing."""
    print("=== Data Processing Example ===")
    
    # Initialize components
    config = ConfigManager()
    config.load_config("config.yaml")
    
    processor = DataProcessor(config)
    
    # Example data in different formats
    sample_data = [
        {"text": "This is a simple text example for training."},
        {
            "instruction": "Translate to French:",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?"
        },
        {
            "conversations": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        }
    ]
    
    # Process different data formats
    for i, data in enumerate(sample_data):
        print(f"Processing sample {i + 1}:")
        
        # Detect format
        data_format = processor.detect_data_format(data)
        print(f"  Detected format: {data_format}")
        
        # Process data
        processed = processor.process_single_item(data, data_format)
        print(f"  Processed: {processed[:100]}...")
        print()
    
    print()


async def example_model_management():
    """Example of model management."""
    print("=== Model Management Example ===")
    
    # Initialize components
    config = ConfigManager()
    config.load_config("config.yaml")
    
    model_manager = ModelManager(config)
    
    try:
        # Load a small model for demonstration
        model_name = "microsoft/DialoGPT-small"  # Smaller model for example
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        tokenizer = model_manager.load_tokenizer(model_name)
        model = model_manager.load_model(model_name)
        
        if tokenizer and model:
            print("Model loaded successfully!")
            
            # Get model info
            info = model_manager.get_model_info()
            print(f"Model info: {info}")
            
            # Example text generation
            prompt = "Hello, how are you?"
            print(f"Generating response for: '{prompt}'")
            
            response = model_manager.generate_text(
                prompt=prompt,
                max_length=50,
                temperature=0.7,
                do_sample=True
            )
            print(f"Generated: {response}")
        else:
            print("Failed to load model")
            
    except Exception as e:
        print(f"Model management example failed: {str(e)}")
        print("This is expected if the model is not available or GPU memory is insufficient")
    
    print()


async def example_serving():
    """Example of model serving."""
    print("=== Model Serving Example ===")
    
    # Initialize components
    config = ConfigManager()
    config.load_config("config.yaml")
    
    # Try different serving backends
    backends = ["pytorch"]
    if VLLM_AVAILABLE:
        backends.append("vllm")
    
    for backend in backends:
        print(f"Testing {backend} backend:")
        
        try:
            if backend == "vllm":
                server = VLLMServer(config)
                # Use a smaller model for example
                model_path = "microsoft/DialoGPT-small"
                success = await server.load_model(
                    model_path=model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.5,
                    max_model_len=512
                )
                
                if success:
                    print("  vLLM server loaded successfully!")
                    
                    # Test generation
                    request = VLLMRequest(
                        prompt="Hello, how are you?",
                        max_tokens=20,
                        temperature=0.7
                    )
                    
                    response = await server.generate(request)
                    print(f"  Generated: {response.text}")
                    
                    # Get stats
                    stats = server.get_stats()
                    print(f"  Stats: {stats}")
                    
                    # Cleanup
                    await server.unload_model()
                else:
                    print("  Failed to load vLLM server")
            
            else:  # pytorch backend
                server = ModelServer(config)
                model_path = "microsoft/DialoGPT-small"
                success = await server.load_model(model_path, "pytorch")
                
                if success:
                    print("  PyTorch server loaded successfully!")
                    
                    # Test generation
                    response = await server.generate(
                        prompt="Hello, how are you?",
                        max_tokens=20,
                        temperature=0.7
                    )
                    print(f"  Generated: {response}")
                    
                    # Get stats
                    stats = server.get_stats()
                    print(f"  Stats: {stats}")
                    
                    # Cleanup
                    await server.unload_model()
                else:
                    print("  Failed to load PyTorch server")
                    
        except Exception as e:
            print(f"  {backend} backend example failed: {str(e)}")
            print("  This is expected if models are not available or resources are insufficient")
        
        print()


def example_experiment_tracking():
    """Example of experiment tracking."""
    print("=== Experiment Tracking Example ===")
    
    # Initialize components
    config = ConfigManager()
    config.load_config("config.yaml")
    
    tracker = ExperimentTracker(config)
    
    try:
        # Start experiment
        experiment_name = "example_experiment"
        run_name = "example_run"
        
        print(f"Starting experiment: {experiment_name}/{run_name}")
        tracker.start_experiment(experiment_name, run_name)
        
        # Log parameters
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_name": "example_model"
        }
        tracker.log_params(params)
        print(f"Logged parameters: {params}")
        
        # Log metrics
        for step in range(5):
            metrics = {
                "loss": 1.0 - (step * 0.1),
                "accuracy": step * 0.2,
                "step": step
            }
            tracker.log_metrics(metrics, step=step)
            print(f"Step {step}: {metrics}")
        
        # Log text
        tracker.log_text("Example generated text", "sample_generation")
        
        # End experiment
        tracker.end_experiment()
        print("Experiment completed!")
        
        # Get experiment URL
        url = tracker.get_experiment_url()
        if url:
            print(f"Experiment URL: {url}")
        
    except Exception as e:
        print(f"Experiment tracking example failed: {str(e)}")
        print("This is expected if experiment tracking services are not configured")
    
    print()


def example_training_setup():
    """Example of training setup (without actual training)."""
    print("=== Training Setup Example ===")
    
    try:
        # Initialize components
        config = ConfigManager()
        config.load_config("config.yaml")
        
        # Override for example (use smaller model and reduced settings)
        config.config.model.model_name = "microsoft/DialoGPT-small"
        config.config.training.num_epochs = 1
        config.config.training.max_steps = 10
        config.config.training.batch_size = 2
        
        trainer = TrainingOrchestrator(config)
        
        print("Training orchestrator initialized successfully!")
        print(f"Model: {config.config.model.model_name}")
        print(f"Training type: {config.config.training.training_type}")
        print(f"Output directory: {config.config.training.output_dir}")
        
        # Note: We don't actually run training in this example
        # as it requires data and significant resources
        print("Training setup complete (not running actual training in example)")
        
    except Exception as e:
        print(f"Training setup example failed: {str(e)}")
        print("This is expected if dependencies are missing or resources are insufficient")
    
    print()


async def main():
    """Run all examples."""
    print("Custom LLM Chatbot - Usage Examples")
    print("===================================\n")
    
    # Check if config file exists
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found. Please ensure the configuration file exists.")
        return
    
    # Run examples
    example_config_usage()
    example_data_processing()
    await example_model_management()
    await example_serving()
    example_experiment_tracking()
    example_training_setup()
    
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Prepare your training data")
    print("2. Adjust configuration as needed")
    print("3. Run training: python train.py --config config.yaml")
    print("4. Start serving: python serve.py --config config.yaml")


if __name__ == "__main__":
    asyncio.run(main())