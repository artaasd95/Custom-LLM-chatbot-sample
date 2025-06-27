#!/usr/bin/env python3
"""Main training script for Custom LLM Chatbot."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import ConfigManager
from src.training.trainer import TrainingOrchestrator
from src.core.model_manager import ModelManager
from src.data.data_processor import DataProcessor
from src.monitoring.experiment_tracker import ExperimentTracker


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Custom LLM Chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    # Training type
    parser.add_argument(
        "--training-type",
        type=str,
        choices=["from_scratch", "fine_tune", "lora", "dpo"],
        help="Type of training to perform (overrides config)"
    )
    
    # Model and data
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name or path (overrides config)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data (overrides config)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for trained model (overrides config)"
    )
    
    # Experiment tracking
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for tracking"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name for tracking"
    )
    
    # Resume training
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    
    parser.add_argument(
        "--eval-steps",
        type=int,
        help="Evaluation frequency in steps (overrides config)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    
    # Distributed training
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data"
    )
    
    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual training"
    )
    
    return parser.parse_args()


def override_config_from_args(config_manager: ConfigManager, args):
    """Override configuration with command line arguments.
    
    Args:
        config_manager: Configuration manager instance.
        args: Parsed command line arguments.
    """
    # Training type
    if args.training_type:
        config_manager.config.training.training_type = args.training_type
    
    # Model
    if args.model_name:
        config_manager.config.model.model_name = args.model_name
    
    # Data
    if args.data_path:
        config_manager.config.data.train_data_path = args.data_path
    
    # Training parameters
    if args.epochs:
        config_manager.config.training.num_epochs = args.epochs
    
    if args.batch_size:
        config_manager.config.training.batch_size = args.batch_size
    
    if args.learning_rate:
        config_manager.config.training.learning_rate = args.learning_rate
    
    if args.eval_steps:
        config_manager.config.training.eval_steps = args.eval_steps
    
    # Output
    if args.output_dir:
        config_manager.config.training.output_dir = args.output_dir
    
    # Experiment tracking
    if args.experiment_name:
        config_manager.config.monitoring.experiment_name = args.experiment_name
    
    if args.run_name:
        config_manager.config.monitoring.run_name = args.run_name
    
    # Debug mode adjustments
    if args.debug:
        config_manager.config.training.num_epochs = min(config_manager.config.training.num_epochs, 1)
        config_manager.config.training.max_steps = min(config_manager.config.training.max_steps or 1000, 100)
        config_manager.config.data.max_samples = min(config_manager.config.data.max_samples or 10000, 1000)
        config_manager.config.training.eval_steps = min(config_manager.config.training.eval_steps, 50)
        config_manager.config.training.save_steps = min(config_manager.config.training.save_steps, 50)


def validate_environment():
    """Validate training environment.
    
    Returns:
        True if environment is valid.
    """
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required packages
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("CUDA not available, training will use CPU")
    except Exception as e:
        logger.warning(f"Could not check CUDA availability: {str(e)}")
    
    return True


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Custom LLM Chatbot Training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config_manager = ConfigManager()
        
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        config_manager.load_config(args.config)
        
        # Override with command line arguments
        override_config_from_args(config_manager, args)
        
        # Validate configuration
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        logger.info("Configuration loaded and validated successfully")
        
        # Print configuration summary
        logger.info(f"Training type: {config_manager.config.training.training_type}")
        logger.info(f"Model: {config_manager.config.model.model_name}")
        logger.info(f"Data path: {config_manager.config.data.train_data_path}")
        logger.info(f"Output directory: {config_manager.config.training.output_dir}")
        logger.info(f"Epochs: {config_manager.config.training.num_epochs}")
        logger.info(f"Batch size: {config_manager.config.training.batch_size}")
        logger.info(f"Learning rate: {config_manager.config.training.learning_rate}")
        
        if args.dry_run:
            logger.info("Dry run mode - configuration validated successfully")
            logger.info("Exiting without training")
            return
        
        # Initialize training orchestrator
        logger.info("Initializing training orchestrator")
        trainer = TrainingOrchestrator(config_manager)
        
        # Prepare training
        logger.info("Preparing training environment")
        trainer.prepare_training()
        
        if args.eval_only:
            # Run evaluation only
            logger.info("Running evaluation only")
            results = trainer.evaluate()
            logger.info(f"Evaluation results: {results}")
        else:
            # Run training
            logger.info("Starting training")
            
            if args.resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
                results = trainer.resume_training(args.resume_from_checkpoint)
            else:
                results = trainer.train()
            
            logger.info("Training completed successfully")
            logger.info(f"Training results: {results}")
            
            # Final evaluation
            logger.info("Running final evaluation")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
        
        # Cleanup
        logger.info("Cleaning up training resources")
        trainer.cleanup()
        
        logger.info("Training pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()