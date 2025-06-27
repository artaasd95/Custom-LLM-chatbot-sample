#!/usr/bin/env python3
"""Main serving script for Custom LLM Chatbot."""

import os
import sys
import argparse
import logging
import asyncio
import signal
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import ConfigManager
from src.serving.model_server import ModelServer
from src.serving.api_server import APIServer
from src.serving.vllm_server import VLLMServer, VLLM_AVAILABLE


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
        description="Serve Custom LLM Chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model (overrides config)"
    )
    
    # Server type
    parser.add_argument(
        "--server-type",
        type=str,
        choices=["pytorch", "onnx", "vllm"],
        help="Type of server backend (overrides config)"
    )
    
    # API settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    
    # Performance settings
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (overrides config)"
    )
    
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        help="Maximum concurrent requests (overrides config)"
    )
    
    # vLLM specific
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallel size for vLLM (overrides config)"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="GPU memory utilization for vLLM (overrides config)"
    )
    
    # Generation settings
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate (overrides config)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature (overrides config)"
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
    
    # Development
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    # Health check
    parser.add_argument(
        "--health-check-only",
        action="store_true",
        help="Only perform health check and exit"
    )
    
    return parser.parse_args()


def override_config_from_args(config_manager: ConfigManager, args):
    """Override configuration with command line arguments.
    
    Args:
        config_manager: Configuration manager instance.
        args: Parsed command line arguments.
    """
    # Model
    if args.model_path:
        config_manager.config.model.model_name = args.model_path
    
    # Server type
    if args.server_type:
        config_manager.config.serving.backend_type = args.server_type
    
    # API settings
    config_manager.config.serving.api.host = args.host
    config_manager.config.serving.api.port = args.port
    
    # Performance
    if args.workers:
        config_manager.config.serving.api.workers = args.workers
    
    if args.max_concurrent_requests:
        config_manager.config.serving.api.max_concurrent_requests = args.max_concurrent_requests
    
    # vLLM specific
    if args.tensor_parallel_size:
        config_manager.config.serving.vllm.tensor_parallel_size = args.tensor_parallel_size
    
    if args.gpu_memory_utilization:
        config_manager.config.serving.vllm.gpu_memory_utilization = args.gpu_memory_utilization
    
    # Generation
    if args.max_tokens:
        config_manager.config.serving.generation.max_tokens = args.max_tokens
    
    if args.temperature:
        config_manager.config.serving.generation.temperature = args.temperature
    
    # Development
    if args.reload:
        config_manager.config.serving.api.reload = True
    
    if args.debug:
        config_manager.config.serving.api.debug = True


def validate_environment():
    """Validate serving environment.
    
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
        "fastapi",
        "uvicorn"
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
            logger.warning("CUDA not available, serving will use CPU")
    except Exception as e:
        logger.warning(f"Could not check CUDA availability: {str(e)}")
    
    return True


async def create_model_server(config_manager: ConfigManager, server_type: str) -> Optional[object]:
    """Create appropriate model server based on type.
    
    Args:
        config_manager: Configuration manager instance.
        server_type: Type of server (pytorch, onnx, vllm).
        
    Returns:
        Model server instance or None if failed.
    """
    logger = logging.getLogger(__name__)
    
    try:
        if server_type == "vllm":
            if not VLLM_AVAILABLE:
                logger.error("vLLM not available. Install with: pip install vllm")
                return None
            
            logger.info("Creating vLLM server")
            server = VLLMServer(config_manager)
            
            # Load model
            model_path = config_manager.config.model.model_name
            tensor_parallel_size = config_manager.config.serving.vllm.tensor_parallel_size
            gpu_memory_utilization = config_manager.config.serving.vllm.gpu_memory_utilization
            max_model_len = config_manager.config.serving.vllm.max_model_len
            
            success = await server.load_model(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len
            )
            
            if not success:
                logger.error("Failed to load model with vLLM")
                return None
            
            return server
            
        else:
            logger.info(f"Creating {server_type} model server")
            server = ModelServer(config_manager)
            
            # Load model
            model_path = config_manager.config.model.model_name
            backend_type = server_type
            
            success = await server.load_model(model_path, backend_type)
            
            if not success:
                logger.error(f"Failed to load model with {server_type}")
                return None
            
            return server
            
    except Exception as e:
        logger.error(f"Failed to create {server_type} server: {str(e)}")
        return None


async def health_check(config_manager: ConfigManager, server_type: str) -> bool:
    """Perform health check on the server.
    
    Args:
        config_manager: Configuration manager instance.
        server_type: Type of server to check.
        
    Returns:
        True if health check passed.
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Performing health check")
        
        # Create model server
        model_server = await create_model_server(config_manager, server_type)
        if not model_server:
            logger.error("Failed to create model server for health check")
            return False
        
        # Check server health
        health_status = model_server.health_check()
        logger.info(f"Health status: {health_status}")
        
        if health_status.get("status") != "healthy":
            logger.error("Health check failed")
            return False
        
        # Test generation
        if hasattr(model_server, 'generate'):
            test_prompt = "Hello, how are you?"
            logger.info(f"Testing generation with prompt: {test_prompt}")
            
            if server_type == "vllm":
                from src.serving.vllm_server import VLLMRequest
                request = VLLMRequest(prompt=test_prompt, max_tokens=10)
                response = await model_server.generate(request)
                logger.info(f"Test generation successful: {response.text[:50]}...")
            else:
                response = await model_server.generate(test_prompt, max_tokens=10)
                logger.info(f"Test generation successful: {response[:50]}...")
        
        # Cleanup
        if hasattr(model_server, 'unload_model'):
            await model_server.unload_model()
        
        logger.info("Health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False


async def run_server(config_manager: ConfigManager, args):
    """Run the serving server.
    
    Args:
        config_manager: Configuration manager instance.
        args: Parsed command line arguments.
    """
    logger = logging.getLogger(__name__)
    
    # Determine server type
    server_type = config_manager.config.serving.backend_type
    logger.info(f"Starting server with backend: {server_type}")
    
    # Create model server
    model_server = await create_model_server(config_manager, server_type)
    if not model_server:
        logger.error("Failed to create model server")
        return
    
    # Create API server
    logger.info("Creating API server")
    api_server = APIServer(config_manager, model_server)
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start API server
        host = config_manager.config.serving.api.host
        port = config_manager.config.serving.api.port
        workers = config_manager.config.serving.api.workers
        reload = config_manager.config.serving.api.reload
        debug = config_manager.config.serving.api.debug
        
        logger.info(f"Starting API server on {host}:{port}")
        logger.info(f"Workers: {workers}, Reload: {reload}, Debug: {debug}")
        
        # Start server
        server_task = asyncio.create_task(
            api_server.start_server(
                host=host,
                port=port,
                workers=workers if not reload else 1,  # Single worker for reload
                reload=reload,
                debug=debug
            )
        )
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Graceful shutdown
        logger.info("Shutting down server")
        server_task.cancel()
        
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        # Cleanup model server
        if hasattr(model_server, 'unload_model'):
            await model_server.unload_model()
        
        logger.info("Server shutdown complete")
        
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise


async def main_async():
    """Main async function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Custom LLM Chatbot Server")
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
        logger.info(f"Model: {config_manager.config.model.model_name}")
        logger.info(f"Backend: {config_manager.config.serving.backend_type}")
        logger.info(f"Host: {config_manager.config.serving.api.host}")
        logger.info(f"Port: {config_manager.config.serving.api.port}")
        
        if args.health_check_only:
            # Run health check only
            logger.info("Running health check only")
            success = await health_check(config_manager, config_manager.config.serving.backend_type)
            if success:
                logger.info("Health check passed")
                sys.exit(0)
            else:
                logger.error("Health check failed")
                sys.exit(1)
        
        # Run server
        await run_server(config_manager, args)
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


def main():
    """Main function."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()