# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when working with the Custom LLM Chatbot system.

## 📋 Table of Contents

- [Installation Issues](#-installation-issues)
- [Configuration Problems](#-configuration-problems)
- [Training Issues](#-training-issues)
- [Serving Problems](#-serving-problems)
- [Performance Issues](#-performance-issues)
- [Memory Problems](#-memory-problems)
- [GPU Issues](#-gpu-issues)
- [API Errors](#-api-errors)
- [Data Processing Issues](#-data-processing-issues)
- [Monitoring Problems](#-monitoring-problems)
- [Common Error Messages](#-common-error-messages)
- [Debugging Tools](#-debugging-tools)
- [Getting Help](#-getting-help)

## 🔧 Installation Issues

### Problem: Package Installation Fails

**Symptoms:**
- `pip install` commands fail
- Missing dependencies errors
- Version conflicts

**Solutions:**

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with verbose output to see detailed errors
pip install -v -r requirements.txt

# Use conda for complex dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Clear pip cache if corrupted
pip cache purge
```

### Problem: CUDA Installation Issues

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- CUDA version mismatches
- Driver compatibility issues

**Solutions:**

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA compatibility
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Environment Setup Issues

**Symptoms:**
- Import errors
- Module not found errors
- Path issues

**Solutions:**

```bash
# Verify Python environment
which python
python --version

# Check installed packages
pip list | grep torch
pip list | grep transformers

# Reinstall in development mode
pip install -e .

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:/path/to/Custom-LLM-chatbot-sample/src"
```

## ⚙️ Configuration Problems

### Problem: Configuration File Not Found

**Symptoms:**
- `FileNotFoundError: config.yaml not found`
- Configuration loading errors

**Solutions:**

```python
# Check file path
import os
print(os.path.abspath('config.yaml'))
print(os.path.exists('config.yaml'))

# Use absolute path
config_path = '/absolute/path/to/config.yaml'
config_manager = ConfigManager(config_path)

# Create default config if missing
from src.core.config import ConfigManager
ConfigManager.create_default_config('config.yaml')
```

### Problem: Invalid Configuration Values

**Symptoms:**
- Validation errors
- Type conversion errors
- Missing required fields

**Solutions:**

```yaml
# Ensure proper YAML syntax
model:
  name: "gpt2"  # Use quotes for strings
  max_length: 512  # Numbers without quotes
  device: "auto"

training:
  batch_size: 4
  learning_rate: 0.0001  # Use decimal notation
  num_epochs: 3
```

```python
# Validate configuration programmatically
from src.core.config import ConfigManager

try:
    config = ConfigManager('config.yaml')
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Problem: Environment Variable Issues

**Symptoms:**
- Environment variables not recognized
- Override not working

**Solutions:**

```bash
# Check environment variables
echo $WANDB_API_KEY
echo $HF_TOKEN

# Set environment variables properly
export WANDB_API_KEY="your_key_here"
export HF_TOKEN="your_token_here"

# Use .env file
echo "WANDB_API_KEY=your_key_here" > .env
echo "HF_TOKEN=your_token_here" >> .env
```

## 🎯 Training Issues

### Problem: Out of Memory During Training

**Symptoms:**
- `CUDA out of memory` errors
- Training crashes
- System freezes

**Solutions:**

```yaml
# Reduce batch size
training:
  batch_size: 1  # Start small
  gradient_accumulation_steps: 8  # Maintain effective batch size

# Enable gradient checkpointing
model:
  gradient_checkpointing: true

# Use mixed precision
training:
  fp16: true  # For older GPUs
  bf16: true  # For newer GPUs (A100, RTX 30xx+)
```

```python
# Monitor memory usage
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Clear cache between runs
torch.cuda.empty_cache()
```

### Problem: Training Not Converging

**Symptoms:**
- Loss not decreasing
- Poor validation metrics
- Unstable training

**Solutions:**

```yaml
# Adjust learning rate
training:
  learning_rate: 0.00001  # Start lower
  lr_scheduler: "cosine"  # Use scheduler
  warmup_steps: 100

# Check data quality
data:
  validation_split: 0.1
  shuffle: true
  max_length: 512  # Ensure consistent length
```

```python
# Debug training data
from src.data.processor import DataProcessor

processor = DataProcessor(config.data)
data = processor.load_data()

# Check data statistics
print(f"Training samples: {len(data['train'])}")
print(f"Validation samples: {len(data['validation'])}")
print(f"Average length: {sum(len(x['input'].split()) for x in data['train']) / len(data['train'])}")
```

### Problem: Training Crashes

**Symptoms:**
- Unexpected training termination
- Error messages during training
- Corrupted checkpoints

**Solutions:**

```python
# Add error handling
try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    # Save current state
    trainer.save_checkpoint("emergency_checkpoint")
    raise

# Enable automatic checkpointing
training_args = TrainingArguments(
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    resume_from_checkpoint="path/to/checkpoint"  # Resume if needed
)
```

## 🚀 Serving Problems

### Problem: Model Loading Fails

**Symptoms:**
- Model not found errors
- Loading timeout
- Corrupted model files

**Solutions:**

```python
# Check model path
import os
model_path = "path/to/model"
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Model files: {os.listdir(model_path)}")

# Load with error handling
from src.core.model_manager import ModelManager

try:
    manager = ModelManager(config.model)
    model = manager.load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")
    # Try loading base model instead
    config.model.name = "gpt2"  # Fallback to base model
    model = manager.load_model()
```

### Problem: API Server Not Starting

**Symptoms:**
- Port already in use
- Permission denied
- Server crashes on startup

**Solutions:**

```bash
# Check port availability
netstat -tulpn | grep :8000
lsof -i :8000

# Kill existing process
kill -9 $(lsof -t -i:8000)

# Use different port
python -m src.serving.api_server --port 8001
```

```python
# Add startup validation
from src.serving.api_server import APIServer

try:
    server = APIServer(config)
    server.start(host="0.0.0.0", port=8000)
except OSError as e:
    if "Address already in use" in str(e):
        print("Port 8000 is already in use. Try a different port.")
    else:
        print(f"Server startup error: {e}")
```

### Problem: Slow Inference

**Symptoms:**
- High response latency
- Timeout errors
- Poor throughput

**Solutions:**

```yaml
# Optimize serving configuration
serving:
  backend: "vllm"  # Use vLLM for better performance
  max_batch_size: 8
  max_sequence_length: 1024
  
  # vLLM specific optimizations
  vllm:
    tensor_parallel_size: 2  # Multi-GPU
    gpu_memory_utilization: 0.9
    swap_space: 4  # GB
```

```python
# Profile inference
import time

start_time = time.time()
response = model.generate(input_text)
end_time = time.time()

print(f"Inference time: {end_time - start_time:.2f} seconds")
print(f"Tokens per second: {len(response.split()) / (end_time - start_time):.2f}")
```

## 📊 Performance Issues

### Problem: High Memory Usage

**Symptoms:**
- System running out of RAM
- Swap usage increasing
- OOM killer activated

**Solutions:**

```python
# Monitor memory usage
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024**3:.2f} GB")
    print(f"VMS: {memory_info.vms / 1024**3:.2f} GB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System memory usage: {system_memory.percent}%")

# Force garbage collection
gc.collect()

# Clear PyTorch cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

```yaml
# Optimize memory usage
model:
  load_in_8bit: true  # Use 8-bit quantization
  device_map: "auto"  # Automatic device placement

serving:
  max_batch_size: 1  # Reduce batch size
  streaming: true  # Enable streaming responses
```

### Problem: Slow Data Loading

**Symptoms:**
- Long data preprocessing times
- Training bottlenecked by data loading
- High I/O wait times

**Solutions:**

```yaml
# Optimize data loading
data:
  num_workers: 4  # Parallel data loading
  pin_memory: true  # Faster GPU transfer
  prefetch_factor: 2  # Prefetch batches
  persistent_workers: true  # Keep workers alive
```

```python
# Use data streaming
from datasets import load_dataset

# Load dataset in streaming mode
dataset = load_dataset("your_dataset", streaming=True)

# Cache processed data
from functools import lru_cache

@lru_cache(maxsize=1000)
def process_sample(text):
    return tokenizer(text, truncation=True, padding=True)
```

## 💾 Memory Problems

### Problem: GPU Memory Leaks

**Symptoms:**
- GPU memory usage keeps increasing
- Eventually runs out of GPU memory
- Performance degrades over time

**Solutions:**

```python
# Track GPU memory leaks
import torch

def track_gpu_memory(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            print(f"Memory change: {(final_memory - initial_memory) / 1024**2:.2f} MB")
            print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
            
        return result
    return wrapper

# Use context manager for memory cleanup
class GPUMemoryManager:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage
with GPUMemoryManager():
    result = model.generate(input_text)
```

### Problem: CPU Memory Leaks

**Symptoms:**
- RAM usage continuously increases
- System becomes unresponsive
- Process killed by OOM killer

**Solutions:**

```python
# Monitor memory usage
import tracemalloc
import gc

# Start memory tracing
tracemalloc.start()

# Your code here

# Get memory statistics
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.2f} MB")
print(f"Peak memory usage: {peak / 1024**2:.2f} MB")

# Stop tracing
tracemalloc.stop()

# Force garbage collection
gc.collect()

# Find memory leaks
import objgraph
objgraph.show_most_common_types(limit=10)
```

## 🖥️ GPU Issues

### Problem: GPU Not Detected

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- Training falls back to CPU
- Poor performance

**Solutions:**

```python
# Comprehensive GPU check
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("No GPU detected. Check CUDA installation.")
```

```bash
# System-level GPU check
nvidia-smi
lspci | grep -i nvidia

# Check CUDA installation
nvcc --version
cat /usr/local/cuda/version.txt

# Check driver version
cat /proc/driver/nvidia/version
```

### Problem: Multi-GPU Issues

**Symptoms:**
- Only one GPU being used
- Uneven GPU utilization
- Synchronization errors

**Solutions:**

```python
# Enable multi-GPU training
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Data Parallel (single machine)
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# Distributed training setup
import torch.distributed as dist

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# Check GPU utilization
def check_gpu_utilization():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} utilization: {torch.cuda.utilization(i)}%")
            print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
```

## 🌐 API Errors

### Problem: HTTP 500 Internal Server Error

**Symptoms:**
- API requests return 500 errors
- Server logs show exceptions
- Inconsistent behavior

**Solutions:**

```python
# Add comprehensive error handling
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Validate input
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty prompt")
        
        if len(request.prompt) > 10000:
            raise HTTPException(status_code=400, detail="Prompt too long")
        
        # Generate response
        response = model.generate(request.prompt)
        return GenerateResponse(text=response)
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Problem: Request Timeout

**Symptoms:**
- API requests timeout
- Client receives no response
- Server appears to hang

**Solutions:**

```python
# Add timeout handling
import asyncio
from fastapi import BackgroundTasks

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Set timeout for generation
        response = await asyncio.wait_for(
            model.generate_async(request.prompt),
            timeout=30.0  # 30 second timeout
        )
        return GenerateResponse(text=response)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
```

```yaml
# Configure server timeouts
serving:
  timeout: 30  # seconds
  max_concurrent_requests: 10
  request_queue_size: 100
```

## 📁 Data Processing Issues

### Problem: Data Format Errors

**Symptoms:**
- JSON parsing errors
- Missing required fields
- Inconsistent data structure

**Solutions:**

```python
# Validate data format
import json
from typing import List, Dict

def validate_chat_data(data: List[Dict]) -> List[str]:
    """Validate chat data format and return errors."""
    errors = []
    required_fields = ['input', 'output']
    
    for i, item in enumerate(data):
        # Check if item is dictionary
        if not isinstance(item, dict):
            errors.append(f"Item {i}: Expected dict, got {type(item)}")
            continue
        
        # Check required fields
        for field in required_fields:
            if field not in item:
                errors.append(f"Item {i}: Missing field '{field}'")
            elif not isinstance(item[field], str):
                errors.append(f"Item {i}: Field '{field}' must be string")
            elif not item[field].strip():
                errors.append(f"Item {i}: Field '{field}' is empty")
    
    return errors

# Usage
with open('data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

errors = validate_chat_data(data)
if errors:
    print("Data validation errors:")
    for error in errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
else:
    print("Data validation passed")
```

### Problem: Encoding Issues

**Symptoms:**
- Unicode decode errors
- Garbled text
- Special characters not displaying correctly

**Solutions:**

```python
# Handle encoding properly
import chardet

def detect_encoding(file_path: str) -> str:
    """Detect file encoding."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result['encoding']

def read_file_safe(file_path: str) -> str:
    """Read file with proper encoding handling."""
    encoding = detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to utf-8 with error handling
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

# Clean text data
def clean_text(text: str) -> str:
    """Clean text data."""
    import re
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove empty lines
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    return text.strip()
```

## 📈 Monitoring Problems

### Problem: Metrics Not Logging

**Symptoms:**
- No metrics in monitoring dashboard
- Experiment tracking not working
- Missing training logs

**Solutions:**

```python
# Debug monitoring setup
from src.monitoring.experiment_tracker import ExperimentTracker

# Test connection
tracker = ExperimentTracker(config.monitoring)

try:
    tracker.start_experiment("test_experiment")
    tracker.log_metrics({"test_metric": 1.0})
    print("Monitoring working correctly")
except Exception as e:
    print(f"Monitoring error: {e}")
    
    # Check API keys
    import os
    print(f"WANDB_API_KEY set: {'WANDB_API_KEY' in os.environ}")
    print(f"COMET_API_KEY set: {'COMET_API_KEY' in os.environ}")
```

```yaml
# Verify monitoring configuration
monitoring:
  enabled: true
  wandb:
    project: "custom-llm-chatbot"
    entity: "your-username"  # Make sure this is correct
  comet:
    project_name: "custom-llm-chatbot"
    workspace: "your-workspace"  # Make sure this is correct
```

### Problem: Dashboard Not Updating

**Symptoms:**
- Metrics dashboard shows old data
- Real-time updates not working
- Graphs not refreshing

**Solutions:**

```python
# Force metric sync
import wandb

# Ensure metrics are synced
wandb.log({"metric": value}, commit=True)

# Check sync status
if wandb.run:
    print(f"Run state: {wandb.run.state}")
    print(f"Run URL: {wandb.run.url}")

# Manual sync if needed
wandb.run.save()
```

## ❌ Common Error Messages

### "RuntimeError: CUDA out of memory"

**Cause:** GPU memory exhausted

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training
4. Clear GPU cache: `torch.cuda.empty_cache()`

### "ModuleNotFoundError: No module named 'src'"

**Cause:** Python path not set correctly

**Solutions:**
1. Install in development mode: `pip install -e .`
2. Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:/path/to/project"`
3. Use absolute imports

### "FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'"

**Cause:** Configuration file not found

**Solutions:**
1. Check file path: `os.path.abspath('config.yaml')`
2. Create default config: `ConfigManager.create_default_config()`
3. Use absolute path in code

### "ValueError: Input contains NaN, infinity or a value too large"

**Cause:** Invalid data in training dataset

**Solutions:**
1. Check for NaN values: `data.isna().sum()`
2. Remove invalid samples: `data.dropna()`
3. Validate data before training

### "ConnectionError: HTTPSConnectionPool"

**Cause:** Network connectivity issues

**Solutions:**
1. Check internet connection
2. Verify API keys
3. Use offline mode if available
4. Check firewall settings

## 🔍 Debugging Tools

### Python Debugger

```python
# Use pdb for debugging
import pdb

def problematic_function():
    pdb.set_trace()  # Breakpoint
    # Your code here
    return result

# Use ipdb for better interface
import ipdb
ipdb.set_trace()
```

### Logging Setup

```python
# Configure comprehensive logging
import logging
import sys

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Usage
logger = setup_logging(logging.DEBUG)
logger.info("Starting application")
```

### Performance Profiling

```python
# Profile code performance
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile function execution."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    return result

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

### System Monitoring

```python
# Monitor system resources
import psutil
import time

def monitor_system(duration=60, interval=5):
    """Monitor system resources."""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # GPU usage (if available)
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        
        print(f"CPU: {cpu_percent}% | RAM: {memory.percent}% | Disk: {disk.percent}% | {gpu_info}")
        time.sleep(interval)

# Usage
monitor_system(duration=300, interval=10)  # Monitor for 5 minutes
```

## 🆘 Getting Help

### Before Asking for Help

1. **Check the logs**: Look for error messages and stack traces
2. **Reproduce the issue**: Create a minimal example that reproduces the problem
3. **Check documentation**: Review relevant documentation sections
4. **Search existing issues**: Look for similar problems in GitHub issues

### Information to Include

```python
# System information script
import sys
import torch
import transformers
import platform

def get_system_info():
    """Get comprehensive system information."""
    info = {
        "Python version": sys.version,
        "Platform": platform.platform(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "CUDA available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["CUDA version"] = torch.version.cuda
        info["GPU count"] = torch.cuda.device_count()
        info["GPU names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info

# Print system info
for key, value in get_system_info().items():
    print(f"{key}: {value}")
```

### Creating Bug Reports

```markdown
## Bug Report Template

### Description
A clear description of what the bug is.

### Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

### Expected Behavior
What you expected to happen.

### Actual Behavior
What actually happened.

### Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 1.13.0]
- CUDA version: [e.g., 11.7]

### Configuration
```yaml
# Your config.yaml content
```

### Error Messages
```
# Full error traceback
```

### Additional Context
Any other context about the problem.
```

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Sample configurations and use cases

---

**Related Documentation**:
- [Installation Guide](./installation.md) - Setup and installation help
- [Configuration Guide](./configuration.md) - Configuration troubleshooting
- [Testing Guide](./testing.md) - Testing and validation
- [User Guide](./user-guide.md) - Usage instructions