# Configuration Guide

Comprehensive guide for configuring the Custom LLM Chatbot system for different use cases and environments.

## 📋 Table of Contents

- [Configuration Overview](#configuration-overview)
- [Configuration Structure](#configuration-structure)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Data Configuration](#data-configuration)
- [Serving Configuration](#serving-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Cloud Configuration](#cloud-configuration)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🔧 Configuration Overview

The Custom LLM Chatbot system uses a hierarchical YAML-based configuration system with the following features:

- **Type Safety**: Dataclass-based configuration with automatic validation
- **Environment Overrides**: Support for environment variable configuration
- **Hierarchical Structure**: Nested configuration sections for different components
- **Hot Reloading**: Runtime configuration updates without restart
- **Validation**: Comprehensive configuration validation with error reporting

### Configuration Files

| File | Purpose | Environment |
|------|---------|-------------|
| `config/config.yaml` | Main configuration file | All |
| `config/development.yaml` | Development settings | Development |
| `config/production.yaml` | Production settings | Production |
| `config/training.yaml` | Training-specific settings | Training |
| `config/serving.yaml` | Serving-specific settings | Serving |
| `.env` | Environment variables | All |

## 🏗️ Configuration Structure

### Main Configuration Schema

```yaml
# config.yaml
model:
  # Model settings
  name: "Qwen/Qwen2.5-3B"
  model_type: "qwen"
  max_length: 2048
  device: "auto"
  torch_dtype: "auto"
  trust_remote_code: true

training:
  # Training settings
  training_type: "dpo"  # from_scratch, dpo, sft, lora
  output_dir: "./models/trained"
  logging_dir: "./logs"
  seed: 42
  
  # Training method specific configs
  from_scratch:
    num_train_epochs: 3
    per_device_train_batch_size: 4
    learning_rate: 5e-5
    
  dpo:
    num_train_epochs: 1
    per_device_train_batch_size: 2
    learning_rate: 1e-6
    beta: 0.1
    
  lora:
    enabled: true
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

data:
  # Data settings
  train_data_path: "./data/train"
  eval_data_path: "./data/eval"
  max_length: 2048
  batch_size: 1000
  
serving:
  # Serving settings
  backend_type: "pytorch"  # pytorch, onnx, vllm
  host: "0.0.0.0"
  port: 8000
  max_concurrent_requests: 100
  
monitoring:
  # Monitoring settings
  wandb:
    enabled: false
    project: "custom-llm-chatbot"
  comet:
    enabled: false
    project_name: "custom-llm-chatbot"
  mlflow:
    enabled: true
    tracking_uri: "./mlruns"

cloud:
  # Cloud deployment settings
  provider: "azure"
  environment: "development"
  replicas: 1
```

## 🤖 Model Configuration

### Basic Model Settings

```yaml
model:
  name: "Qwen/Qwen2.5-3B"           # HuggingFace model name or local path
  model_type: "qwen"                 # Model architecture type
  max_length: 2048                   # Maximum sequence length
  device: "auto"                     # Device selection
  torch_dtype: "auto"                # PyTorch data type
  trust_remote_code: true            # Allow remote code execution
```

### Supported Models

| Model Family | Model Name | Parameters | VRAM Required |
|--------------|------------|------------|---------------|
| **Qwen** | `Qwen/Qwen2.5-1.5B` | 1.5B | 4GB |
| **Qwen** | `Qwen/Qwen2.5-3B` | 3B | 8GB |
| **Qwen** | `Qwen/Qwen2.5-7B` | 7B | 16GB |
| **Qwen** | `Qwen/Qwen2.5-14B` | 14B | 32GB |
| **Llama** | `meta-llama/Llama-2-7b-hf` | 7B | 16GB |
| **Llama** | `meta-llama/Llama-2-13b-hf` | 13B | 32GB |
| **Mistral** | `mistralai/Mistral-7B-v0.1` | 7B | 16GB |

### Device Configuration

```yaml
model:
  device: "auto"        # Automatic device selection
  # device: "cpu"       # Force CPU usage
  # device: "cuda"      # Use first available GPU
  # device: "cuda:0"    # Use specific GPU
  # device: "cuda:0,1"  # Use multiple GPUs
```

### Data Type Configuration

```yaml
model:
  torch_dtype: "auto"      # Automatic selection
  # torch_dtype: "float32" # Full precision
  # torch_dtype: "float16" # Half precision
  # torch_dtype: "bfloat16" # Brain float 16
```

## 🎓 Training Configuration

### Training Type Selection

```yaml
training:
  training_type: "dpo"  # Options: from_scratch, dpo, sft, lora
  output_dir: "./models/trained"
  logging_dir: "./logs"
  seed: 42
```

### From-Scratch Training

```yaml
training:
  training_type: "from_scratch"
  
  from_scratch:
    # Basic training parameters
    num_train_epochs: 3
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 4
    gradient_accumulation_steps: 8
    
    # Optimization parameters
    learning_rate: 5e-5
    weight_decay: 0.01
    warmup_steps: 500
    max_grad_norm: 1.0
    
    # Advanced settings
    optimizer: "adamw_torch"
    lr_scheduler_type: "cosine"
    fp16: false
    bf16: true
    gradient_checkpointing: true
    dataloader_num_workers: 4
    
    # Evaluation and saving
    eval_strategy: "steps"
    eval_steps: 500
    save_strategy: "steps"
    save_steps: 1000
    save_total_limit: 3
    load_best_model_at_end: true
    metric_for_best_model: "eval_loss"
    greater_is_better: false
```

### DPO Training

```yaml
training:
  training_type: "dpo"
  
  dpo:
    # DPO-specific parameters
    num_train_epochs: 1
    per_device_train_batch_size: 2
    per_device_eval_batch_size: 2
    gradient_accumulation_steps: 16
    
    # DPO hyperparameters
    learning_rate: 1e-6
    beta: 0.1                    # DPO temperature parameter
    max_length: 1024             # Maximum sequence length
    max_prompt_length: 512       # Maximum prompt length
    
    # Reference model settings
    reference_model: null        # Use base model as reference
    reference_model_path: null   # Path to custom reference model
    
    # Loss configuration
    loss_type: "sigmoid"         # Loss function type
    label_smoothing: 0.0         # Label smoothing factor
```

### LoRA Training

```yaml
training:
  training_type: "lora"
  
  lora:
    enabled: true
    
    # LoRA parameters
    r: 16                        # LoRA rank
    lora_alpha: 32               # LoRA alpha parameter
    lora_dropout: 0.1            # LoRA dropout rate
    
    # Target modules
    target_modules:
      - "q_proj"
      - "v_proj"
      - "k_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    
    # Advanced LoRA settings
    bias: "none"                 # Bias handling: none, all, lora_only
    task_type: "CAUSAL_LM"       # Task type
    inference_mode: false        # Inference mode
    
    # Training parameters
    num_train_epochs: 5
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4
    learning_rate: 1e-4
    warmup_steps: 100
    weight_decay: 0.01
```

### Supervised Fine-Tuning

```yaml
training:
  training_type: "sft"
  
  sft:
    # Basic parameters
    num_train_epochs: 3
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
    gradient_accumulation_steps: 2
    
    # Optimization
    learning_rate: 2e-5
    warmup_ratio: 0.1
    lr_scheduler_type: "cosine"
    weight_decay: 0.01
    
    # Data handling
    max_seq_length: 2048
    packing: false               # Pack multiple samples
    dataset_text_field: "text"   # Text field name
    
    # Advanced settings
    fp16: false
    bf16: true
    gradient_checkpointing: true
    remove_unused_columns: false
```

## 📊 Data Configuration

### Basic Data Settings

```yaml
data:
  # Data paths
  train_data_path: "./data/train"
  eval_data_path: "./data/eval"
  test_data_path: "./data/test"
  
  # Data types
  data_types:
    - "chat"
    - "documents"
    - "reports"
  
  # Preprocessing settings
  max_length: 2048
  truncation: true
  padding: "max_length"
  remove_columns: ["id"]
  text_column: "text"
  
  # Pipeline settings
  batch_size: 1000
  num_workers: 4
  cache_dir: "./cache"
```

### Data Format Specifications

#### Chat Data Format

```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
  ]
}
```

#### DPO Data Format

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I think it might be Lyon or Marseille."
}
```

#### Instruction Data Format

```json
{
  "instruction": "Translate the following English text to French:",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

## 🚀 Serving Configuration

### Backend Selection

```yaml
serving:
  backend_type: "pytorch"  # Options: pytorch, onnx, vllm
  
  # Server settings
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 300
  max_concurrent_requests: 100
```

### PyTorch Backend

```yaml
serving:
  backend_type: "pytorch"
  
  pytorch:
    device: "auto"
    torch_dtype: "auto"
    compile_model: false
    optimization_level: "O1"
    max_batch_size: 32
    enable_streaming: true
```

### vLLM Backend

```yaml
serving:
  backend_type: "vllm"
  
  vllm:
    # Model settings
    model_path: "./models/trained"
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    
    # Memory settings
    gpu_memory_utilization: 0.9
    max_model_len: 2048
    max_num_seqs: 256
    
    # Performance settings
    enable_streaming: true
    enable_chunked_prefill: true
    max_num_batched_tokens: 8192
    
    # Advanced settings
    trust_remote_code: true
    enforce_eager: false
    swap_space: 4
```

### ONNX Backend

```yaml
serving:
  backend_type: "onnx"
  
  onnx:
    model_path: "./models/onnx/model.onnx"
    providers:
      - "CUDAExecutionProvider"
      - "CPUExecutionProvider"
    
    # Optimization settings
    optimization_level: "all"
    intra_op_num_threads: 4
    inter_op_num_threads: 2
    
    # Session settings
    enable_profiling: false
    enable_mem_pattern: true
    enable_cpu_mem_arena: true
```

## 📈 Monitoring Configuration

### Experiment Tracking

```yaml
monitoring:
  experiment_tracking:
    platform: "wandb"  # Options: wandb, mlflow, comet, local
    
  # Weights & Biases
  wandb:
    enabled: true
    project: "custom-llm-chatbot"
    entity: "your-team"
    api_key: "${WANDB_API_KEY}"
    tags: ["llm", "training"]
    group: "experiment-group"
    job_type: "training"
    notes: "Training experiment"
    
  # MLflow
  mlflow:
    enabled: false
    tracking_uri: "./mlruns"
    experiment_name: "llm-training"
    registry_uri: "sqlite:///mlflow.db"
    
  # Comet ML
  comet:
    enabled: false
    project_name: "custom-llm-chatbot"
    workspace: "your-workspace"
    api_key: "${COMET_API_KEY}"
    
  # Metrics collection
  metrics:
    enabled: true
    log_interval: 10
    save_interval: 100
    
  # Logging
  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: "./logs/app.log"
    max_file_size: "10MB"
    backup_count: 5
```

## ☁️ Cloud Configuration

### Azure Configuration

```yaml
cloud:
  provider: "azure"
  
  # Azure settings
  azure:
    subscription_id: "${AZURE_SUBSCRIPTION_ID}"
    resource_group: "llm-training-rg"
    location: "eastus"
    
    # Storage
    storage_account: "llmtrainingstorage"
    container_name: "models"
    
    # Compute
    vm_size: "Standard_NC24s_v3"
    disk_size: 512
    
  # Deployment settings
  deployment:
    environment: "development"
    replicas: 1
    
    # Resource limits
    cpu_request: "2"
    cpu_limit: "4"
    memory_request: "8Gi"
    memory_limit: "16Gi"
    gpu_request: 1
    gpu_limit: 1
```

### AWS Configuration

```yaml
cloud:
  provider: "aws"
  
  # AWS settings
  aws:
    region: "us-west-2"
    access_key_id: "${AWS_ACCESS_KEY_ID}"
    secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    
    # S3 storage
    s3_bucket: "llm-training-bucket"
    s3_prefix: "models/"
    
    # EC2 compute
    instance_type: "p3.8xlarge"
    ami_id: "ami-0abcdef1234567890"
    key_pair: "llm-training-key"
    security_group: "sg-0123456789abcdef0"
```

## 🌍 Environment Variables

### Core Environment Variables

```bash
# Model Configuration
MODEL_NAME=Qwen/Qwen2.5-3B
MODEL_PATH=./models
MAX_MODEL_LEN=2048
DEVICE=auto
TORCH_DTYPE=auto

# Training Configuration
TRAINING_TYPE=dpo
OUTPUT_DIR=./outputs
LOGGING_DIR=./logs
NUM_TRAIN_EPOCHS=3
PER_DEVICE_BATCH_SIZE=4
LEARNING_RATE=5e-5

# Serving Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
BACKEND_TYPE=pytorch
WORKERS=4
MAX_CONCURRENT_REQUESTS=100

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

# Data Configuration
TRAIN_DATA_PATH=./data/train
EVAL_DATA_PATH=./data/eval
BATCH_SIZE=1000
MAX_LENGTH=2048
CACHE_DIR=./cache

# Monitoring
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=llm-training
MLFLOW_TRACKING_URI=./mlruns
COMET_API_KEY=your-comet-key

# Cloud
AZURE_SUBSCRIPTION_ID=your-subscription-id
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Security
API_KEY=your-api-key
JWT_SECRET=your-jwt-secret
AUTH_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## 📝 Configuration Examples

### Development Configuration

```yaml
# config/development.yaml
model:
  name: "Qwen/Qwen2.5-1.5B"  # Smaller model for development
  device: "auto"
  torch_dtype: "float32"

training:
  training_type: "sft"
  output_dir: "./dev_outputs"
  
  sft:
    num_train_epochs: 1
    per_device_train_batch_size: 2
    learning_rate: 5e-5
    eval_steps: 50
    save_steps: 100

data:
  train_data_path: "./data/dev_train"
  eval_data_path: "./data/dev_eval"
  max_length: 512
  batch_size: 100

serving:
  backend_type: "pytorch"
  port: 8000
  workers: 1

monitoring:
  wandb:
    enabled: false
  mlflow:
    enabled: true
    tracking_uri: "./dev_mlruns"
```

### Production Configuration

```yaml
# config/production.yaml
model:
  name: "Qwen/Qwen2.5-7B"
  device: "auto"
  torch_dtype: "bfloat16"

training:
  training_type: "dpo"
  output_dir: "/data/models/trained"
  
  dpo:
    num_train_epochs: 3
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 8
    learning_rate: 1e-6
    beta: 0.1

data:
  train_data_path: "/data/train"
  eval_data_path: "/data/eval"
  max_length: 2048
  batch_size: 10000

serving:
  backend_type: "vllm"
  port: 8000
  workers: 4
  
  vllm:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
    max_model_len: 2048
    max_num_seqs: 256

monitoring:
  wandb:
    enabled: true
    project: "llm-production"
  
  logging:
    level: "WARNING"
    file: "/var/log/llm-chatbot.log"
```

### Training-Only Configuration

```yaml
# config/training.yaml
model:
  name: "Qwen/Qwen2.5-3B"
  device: "auto"
  torch_dtype: "bfloat16"

training:
  training_type: "lora"
  output_dir: "./models/lora_trained"
  
  lora:
    enabled: true
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    num_train_epochs: 5
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4
    learning_rate: 1e-4
    warmup_steps: 100
    
    eval_strategy: "steps"
    eval_steps: 200
    save_strategy: "steps"
    save_steps: 500
    logging_steps: 10

data:
  train_data_path: "./data/instruction_train.jsonl"
  eval_data_path: "./data/instruction_eval.jsonl"
  max_length: 2048
  text_column: "text"

monitoring:
  wandb:
    enabled: true
    project: "lora-training"
    tags: ["lora", "instruction-tuning"]
  
  metrics:
    enabled: true
    log_interval: 10
```

## ✅ Best Practices

### Configuration Management

1. **Environment-Specific Configs**: Use separate configuration files for different environments
2. **Secret Management**: Store sensitive information in environment variables
3. **Version Control**: Track configuration changes in version control
4. **Validation**: Always validate configuration before training/serving
5. **Documentation**: Document custom configuration parameters

### Performance Optimization

1. **Batch Size**: Start with small batch sizes and increase gradually
2. **Gradient Accumulation**: Use gradient accumulation for effective larger batch sizes
3. **Mixed Precision**: Enable bf16 or fp16 for faster training
4. **Gradient Checkpointing**: Enable for memory-constrained environments
5. **Data Loading**: Optimize num_workers for your system

### Memory Management

1. **GPU Memory**: Monitor GPU memory usage and adjust accordingly
2. **Sequence Length**: Use appropriate max_length for your use case
3. **Model Size**: Choose model size based on available resources
4. **Quantization**: Consider quantization for inference

## 🔧 Troubleshooting

### Common Configuration Issues

#### 1. Invalid Configuration

```bash
# Validate configuration
python -c "from src.core.config import ConfigManager; config = ConfigManager('config.yaml'); print('Configuration valid')"
```

#### 2. Environment Variable Issues

```bash
# Check environment variables
env | grep -E "(MODEL|TRAINING|SERVING|WANDB|CUDA)"

# Load .env file
source .env
```

#### 3. Path Issues

```bash
# Check paths exist
ls -la ./data/train
ls -la ./models
mkdir -p ./outputs ./logs ./cache
```

#### 4. Permission Issues

```bash
# Fix permissions
chmod -R 755 ./data ./models ./outputs
chown -R $USER:$USER ./
```

### Configuration Debugging

```python
# Debug configuration loading
from src.core.config import ConfigManager
import yaml

# Load and print configuration
config = ConfigManager('config.yaml')
print("Model config:", config.model)
print("Training config:", config.training)
print("Data config:", config.data)

# Check for missing values
if not config.model.name:
    print("Warning: Model name not set")
if not config.data.train_data_path:
    print("Warning: Training data path not set")
```

---

**Next Steps**: After configuration, proceed to the [Quick Start Guide](./quick-start.md) to begin using the system.