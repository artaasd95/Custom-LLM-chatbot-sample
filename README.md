# Custom LLM Chatbot

A comprehensive, scalable system for training and serving custom Large Language Models (LLMs) with support for multiple training methods, high-performance inference, and production-ready deployment.

## 🚀 Features

### Training Capabilities
- **Multiple Training Methods**: Fine-tuning, LoRA, DPO, and training from scratch
- **Distributed Training**: Multi-GPU support with Accelerate and DeepSpeed
- **Advanced Optimizations**: Gradient checkpointing, mixed precision, memory optimization
- **Experiment Tracking**: Integration with Weights & Biases, Comet ML, and MLflow
- **Comprehensive Metrics**: Training loss, perplexity, BLEU, ROUGE, and custom metrics

### Serving & Inference
- **Multiple Backends**: PyTorch, ONNX Runtime, and vLLM for high-performance serving
- **API Server**: FastAPI-based REST API with streaming support
- **Load Balancing**: Built-in request queuing and concurrent processing
- **Performance Monitoring**: Real-time metrics and health checks

### Data Processing
- **Multi-format Support**: JSON, JSONL, CSV, TXT, PDF, DOCX
- **Flexible Data Structures**: Text, instruction, conversation, and DPO formats
- **Preprocessing Pipeline**: Text cleaning, chunking, and normalization
- **Data Validation**: Automatic format detection and validation

### Production Ready
- **Configuration Management**: YAML-based configuration with validation
- **Logging & Monitoring**: Comprehensive logging and experiment tracking
- **Error Handling**: Robust error handling and recovery mechanisms
- **Security**: Best practices for API security and model protection

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU training/inference)
- 16GB+ RAM (32GB+ recommended for large models)
- 50GB+ storage space

### Dependencies
See `requirements.txt` for complete list. Key dependencies:
- PyTorch 2.0+
- Transformers 4.30+
- Datasets
- Accelerate
- PEFT (for LoRA)
- FastAPI
- vLLM (optional, for high-performance serving)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Custom-LLM-chatbot-sample
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install optional dependencies**:
   ```bash
   # For vLLM high-performance serving
   pip install vllm
   
   # For ONNX runtime
   pip install onnxruntime-gpu  # or onnxruntime for CPU
   
   # For experiment tracking
   pip install wandb comet-ml mlflow
   ```

## ⚙️ Configuration

The system uses a YAML configuration file (`config.yaml`) for all settings. Key sections:

### Model Configuration
```yaml
model:
  model_name: "microsoft/DialoGPT-medium"  # HuggingFace model or local path
  tokenizer_name: null  # Uses model_name if null
  cache_dir: "./models"
  trust_remote_code: false
```

### Training Configuration
```yaml
training:
  training_type: "fine_tune"  # fine_tune, lora, dpo, from_scratch
  num_epochs: 3
  batch_size: 4
  learning_rate: 5e-5
  output_dir: "./outputs"
  
  # LoRA settings
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
```

### Serving Configuration
```yaml
serving:
  backend_type: "pytorch"  # pytorch, onnx, vllm
  
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 1
    max_concurrent_requests: 10
  
  vllm:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 2048
```

## 🎯 Usage

### Training

#### Basic Training
```bash
# Fine-tune a model
python train.py --config config.yaml --training-type fine_tune

# LoRA training
python train.py --config config.yaml --training-type lora

# DPO training
python train.py --config config.yaml --training-type dpo
```

#### Advanced Training Options
```bash
# Custom parameters
python train.py \
  --config config.yaml \
  --model-name "microsoft/DialoGPT-large" \
  --data-path "./data/train.jsonl" \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 3e-5 \
  --output-dir "./custom_output"

# Resume from checkpoint
python train.py \
  --config config.yaml \
  --resume-from-checkpoint "./outputs/checkpoint-1000"

# Debug mode (reduced data)
python train.py --config config.yaml --debug

# Evaluation only
python train.py --config config.yaml --eval-only
```

### Serving

#### Start API Server
```bash
# Basic serving
python serve.py --config config.yaml

# Custom host/port
python serve.py --config config.yaml --host 0.0.0.0 --port 8080

# vLLM backend for high performance
python serve.py --config config.yaml --server-type vllm

# Development mode with auto-reload
python serve.py --config config.yaml --reload --debug
```

#### Health Check
```bash
# Check server health
python serve.py --config config.yaml --health-check-only
```

### API Usage

Once the server is running, you can interact with it via REST API:

#### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Streaming Generation
```bash
curl -X POST "http://localhost:8000/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a story about",
    "max_tokens": 200,
    "temperature": 0.8
  }'
```

#### Server Status
```bash
# Health check
curl "http://localhost:8000/health"

# Server statistics
curl "http://localhost:8000/stats"
```

## 📊 Data Formats

The system supports multiple data formats for training:

### Text Format
```json
{"text": "This is a sample text for training."}
```

### Instruction Format
```json
{
  "instruction": "Translate the following text to French:",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

### Conversation Format
```json
{
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### DPO Format
```json
{
  "prompt": "What is the best programming language?",
  "chosen": "Python is great for beginners and has extensive libraries.",
  "rejected": "Assembly is the only real programming language."
}
```

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
src/
├── core/                 # Core functionality
│   ├── config.py        # Configuration management
│   └── model_manager.py # Model loading and management
├── data/                # Data processing
│   └── data_processor.py
├── training/            # Training pipeline
│   ├── trainer.py       # Main training orchestrator
│   ├── dpo_trainer.py   # DPO-specific training
│   └── metrics.py       # Training metrics
├── serving/             # Inference serving
│   ├── model_server.py  # Model serving backend
│   ├── api_server.py    # FastAPI REST API
│   └── vllm_server.py   # vLLM high-performance serving
└── monitoring/          # Experiment tracking
    └── experiment_tracker.py
```

### Key Components

1. **ConfigManager**: Centralized configuration management with validation
2. **ModelManager**: Model loading, LoRA application, and management
3. **DataProcessor**: Multi-format data loading and preprocessing
4. **TrainingOrchestrator**: Main training pipeline coordinator
5. **ModelServer**: Inference backend with multiple engine support
6. **APIServer**: REST API with FastAPI
7. **ExperimentTracker**: Multi-platform experiment tracking

## 🔧 Advanced Features

### Distributed Training
```bash
# Multi-GPU training with Accelerate
accelerate config  # Configure accelerate
accelerate launch train.py --config config.yaml

# DeepSpeed integration
deepspeed train.py --config config.yaml --deepspeed ds_config.json
```

### Custom Data Processing
```python
from src.data.data_processor import DataProcessor
from src.core.config import ConfigManager

config = ConfigManager()
processor = DataProcessor(config)

# Load and process custom data
data = processor.load_data("path/to/data")
processed = processor.process_data(data, "instruction")
```

### Custom Training Loop
```python
from src.training.trainer import TrainingOrchestrator
from src.core.config import ConfigManager

config = ConfigManager()
config.load_config("config.yaml")

trainer = TrainingOrchestrator(config)
trainer.prepare_training()
results = trainer.train()
```

## 📈 Monitoring & Logging

### Experiment Tracking
The system integrates with multiple experiment tracking platforms:

- **Weights & Biases**: Set `WANDB_API_KEY` environment variable
- **Comet ML**: Set `COMET_API_KEY` environment variable
- **MLflow**: Automatic local tracking, configure for remote

### Logging
Comprehensive logging at multiple levels:
- Training progress and metrics
- Model loading and serving status
- API request/response logging
- Error tracking and debugging

### Metrics
- Training: Loss, perplexity, learning rate, gradient norms
- Evaluation: BLEU, ROUGE, METEOR, accuracy
- Serving: Request latency, throughput, error rates
- System: GPU/CPU usage, memory consumption

## 🚀 Performance Optimization

### Training Optimizations
- **Mixed Precision**: Automatic with Accelerate
- **Gradient Checkpointing**: Reduce memory usage
- **DataLoader Optimization**: Multi-worker data loading
- **LoRA**: Parameter-efficient fine-tuning

### Serving Optimizations
- **vLLM Backend**: High-throughput inference
- **ONNX Runtime**: Optimized inference engine
- **Request Batching**: Automatic batching for efficiency
- **Caching**: Model and tokenizer caching

## 🔒 Security

- API rate limiting and authentication ready
- Model file validation
- Input sanitization
- Secure configuration management
- No hardcoded secrets

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use LoRA instead of full fine-tuning

2. **Model Loading Errors**:
   - Check model path and permissions
   - Verify HuggingFace token for private models
   - Ensure sufficient disk space

3. **Training Convergence Issues**:
   - Adjust learning rate
   - Check data quality and format
   - Monitor gradient norms

4. **API Server Issues**:
   - Check port availability
   - Verify model loading
   - Review server logs

### Debug Mode
```bash
# Enable debug logging
python train.py --config config.yaml --log-level DEBUG
python serve.py --config config.yaml --log-level DEBUG --debug
```

## 📚 Examples

See the `examples/` directory for:
- Training scripts for different scenarios
- API client examples
- Custom data processing examples
- Deployment configurations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers team
- vLLM development team
- FastAPI framework
- PyTorch community

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions

---

**Happy Training and Serving! 🚀**