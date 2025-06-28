# Custom LLM Chatbot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](./docs/README.md)

A comprehensive, production-ready Large Language Model (LLM) chatbot system with advanced training capabilities, multiple serving backends, and enterprise-grade monitoring.

## 🌟 Key Features

### 🎯 **Training & Fine-tuning**
- **Multiple Training Methods**: From-scratch training, Supervised Fine-Tuning (SFT), LoRA, and Direct Preference Optimization (DPO)
- **Advanced Optimizations**: Memory-efficient training with gradient checkpointing, mixed precision, and DeepSpeed integration
- **Flexible Data Processing**: Support for various data formats with built-in preprocessing and validation
- **Experiment Tracking**: Integrated support for Weights & Biases, MLflow, and Comet ML

### 🚀 **High-Performance Serving**
- **Multiple Backends**: PyTorch, ONNX Runtime, and vLLM for optimal performance
- **Production-Ready API**: FastAPI-based REST API with OpenAI compatibility
- **Real-time Streaming**: WebSocket and Server-Sent Events support
- **Scalable Architecture**: Load balancing, request batching, and GPU optimization

### 📊 **Monitoring & Observability**
- **Comprehensive Metrics**: Performance, resource utilization, and business metrics
- **Real-time Dashboard**: Streamlit-based monitoring interface
- **Health Checks**: Automated system health monitoring and alerting
- **Prometheus Integration**: Industry-standard metrics collection

### 🔧 **Developer Experience**
- **Modular Design**: Clean, extensible architecture with clear separation of concerns
- **Configuration Management**: YAML-based configuration with validation
- **Comprehensive Testing**: Unit tests, integration tests, and performance benchmarks
- **Rich Documentation**: Detailed guides, API reference, and examples

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

### Web UI (Streamlit)

The system includes a modern web interface built with Streamlit for easy interaction with your trained models.

#### Quick Start
```bash
# Start the LLM server first
python serve.py --server-type vllm --model-path your-model-path

# In another terminal, start the UI
python run_ui.py
```

The UI will be available at `http://localhost:8501`

#### Advanced UI Usage
```bash
# Custom host and port
python run_ui.py --host 0.0.0.0 --port 8502

# Connect to remote LLM server
python run_ui.py --server-url http://remote-server:8000

# Use dark theme
python run_ui.py --theme dark

# Auto-open browser
python run_ui.py --browser
```

#### UI Features

- **Real-time Chat Interface**: Interactive conversation with your LLM
- **Server Status Monitoring**: Live server health and performance metrics
- **Generation Controls**: Adjust temperature, top-p, top-k, and other parameters
- **Prompt Templates**: Pre-configured templates for different use cases:
  - General Chat
  - Code Assistant
  - Creative Writing
  - Question Answering
  - Summarization
- **Conversation Export**: Save chat history as JSON
- **Responsive Design**: Works on desktop and mobile devices

#### UI Configuration

Customize the UI behavior by editing `ui_config.yaml`:

```yaml
# Server connection
server:
  base_url: "http://localhost:8000"
  timeout: 60

# UI settings
ui:
  title: "Custom LLM Chatbot"
  page_icon: "🤖"
  layout: "wide"

# Default generation parameters
generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1

# Custom prompt templates
prompt_templates:
  - name: "Code Assistant"
    template: "You are a helpful coding assistant. Please help with: {user_input}"
    description: "Get help with coding tasks"
```

#### Alternative UI Files

- `streamlit_ui.py`: Basic UI with core functionality
- `streamlit_app.py`: Enhanced UI with configuration support and advanced features

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