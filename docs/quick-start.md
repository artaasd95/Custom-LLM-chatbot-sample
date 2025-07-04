# Quick Start Guide

Get up and running with the Custom LLM Chatbot in minutes. This guide covers the essential steps to start training and serving your custom language model.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Basic Setup](#basic-setup)
- [First Training Run](#first-training-run)
- [Model Serving](#model-serving)
- [Testing the API](#testing-the-api)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 50GB free space

### Quick Check

```bash
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi
```

## ⚡ Quick Installation

### Option 1: One-Command Setup

```bash
# Clone and setup in one go
git clone https://github.com/your-org/Custom-LLM-chatbot-sample.git
cd Custom-LLM-chatbot-sample
./scripts/quick_setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/Custom-LLM-chatbot-sample.git
cd Custom-LLM-chatbot-sample

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your settings
```

### Option 3: Docker Setup

```bash
# Quick Docker setup
docker-compose up -d

# Or build from scratch
docker build -t custom-llm-chatbot .
docker run -p 8000:8000 custom-llm-chatbot
```

## 🛠️ Basic Setup

### 1. Environment Configuration

Create and configure your environment file:

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` with your basic settings:

```bash
# Basic Configuration
MODEL_NAME=Qwen/Qwen2.5-1.5B
DEVICE=auto
TRAINING_TYPE=sft
OUTPUT_DIR=./outputs
DATA_PATH=./data

# Optional: Monitoring
WANDB_API_KEY=your-wandb-key
MLFLOW_TRACKING_URI=./mlruns
```

### 2. Data Preparation

Prepare your training data in the correct format:

```bash
# Create data directory
mkdir -p data/train data/eval

# Example: Chat format
cat > data/train/sample.jsonl << EOF
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help you?"}]}
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}]}
EOF

# Example: Instruction format
cat > data/train/instructions.jsonl << EOF
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Summarize this text", "input": "Long text here...", "output": "Summary here..."}
EOF
```

### 3. Configuration File

Create a basic configuration file:

```bash
# Create config directory
mkdir -p config

# Create basic config
cat > config/quick_start.yaml << EOF
model:
  name: "Qwen/Qwen2.5-1.5B"
  device: "auto"
  torch_dtype: "auto"

training:
  training_type: "sft"
  output_dir: "./outputs"
  
  sft:
    num_train_epochs: 1
    per_device_train_batch_size: 2
    learning_rate: 5e-5
    warmup_steps: 100
    eval_steps: 50
    save_steps: 100
    logging_steps: 10

data:
  train_data_path: "./data/train"
  eval_data_path: "./data/eval"
  max_length: 512
  batch_size: 100

serving:
  backend_type: "pytorch"
  host: "0.0.0.0"
  port: 8000

monitoring:
  mlflow:
    enabled: true
    tracking_uri: "./mlruns"
EOF
```

## 🎓 First Training Run

### Quick Training (5 minutes)

```bash
# Start training with minimal configuration
python -m src.training.train \
  --config config/quick_start.yaml \
  --model-name Qwen/Qwen2.5-1.5B \
  --training-type sft \
  --num-epochs 1 \
  --batch-size 2
```

### Monitor Training Progress

```bash
# In another terminal, monitor logs
tail -f logs/training.log

# Or use MLflow UI
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

### Training Output

You should see output like:

```
[INFO] Loading model: Qwen/Qwen2.5-1.5B
[INFO] Model loaded successfully
[INFO] Loading training data from ./data/train
[INFO] Found 100 training samples
[INFO] Starting training...
Epoch 1/1: 100%|██████████| 50/50 [02:30<00:00,  3.00s/it]
[INFO] Training completed!
[INFO] Model saved to ./outputs/checkpoint-100
```

## 🚀 Model Serving

### Start the Server

```bash
# Start serving the trained model
python -m src.serving.serve \
  --config config/quick_start.yaml \
  --model-path ./outputs/checkpoint-100
```

### Alternative: Serve Base Model

```bash
# Serve the base model without training
python -m src.serving.serve \
  --config config/quick_start.yaml \
  --model-name Qwen/Qwen2.5-1.5B
```

### Server Startup

You should see:

```
[INFO] Loading model from ./outputs/checkpoint-100
[INFO] Model loaded successfully
[INFO] Starting API server on http://0.0.0.0:8000
[INFO] Server ready! 🚀
[INFO] API documentation: http://localhost:8000/docs
```

## 🧪 Testing the API

### Health Check

```bash
# Test server health
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Generate Text

```bash
# Simple text generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Expected response:
```json
{
  "generated_text": "Hello, how are you? I'm doing well, thank you for asking! How can I help you today?",
  "tokens_generated": 18,
  "generation_time": 0.85,
  "model_name": "Qwen/Qwen2.5-1.5B"
}
```

### Interactive Testing

```python
# Python client example
import requests

def chat_with_model(prompt, max_tokens=100):
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    return response.json()["generated_text"]

# Test the model
print(chat_with_model("What is machine learning?"))
print(chat_with_model("Explain quantum computing in simple terms."))
```

### Web Interface

Open your browser and visit:

- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Health Status**: http://localhost:8000/health
- **Server Stats**: http://localhost:8000/stats

## 🎯 Next Steps

### 1. Advanced Training

```bash
# Try LoRA fine-tuning
python -m src.training.train \
  --config config/lora_config.yaml \
  --training-type lora

# Try DPO training
python -m src.training.train \
  --config config/dpo_config.yaml \
  --training-type dpo
```

### 2. Production Deployment

```bash
# Use production configuration
cp config/production.yaml config/config.yaml

# Deploy with Docker
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to cloud
./scripts/deploy_azure.sh
```

### 3. Monitoring Setup

```bash
# Setup Weights & Biases
pip install wandb
wandb login

# Enable in config
echo "wandb:\n  enabled: true\n  project: my-llm-project" >> config/config.yaml
```

### 4. Data Pipeline

```bash
# Process your own data
python -m src.data.process \
  --input-dir ./raw_data \
  --output-dir ./data \
  --format chat

# Validate data quality
python -m src.data.validate --data-dir ./data
```

## 📚 Learn More

### Essential Documentation

- **[Configuration Guide](./configuration.md)**: Detailed configuration options
- **[Training Methods](./training-methods.md)**: Different training approaches
- **[Data Processing](./data-processing.md)**: Data preparation and formats
- **[Deployment Guide](./deployment-guide.md)**: Production deployment
- **[API Reference](./api-reference.md)**: Complete API documentation

### Example Projects

```bash
# Clone example projects
git clone https://github.com/your-org/llm-examples.git
cd llm-examples

# Customer service chatbot
cd examples/customer-service
python train.py

# Code assistant
cd examples/code-assistant
python train.py

# Document summarizer
cd examples/document-summarizer
python train.py
```

### Community Resources

- **Discord**: [Join our community](https://discord.gg/llm-chatbot)
- **GitHub Discussions**: [Ask questions](https://github.com/your-org/Custom-LLM-chatbot-sample/discussions)
- **Blog**: [Latest tutorials](https://blog.example.com/llm-tutorials)
- **YouTube**: [Video tutorials](https://youtube.com/llm-tutorials)

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8

# Use smaller model
export MODEL_NAME=Qwen/Qwen2.5-1.5B

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=true
```

#### 2. Model Loading Issues

```bash
# Clear cache
rm -rf ~/.cache/huggingface

# Download model manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B')"

# Check disk space
df -h
```

#### 3. Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
export SERVER_PORT=8001
```

#### 4. Permission Errors

```bash
# Fix permissions
sudo chown -R $USER:$USER ./
chmod -R 755 ./data ./outputs ./logs

# Create directories
mkdir -p data outputs logs cache
```

### Getting Help

```bash
# Check system info
python -m src.utils.system_info

# Run diagnostics
python -m src.utils.diagnostics

# Enable debug logging
export LOG_LEVEL=DEBUG
```

### Performance Tips

1. **Start Small**: Begin with small models and datasets
2. **Monitor Resources**: Watch GPU memory and CPU usage
3. **Batch Size**: Adjust based on available memory
4. **Mixed Precision**: Enable bf16 for faster training
5. **Data Loading**: Optimize num_workers for your system

---

**Congratulations!** 🎉 You've successfully set up and tested the Custom LLM Chatbot. You're now ready to train your own models and build amazing AI applications.

**What's Next?**
- Explore [advanced training methods](./training-methods.md)
- Learn about [production deployment](./deployment-guide.md)
- Join our [community](https://discord.gg/llm-chatbot) for support and discussions