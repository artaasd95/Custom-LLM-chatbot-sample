# Installation Guide

Comprehensive installation guide for the Custom LLM Chatbot system across different platforms and environments.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Development Installation](#development-installation)
- [Docker Installation](#docker-installation)
- [GPU Setup](#gpu-setup)
- [Environment Configuration](#environment-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## 🖥️ System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **Python**: 3.9 or higher
- **RAM**: 16GB (32GB recommended for training)
- **Storage**: 50GB free space
- **Internet**: Stable connection for model downloads

### Recommended Requirements

- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ (64GB for large models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 200GB+ NVMe SSD
- **CUDA**: 11.8+ (for GPU acceleration)

### GPU Requirements

| Model Size | Minimum VRAM | Recommended VRAM | Training VRAM |
|------------|---------------|------------------|---------------|
| **3B parameters** | 6GB | 8GB | 16GB |
| **7B parameters** | 14GB | 16GB | 32GB |
| **13B parameters** | 26GB | 32GB | 64GB |
| **70B parameters** | 140GB | 160GB | 320GB |

## ⚡ Quick Installation

### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/artaasd95/Custom-LLM-chatbot-sample.git
cd custom-llm-chatbot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Run basic test
python -m pytest tests/test_project_structure.py -v
```

## 🛠️ Development Installation

### Full Development Setup

```bash
# Clone with development branch
git clone -b develop https://github.com/artaasd95/Custom-LLM-chatbot-sample.git
cd custom-llm-chatbot

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run full test suite
pytest tests/ -v --cov=src
```

### Development Tools

```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
mypy src/

# Documentation
sphinx-build -b html docs/ docs/_build/
```

## 🐳 Docker Installation

### Using Pre-built Image

```bash
# Pull the latest image
docker pull custom-llm-chatbot:latest

# Run container with GPU support
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name llm-chatbot \
  custom-llm-chatbot:latest
```

### Building from Source

```bash
# Build the image
docker build -t custom-llm-chatbot:local .

# Run with custom configuration
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -e MODEL_NAME="Qwen/Qwen2.5-3B" \
  -e BACKEND_TYPE="vllm" \
  custom-llm-chatbot:local
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🎮 GPU Setup

### NVIDIA GPU Setup (Linux)

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-535

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Test PyTorch GPU
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name()}')"
```

### Windows GPU Setup

1. **Install NVIDIA Drivers**:
   - Download from [NVIDIA website](https://www.nvidia.com/drivers/)
   - Install the latest Game Ready or Studio drivers

2. **Install CUDA Toolkit**:
   - Download CUDA 11.8+ from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation wizard

3. **Verify Installation**:
   ```cmd
   nvidia-smi
   nvcc --version
   ```

## ⚙️ Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model Configuration
MODEL_NAME=Qwen/Qwen2.5-3B
MODEL_PATH=./models
MAX_MODEL_LEN=2048

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
BACKEND_TYPE=pytorch

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

# Training Configuration
OUTPUT_DIR=./outputs
LOGGING_DIR=./logs
CACHE_DIR=./cache

# Monitoring
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=llm-training

# Logging
LOG_LEVEL=INFO
```

### Configuration Files

Copy and customize configuration templates:

```bash
# Copy example configuration
cp .env.example .env
cp config/config.example.yaml config/config.yaml

# Edit configuration
nano config/config.yaml
```

## ✅ Verification

### Basic Functionality Test

```bash
# Test configuration loading
python -c "from src.core.config import ConfigManager; config = ConfigManager(); print('Config loaded successfully')"

# Test model loading
python -c "from src.core.model_manager import ModelManager; from src.core.config import ConfigManager; config = ConfigManager(); manager = ModelManager(config); print('Model manager initialized')"

# Run health check
python -c "from src.serving.api_server import APIServer; from src.core.config import ConfigManager; config = ConfigManager(); server = APIServer(config); print('API server initialized')"
```

### Full System Test

```bash
# Run all tests
pytest tests/ -v

# Test specific components
pytest tests/test_config.py -v
pytest tests/test_serving.py -v
pytest tests/test_training.py -v
```

### Performance Test

```bash
# Start server
python serve.py --config config/config.yaml &

# Wait for startup
sleep 10

# Test inference
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'

# Stop server
kill %1
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=true

# Use smaller model
export MODEL_NAME="Qwen/Qwen2.5-1.5B"
```

#### 2. Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
pip install -e .
```

#### 3. Model Download Issues

```bash
# Set HuggingFace cache directory
export HF_HOME=./cache/huggingface

# Use mirror (if needed)
export HF_ENDPOINT=https://hf-mirror.com

# Manual download
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B')"
```

#### 4. Permission Issues (Linux/macOS)

```bash
# Fix permissions
sudo chown -R $USER:$USER ./
chmod +x *.sh

# Use virtual environment
python -m venv venv
source venv/bin/activate
pip install --user -r requirements.txt
```

### Getting Help

- **Documentation**: Check the [docs](./README.md) for detailed guides
- **Issues**: Report bugs on [GitHub Issues](https://github.com/artaasd95/Custom-LLM-chatbot-sample/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/artaasd95/Custom-LLM-chatbot-sample/discussions)
- **Community**: Join our [Discord server](https://discord.gg/your-server)

### System Information

Collect system information for troubleshooting:

```bash
# Create system info script
cat > system_info.py << 'EOF'
import sys
import torch
import platform
import subprocess

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print("\nNVIDIA-SMI:")
    print(result.stdout)
except:
    print("nvidia-smi not available")
EOF

python system_info.py
```

---

**Next Steps**: After installation, proceed to the [Configuration Guide](./configuration.md) to set up your system for training and serving.