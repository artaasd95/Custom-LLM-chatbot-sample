# User Guide

Comprehensive guide for using the Custom LLM Chatbot system, from basic setup to advanced features.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training Your Model](#training-your-model)
- [Running Inference](#running-inference)
- [Using the Web Interface](#using-the-web-interface)
- [API Usage](#api-usage)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## 🚀 Getting Started

### What is Custom LLM Chatbot?

The Custom LLM Chatbot is a comprehensive system for training, fine-tuning, and deploying large language models. It supports various training methods including:

- **From-scratch training**: Train models from the ground up
- **Supervised Fine-tuning (SFT)**: Fine-tune pre-trained models on your data
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with minimal parameters
- **DPO (Direct Preference Optimization)**: Align models with human preferences

### Key Features

- 🎯 **Multiple Training Methods**: Support for various training approaches
- 🚀 **High-Performance Serving**: Optimized inference with vLLM, ONNX, and PyTorch
- 📊 **Comprehensive Monitoring**: Built-in experiment tracking and metrics
- 🔧 **Easy Configuration**: YAML-based configuration system
- 🌐 **Web Interface**: User-friendly Streamlit interface
- 🔌 **REST API**: Complete API for integration
- 📈 **Production Ready**: Scalable deployment options

### Prerequisites

- **Python 3.9+**
- **CUDA 11.8+** (for GPU acceleration)
- **16GB+ RAM** (32GB+ recommended)
- **NVIDIA GPU** with 8GB+ VRAM (optional but recommended)

## 📦 Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/artaasd95/Custom-LLM-chatbot-sample.git
cd custom-llm-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .

# Run tests to verify installation
pytest tests/
```

### Docker Installation

```bash
# Pull pre-built image
docker pull custom-llm-chatbot:latest

# Or build from source
docker build -t custom-llm-chatbot:latest .

# Run container
docker run -d --gpus all -p 8000:8000 custom-llm-chatbot:latest
```

## ⚙️ Configuration

### Basic Configuration

Create a configuration file `config/my_config.yaml`:

```yaml
# Basic model configuration
model:
  name: "Qwen/Qwen2.5-3B"  # Hugging Face model name
  type: "causal_lm"         # Model type
  max_length: 2048          # Maximum sequence length
  device: "auto"            # Device selection (auto/cpu/cuda)

# Training configuration
training:
  type: "sft"               # Training type: from_scratch/sft/lora/dpo
  output_dir: "./outputs"   # Output directory
  num_epochs: 3             # Number of training epochs
  batch_size: 4             # Training batch size
  learning_rate: 5e-5       # Learning rate
  save_steps: 500           # Save checkpoint every N steps
  eval_steps: 100           # Evaluate every N steps
  logging_steps: 10         # Log every N steps

# Data configuration
data:
  train_path: "data/train.jsonl"  # Training data path
  eval_path: "data/eval.jsonl"    # Evaluation data path
  data_type: "instruction"        # Data format type
  max_samples: null               # Limit training samples (null = all)

# Serving configuration
serving:
  backend_type: "pytorch"   # Backend: pytorch/onnx/vllm
  host: "0.0.0.0"          # Server host
  port: 8000               # Server port
  workers: 1               # Number of workers

# Monitoring configuration
monitoring:
  experiment_tracking:
    platform: "local"       # Tracking platform: wandb/mlflow/comet/local
    project: "my-llm"       # Project name
  
  metrics:
    enabled: true           # Enable metrics collection
    log_interval: 10        # Metrics logging interval
```

### Advanced Configuration

#### LoRA Configuration

```yaml
model:
  name: "Qwen/Qwen2.5-7B"
  type: "causal_lm"
  max_length: 4096

training:
  type: "lora"
  
  # LoRA-specific settings
  lora:
    r: 16                   # LoRA rank
    alpha: 32               # LoRA alpha
    dropout: 0.1            # LoRA dropout
    target_modules:         # Target modules for LoRA
      - "q_proj"
      - "v_proj"
      - "k_proj"
      - "o_proj"
    bias: "none"            # Bias handling
    task_type: "CAUSAL_LM"  # Task type

  # Training parameters
  num_epochs: 5
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  warmup_steps: 100
  weight_decay: 0.01
  
  # Optimization
  optimizer: "adamw"
  scheduler: "cosine"
  fp16: true              # Mixed precision training
  gradient_checkpointing: true
```

#### DPO Configuration

```yaml
model:
  name: "Qwen/Qwen2.5-3B"
  type: "causal_lm"

training:
  type: "dpo"
  
  # DPO-specific settings
  dpo:
    beta: 0.1               # DPO temperature parameter
    reference_model: null   # Reference model (null = use base model)
    loss_type: "sigmoid"    # Loss function type
    label_smoothing: 0.0    # Label smoothing
    
  # Training parameters
  num_epochs: 3
  batch_size: 2
  learning_rate: 5e-7
  max_length: 2048
  max_prompt_length: 1024

data:
  train_path: "data/dpo_train.jsonl"
  eval_path: "data/dpo_eval.jsonl"
  data_type: "dpo"          # DPO data format
```

#### High-Performance Serving

```yaml
serving:
  backend_type: "vllm"
  host: "0.0.0.0"
  port: 8000
  
  # vLLM-specific settings
  vllm:
    tensor_parallel_size: 2     # Number of GPUs for tensor parallelism
    gpu_memory_utilization: 0.9 # GPU memory utilization
    max_model_len: 4096         # Maximum model length
    max_num_seqs: 256           # Maximum number of sequences
    enable_streaming: true      # Enable streaming responses
    quantization: null          # Quantization method (awq/gptq/null)
    
  # API settings
  api:
    max_tokens: 2048            # Default max tokens
    temperature: 0.7            # Default temperature
    top_p: 0.9                  # Default top-p
    frequency_penalty: 0.0      # Default frequency penalty
    presence_penalty: 0.0       # Default presence penalty
```

## 🎓 Training Your Model

### Preparing Your Data

#### Instruction-Following Format

```jsonl
{"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."}
{"instruction": "Translate to Spanish", "input": "Hello, how are you?", "output": "Hola, ¿cómo estás?"}
{"instruction": "Summarize this text", "input": "Long text here...", "output": "Summary here..."}
```

#### Conversational Format

```jsonl
{"conversations": [{"from": "human", "value": "Hello!"}, {"from": "assistant", "value": "Hi there! How can I help you today?"}]}
{"conversations": [{"from": "human", "value": "What's 2+2?"}, {"from": "assistant", "value": "2+2 equals 4."}]}
```

#### DPO Format

```jsonl
{"prompt": "What is the best programming language?", "chosen": "The best programming language depends on your specific needs and use case.", "rejected": "Python is definitely the best programming language for everything."}
```

### Running Training

#### Basic Training

```bash
# Supervised Fine-tuning
python train.py \
  --config config/sft_config.yaml \
  --model_name "Qwen/Qwen2.5-3B" \
  --train_data "data/train.jsonl" \
  --eval_data "data/eval.jsonl" \
  --output_dir "./outputs/sft_model"
```

#### LoRA Training

```bash
# LoRA fine-tuning
python train.py \
  --config config/lora_config.yaml \
  --training_type "lora" \
  --model_name "Qwen/Qwen2.5-7B" \
  --train_data "data/train.jsonl" \
  --lora_r 16 \
  --lora_alpha 32 \
  --output_dir "./outputs/lora_model"
```

#### DPO Training

```bash
# DPO training
python train.py \
  --config config/dpo_config.yaml \
  --training_type "dpo" \
  --model_name "Qwen/Qwen2.5-3B" \
  --train_data "data/dpo_train.jsonl" \
  --dpo_beta 0.1 \
  --output_dir "./outputs/dpo_model"
```

#### Multi-GPU Training

```bash
# Using accelerate for multi-GPU training
accelerate config  # Configure accelerate settings

accelerate launch train.py \
  --config config/multi_gpu_config.yaml \
  --model_name "Qwen/Qwen2.5-7B" \
  --train_data "data/train.jsonl"
```

### Monitoring Training

#### Using Weights & Biases

```yaml
# Add to your config
monitoring:
  experiment_tracking:
    platform: "wandb"
    project: "my-llm-project"
    entity: "your-username"
    tags: ["sft", "qwen", "experiment-1"]
```

```bash
# Set API key
export WANDB_API_KEY="your-wandb-api-key"

# Run training with W&B logging
python train.py --config config/wandb_config.yaml
```

#### Using MLflow

```yaml
# Add to your config
monitoring:
  experiment_tracking:
    platform: "mlflow"
    tracking_uri: "http://localhost:5000"
    experiment_name: "llm-training"
```

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Run training with MLflow logging
python train.py --config config/mlflow_config.yaml
```

## 🚀 Running Inference

### Starting the Server

#### Basic Server

```bash
# Start PyTorch server
python serve.py \
  --model_path "./outputs/sft_model" \
  --backend_type "pytorch" \
  --host "0.0.0.0" \
  --port 8000
```

#### High-Performance Server

```bash
# Start vLLM server
python serve.py \
  --model_path "./outputs/sft_model" \
  --backend_type "vllm" \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.9 \
  --host "0.0.0.0" \
  --port 8000
```

#### ONNX Server

```bash
# Convert model to ONNX first
python scripts/convert_to_onnx.py \
  --model_path "./outputs/sft_model" \
  --output_path "./outputs/sft_model.onnx"

# Start ONNX server
python serve.py \
  --model_path "./outputs/sft_model.onnx" \
  --backend_type "onnx" \
  --host "0.0.0.0" \
  --port 8000
```

### Testing the Server

```bash
# Health check
curl http://localhost:8000/health

# Basic generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completion
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50
  }'
```

## 🌐 Using the Web Interface

### Starting the Streamlit App

```bash
# Start the web interface
streamlit run streamlit_app.py --server.port 8501
```

### Features

#### Chat Interface
- **Real-time Chat**: Interactive conversation with your model
- **Message History**: Persistent conversation history
- **Model Selection**: Switch between different models
- **Parameter Control**: Adjust temperature, top-p, max tokens

#### Model Management
- **Model Status**: View loaded models and their status
- **Performance Metrics**: Real-time performance statistics
- **Resource Usage**: Monitor GPU and memory usage

#### Configuration
- **Server Settings**: Configure API endpoint and authentication
- **UI Preferences**: Customize interface appearance
- **Export/Import**: Save and load conversation history

### Configuration File

Create `config/streamlit_config.yaml`:

```yaml
ui:
  title: "My Custom LLM Chatbot"
  description: "Powered by Custom LLM Framework"
  theme: "dark"  # light/dark
  
  # Chat settings
  chat:
    max_history: 100
    auto_scroll: true
    show_timestamps: true
    
  # Model settings
  models:
    default_temperature: 0.7
    default_max_tokens: 512
    default_top_p: 0.9

server:
  api_base_url: "http://localhost:8000"
  timeout: 30
  
  # Authentication (if enabled)
  auth:
    enabled: false
    api_key: "your-api-key"
```

## 🔌 API Usage

### Python Client

```python
import requests
import json

class LLMClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def generate(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
        """Generate text from a prompt."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def chat(self, messages, max_tokens=100, temperature=0.7, **kwargs):
        """Chat completion."""
        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def stream_generate(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
        """Stream text generation."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data != '[DONE]':
                        yield json.loads(data)

# Usage examples
client = LLMClient()

# Simple generation
result = client.generate("What is machine learning?")
print(result['text'])

# Chat
messages = [
    {"role": "user", "content": "Hello!"}
]
response = client.chat(messages)
print(response['choices'][0]['message']['content'])

# Streaming
for chunk in client.stream_generate("Tell me a story"):
    print(chunk['text'], end='', flush=True)
```

### JavaScript Client

```javascript
class LLMClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }
    
    async generate(prompt, options = {}) {
        const data = {
            prompt,
            max_tokens: 100,
            temperature: 0.7,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async chat(messages, options = {}) {
        const data = {
            messages,
            max_tokens: 100,
            temperature: 0.7,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async* streamGenerate(prompt, options = {}) {
        const data = {
            prompt,
            max_tokens: 100,
            temperature: 0.7,
            stream: true,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data !== '[DONE]') {
                            yield JSON.parse(data);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }
}

// Usage examples
const client = new LLMClient();

// Simple generation
client.generate('What is artificial intelligence?')
    .then(result => console.log(result.text));

// Chat
const messages = [
    { role: 'user', content: 'Hello!' }
];
client.chat(messages)
    .then(response => console.log(response.choices[0].message.content));

// Streaming
(async () => {
    for await (const chunk of client.streamGenerate('Tell me a joke')) {
        process.stdout.write(chunk.text);
    }
})();
```

## 🔧 Advanced Features

### Custom Data Processing

```python
# Custom data processor
from src.data.data_processor import DataProcessor

class CustomDataProcessor(DataProcessor):
    def preprocess_text(self, text):
        """Custom text preprocessing."""
        # Add your custom preprocessing logic
        text = text.strip()
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        return text
    
    def custom_filter(self, example):
        """Custom filtering logic."""
        # Filter based on length
        if len(example['text']) < 10:
            return False
        
        # Filter based on content
        if any(word in example['text'].lower() for word in ['spam', 'advertisement']):
            return False
        
        return True

# Usage
processor = CustomDataProcessor(config)
processed_data = processor.process_dataset('data/raw_data.jsonl')
```

### Custom Model Architecture

```python
# Custom model configuration
from transformers import AutoConfig, AutoModelForCausalLM

class CustomModelConfig:
    def __init__(self):
        self.model_config = {
            'vocab_size': 32000,
            'hidden_size': 4096,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'intermediate_size': 11008,
            'max_position_embeddings': 4096,
            'rms_norm_eps': 1e-6,
            'tie_word_embeddings': False,
        }
    
    def create_model(self):
        config = AutoConfig.from_dict(self.model_config)
        model = AutoModelForCausalLM.from_config(config)
        return model

# Usage in training
model_config = CustomModelConfig()
custom_model = model_config.create_model()
```

### Custom Training Loop

```python
# Custom training orchestrator
from src.training.trainer import TrainingOrchestrator

class CustomTrainingOrchestrator(TrainingOrchestrator):
    def custom_training_step(self, batch):
        """Custom training step with additional logic."""
        # Standard forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Add custom regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss += 0.01 * l2_reg
        
        # Custom gradient clipping
        if self.config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        return loss
    
    def custom_evaluation(self, eval_dataloader):
        """Custom evaluation with additional metrics."""
        self.model.eval()
        total_loss = 0
        perplexity_scores = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Calculate perplexity
                perplexity = torch.exp(loss)
                perplexity_scores.append(perplexity.item())
        
        avg_loss = total_loss / len(eval_dataloader)
        avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': avg_perplexity
        }

# Usage
trainer = CustomTrainingOrchestrator(config)
trainer.train()
```

### Custom Serving Backend

```python
# Custom serving backend
from src.serving.model_server import ModelServer

class CustomModelServer(ModelServer):
    def __init__(self, config):
        super().__init__(config)
        self.custom_cache = {}
    
    def preprocess_input(self, text):
        """Custom input preprocessing."""
        # Add custom preprocessing
        text = text.strip()
        text = self.apply_custom_formatting(text)
        return text
    
    def postprocess_output(self, text):
        """Custom output postprocessing."""
        # Add custom postprocessing
        text = self.clean_output(text)
        text = self.apply_custom_filters(text)
        return text
    
    def generate_with_cache(self, prompt, **kwargs):
        """Generation with custom caching."""
        cache_key = f"{prompt}_{kwargs}"
        
        if cache_key in self.custom_cache:
            return self.custom_cache[cache_key]
        
        result = self.generate(prompt, **kwargs)
        self.custom_cache[cache_key] = result
        
        return result

# Usage
server = CustomModelServer(config)
result = server.generate_with_cache("Hello, world!")
```

## 📋 Best Practices

### Data Preparation

1. **Quality over Quantity**
   - Use high-quality, diverse training data
   - Remove duplicates and low-quality samples
   - Balance different types of examples

2. **Data Format Consistency**
   - Ensure consistent formatting across all examples
   - Validate data before training
   - Use proper tokenization

3. **Data Privacy**
   - Remove sensitive information
   - Anonymize personal data
   - Follow data protection regulations

### Training Optimization

1. **Hyperparameter Tuning**
   - Start with recommended defaults
   - Use learning rate scheduling
   - Monitor training metrics closely

2. **Memory Management**
   - Use gradient accumulation for large batches
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Monitoring and Debugging**
   - Log training metrics regularly
   - Save checkpoints frequently
   - Monitor for overfitting

### Serving Optimization

1. **Performance Tuning**
   - Use appropriate batch sizes
   - Enable model compilation
   - Optimize memory usage

2. **Scalability**
   - Use load balancing
   - Implement caching strategies
   - Monitor resource usage

3. **Security**
   - Enable authentication
   - Implement rate limiting
   - Validate inputs

## 🔍 Troubleshooting

### Common Training Issues

#### Out of Memory (OOM)

```bash
# Reduce batch size
python train.py --batch_size 2 --gradient_accumulation_steps 8

# Enable gradient checkpointing
python train.py --gradient_checkpointing true

# Use mixed precision
python train.py --fp16 true
```

#### Slow Training

```bash
# Use multiple GPUs
accelerate launch --multi_gpu train.py

# Optimize data loading
python train.py --dataloader_num_workers 4

# Use faster optimizers
python train.py --optimizer adamw_torch_fused
```

#### Poor Model Performance

```yaml
# Adjust learning rate
training:
  learning_rate: 1e-5  # Try different values
  warmup_steps: 500    # Add warmup
  
# Increase training data
data:
  max_samples: null    # Use all available data
  
# Try different training methods
training:
  type: "lora"         # Try LoRA instead of full fine-tuning
```

### Common Serving Issues

#### Model Loading Errors

```bash
# Check model path
ls -la ./outputs/model_name/

# Verify model format
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('./outputs/model_name')"

# Check permissions
chmod -R 755 ./outputs/model_name/
```

#### API Connection Issues

```bash
# Check server status
curl http://localhost:8000/health

# Check port availability
netstat -tulpn | grep 8000

# Check firewall settings
sudo ufw status
```

#### Performance Issues

```yaml
# Optimize serving configuration
serving:
  backend_type: "vllm"  # Use vLLM for better performance
  
  vllm:
    tensor_parallel_size: 2     # Use multiple GPUs
    gpu_memory_utilization: 0.9 # Increase memory usage
    max_num_seqs: 256          # Increase batch size
```

## ❓ FAQ

### General Questions

**Q: What models are supported?**
A: The system supports any Hugging Face compatible model, including Qwen, Llama, Mistral, and custom models.

**Q: Can I use CPU-only training?**
A: Yes, but it will be significantly slower. GPU training is highly recommended.

**Q: How much GPU memory do I need?**
A: Minimum 8GB for small models (3B parameters), 16GB+ for larger models (7B+).

### Training Questions

**Q: Which training method should I use?**
A: 
- **SFT**: For general fine-tuning with instruction data
- **LoRA**: For efficient fine-tuning with limited resources
- **DPO**: For aligning models with human preferences
- **From-scratch**: For training completely new models

**Q: How long does training take?**
A: Depends on model size, data size, and hardware. Typical ranges:
- LoRA: 1-4 hours for small datasets
- SFT: 4-24 hours for medium datasets
- From-scratch: Days to weeks

**Q: How do I know if my model is overfitting?**
A: Monitor the gap between training and validation loss. If validation loss starts increasing while training loss decreases, you're overfitting.

### Serving Questions

**Q: Which backend should I use?**
A:
- **PyTorch**: Good for development and small-scale deployment
- **vLLM**: Best for high-performance production serving
- **ONNX**: Good for CPU inference and edge deployment

**Q: How do I scale my deployment?**
A: Use Kubernetes with horizontal pod autoscaling, or deploy multiple instances behind a load balancer.

**Q: Can I serve multiple models simultaneously?**
A: Yes, you can run multiple server instances on different ports or use a model router.

This comprehensive user guide covers all aspects of using the Custom LLM Chatbot system effectively.