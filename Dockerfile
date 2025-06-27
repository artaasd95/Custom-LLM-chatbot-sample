# Multi-stage Dockerfile for Custom LLM Chatbot
# Supports both training and serving modes

# Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for production
RUN pip install --no-cache-dir \
    gunicorn \
    supervisor \
    redis \
    celery

# Training stage
FROM base as training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    deepspeed \
    flash-attn \
    xformers

# Copy source code
COPY src/ ./src/
COPY train.py .
COPY config.yaml .
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs

# Set permissions
RUN chmod +x train.py

# Default command for training
CMD ["python", "train.py", "--config", "config.yaml"]

# Serving stage
FROM base as serving

# Install serving-specific dependencies
RUN pip install --no-cache-dir \
    vllm \
    onnxruntime-gpu \
    triton

# Copy source code
COPY src/ ./src/
COPY serve.py .
COPY config.yaml .
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/models /app/logs

# Set permissions
RUN chmod +x serve.py

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for serving
CMD ["python", "serve.py", "--config", "config.yaml", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    pre-commit

# Copy all files
COPY . .

# Install the package in development mode
RUN pip install -e .[dev,all]

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/notebooks

# Set permissions
RUN chmod +x train.py serve.py example_usage.py

# Expose ports for development
EXPOSE 8000 8001 8002 8888

# Default command for development
CMD ["bash"]

# Production stage (default)
FROM serving as production

# Copy supervisor configuration
COPY <<EOF /etc/supervisor/conf.d/llm-server.conf
[supervisord]
nodaemon=true
user=root

[program:llm-server]
command=python serve.py --config config.yaml --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stderr_logfile=/app/logs/server.err.log
stdout_logfile=/app/logs/server.out.log
user=root

[program:health-monitor]
command=python -c "import time; import requests; [time.sleep(30) or requests.get('http://localhost:8000/health') for _ in iter(int, 1)]"
directory=/app
autostart=true
autorestart=true
stderr_logfile=/app/logs/health.err.log
stdout_logfile=/app/logs/health.out.log
user=root
EOF

# Use supervisor for production
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]

# Labels
LABEL maintainer="Custom LLM Team <team@customllm.com>"
LABEL version="1.0.0"
LABEL description="Custom LLM Chatbot - Training and Serving Framework"
LABEL org.opencontainers.image.source="https://github.com/your-org/custom-llm-chatbot"
LABEL org.opencontainers.image.documentation="https://custom-llm-chatbot.readthedocs.io/"
LABEL org.opencontainers.image.licenses="MIT"