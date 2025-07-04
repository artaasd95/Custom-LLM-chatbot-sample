# Deployment Guide

Comprehensive deployment guide for the Custom LLM Chatbot system, covering various deployment scenarios from development to production environments.

## 📚 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Local Development Deployment](#local-development-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### Hardware Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 8GB VRAM (optional but recommended)
- **Storage**: 50GB free space
- **Network**: Stable internet connection

#### Recommended Requirements (Production)
- **CPU**: 16+ cores, 3.0GHz+
- **RAM**: 64GB+
- **GPU**: NVIDIA A100/H100 or multiple RTX 4090s
- **Storage**: 500GB+ NVMe SSD
- **Network**: High-bandwidth, low-latency connection

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU acceleration)
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.25+ (for orchestrated deployment)
- **Git**: For source code management

### NVIDIA GPU Setup

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

# Verify installation
nvidia-smi
nvcc --version
```

## 🌍 Environment Setup

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Model Configuration
MODEL_NAME=Qwen/Qwen2.5-3B
MODEL_PATH=/models/qwen2.5-3b
MAX_MODEL_LEN=2048

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
WORKERS=4
BACKEND_TYPE=vllm

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9

# Authentication
API_KEY=your-secure-api-key
JWT_SECRET=your-jwt-secret
AUTH_ENABLED=true

# Monitoring
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=llm-deployment
MLFLOW_TRACKING_URI=http://localhost:5000

# Database
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/llm_db

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
```

### Configuration Files

**config/production.yaml**:
```yaml
model:
  name: "Qwen/Qwen2.5-3B"
  type: "causal_lm"
  max_length: 2048
  device: "auto"

serving:
  backend_type: "vllm"
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
  vllm:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
    max_model_len: 2048
    max_num_seqs: 256
    enable_streaming: true

auth:
  enabled: true
  auth_type: "bearer_token"
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    tokens_per_minute: 100000

monitoring:
  experiment_tracking:
    platform: "wandb"
    project: "llm-production"
  
  metrics:
    enabled: true
    endpoint: "/metrics"
    prometheus_gateway: "http://prometheus:9091"

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/llm-chatbot.log"
```

## 🏠 Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/artaasd95/Custom-LLM-chatbot-sample.git
cd custom-llm-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_NAME="Qwen/Qwen2.5-3B"
export BACKEND_TYPE="pytorch"
export SERVER_PORT="8000"

# Start the server
python serve.py --config config/development.yaml
```

### Development with Hot Reload

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Start with auto-reload
uvicorn src.serving.api_server:app --reload --host 0.0.0.0 --port 8000
```

### Testing the Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'

# Start Streamlit UI
streamlit run streamlit_app.py --server.port 8501
```

## 🐳 Docker Deployment

### Single Container Deployment

**Dockerfile**:
```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 llmuser && chown -R llmuser:llmuser /app
USER llmuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "serve.py", "--config", "config/production.yaml"]
```

**Build and Run**:
```bash
# Build image
docker build -t custom-llm-chatbot:latest .

# Run container
docker run -d \
  --name llm-chatbot \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_NAME="Qwen/Qwen2.5-3B" \
  -e BACKEND_TYPE="vllm" \
  -e TENSOR_PARALLEL_SIZE="2" \
  custom-llm-chatbot:latest

# Check logs
docker logs -f llm-chatbot

# Stop container
docker stop llm-chatbot
```

### Multi-Container Deployment with Docker Compose

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  llm-server:
    build: .
    container_name: llm-server
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=Qwen/Qwen2.5-3B
      - BACKEND_TYPE=vllm
      - TENSOR_PARALLEL_SIZE=2
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:password@postgres:5432/llm_db
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    container_name: llm-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    container_name: llm-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=llm_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: llm-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - llm-server
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: llm-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: llm-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: llm-network
```

**Start the Stack**:
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f llm-server

# Scale the LLM server
docker-compose up -d --scale llm-server=3

# Stop all services
docker-compose down
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-chatbot
  labels:
    name: llm-chatbot
```

**configmap.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-config
  namespace: llm-chatbot
data:
  config.yaml: |
    model:
      name: "Qwen/Qwen2.5-3B"
      type: "causal_lm"
      max_length: 2048
    
    serving:
      backend_type: "vllm"
      host: "0.0.0.0"
      port: 8000
      
      vllm:
        tensor_parallel_size: 2
        gpu_memory_utilization: 0.9
        max_model_len: 2048
    
    monitoring:
      experiment_tracking:
        platform: "wandb"
        project: "llm-k8s"
```

### Secrets

**secrets.yaml**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-secrets
  namespace: llm-chatbot
type: Opaque
data:
  api-key: eW91ci1hcGkta2V5LWhlcmU=  # base64 encoded
  jwt-secret: eW91ci1qd3Qtc2VjcmV0LWhlcmU=  # base64 encoded
  wandb-api-key: eW91ci13YW5kYi1rZXktaGVyZQ==  # base64 encoded
```

### Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-chatbot
  namespace: llm-chatbot
  labels:
    app: llm-chatbot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-chatbot
  template:
    metadata:
      labels:
        app: llm-chatbot
    spec:
      containers:
      - name: llm-chatbot
        image: custom-llm-chatbot:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "Qwen/Qwen2.5-3B"
        - name: BACKEND_TYPE
          value: "vllm"
        - name: TENSOR_PARALLEL_SIZE
          value: "2"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: api-key
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: wandb-api-key
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: model-storage
          mountPath: /app/models
        - name: logs-storage
          mountPath: /app/logs
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "2"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
      - name: config-volume
        configMap:
          name: llm-config
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: logs-storage
        persistentVolumeClaim:
          claimName: logs-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service and Ingress

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-chatbot-service
  namespace: llm-chatbot
  labels:
    app: llm-chatbot
spec:
  selector:
    app: llm-chatbot
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
```

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-chatbot-ingress
  namespace: llm-chatbot
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: llm-chatbot-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-chatbot-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

**hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-chatbot-hpa
  namespace: llm-chatbot
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-chatbot
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n llm-chatbot
kubectl get services -n llm-chatbot
kubectl get ingress -n llm-chatbot

# View logs
kubectl logs -f deployment/llm-chatbot -n llm-chatbot

# Scale deployment
kubectl scale deployment llm-chatbot --replicas=5 -n llm-chatbot

# Update deployment
kubectl set image deployment/llm-chatbot llm-chatbot=custom-llm-chatbot:v2.0.0 -n llm-chatbot
```

## ☁️ Cloud Platform Deployment

### AWS Deployment

#### EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster \
  --name llm-chatbot-cluster \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy application
kubectl apply -f k8s/
```

#### ECS Deployment

**task-definition.json**:
```json
{
  "family": "llm-chatbot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "llm-chatbot",
      "image": "your-account.dkr.ecr.us-west-2.amazonaws.com/llm-chatbot:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_NAME",
          "value": "Qwen/Qwen2.5-3B"
        },
        {
          "name": "BACKEND_TYPE",
          "value": "vllm"
        }
      ],
      "secrets": [
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:account:secret:llm-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-chatbot",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Google Cloud Platform Deployment

#### GKE Deployment

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create llm-chatbot-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 1 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5 \
  --accelerator type=nvidia-tesla-v100,count=2 \
  --enable-autorepair \
  --enable-autoupgrade

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy application
kubectl apply -f k8s/
```

#### Cloud Run Deployment

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/llm-chatbot .

# Deploy to Cloud Run
gcloud run deploy llm-chatbot \
  --image gcr.io/PROJECT_ID/llm-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --concurrency 10 \
  --set-env-vars MODEL_NAME="Qwen/Qwen2.5-3B",BACKEND_TYPE="pytorch"
```

### Azure Deployment

#### AKS Deployment

```bash
# Create resource group
az group create --name llm-chatbot-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group llm-chatbot-rg \
  --name llm-chatbot-cluster \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group llm-chatbot-rg --name llm-chatbot-cluster

# Deploy application
kubectl apply -f k8s/
```

## 🔒 Production Considerations

### Security Hardening

#### Container Security

```dockerfile
# Use minimal base image
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

# Set security options
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove unnecessary packages
RUN apt-get autoremove -y && apt-get autoclean

# Set file permissions
COPY --chown=llmuser:llmuser . /app
WORKDIR /app
USER llmuser

# Use specific versions
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
```

#### Network Security

```yaml
# NetworkPolicy for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-chatbot-netpol
  namespace: llm-chatbot
spec:
  podSelector:
    matchLabels:
      app: llm-chatbot
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Performance Optimization

#### Resource Limits

```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "2"
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "2"
```

#### Caching Strategy

```yaml
# Redis for caching
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
```

### High Availability

#### Multi-Region Deployment

```yaml
# Global load balancer configuration
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: llm-chatbot-ssl
spec:
  domains:
    - api.yourdomain.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-chatbot-global-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: llm-chatbot-ip
    networking.gke.io/managed-certificates: llm-chatbot-ssl
    kubernetes.io/ingress.class: gce
spec:
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: llm-chatbot-service
            port:
              number: 80
```

#### Disaster Recovery

```bash
#!/bin/bash
# Backup script

# Backup models
gsutil -m cp -r /app/models gs://llm-chatbot-backup/models/$(date +%Y%m%d)

# Backup configuration
kubectl get configmap llm-config -o yaml > config-backup-$(date +%Y%m%d).yaml
gsutil cp config-backup-$(date +%Y%m%d).yaml gs://llm-chatbot-backup/config/

# Backup database
pg_dump $DATABASE_URL | gzip > db-backup-$(date +%Y%m%d).sql.gz
gsutil cp db-backup-$(date +%Y%m%d).sql.gz gs://llm-chatbot-backup/database/
```

## 📊 Monitoring and Observability

### Prometheus Configuration

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "llm_rules.yml"

scrape_configs:
  - job_name: 'llm-chatbot'
    static_configs:
      - targets: ['llm-chatbot-service:80']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana Dashboard

**dashboard.json** (excerpt):
```json
{
  "dashboard": {
    "title": "LLM Chatbot Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Memory Issues

```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce GPU memory utilization
export GPU_MEMORY_UTILIZATION=0.7
```

#### Model Loading Issues

```bash
# Check model files
ls -la /app/models/

# Verify model format
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')"

# Check disk space
df -h
```

#### Network Connectivity

```bash
# Test internal connectivity
kubectl exec -it pod-name -- curl http://llm-chatbot-service/health

# Check DNS resolution
nslookup llm-chatbot-service

# Verify ingress
curl -H "Host: api.yourdomain.com" http://ingress-ip/health
```

### Debugging Commands

```bash
# Kubernetes debugging
kubectl describe pod pod-name -n llm-chatbot
kubectl logs pod-name -n llm-chatbot --previous
kubectl exec -it pod-name -n llm-chatbot -- /bin/bash

# Docker debugging
docker logs container-name
docker exec -it container-name /bin/bash
docker stats container-name

# System debugging
top -p $(pgrep -f "python serve.py")
strace -p $(pgrep -f "python serve.py")
lsof -p $(pgrep -f "python serve.py")
```

This comprehensive deployment guide covers all aspects of deploying the Custom LLM Chatbot system from development to production environments across various platforms.