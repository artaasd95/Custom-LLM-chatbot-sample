# API Reference

Complete API documentation for the Custom LLM Chatbot system.

## 📚 Table of Contents

- [Authentication](#authentication)
- [Base URL and Versioning](#base-url-and-versioning)
- [Common Response Formats](#common-response-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Text Generation API](#text-generation-api)
- [Chat API](#chat-api)
- [Model Management API](#model-management-api)
- [Health and Status API](#health-and-status-api)
- [Metrics and Monitoring API](#metrics-and-monitoring-api)
- [WebSocket API](#websocket-api)
- [SDK and Client Libraries](#sdk-and-client-libraries)

## 🔐 Authentication

### Bearer Token Authentication

**Header Format**:
```http
Authorization: Bearer <your-api-token>
```

**Example**:
```bash
curl -H "Authorization: Bearer sk-1234567890abcdef" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8000/generate
```

### API Key Authentication

**Header Format**:
```http
X-API-Key: <your-api-key>
```

**Example**:
```bash
curl -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8000/generate
```

## 🌐 Base URL and Versioning

**Base URL**: `http://localhost:8000` (development) or `https://api.yourcompany.com` (production)

**API Version**: All endpoints support versioning through the `Accept` header:
```http
Accept: application/vnd.api+json;version=1
```

**Default Version**: v1 (if no version specified)

## 📋 Common Response Formats

### Success Response
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid",
    "details": {
      "field": "prompt",
      "issue": "Prompt cannot be empty"
    }
  },
  "metadata": {
    "request_id": "req_def456",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## ❌ Error Handling

### HTTP Status Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid request format or parameters |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Endpoint or resource not found |
| `422` | Unprocessable Entity | Valid format but invalid data |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error |
| `503` | Service Unavailable | Server overloaded or maintenance |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `INVALID_REQUEST` | Request format or parameters are invalid |
| `AUTHENTICATION_FAILED` | Authentication credentials are invalid |
| `RATE_LIMIT_EXCEEDED` | Too many requests in time window |
| `MODEL_NOT_LOADED` | Requested model is not loaded |
| `GENERATION_FAILED` | Text generation failed |
| `TIMEOUT` | Request timed out |
| `INSUFFICIENT_RESOURCES` | Server lacks resources to process request |

## ⏱️ Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 3600
```

### Rate Limit Tiers

| Tier | Requests/Hour | Tokens/Hour | Concurrent Requests |
|------|---------------|-------------|--------------------|
| **Free** | 100 | 10,000 | 1 |
| **Basic** | 1,000 | 100,000 | 5 |
| **Pro** | 10,000 | 1,000,000 | 20 |
| **Enterprise** | Unlimited | Unlimited | 100 |

## 🤖 Text Generation API

### Generate Text

**Endpoint**: `POST /generate`

**Description**: Generate text completion for a given prompt.

#### Request Format

```json
{
  "prompt": "What is artificial intelligence?",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop_sequences": ["\n\n", "Human:", "AI:"],
  "stream": false,
  "seed": 42,
  "logprobs": false,
  "echo": false,
  "metadata": {
    "user_id": "user123",
    "session_id": "session456",
    "application": "chatbot"
  }
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | ✅ | - | Input text prompt |
| `max_tokens` | integer | ❌ | 512 | Maximum tokens to generate |
| `temperature` | float | ❌ | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | ❌ | 0.9 | Nucleus sampling parameter |
| `top_k` | integer | ❌ | 50 | Top-k sampling parameter |
| `repetition_penalty` | float | ❌ | 1.1 | Repetition penalty (1.0-2.0) |
| `frequency_penalty` | float | ❌ | 0.0 | Frequency penalty (-2.0-2.0) |
| `presence_penalty` | float | ❌ | 0.0 | Presence penalty (-2.0-2.0) |
| `stop_sequences` | array | ❌ | [] | Stop generation at these sequences |
| `stream` | boolean | ❌ | false | Enable streaming response |
| `seed` | integer | ❌ | null | Random seed for reproducibility |
| `logprobs` | boolean | ❌ | false | Return log probabilities |
| `echo` | boolean | ❌ | false | Include prompt in response |
| `metadata` | object | ❌ | {} | Additional metadata |

#### Response Format

```json
{
  "text": "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence...",
  "request_id": "req_abc123",
  "tokens_generated": 128,
  "generation_time": 2.34,
  "tokens_per_second": 54.7,
  "finish_reason": "stop",
  "logprobs": [
    {
      "token": "Artificial",
      "logprob": -0.123,
      "top_logprobs": {
        "Artificial": -0.123,
        "AI": -2.456,
        "Machine": -3.789
      }
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 128,
    "total_tokens": 134
  },
  "metadata": {
    "model_name": "Qwen/Qwen2.5-3B",
    "backend": "vllm",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Streaming Response

When `stream: true`, the response is sent as Server-Sent Events:

```
data: {"text": "Artificial", "request_id": "req_abc123", "tokens_generated": 1}

data: {"text": " intelligence", "request_id": "req_abc123", "tokens_generated": 2}

data: {"text": " is", "request_id": "req_abc123", "tokens_generated": 3}

data: {"text": "", "request_id": "req_abc123", "finish_reason": "stop", "final": true}
```

#### Example Usage

**cURL**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer your-token"
    },
    json={
        "prompt": "Explain quantum computing in simple terms",
        "max_tokens": 256,
        "temperature": 0.7
    }
)

result = response.json()
print(result["text"])
```

**JavaScript**:
```javascript
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your-token'
  },
  body: JSON.stringify({
    prompt: 'Explain quantum computing in simple terms',
    max_tokens: 256,
    temperature: 0.7
  })
});

const result = await response.json();
console.log(result.text);
```

## 💬 Chat API

### Chat Completion

**Endpoint**: `POST /chat`

**Description**: Generate chat responses using conversation format.

#### Request Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant specialized in explaining complex topics."
    },
    {
      "role": "user",
      "content": "What is machine learning?"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop_sequences": ["Human:", "Assistant:"],
  "metadata": {
    "conversation_id": "conv_123",
    "user_id": "user_456"
  }
}
```

#### Message Roles

| Role | Description | Example |
|------|-------------|----------|
| `system` | System instructions | "You are a helpful assistant" |
| `user` | User message | "What is AI?" |
| `assistant` | Assistant response | "AI is artificial intelligence..." |
| `function` | Function call result | `{"result": "data"}` |

#### Response Format

```json
{
  "message": {
    "role": "assistant",
    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed..."
  },
  "request_id": "req_def456",
  "tokens_generated": 89,
  "generation_time": 1.23,
  "tokens_per_second": 72.4,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 89,
    "total_tokens": 134
  },
  "metadata": {
    "model_name": "Qwen/Qwen2.5-3B",
    "conversation_turns": 2
  }
}
```

#### Example Usage

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    headers={"Authorization": "Bearer your-token"},
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain photosynthesis"}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
)

result = response.json()
print(result["message"]["content"])
```

### Chat with Function Calling

**Endpoint**: `POST /chat/functions`

**Description**: Chat with function calling capabilities.

#### Request Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in New York?"
    }
  ],
  "functions": [
    {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius"
          }
        },
        "required": ["location"]
      }
    }
  ],
  "function_call": "auto"
}
```

## 🎛️ Model Management API

### List Available Models

**Endpoint**: `GET /models`

**Description**: Get list of available models.

#### Response Format

```json
{
  "models": [
    {
      "id": "qwen2.5-3b",
      "name": "Qwen/Qwen2.5-3B",
      "type": "causal_lm",
      "parameters": "3B",
      "context_length": 2048,
      "loaded": true,
      "backend": "vllm",
      "capabilities": ["text_generation", "chat"],
      "metadata": {
        "architecture": "transformer",
        "training_data": "multilingual",
        "license": "apache-2.0"
      }
    }
  ],
  "default_model": "qwen2.5-3b",
  "total_models": 1
}
```

### Get Model Information

**Endpoint**: `GET /models/{model_id}`

**Description**: Get detailed information about a specific model.

#### Response Format

```json
{
  "id": "qwen2.5-3b",
  "name": "Qwen/Qwen2.5-3B",
  "type": "causal_lm",
  "parameters": "3B",
  "context_length": 2048,
  "loaded": true,
  "backend": "vllm",
  "load_time": "2024-01-15T10:25:00Z",
  "memory_usage": {
    "model_size_gb": 6.2,
    "kv_cache_gb": 2.1,
    "total_gb": 8.3
  },
  "performance": {
    "avg_tokens_per_second": 85.3,
    "avg_latency_ms": 234,
    "total_requests": 1547
  },
  "capabilities": ["text_generation", "chat", "streaming"],
  "configuration": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048
  }
}
```

### Load Model

**Endpoint**: `POST /models/{model_id}/load`

**Description**: Load a specific model.

#### Request Format

```json
{
  "backend": "vllm",
  "configuration": {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048
  }
}
```

### Unload Model

**Endpoint**: `POST /models/{model_id}/unload`

**Description**: Unload a specific model to free resources.

## 🏥 Health and Status API

### Health Check

**Endpoint**: `GET /health`

**Description**: Check server health and readiness.

#### Response Format

```json
{
  "status": "healthy",
  "is_loaded": true,
  "is_ready": true,
  "backend_type": "vllm",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "gpu_available": true,
  "gpu_count": 2,
  "gpu_memory": {
    "total_gb": 48,
    "used_gb": 18,
    "free_gb": 30,
    "utilization": 0.375
  },
  "system_memory": {
    "total_gb": 64,
    "used_gb": 24,
    "free_gb": 40,
    "utilization": 0.375
  },
  "uptime": "2 days, 14 hours, 32 minutes",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Readiness Check

**Endpoint**: `GET /ready`

**Description**: Check if server is ready to accept requests.

#### Response Format

```json
{
  "ready": true,
  "model_loaded": true,
  "backend_initialized": true,
  "dependencies_available": true,
  "estimated_warmup_time": 0
}
```

### Liveness Check

**Endpoint**: `GET /live`

**Description**: Check if server is alive (for Kubernetes liveness probes).

#### Response Format

```json
{
  "alive": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 📊 Metrics and Monitoring API

### Server Statistics

**Endpoint**: `GET /stats`

**Description**: Get comprehensive server statistics.

#### Response Format

```json
{
  "is_loaded": true,
  "is_ready": true,
  "backend_type": "vllm",
  "model_info": {
    "name": "Qwen/Qwen2.5-3B",
    "parameters": "3B",
    "context_length": 2048
  },
  "request_stats": {
    "total_requests": 1547,
    "successful_requests": 1523,
    "failed_requests": 24,
    "error_rate": 0.0155,
    "requests_per_minute": 12.3
  },
  "generation_stats": {
    "total_tokens_generated": 892341,
    "avg_tokens_per_request": 576.8,
    "avg_generation_time": 2.21,
    "avg_tokens_per_second": 260.9,
    "p50_latency": 1.8,
    "p95_latency": 4.2,
    "p99_latency": 7.1
  },
  "resource_usage": {
    "gpu_utilization": 0.75,
    "gpu_memory_used_gb": 18.2,
    "cpu_utilization": 0.45,
    "memory_used_gb": 24.1
  },
  "uptime": "2 days, 14 hours, 32 minutes",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Prometheus Metrics

**Endpoint**: `GET /metrics`

**Description**: Get metrics in Prometheus format.

#### Response Format

```
# HELP llm_requests_total Total number of requests
# TYPE llm_requests_total counter
llm_requests_total{status="success"} 1523
llm_requests_total{status="error"} 24

# HELP llm_request_duration_seconds Request duration in seconds
# TYPE llm_request_duration_seconds histogram
llm_request_duration_seconds_bucket{le="0.5"} 234
llm_request_duration_seconds_bucket{le="1.0"} 567
llm_request_duration_seconds_bucket{le="2.0"} 890
llm_request_duration_seconds_bucket{le="5.0"} 1234
llm_request_duration_seconds_bucket{le="+Inf"} 1547
llm_request_duration_seconds_sum 3421.7
llm_request_duration_seconds_count 1547

# HELP llm_tokens_generated_total Total tokens generated
# TYPE llm_tokens_generated_total counter
llm_tokens_generated_total 892341

# HELP llm_gpu_memory_usage_bytes GPU memory usage in bytes
# TYPE llm_gpu_memory_usage_bytes gauge
llm_gpu_memory_usage_bytes{gpu="0"} 19595788288
llm_gpu_memory_usage_bytes{gpu="1"} 18874368000
```

## 🔌 WebSocket API

### Real-time Generation

**Endpoint**: `WS /ws/generate`

**Description**: Real-time text generation via WebSocket.

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/generate?token=your-token');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

#### Message Format

**Send Request**:
```json
{
  "type": "generate",
  "data": {
    "prompt": "Tell me about space exploration",
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  },
  "request_id": "req_123"
}
```

**Receive Response**:
```json
{
  "type": "token",
  "data": {
    "text": "Space",
    "token_id": 12345,
    "logprob": -0.123
  },
  "request_id": "req_123"
}
```

**Final Response**:
```json
{
  "type": "complete",
  "data": {
    "finish_reason": "stop",
    "total_tokens": 89,
    "generation_time": 2.34
  },
  "request_id": "req_123"
}
```

## 📚 SDK and Client Libraries

### Python SDK

**Installation**:
```bash
pip install custom-llm-client
```

**Usage**:
```python
from custom_llm_client import LLMClient

# Initialize client
client = LLMClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Generate text
response = client.generate(
    prompt="Explain artificial intelligence",
    max_tokens=512,
    temperature=0.7
)
print(response.text)

# Chat completion
response = client.chat([
    {"role": "user", "content": "What is machine learning?"}
])
print(response.message.content)

# Streaming generation
for chunk in client.generate_stream(
    prompt="Write a story about robots",
    max_tokens=1024
):
    print(chunk.text, end="", flush=True)
```

### JavaScript SDK

**Installation**:
```bash
npm install custom-llm-client
```

**Usage**:
```javascript
import { LLMClient } from 'custom-llm-client';

// Initialize client
const client = new LLMClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Generate text
const response = await client.generate({
  prompt: 'Explain artificial intelligence',
  maxTokens: 512,
  temperature: 0.7
});
console.log(response.text);

// Streaming generation
for await (const chunk of client.generateStream({
  prompt: 'Write a story about robots',
  maxTokens: 1024
})) {
  process.stdout.write(chunk.text);
}
```

### cURL Examples

**Basic Generation**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "prompt": "What is quantum computing?",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Streaming Generation**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Tell me a joke",
    "max_tokens": 100,
    "stream": true
  }'
```

**Chat Completion**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain photosynthesis"}
    ],
    "max_tokens": 300
  }'
```

**Health Check**:
```bash
curl -X GET "http://localhost:8000/health" \
  -H "Authorization: Bearer your-token"
```

**Server Stats**:
```bash
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer your-token"
```

This comprehensive API reference provides all the information needed to integrate with and use the Custom LLM Chatbot system effectively.