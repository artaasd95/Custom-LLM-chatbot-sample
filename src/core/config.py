"""Configuration management for the LLM training and serving system."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str = "Qwen/Qwen2.5-3B"
    model_type: str = "qwen"
    max_length: int = 2048
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration for parameter-efficient fine-tuning."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class FromScratchConfig:
    """Configuration for training from scratch."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-6
    beta: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    output_dir: str = "./models/trained"
    logging_dir: str = "./logs"
    seed: int = 42
    training_type: str = "dpo"  # "from_scratch", "dpo", "sft"
    
    # Sub-configurations
    from_scratch: FromScratchConfig = field(default_factory=FromScratchConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Optimization settings
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Data configuration settings."""
    train_data_path: str = "./data/train"
    eval_data_path: str = "./data/eval"
    test_data_path: str = "./data/test"
    data_types: list = field(default_factory=lambda: ["chat", "documents", "reports"])
    
    # Preprocessing settings
    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"
    remove_columns: list = field(default_factory=lambda: ["id"])
    text_column: str = "text"
    
    # Pipeline settings
    batch_size: int = 1000
    num_workers: int = 4
    cache_dir: str = "./cache"


@dataclass
class ServingConfig:
    """Serving configuration settings."""
    # vLLM settings
    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000
    model_path: str = "./models/trained"
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    workers: int = 4
    timeout: int = 300
    
    # ONNX export
    onnx_enabled: bool = False
    onnx_export_path: str = "./models/onnx"
    onnx_opset_version: int = 14
    onnx_optimize: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    # CometML
    comet_enabled: bool = True
    comet_project_name: str = "custom-llm-training"
    comet_workspace: str = "your-workspace"
    comet_api_key: str = ""
    
    # MLflow
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "llm-training"
    
    # Prometheus
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_metrics_path: str = "/metrics"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "./logs/app.log"


@dataclass
class CloudConfig:
    """Cloud configuration settings."""
    provider: str = "azure"
    
    # Azure settings
    azure_subscription_id: str = ""
    azure_resource_group: str = "llm-training-rg"
    azure_storage_account: str = "llmtrainingstorage"
    azure_container_name: str = "models"
    
    # Deployment
    environment: str = "development"
    replicas: int = 1
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "8Gi"
    memory_limit: str = "16Gi"
    gpu_request: int = 1


class ConfigManager:
    """Centralized configuration manager for the entire system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        self.config_path = config_path or "config.yaml"
        self._config_data = {}
        self.load_config()
        
        # Initialize configuration objects
        self.model = self._create_model_config()
        self.training = self._create_training_config()
        self.data = self._create_data_config()
        self.serving = self._create_serving_config()
        self.monitoring = self._create_monitoring_config()
        self.cloud = self._create_cloud_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self._config_data = {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            self._config_data = {}
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_path: Path to save the config. If None, uses the original path.
        """
        output_path = output_path or self.config_path
        
        # Convert dataclasses back to dict
        config_dict = {
            'model': self._dataclass_to_dict(self.model),
            'training': self._dataclass_to_dict(self.training),
            'data': self._dataclass_to_dict(self.data),
            'serving': self._dataclass_to_dict(self.serving),
            'monitoring': self._dataclass_to_dict(self.monitoring),
            'cloud': self._dataclass_to_dict(self.cloud)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary recursively."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_def in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._dataclass_to_dict(value)
                else:
                    result[field_name] = value
            return result
        return obj
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration from loaded data."""
        model_data = self._config_data.get('model', {})
        return ModelConfig(
            name=model_data.get('name', ModelConfig.name),
            model_type=model_data.get('model_type', ModelConfig.model_type),
            max_length=model_data.get('max_length', ModelConfig.max_length),
            device=model_data.get('device', ModelConfig.device),
            torch_dtype=model_data.get('torch_dtype', ModelConfig.torch_dtype),
            trust_remote_code=model_data.get('trust_remote_code', ModelConfig.trust_remote_code)
        )
    
    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration from loaded data."""
        training_data = self._config_data.get('training', {})
        
        # Create sub-configurations
        from_scratch_data = training_data.get('from_scratch', {})
        from_scratch_config = FromScratchConfig(
            num_train_epochs=from_scratch_data.get('num_train_epochs', FromScratchConfig.num_train_epochs),
            per_device_train_batch_size=from_scratch_data.get('per_device_train_batch_size', FromScratchConfig.per_device_train_batch_size),
            per_device_eval_batch_size=from_scratch_data.get('per_device_eval_batch_size', FromScratchConfig.per_device_eval_batch_size),
            gradient_accumulation_steps=from_scratch_data.get('gradient_accumulation_steps', FromScratchConfig.gradient_accumulation_steps),
            learning_rate=from_scratch_data.get('learning_rate', FromScratchConfig.learning_rate),
            weight_decay=from_scratch_data.get('weight_decay', FromScratchConfig.weight_decay),
            warmup_steps=from_scratch_data.get('warmup_steps', FromScratchConfig.warmup_steps),
            max_grad_norm=from_scratch_data.get('max_grad_norm', FromScratchConfig.max_grad_norm)
        )
        
        dpo_data = training_data.get('dpo', {})
        dpo_config = DPOConfig(
            num_train_epochs=dpo_data.get('num_train_epochs', DPOConfig.num_train_epochs),
            per_device_train_batch_size=dpo_data.get('per_device_train_batch_size', DPOConfig.per_device_train_batch_size),
            per_device_eval_batch_size=dpo_data.get('per_device_eval_batch_size', DPOConfig.per_device_eval_batch_size),
            gradient_accumulation_steps=dpo_data.get('gradient_accumulation_steps', DPOConfig.gradient_accumulation_steps),
            learning_rate=dpo_data.get('learning_rate', DPOConfig.learning_rate),
            beta=dpo_data.get('beta', DPOConfig.beta),
            max_length=dpo_data.get('max_length', DPOConfig.max_length),
            max_prompt_length=dpo_data.get('max_prompt_length', DPOConfig.max_prompt_length)
        )
        
        lora_data = training_data.get('lora', {})
        lora_config = LoRAConfig(
            enabled=lora_data.get('enabled', LoRAConfig.enabled),
            r=lora_data.get('r', LoRAConfig.r),
            lora_alpha=lora_data.get('lora_alpha', LoRAConfig.lora_alpha),
            lora_dropout=lora_data.get('lora_dropout', LoRAConfig.lora_dropout),
            target_modules=lora_data.get('target_modules', LoRAConfig.target_modules)
        )
        
        return TrainingConfig(
            output_dir=training_data.get('output_dir', TrainingConfig.output_dir),
            logging_dir=training_data.get('logging_dir', TrainingConfig.logging_dir),
            seed=training_data.get('seed', TrainingConfig.seed),
            training_type=training_data.get('training_type', TrainingConfig.training_type),
            from_scratch=from_scratch_config,
            dpo=dpo_config,
            lora=lora_config,
            optimizer=training_data.get('optimizer', TrainingConfig.optimizer),
            lr_scheduler_type=training_data.get('lr_scheduler_type', TrainingConfig.lr_scheduler_type),
            fp16=training_data.get('fp16', TrainingConfig.fp16),
            bf16=training_data.get('bf16', TrainingConfig.bf16),
            gradient_checkpointing=training_data.get('gradient_checkpointing', TrainingConfig.gradient_checkpointing),
            dataloader_num_workers=training_data.get('dataloader_num_workers', TrainingConfig.dataloader_num_workers)
        )
    
    def _create_data_config(self) -> DataConfig:
        """Create data configuration from loaded data."""
        data_config = self._config_data.get('data', {})
        preprocessing = data_config.get('preprocessing', {})
        pipeline = data_config.get('pipeline', {})
        
        return DataConfig(
            train_data_path=data_config.get('train_data_path', DataConfig.train_data_path),
            eval_data_path=data_config.get('eval_data_path', DataConfig.eval_data_path),
            test_data_path=data_config.get('test_data_path', DataConfig.test_data_path),
            data_types=data_config.get('data_types', DataConfig.data_types),
            max_length=preprocessing.get('max_length', DataConfig.max_length),
            truncation=preprocessing.get('truncation', DataConfig.truncation),
            padding=preprocessing.get('padding', DataConfig.padding),
            remove_columns=preprocessing.get('remove_columns', DataConfig.remove_columns),
            text_column=preprocessing.get('text_column', DataConfig.text_column),
            batch_size=pipeline.get('batch_size', DataConfig.batch_size),
            num_workers=pipeline.get('num_workers', DataConfig.num_workers),
            cache_dir=pipeline.get('cache_dir', DataConfig.cache_dir)
        )
    
    def _create_serving_config(self) -> ServingConfig:
        """Create serving configuration from loaded data."""
        serving_data = self._config_data.get('serving', {})
        vllm_data = serving_data.get('vllm', {})
        api_data = serving_data.get('api', {})
        onnx_data = serving_data.get('onnx', {})
        
        return ServingConfig(
            vllm_host=vllm_data.get('host', ServingConfig.vllm_host),
            vllm_port=vllm_data.get('port', ServingConfig.vllm_port),
            model_path=vllm_data.get('model_path', ServingConfig.model_path),
            tensor_parallel_size=vllm_data.get('tensor_parallel_size', ServingConfig.tensor_parallel_size),
            max_num_seqs=vllm_data.get('max_num_seqs', ServingConfig.max_num_seqs),
            max_model_len=vllm_data.get('max_model_len', ServingConfig.max_model_len),
            gpu_memory_utilization=vllm_data.get('gpu_memory_utilization', ServingConfig.gpu_memory_utilization),
            api_host=api_data.get('host', ServingConfig.api_host),
            api_port=api_data.get('port', ServingConfig.api_port),
            workers=api_data.get('workers', ServingConfig.workers),
            timeout=api_data.get('timeout', ServingConfig.timeout),
            onnx_enabled=onnx_data.get('enabled', ServingConfig.onnx_enabled),
            onnx_export_path=onnx_data.get('export_path', ServingConfig.onnx_export_path),
            onnx_opset_version=onnx_data.get('opset_version', ServingConfig.onnx_opset_version),
            onnx_optimize=onnx_data.get('optimize', ServingConfig.onnx_optimize)
        )
    
    def _create_monitoring_config(self) -> MonitoringConfig:
        """Create monitoring configuration from loaded data."""
        monitoring_data = self._config_data.get('monitoring', {})
        comet_data = monitoring_data.get('comet', {})
        mlflow_data = monitoring_data.get('mlflow', {})
        prometheus_data = monitoring_data.get('prometheus', {})
        logging_data = monitoring_data.get('logging', {})
        
        return MonitoringConfig(
            comet_enabled=comet_data.get('enabled', MonitoringConfig.comet_enabled),
            comet_project_name=comet_data.get('project_name', MonitoringConfig.comet_project_name),
            comet_workspace=comet_data.get('workspace', MonitoringConfig.comet_workspace),
            comet_api_key=os.getenv('COMET_API_KEY', comet_data.get('api_key', MonitoringConfig.comet_api_key)),
            mlflow_enabled=mlflow_data.get('enabled', MonitoringConfig.mlflow_enabled),
            mlflow_tracking_uri=mlflow_data.get('tracking_uri', MonitoringConfig.mlflow_tracking_uri),
            mlflow_experiment_name=mlflow_data.get('experiment_name', MonitoringConfig.mlflow_experiment_name),
            prometheus_enabled=prometheus_data.get('enabled', MonitoringConfig.prometheus_enabled),
            prometheus_port=prometheus_data.get('port', MonitoringConfig.prometheus_port),
            prometheus_metrics_path=prometheus_data.get('metrics_path', MonitoringConfig.prometheus_metrics_path),
            log_level=logging_data.get('level', MonitoringConfig.log_level),
            log_format=logging_data.get('format', MonitoringConfig.log_format),
            log_file=logging_data.get('file', MonitoringConfig.log_file)
        )
    
    def _create_cloud_config(self) -> CloudConfig:
        """Create cloud configuration from loaded data."""
        cloud_data = self._config_data.get('cloud', {})
        azure_data = cloud_data.get('azure', {})
        deployment_data = cloud_data.get('deployment', {})
        
        return CloudConfig(
            provider=cloud_data.get('provider', CloudConfig.provider),
            azure_subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID', azure_data.get('subscription_id', CloudConfig.azure_subscription_id)),
            azure_resource_group=azure_data.get('resource_group', CloudConfig.azure_resource_group),
            azure_storage_account=azure_data.get('storage_account', CloudConfig.azure_storage_account),
            azure_container_name=azure_data.get('container_name', CloudConfig.azure_container_name),
            environment=deployment_data.get('environment', CloudConfig.environment),
            replicas=deployment_data.get('replicas', CloudConfig.replicas),
            cpu_request=deployment_data.get('cpu_request', CloudConfig.cpu_request),
            cpu_limit=deployment_data.get('cpu_limit', CloudConfig.cpu_limit),
            memory_request=deployment_data.get('memory_request', CloudConfig.memory_request),
            memory_limit=deployment_data.get('memory_limit', CloudConfig.memory_limit),
            gpu_request=deployment_data.get('gpu_request', CloudConfig.gpu_request)
        )
    
    def get_device(self) -> str:
        """Get the appropriate device for training/inference."""
        import torch
        
        if self.model.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.model.device
    
    def get_torch_dtype(self):
        """Get the appropriate torch dtype."""
        import torch
        
        if self.model.torch_dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if self.training.bf16 else torch.float16
            else:
                return torch.float32
        
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        return dtype_mapping.get(self.model.torch_dtype, torch.float32)
    
    def update_model_name(self, model_name: str) -> None:
        """Update the model name configuration.
        
        Args:
            model_name: New model name to use.
        """
        self.model.name = model_name
        print(f"Model name updated to: {model_name}")
    
    def update_training_type(self, training_type: str) -> None:
        """Update the training type configuration.
        
        Args:
            training_type: New training type ('from_scratch', 'dpo', 'sft').
        """
        if training_type not in ['from_scratch', 'dpo', 'sft']:
            raise ValueError(f"Invalid training type: {training_type}")
        
        self.training.training_type = training_type
        print(f"Training type updated to: {training_type}")
    
    def validate_config(self) -> bool:
        """Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        errors = []
        
        # Validate paths
        required_dirs = [
            self.training.output_dir,
            self.training.logging_dir,
            self.data.cache_dir
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate training type
        if self.training.training_type not in ['from_scratch', 'dpo', 'sft']:
            errors.append(f"Invalid training type: {self.training.training_type}")
        
        # Validate device
        if self.model.device not in ['auto', 'cuda', 'cpu']:
            errors.append(f"Invalid device: {self.model.device}")
        
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            return False
        
        return True


# Global configuration instance
config = ConfigManager()