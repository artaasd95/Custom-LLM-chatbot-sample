"""Model management for loading, configuring, and managing LLM models."""

import os
import torch
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator

from .config import ConfigManager, ModelConfig, LoRAConfig


class ModelManager:
    """Manages model loading, configuration, and optimization."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the model manager.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.accelerator = None
        
        # Model metadata
        self.model_loaded = False
        self.is_peft_model = False
        self.device = self.config.get_device()
        self.torch_dtype = self.config.get_torch_dtype()
        
        self.logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_model(self, 
                   model_name: Optional[str] = None,
                   force_reload: bool = False) -> bool:
        """Load the specified model and tokenizer.
        
        Args:
            model_name: Model name to load. If None, uses config default.
            force_reload: Whether to force reload even if model is already loaded.
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.model_loaded and not force_reload:
            self.logger.info("Model already loaded. Use force_reload=True to reload.")
            return True
        
        model_name = model_name or self.config.model.name
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = self._load_tokenizer(model_name)
            
            # Load model
            self.model = self._load_base_model(model_name)
            
            # Apply LoRA if enabled
            if self.config.training.lora.enabled:
                self._apply_lora()
            
            # Initialize accelerator for distributed training
            self._initialize_accelerator()
            
            self.model_loaded = True
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load the tokenizer for the specified model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Loaded tokenizer.
        """
        self.logger.info(f"Loading tokenizer for {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="left"  # Important for generation
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        return tokenizer
    
    def _load_base_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load the base model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Loaded model.
        """
        self.logger.info(f"Loading base model: {model_name}")
        
        # Configure quantization if needed
        quantization_config = None
        if self.device == "cuda" and self._should_use_quantization():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=self.config.model.trust_remote_code,
            low_cpu_mem_usage=True
        )
        
        # Move to device if not using device_map
        if self.device != "cuda" or quantization_config is None:
            model = model.to(self.device)
        
        return model
    
    def _should_use_quantization(self) -> bool:
        """Determine if quantization should be used based on available memory.
        
        Returns:
            True if quantization should be used.
        """
        if self.device != "cuda":
            return False
        
        try:
            # Check available GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Use quantization if less than 16GB available
            return total_memory < 16 * 1024**3
        except:
            return True  # Default to quantization if can't determine memory
    
    def _apply_lora(self) -> None:
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        if not self.config.training.lora.enabled:
            return
        
        self.logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.training.lora.r,
            lora_alpha=self.config.training.lora.lora_alpha,
            lora_dropout=self.config.training.lora.lora_dropout,
            target_modules=self.config.training.lora.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.is_peft_model = True
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def _initialize_accelerator(self) -> None:
        """Initialize accelerator for distributed training."""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self._get_gradient_accumulation_steps(),
            mixed_precision="bf16" if self.config.training.bf16 else "fp16" if self.config.training.fp16 else "no",
            log_with=["tensorboard"] if self.config.monitoring.mlflow_enabled else None,
            project_dir=self.config.training.logging_dir
        )
        
        self.logger.info(f"Accelerator initialized with {self.accelerator.num_processes} processes")
    
    def _get_gradient_accumulation_steps(self) -> int:
        """Get gradient accumulation steps based on training type.
        
        Returns:
            Number of gradient accumulation steps.
        """
        if self.config.training.training_type == "from_scratch":
            return self.config.training.from_scratch.gradient_accumulation_steps
        elif self.config.training.training_type == "dpo":
            return self.config.training.dpo.gradient_accumulation_steps
        else:
            return 8  # Default value
    
    def prepare_for_training(self) -> Dict[str, Any]:
        """Prepare model and tokenizer for training.
        
        Returns:
            Dictionary containing prepared components.
        """
        if not self.model_loaded:
            raise RuntimeError("Model must be loaded before preparing for training")
        
        # Enable gradient checkpointing if configured
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Prepare model and tokenizer with accelerator
        if self.accelerator:
            self.model = self.accelerator.prepare(self.model)
        
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "accelerator": self.accelerator
        }
    
    def save_model(self, 
                   output_dir: str,
                   save_tokenizer: bool = True,
                   save_config: bool = True) -> bool:
        """Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model.
            save_tokenizer: Whether to save the tokenizer.
            save_config: Whether to save the model configuration.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        if not self.model_loaded:
            self.logger.error("No model loaded to save")
            return False
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving model to {output_dir}")
            
            # Save model
            if self.is_peft_model:
                # Save LoRA adapter
                self.model.save_pretrained(output_dir)
            else:
                # Save full model
                if self.accelerator:
                    self.accelerator.unwrap_model(self.model).save_pretrained(output_dir)
                else:
                    self.model.save_pretrained(output_dir)
            
            # Save tokenizer
            if save_tokenizer and self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            
            # Save configuration
            if save_config:
                self.config.save_config(output_path / "training_config.yaml")
            
            self.logger.info(f"Model saved successfully to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_trained_model(self, model_path: str) -> bool:
        """Load a previously trained model.
        
        Args:
            model_path: Path to the trained model.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            self.logger.info(f"Loading trained model from {model_path}")
            
            # Check if it's a PEFT model
            peft_config_path = Path(model_path) / "adapter_config.json"
            
            if peft_config_path.exists():
                # Load PEFT model
                self.logger.info("Loading PEFT model")
                
                # First load base model
                base_model_name = self.config.model.name
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=self.config.model.trust_remote_code
                )
                
                # Load PEFT adapter
                self.model = PeftModel.from_pretrained(self.model, model_path)
                self.is_peft_model = True
            else:
                # Load full model
                self.logger.info("Loading full model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=self.config.model.trust_remote_code
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code
            )
            
            self.model_loaded = True
            self.logger.info("Trained model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information.
        """
        if not self.model_loaded:
            return {"status": "No model loaded"}
        
        info = {
            "model_name": self.config.model.name,
            "model_type": self.config.model.model_type,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "is_peft_model": self.is_peft_model,
            "model_loaded": self.model_loaded
        }
        
        if self.model:
            try:
                # Get model parameters count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
                })
            except Exception as e:
                self.logger.warning(f"Could not get parameter count: {str(e)}")
        
        return info
    
    def generate_text(self, 
                      prompt: str,
                      max_length: int = 512,
                      temperature: float = 0.7,
                      do_sample: bool = True,
                      top_p: float = 0.9,
                      top_k: int = 50) -> str:
        """Generate text using the loaded model.
        
        Args:
            prompt: Input prompt for generation.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            
        Returns:
            Generated text.
        """
        if not self.model_loaded:
            raise RuntimeError("Model must be loaded before generation")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.accelerator:
            del self.accelerator
            self.accelerator = None
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        self.is_peft_model = False
        
        self.logger.info("Model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()