"""Training orchestrator for LLM training with support for from-scratch and DPO training."""

import os
import json
import logging
import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import numpy as np

from ..core.config import ConfigManager
from ..core.model_manager import ModelManager
from ..monitoring.experiment_tracker import ExperimentTracker
from ..data.data_processor import DataProcessor


class TrainingOrchestrator:
    """Orchestrates the training process for different training types."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the training orchestrator.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_manager = ModelManager(config_manager)
        self.experiment_tracker = ExperimentTracker(config_manager)
        self.data_processor = DataProcessor(config_manager)
        
        # Training state
        self.trainer = None
        self.training_args = None
        self.current_experiment = None
        
        # Create output directories
        self._setup_directories()
        
        self.logger.info("TrainingOrchestrator initialized")
    
    def _setup_directories(self) -> None:
        """Setup required directories for training."""
        directories = [
            self.config.training.output_dir,
            self.config.training.logging_dir,
            self.config.data.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def prepare_training(self, 
                        training_type: Optional[str] = None,
                        model_name: Optional[str] = None) -> bool:
        """Prepare for training by loading model and data.
        
        Args:
            training_type: Type of training ('from_scratch', 'dpo', 'sft').
            model_name: Model name to use for training.
            
        Returns:
            True if preparation successful, False otherwise.
        """
        try:
            # Update configuration if provided
            if training_type:
                self.config.update_training_type(training_type)
            if model_name:
                self.config.update_model_name(model_name)
            
            self.logger.info(f"Preparing for {self.config.training.training_type} training")
            
            # Load model
            if not self.model_manager.load_model():
                self.logger.error("Failed to load model")
                return False
            
            # Prepare model for training
            self.model_components = self.model_manager.prepare_for_training()
            
            # Initialize experiment tracking
            self.current_experiment = self.experiment_tracker.start_experiment(
                experiment_name=f"{self.config.training.training_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=[self.config.training.training_type, self.config.model.name.split('/')[-1]]
            )
            
            self.logger.info("Training preparation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training: {str(e)}")
            return False
    
    def train(self, 
              train_dataset: Optional[Dataset] = None,
              eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Execute training based on the configured training type.
        
        Args:
            train_dataset: Training dataset. If None, loads from config.
            eval_dataset: Evaluation dataset. If None, loads from config.
            
        Returns:
            Training results and metrics.
        """
        if not self.model_components:
            raise RuntimeError("Must call prepare_training() before train()")
        
        # Load datasets if not provided
        if train_dataset is None or eval_dataset is None:
            datasets = self.data_processor.load_and_process_data()
            train_dataset = train_dataset or datasets.get('train')
            eval_dataset = eval_dataset or datasets.get('eval')
        
        if train_dataset is None:
            raise ValueError("No training dataset available")
        
        # Execute training based on type
        training_type = self.config.training.training_type
        
        if training_type == "from_scratch":
            return self._train_from_scratch(train_dataset, eval_dataset)
        elif training_type == "dpo":
            return self._train_dpo(train_dataset, eval_dataset)
        elif training_type == "sft":
            return self._train_supervised_fine_tuning(train_dataset, eval_dataset)
        else:
            raise ValueError(f"Unsupported training type: {training_type}")
    
    def _train_from_scratch(self, 
                           train_dataset: Dataset,
                           eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Train model from scratch.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            
        Returns:
            Training results.
        """
        self.logger.info("Starting from-scratch training")
        
        # Create training arguments
        training_args = self._create_training_arguments("from_scratch")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_components["tokenizer"],
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model_components["model"],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model_components["tokenizer"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Log training info
        self._log_training_info(train_dataset, eval_dataset)
        
        # Start training
        train_result = self.trainer.train()
        
        # Log results
        self._log_training_results(train_result)
        
        # Save model
        self._save_trained_model()
        
        return {
            "training_type": "from_scratch",
            "train_result": train_result,
            "model_path": self.config.training.output_dir
        }
    
    def _train_dpo(self, 
                   train_dataset: Dataset,
                   eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Train model using DPO (Direct Preference Optimization).
        
        Args:
            train_dataset: Training dataset with preference pairs.
            eval_dataset: Evaluation dataset.
            
        Returns:
            Training results.
        """
        self.logger.info("Starting DPO training")
        
        # Create DPO training arguments
        training_args = self._create_dpo_training_arguments()
        
        # Create DPO trainer
        self.trainer = DPOTrainer(
            model=self.model_components["model"],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model_components["tokenizer"],
            beta=self.config.training.dpo.beta,
            max_length=self.config.training.dpo.max_length,
            max_prompt_length=self.config.training.dpo.max_prompt_length
        )
        
        # Log training info
        self._log_training_info(train_dataset, eval_dataset)
        
        # Start training
        train_result = self.trainer.train()
        
        # Log results
        self._log_training_results(train_result)
        
        # Save model
        self._save_trained_model()
        
        return {
            "training_type": "dpo",
            "train_result": train_result,
            "model_path": self.config.training.output_dir
        }
    
    def _train_supervised_fine_tuning(self, 
                                     train_dataset: Dataset,
                                     eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Train model using supervised fine-tuning.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            
        Returns:
            Training results.
        """
        self.logger.info("Starting supervised fine-tuning")
        
        # Use from_scratch config for SFT (similar parameters)
        training_args = self._create_training_arguments("from_scratch")
        
        # Create data collator for instruction tuning
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_components["tokenizer"],
            mlm=False
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model_components["model"],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model_components["tokenizer"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Log training info
        self._log_training_info(train_dataset, eval_dataset)
        
        # Start training
        train_result = self.trainer.train()
        
        # Log results
        self._log_training_results(train_result)
        
        # Save model
        self._save_trained_model()
        
        return {
            "training_type": "sft",
            "train_result": train_result,
            "model_path": self.config.training.output_dir
        }
    
    def _create_training_arguments(self, training_type: str) -> TrainingArguments:
        """Create training arguments based on training type.
        
        Args:
            training_type: Type of training configuration to use.
            
        Returns:
            TrainingArguments instance.
        """
        if training_type == "from_scratch":
            config = self.config.training.from_scratch
        else:
            config = self.config.training.from_scratch  # Default fallback
        
        return TrainingArguments(
            output_dir=self.config.training.output_dir,
            logging_dir=self.config.training.logging_dir,
            
            # Training parameters
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            optim=self.config.training.optimizer,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            
            # Mixed precision
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            
            # Checkpointing and logging
            save_strategy="steps",
            save_steps=500,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500,
            logging_steps=100,
            
            # Other settings
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            seed=self.config.training.seed,
            
            # Evaluation and saving
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            
            # Reporting
            report_to=["tensorboard"] if self.config.monitoring.mlflow_enabled else [],
            run_name=f"{training_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def _create_dpo_training_arguments(self) -> DPOConfig:
        """Create DPO-specific training arguments.
        
        Returns:
            DPOConfig instance.
        """
        config = self.config.training.dpo
        
        return DPOConfig(
            output_dir=self.config.training.output_dir,
            logging_dir=self.config.training.logging_dir,
            
            # Training parameters
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=config.learning_rate,
            optim=self.config.training.optimizer,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            
            # Mixed precision
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            
            # DPO specific
            beta=config.beta,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            
            # Checkpointing and logging
            save_strategy="steps",
            save_steps=500,
            eval_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            
            # Other settings
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            seed=self.config.training.seed,
            
            # Reporting
            report_to=["tensorboard"] if self.config.monitoring.mlflow_enabled else [],
            run_name=f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def _log_training_info(self, 
                          train_dataset: Dataset,
                          eval_dataset: Optional[Dataset] = None) -> None:
        """Log training information to experiment tracker.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
        """
        # Log model info
        model_info = self.model_manager.get_model_info()
        self.experiment_tracker.log_parameters(model_info)
        
        # Log training configuration
        training_config = {
            "training_type": self.config.training.training_type,
            "model_name": self.config.model.name,
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
            "device": self.model_manager.device,
            "torch_dtype": str(self.model_manager.torch_dtype)
        }
        
        if self.config.training.training_type == "from_scratch":
            training_config.update({
                "num_train_epochs": self.config.training.from_scratch.num_train_epochs,
                "learning_rate": self.config.training.from_scratch.learning_rate,
                "batch_size": self.config.training.from_scratch.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.training.from_scratch.gradient_accumulation_steps
            })
        elif self.config.training.training_type == "dpo":
            training_config.update({
                "num_train_epochs": self.config.training.dpo.num_train_epochs,
                "learning_rate": self.config.training.dpo.learning_rate,
                "batch_size": self.config.training.dpo.per_device_train_batch_size,
                "beta": self.config.training.dpo.beta,
                "max_length": self.config.training.dpo.max_length
            })
        
        self.experiment_tracker.log_parameters(training_config)
        
        self.logger.info(f"Training info logged: {json.dumps(training_config, indent=2)}")
    
    def _log_training_results(self, train_result) -> None:
        """Log training results to experiment tracker.
        
        Args:
            train_result: Training result from trainer.
        """
        if hasattr(train_result, 'training_loss'):
            self.experiment_tracker.log_metric("final_training_loss", train_result.training_loss)
        
        if hasattr(train_result, 'metrics'):
            for key, value in train_result.metrics.items():
                self.experiment_tracker.log_metric(f"final_{key}", value)
        
        # Log training time
        if hasattr(train_result, 'training_time'):
            self.experiment_tracker.log_metric("training_time_seconds", train_result.training_time)
        
        self.logger.info("Training results logged to experiment tracker")
    
    def _save_trained_model(self) -> None:
        """Save the trained model and related artifacts."""
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(self.config.training.output_dir) / f"{self.config.training.training_type}_{timestamp}"
            
            # Save model
            success = self.model_manager.save_model(
                str(output_dir),
                save_tokenizer=True,
                save_config=True
            )
            
            if success:
                # Log model artifact to experiment tracker
                self.experiment_tracker.log_model(str(output_dir))
                
                # Save training metadata
                metadata = {
                    "training_type": self.config.training.training_type,
                    "model_name": self.config.model.name,
                    "training_timestamp": timestamp,
                    "output_directory": str(output_dir),
                    "experiment_id": self.current_experiment
                }
                
                metadata_path = output_dir / "training_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"Model saved successfully to {output_dir}")
            else:
                self.logger.error("Failed to save model")
                
        except Exception as e:
            self.logger.error(f"Error saving trained model: {str(e)}")
    
    def evaluate_model(self, 
                      eval_dataset: Optional[Dataset] = None,
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the trained model.
        
        Args:
            eval_dataset: Evaluation dataset. If None, uses default.
            metrics: List of metrics to compute.
            
        Returns:
            Evaluation results.
        """
        if not self.trainer:
            raise RuntimeError("No trainer available. Must train model first.")
        
        if eval_dataset is None:
            datasets = self.data_processor.load_and_process_data()
            eval_dataset = datasets.get('eval')
        
        if eval_dataset is None:
            raise ValueError("No evaluation dataset available")
        
        self.logger.info("Starting model evaluation")
        
        # Run evaluation
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # Log evaluation results
        for key, value in eval_results.items():
            self.experiment_tracker.log_metric(f"eval_{key}", value)
        
        self.logger.info(f"Evaluation completed: {json.dumps(eval_results, indent=2)}")
        
        return eval_results
    
    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        """Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory.
            
        Returns:
            Training results.
        """
        if not self.trainer:
            raise RuntimeError("No trainer available. Must prepare training first.")
        
        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Resume training
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        
        # Log results
        self._log_training_results(train_result)
        
        # Save model
        self._save_trained_model()
        
        return {
            "training_type": self.config.training.training_type,
            "train_result": train_result,
            "resumed_from": checkpoint_path,
            "model_path": self.config.training.output_dir
        }
    
    def cleanup(self) -> None:
        """Clean up training resources."""
        if self.trainer:
            del self.trainer
            self.trainer = None
        
        if self.model_manager:
            self.model_manager.cleanup()
        
        if self.experiment_tracker:
            self.experiment_tracker.end_experiment()
        
        self.logger.info("Training resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()