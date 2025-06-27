"""DPO (Direct Preference Optimization) trainer implementation."""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
from accelerate import Accelerator

from ..core.config import ConfigManager


@dataclass
class DPOBatch:
    """Batch data structure for DPO training."""
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_labels: torch.Tensor
    prompt_input_ids: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None


class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: ConfigManager,
                 accelerator: Optional[Accelerator] = None):
        """Initialize DPO trainer.
        
        Args:
            model: The model to train.
            tokenizer: Tokenizer for the model.
            config: Configuration manager.
            accelerator: Accelerator for distributed training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator or Accelerator()
        self.logger = logging.getLogger(__name__)
        
        # DPO hyperparameters
        self.beta = config.training.dpo.beta
        self.max_length = config.training.dpo.max_length
        self.max_prompt_length = config.training.dpo.max_prompt_length
        
        # Reference model for DPO (frozen copy of the original model)
        self.ref_model = None
        self._setup_reference_model()
        
        self.logger.info("DPO Trainer initialized")
    
    def _setup_reference_model(self) -> None:
        """Setup reference model for DPO."""
        # Create a copy of the model for reference
        self.ref_model = type(self.model)(self.model.config)
        self.ref_model.load_state_dict(self.model.state_dict())
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.ref_model.eval()
        self.logger.info("Reference model setup completed")
    
    def train(self, 
              train_dataset: Dataset,
              eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Execute DPO training.
        
        Args:
            train_dataset: Training dataset with preference pairs.
            eval_dataset: Optional evaluation dataset.
            
        Returns:
            Training results.
        """
        self.logger.info("Starting DPO training")
        
        # Prepare datasets
        train_dataloader = self._prepare_dataloader(train_dataset, shuffle=True)
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = self._prepare_dataloader(eval_dataset, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer, len(train_dataloader))
        
        # Prepare for training with accelerator
        self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Training loop
        num_epochs = self.config.training.dpo.num_train_epochs
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch(
                train_dataloader, optimizer, scheduler, epoch, global_step
            )
            
            # Evaluation phase
            eval_loss = None
            if eval_dataloader:
                eval_loss = self._evaluate_epoch(eval_dataloader, epoch)
                
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self._save_checkpoint(epoch, global_step, is_best=True)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1} completed - "
                f"Train Loss: {train_loss:.4f}"
                f"{f', Eval Loss: {eval_loss:.4f}' if eval_loss else ''}"
            )
            
            global_step += len(train_dataloader)
        
        self.logger.info("DPO training completed")
        
        return {
            'training_type': 'dpo',
            'final_train_loss': train_loss,
            'final_eval_loss': eval_loss,
            'best_eval_loss': best_eval_loss,
            'total_steps': global_step
        }
    
    def _train_epoch(self, 
                     dataloader,
                     optimizer,
                     scheduler,
                     epoch: int,
                     global_step: int) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training dataloader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            epoch: Current epoch number.
            global_step: Global training step.
            
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            # Compute DPO loss
            loss = self._compute_dpo_loss(batch)
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if step % self.config.training.logging_steps == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}, Step {step}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _evaluate_epoch(self, dataloader, epoch: int) -> float:
        """Evaluate for one epoch.
        
        Args:
            dataloader: Evaluation dataloader.
            epoch: Current epoch number.
            
        Returns:
            Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self._compute_dpo_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"Evaluation epoch {epoch + 1} - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _compute_dpo_loss(self, batch: DPOBatch) -> torch.Tensor:
        """Compute DPO loss for a batch.
        
        Args:
            batch: Batch of DPO data.
            
        Returns:
            DPO loss tensor.
        """
        # Get logits for chosen and rejected responses
        chosen_logits = self._get_logits(
            batch.chosen_input_ids,
            batch.chosen_attention_mask
        )
        rejected_logits = self._get_logits(
            batch.rejected_input_ids,
            batch.rejected_attention_mask
        )
        
        # Get reference logits (frozen model)
        with torch.no_grad():
            chosen_ref_logits = self._get_logits(
                batch.chosen_input_ids,
                batch.chosen_attention_mask,
                model=self.ref_model
            )
            rejected_ref_logits = self._get_logits(
                batch.rejected_input_ids,
                batch.rejected_attention_mask,
                model=self.ref_model
            )
        
        # Compute log probabilities
        chosen_logprobs = self._get_logprobs(
            chosen_logits, batch.chosen_input_ids, batch.chosen_labels
        )
        rejected_logprobs = self._get_logprobs(
            rejected_logits, batch.rejected_input_ids, batch.rejected_labels
        )
        
        chosen_ref_logprobs = self._get_logprobs(
            chosen_ref_logits, batch.chosen_input_ids, batch.chosen_labels
        )
        rejected_ref_logprobs = self._get_logprobs(
            rejected_ref_logits, batch.rejected_input_ids, batch.rejected_labels
        )
        
        # Compute DPO loss
        pi_logratios = chosen_logprobs - rejected_logprobs
        ref_logratios = chosen_ref_logprobs - rejected_ref_logprobs
        
        logits = pi_logratios - ref_logratios
        
        # DPO loss: -log(sigmoid(beta * logits))
        loss = -F.logsigmoid(self.beta * logits).mean()
        
        return loss
    
    def _get_logits(self, 
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    model: Optional[PreTrainedModel] = None) -> torch.Tensor:
        """Get model logits for input.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            model: Model to use (defaults to self.model).
            
        Returns:
            Model logits.
        """
        if model is None:
            model = self.model
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.logits
    
    def _get_logprobs(self, 
                      logits: torch.Tensor,
                      input_ids: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for labels.
        
        Args:
            logits: Model logits.
            input_ids: Input token IDs.
            labels: Target labels.
            
        Returns:
            Log probabilities.
        """
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for target tokens
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens
        mask = (shift_labels != -100).float()
        masked_log_probs = gathered_log_probs * mask
        
        # Sum over sequence length
        return masked_log_probs.sum(dim=-1)
    
    def _prepare_dataloader(self, dataset: Dataset, shuffle: bool = True):
        """Prepare dataloader for DPO training.
        
        Args:
            dataset: Dataset to create dataloader for.
            shuffle: Whether to shuffle the data.
            
        Returns:
            DataLoader instance.
        """
        from torch.utils.data import DataLoader
        
        def collate_fn(examples):
            # Process examples into DPO format
            batch = self._process_dpo_examples(examples)
            return batch
        
        return DataLoader(
            dataset,
            batch_size=self.config.training.dpo.per_device_train_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.config.training.dataloader_num_workers
        )
    
    def _process_dpo_examples(self, examples: List[Dict[str, Any]]) -> DPOBatch:
        """Process examples into DPO batch format.
        
        Args:
            examples: List of examples with chosen/rejected pairs.
            
        Returns:
            DPO batch.
        """
        chosen_texts = []
        rejected_texts = []
        prompts = []
        
        for example in examples:
            prompt = example.get('input', example.get('prompt', ''))
            chosen = example.get('chosen', '')
            rejected = example.get('rejected', '')
            
            prompts.append(prompt)
            chosen_texts.append(prompt + chosen)
            rejected_texts.append(prompt + rejected)
        
        # Tokenize chosen responses
        chosen_encodings = self.tokenizer(
            chosen_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize rejected responses
        rejected_encodings = self.tokenizer(
            rejected_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        chosen_labels = chosen_encodings['input_ids'].clone()
        rejected_labels = rejected_encodings['input_ids'].clone()
        
        # Mask prompt tokens in labels if needed
        if prompts[0]:  # If prompts exist
            prompt_encodings = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=self.max_prompt_length,
                return_tensors='pt'
            )
            
            # Mask prompt tokens in labels
            for i, prompt_len in enumerate(prompt_encodings['attention_mask'].sum(dim=1)):
                chosen_labels[i, :prompt_len] = -100
                rejected_labels[i, :prompt_len] = -100
        
        return DPOBatch(
            chosen_input_ids=chosen_encodings['input_ids'],
            chosen_attention_mask=chosen_encodings['attention_mask'],
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_encodings['input_ids'],
            rejected_attention_mask=rejected_encodings['attention_mask'],
            rejected_labels=rejected_labels
        )
    
    def _setup_optimizer(self):
        """Setup optimizer for training.
        
        Returns:
            Optimizer instance.
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.dpo.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.95)
        )
    
    def _setup_scheduler(self, optimizer, num_training_steps_per_epoch: int):
        """Setup learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance.
            num_training_steps_per_epoch: Number of training steps per epoch.
            
        Returns:
            Scheduler instance.
        """
        from transformers import get_linear_schedule_with_warmup
        
        total_steps = (
            num_training_steps_per_epoch * 
            self.config.training.dpo.num_train_epochs
        )
        
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
    
    def _save_checkpoint(self, epoch: int, step: int, is_best: bool = False) -> None:
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch.
            step: Current step.
            is_best: Whether this is the best checkpoint.
        """
        from pathlib import Path
        
        checkpoint_dir = Path(self.config.training.output_dir) / f"checkpoint-epoch-{epoch}"
        if is_best:
            checkpoint_dir = Path(self.config.training.output_dir) / "best_model"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.accelerator.save_model(self.model, str(checkpoint_dir))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict()
        }
        
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")