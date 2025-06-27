"""Training metrics and evaluation utilities."""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
import math

from transformers import EvalPrediction
from datasets import Dataset
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TrainingMetrics:
    """Handles computation and tracking of training metrics."""
    
    def __init__(self):
        """Initialize training metrics."""
        self.logger = logging.getLogger(__name__)
        
        # Load evaluation metrics
        self.metrics = {}
        self._load_metrics()
        
        # Metric history
        self.metric_history = defaultdict(list)
        
        self.logger.info("TrainingMetrics initialized")
    
    def _load_metrics(self) -> None:
        """Load evaluation metrics from HuggingFace evaluate library."""
        try:
            # Language modeling metrics
            self.metrics['perplexity'] = evaluate.load('perplexity', module_type='metric')
            
            # Text generation metrics
            self.metrics['bleu'] = evaluate.load('bleu')
            self.metrics['rouge'] = evaluate.load('rouge')
            self.metrics['meteor'] = evaluate.load('meteor')
            
            # General metrics
            self.metrics['accuracy'] = evaluate.load('accuracy')
            
            self.logger.info("Evaluation metrics loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Some metrics could not be loaded: {str(e)}")
            # Fallback to basic metrics
            self.metrics = {}
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for evaluation predictions.
        
        Args:
            eval_pred: Evaluation predictions from trainer.
            
        Returns:
            Dictionary of computed metrics.
        """
        predictions, labels = eval_pred
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        
        metrics = {}
        
        # Compute perplexity
        perplexity = self._compute_perplexity(predictions, labels)
        if perplexity is not None:
            metrics['perplexity'] = perplexity
        
        # Compute accuracy (for classification tasks)
        accuracy = self._compute_accuracy(predictions, labels)
        if accuracy is not None:
            metrics['accuracy'] = accuracy
        
        # Compute loss if available
        if hasattr(eval_pred, 'loss') and eval_pred.loss is not None:
            metrics['eval_loss'] = float(eval_pred.loss)
        
        # Store metrics in history
        for key, value in metrics.items():
            self.metric_history[key].append(value)
        
        self.logger.info(f"Computed metrics: {metrics}")
        return metrics
    
    def _compute_perplexity(self, predictions: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Compute perplexity from predictions and labels.
        
        Args:
            predictions: Model predictions (logits).
            labels: True labels.
            
        Returns:
            Perplexity value or None if computation fails.
        """
        try:
            # Handle different shapes
            if len(predictions.shape) == 3:  # (batch, seq_len, vocab_size)
                # Shift predictions and labels for causal LM
                shift_predictions = predictions[:, :-1, :]
                shift_labels = labels[:, 1:]
                
                # Flatten
                shift_predictions = shift_predictions.reshape(-1, shift_predictions.shape[-1])
                shift_labels = shift_labels.reshape(-1)
                
                # Remove padding tokens
                mask = shift_labels != -100
                shift_predictions = shift_predictions[mask]
                shift_labels = shift_labels[mask]
                
                if len(shift_labels) == 0:
                    return None
                
                # Compute cross entropy loss
                log_probs = torch.log_softmax(torch.from_numpy(shift_predictions), dim=-1)
                nll_loss = torch.nn.functional.nll_loss(
                    log_probs, 
                    torch.from_numpy(shift_labels), 
                    reduction='mean'
                )
                
                # Compute perplexity
                perplexity = torch.exp(nll_loss).item()
                return perplexity
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to compute perplexity: {str(e)}")
            return None
    
    def _compute_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Compute accuracy from predictions and labels.
        
        Args:
            predictions: Model predictions.
            labels: True labels.
            
        Returns:
            Accuracy value or None if computation fails.
        """
        try:
            if len(predictions.shape) == 3:  # (batch, seq_len, vocab_size)
                # Get predicted tokens
                predicted_tokens = np.argmax(predictions, axis=-1)
                
                # Shift for causal LM
                predicted_tokens = predicted_tokens[:, :-1]
                true_labels = labels[:, 1:]
                
                # Flatten and remove padding
                predicted_tokens = predicted_tokens.reshape(-1)
                true_labels = true_labels.reshape(-1)
                
                mask = true_labels != -100
                predicted_tokens = predicted_tokens[mask]
                true_labels = true_labels[mask]
                
                if len(true_labels) == 0:
                    return None
                
                # Compute accuracy
                accuracy = accuracy_score(true_labels, predicted_tokens)
                return accuracy
            
            elif len(predictions.shape) == 2:  # (batch, num_classes)
                # Classification task
                predicted_classes = np.argmax(predictions, axis=-1)
                accuracy = accuracy_score(labels, predicted_classes)
                return accuracy
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to compute accuracy: {str(e)}")
            return None
    
    def compute_text_generation_metrics(self, 
                                       predictions: List[str], 
                                       references: List[str]) -> Dict[str, float]:
        """Compute text generation metrics.
        
        Args:
            predictions: Generated text predictions.
            references: Reference texts.
            
        Returns:
            Dictionary of text generation metrics.
        """
        metrics = {}
        
        try:
            # BLEU score
            if 'bleu' in self.metrics:
                bleu_result = self.metrics['bleu'].compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                metrics['bleu'] = bleu_result['bleu']
            
            # ROUGE scores
            if 'rouge' in self.metrics:
                rouge_result = self.metrics['rouge'].compute(
                    predictions=predictions,
                    references=references
                )
                metrics.update({
                    'rouge1': rouge_result['rouge1'],
                    'rouge2': rouge_result['rouge2'],
                    'rougeL': rouge_result['rougeL']
                })
            
            # METEOR score
            if 'meteor' in self.metrics:
                meteor_result = self.metrics['meteor'].compute(
                    predictions=predictions,
                    references=references
                )
                metrics['meteor'] = meteor_result['meteor']
            
        except Exception as e:
            self.logger.warning(f"Failed to compute text generation metrics: {str(e)}")
        
        return metrics
    
    def compute_dpo_metrics(self, 
                           chosen_rewards: torch.Tensor,
                           rejected_rewards: torch.Tensor) -> Dict[str, float]:
        """Compute DPO-specific metrics.
        
        Args:
            chosen_rewards: Rewards for chosen responses.
            rejected_rewards: Rewards for rejected responses.
            
        Returns:
            Dictionary of DPO metrics.
        """
        metrics = {}
        
        try:
            # Convert to numpy
            if torch.is_tensor(chosen_rewards):
                chosen_rewards = chosen_rewards.detach().cpu().numpy()
            if torch.is_tensor(rejected_rewards):
                rejected_rewards = rejected_rewards.detach().cpu().numpy()
            
            # Reward accuracy (how often chosen > rejected)
            reward_accuracy = np.mean(chosen_rewards > rejected_rewards)
            metrics['reward_accuracy'] = reward_accuracy
            
            # Reward margin
            reward_margin = np.mean(chosen_rewards - rejected_rewards)
            metrics['reward_margin'] = reward_margin
            
            # Reward statistics
            metrics['chosen_reward_mean'] = np.mean(chosen_rewards)
            metrics['chosen_reward_std'] = np.std(chosen_rewards)
            metrics['rejected_reward_mean'] = np.mean(rejected_rewards)
            metrics['rejected_reward_std'] = np.std(rejected_rewards)
            
        except Exception as e:
            self.logger.warning(f"Failed to compute DPO metrics: {str(e)}")
        
        return metrics
    
    def compute_loss_metrics(self, losses: List[float]) -> Dict[str, float]:
        """Compute loss-based metrics.
        
        Args:
            losses: List of loss values.
            
        Returns:
            Dictionary of loss metrics.
        """
        if not losses:
            return {}
        
        losses = np.array(losses)
        
        return {
            'loss_mean': np.mean(losses),
            'loss_std': np.std(losses),
            'loss_min': np.min(losses),
            'loss_max': np.max(losses),
            'loss_median': np.median(losses)
        }
    
    def compute_gradient_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute gradient-based metrics.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Dictionary of gradient metrics.
        """
        metrics = {}
        
        try:
            total_norm = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                metrics['grad_norm'] = total_norm
                metrics['grad_norm_avg'] = total_norm / param_count
            
        except Exception as e:
            self.logger.warning(f"Failed to compute gradient metrics: {str(e)}")
        
        return metrics
    
    def compute_model_metrics(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Compute model-specific metrics.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Dictionary of model metrics.
        """
        metrics = {}
        
        try:
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metrics['total_parameters'] = total_params
            metrics['trainable_parameters'] = trainable_params
            metrics['frozen_parameters'] = total_params - trainable_params
            
            # Memory usage (if CUDA available)
            if torch.cuda.is_available():
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated()
                metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved()
            
            # Model size estimation (MB)
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            metrics['model_size_mb'] = model_size_mb
            
        except Exception as e:
            self.logger.warning(f"Failed to compute model metrics: {str(e)}")
        
        return metrics
    
    def get_metric_history(self, metric_name: Optional[str] = None) -> Union[List[float], Dict[str, List[float]]]:
        """Get metric history.
        
        Args:
            metric_name: Specific metric name. If None, returns all metrics.
            
        Returns:
            Metric history for specified metric or all metrics.
        """
        if metric_name:
            return self.metric_history.get(metric_name, [])
        return dict(self.metric_history)
    
    def reset_history(self) -> None:
        """Reset metric history."""
        self.metric_history.clear()
        self.logger.info("Metric history reset")
    
    def compute_training_stability_metrics(self, losses: List[float]) -> Dict[str, float]:
        """Compute training stability metrics.
        
        Args:
            losses: List of training losses.
            
        Returns:
            Dictionary of stability metrics.
        """
        if len(losses) < 2:
            return {}
        
        losses = np.array(losses)
        
        # Loss variance
        loss_variance = np.var(losses)
        
        # Loss trend (slope of linear regression)
        x = np.arange(len(losses))
        loss_trend = np.polyfit(x, losses, 1)[0]
        
        # Loss smoothness (average absolute difference between consecutive losses)
        loss_smoothness = np.mean(np.abs(np.diff(losses)))
        
        # Convergence indicator (ratio of recent variance to overall variance)
        if len(losses) >= 10:
            recent_variance = np.var(losses[-10:])
            convergence_ratio = recent_variance / (loss_variance + 1e-8)
        else:
            convergence_ratio = 1.0
        
        return {
            'loss_variance': loss_variance,
            'loss_trend': loss_trend,
            'loss_smoothness': loss_smoothness,
            'convergence_ratio': convergence_ratio
        }
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics with optional step information.
        
        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.
        """
        log_msg = f"Metrics{f' (step {step})' if step else ''}: "
        log_msg += ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                              for k, v in metrics.items()])
        
        self.logger.info(log_msg)
    
    def export_metrics(self, filepath: str) -> None:
        """Export metric history to file.
        
        Args:
            filepath: Path to save metrics.
        """
        import json
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            exportable_metrics = {}
            for key, values in self.metric_history.items():
                if isinstance(values, np.ndarray):
                    exportable_metrics[key] = values.tolist()
                else:
                    exportable_metrics[key] = values
            
            with open(filepath, 'w') as f:
                json.dump(exportable_metrics, f, indent=2)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")