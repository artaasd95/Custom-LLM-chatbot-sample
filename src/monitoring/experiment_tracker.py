"""Experiment tracking and monitoring for training runs."""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    comet_ml = None

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from ..core.config import ConfigManager


class ExperimentTracker:
    """Unified experiment tracking across multiple platforms."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize experiment tracker.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Tracking state
        self.current_experiment_id = None
        self.current_run_id = None
        self.active_trackers = []
        
        # Platform instances
        self.wandb_run = None
        self.comet_experiment = None
        self.mlflow_run = None
        
        # Local logging
        self.local_log_dir = None
        self.local_metrics = []
        self.local_parameters = {}
        
        self.logger.info("ExperimentTracker initialized")
    
    def start_experiment(self, 
                        experiment_name: str,
                        tags: Optional[List[str]] = None,
                        description: Optional[str] = None) -> str:
        """Start a new experiment across all configured platforms.
        
        Args:
            experiment_name: Name of the experiment.
            tags: Optional list of tags.
            description: Optional experiment description.
            
        Returns:
            Experiment ID.
        """
        self.current_experiment_id = str(uuid.uuid4())
        self.current_run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        # Setup local logging
        self._setup_local_logging(experiment_name)
        
        # Initialize tracking platforms
        if self.config.monitoring.wandb.enabled and WANDB_AVAILABLE:
            self._start_wandb_experiment(experiment_name, tags, description)
        
        if self.config.monitoring.comet.enabled and COMET_AVAILABLE:
            self._start_comet_experiment(experiment_name, tags, description)
        
        if self.config.monitoring.mlflow.enabled and MLFLOW_AVAILABLE:
            self._start_mlflow_experiment(experiment_name, tags, description)
        
        # Log experiment metadata
        self._log_experiment_metadata(experiment_name, tags, description)
        
        self.logger.info(f"Experiment started with ID: {self.current_experiment_id}")
        return self.current_experiment_id
    
    def _setup_local_logging(self, experiment_name: str) -> None:
        """Setup local logging directory and files.
        
        Args:
            experiment_name: Name of the experiment.
        """
        self.local_log_dir = Path(self.config.training.output_dir) / "experiments" / self.current_run_id
        self.local_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize local tracking files
        self.local_metrics = []
        self.local_parameters = {}
        
        self.logger.info(f"Local logging setup at: {self.local_log_dir}")
    
    def _start_wandb_experiment(self, 
                               experiment_name: str,
                               tags: Optional[List[str]] = None,
                               description: Optional[str] = None) -> None:
        """Start Weights & Biases experiment.
        
        Args:
            experiment_name: Name of the experiment.
            tags: Optional list of tags.
            description: Optional experiment description.
        """
        try:
            self.wandb_run = wandb.init(
                project=self.config.monitoring.wandb.project,
                name=self.current_run_id,
                tags=tags or [],
                notes=description,
                config=self.config.to_dict(),
                reinit=True
            )
            
            self.active_trackers.append('wandb')
            self.logger.info("Weights & Biases experiment started")
            
        except Exception as e:
            self.logger.error(f"Failed to start Weights & Biases experiment: {str(e)}")
    
    def _start_comet_experiment(self, 
                               experiment_name: str,
                               tags: Optional[List[str]] = None,
                               description: Optional[str] = None) -> None:
        """Start Comet ML experiment.
        
        Args:
            experiment_name: Name of the experiment.
            tags: Optional list of tags.
            description: Optional experiment description.
        """
        try:
            self.comet_experiment = comet_ml.Experiment(
                api_key=self.config.monitoring.comet.api_key,
                project_name=self.config.monitoring.comet.project_name,
                workspace=self.config.monitoring.comet.workspace
            )
            
            self.comet_experiment.set_name(self.current_run_id)
            if tags:
                self.comet_experiment.add_tags(tags)
            if description:
                self.comet_experiment.log_other("description", description)
            
            # Log configuration
            self.comet_experiment.log_parameters(self.config.to_dict())
            
            self.active_trackers.append('comet')
            self.logger.info("Comet ML experiment started")
            
        except Exception as e:
            self.logger.error(f"Failed to start Comet ML experiment: {str(e)}")
    
    def _start_mlflow_experiment(self, 
                                experiment_name: str,
                                tags: Optional[List[str]] = None,
                                description: Optional[str] = None) -> None:
        """Start MLflow experiment.
        
        Args:
            experiment_name: Name of the experiment.
            tags: Optional list of tags.
            description: Optional experiment description.
        """
        try:
            # Set tracking URI if configured
            if self.config.monitoring.mlflow.tracking_uri:
                mlflow.set_tracking_uri(self.config.monitoring.mlflow.tracking_uri)
            
            # Set or create experiment
            try:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    tags={"description": description} if description else None
                )
            except mlflow.exceptions.MlflowException:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id
            
            # Start run
            self.mlflow_run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.current_run_id,
                tags=dict(zip(tags or [], [True] * len(tags or [])))
            )
            
            # Log configuration
            mlflow.log_params(self._flatten_dict(self.config.to_dict()))
            
            self.active_trackers.append('mlflow')
            self.logger.info("MLflow experiment started")
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow experiment: {str(e)}")
    
    def _log_experiment_metadata(self, 
                                experiment_name: str,
                                tags: Optional[List[str]] = None,
                                description: Optional[str] = None) -> None:
        """Log experiment metadata locally.
        
        Args:
            experiment_name: Name of the experiment.
            tags: Optional list of tags.
            description: Optional experiment description.
        """
        metadata = {
            "experiment_id": self.current_experiment_id,
            "run_id": self.current_run_id,
            "experiment_name": experiment_name,
            "tags": tags or [],
            "description": description,
            "start_time": datetime.now().isoformat(),
            "active_trackers": self.active_trackers,
            "config": self.config.to_dict()
        }
        
        metadata_file = self.local_log_dir / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log_metric(self, 
                   name: str, 
                   value: Union[int, float], 
                   step: Optional[int] = None) -> None:
        """Log a metric across all active trackers.
        
        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        # Log to local storage
        metric_entry = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.local_metrics.append(metric_entry)
        
        # Log to active trackers
        if 'wandb' in self.active_trackers and self.wandb_run:
            try:
                self.wandb_run.log({name: value}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log metric to Weights & Biases: {str(e)}")
        
        if 'comet' in self.active_trackers and self.comet_experiment:
            try:
                self.comet_experiment.log_metric(name, value, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log metric to Comet ML: {str(e)}")
        
        if 'mlflow' in self.active_trackers and self.mlflow_run:
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log metric to MLflow: {str(e)}")
    
    def log_metrics(self, 
                    metrics: Dict[str, Union[int, float]], 
                    step: Optional[int] = None) -> None:
        """Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_parameter(self, name: str, value: Any) -> None:
        """Log a parameter across all active trackers.
        
        Args:
            name: Parameter name.
            value: Parameter value.
        """
        # Store locally
        self.local_parameters[name] = value
        
        # Log to active trackers
        if 'wandb' in self.active_trackers and self.wandb_run:
            try:
                wandb.config.update({name: value})
            except Exception as e:
                self.logger.warning(f"Failed to log parameter to Weights & Biases: {str(e)}")
        
        if 'comet' in self.active_trackers and self.comet_experiment:
            try:
                self.comet_experiment.log_parameter(name, value)
            except Exception as e:
                self.logger.warning(f"Failed to log parameter to Comet ML: {str(e)}")
        
        if 'mlflow' in self.active_trackers and self.mlflow_run:
            try:
                mlflow.log_param(name, value)
            except Exception as e:
                self.logger.warning(f"Failed to log parameter to MLflow: {str(e)}")
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log multiple parameters at once.
        
        Args:
            parameters: Dictionary of parameter names and values.
        """
        for name, value in parameters.items():
            self.log_parameter(name, value)
    
    def log_artifact(self, 
                     artifact_path: str, 
                     artifact_name: Optional[str] = None) -> None:
        """Log an artifact (file) across all active trackers.
        
        Args:
            artifact_path: Path to the artifact file.
            artifact_name: Optional name for the artifact.
        """
        artifact_name = artifact_name or Path(artifact_path).name
        
        # Copy to local log directory
        try:
            import shutil
            local_artifact_path = self.local_log_dir / "artifacts" / artifact_name
            local_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, local_artifact_path)
        except Exception as e:
            self.logger.warning(f"Failed to copy artifact locally: {str(e)}")
        
        # Log to active trackers
        if 'wandb' in self.active_trackers and self.wandb_run:
            try:
                self.wandb_run.log_artifact(artifact_path, name=artifact_name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to Weights & Biases: {str(e)}")
        
        if 'comet' in self.active_trackers and self.comet_experiment:
            try:
                self.comet_experiment.log_asset(artifact_path, file_name=artifact_name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to Comet ML: {str(e)}")
        
        if 'mlflow' in self.active_trackers and self.mlflow_run:
            try:
                mlflow.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to MLflow: {str(e)}")
    
    def log_model(self, model_path: str, model_name: Optional[str] = None) -> None:
        """Log a trained model across all active trackers.
        
        Args:
            model_path: Path to the model directory.
            model_name: Optional name for the model.
        """
        model_name = model_name or "trained_model"
        
        # Log to active trackers
        if 'wandb' in self.active_trackers and self.wandb_run:
            try:
                model_artifact = wandb.Artifact(model_name, type="model")
                model_artifact.add_dir(model_path)
                self.wandb_run.log_artifact(model_artifact)
            except Exception as e:
                self.logger.warning(f"Failed to log model to Weights & Biases: {str(e)}")
        
        if 'comet' in self.active_trackers and self.comet_experiment:
            try:
                self.comet_experiment.log_model(model_name, model_path)
            except Exception as e:
                self.logger.warning(f"Failed to log model to Comet ML: {str(e)}")
        
        if 'mlflow' in self.active_trackers and self.mlflow_run:
            try:
                mlflow.log_artifacts(model_path, "model")
            except Exception as e:
                self.logger.warning(f"Failed to log model to MLflow: {str(e)}")
    
    def log_text(self, 
                 name: str, 
                 text: str, 
                 step: Optional[int] = None) -> None:
        """Log text content across all active trackers.
        
        Args:
            name: Name for the text log.
            text: Text content to log.
            step: Optional step number.
        """
        # Save locally
        text_file = self.local_log_dir / "text_logs" / f"{name}_{step or 'final'}.txt"
        text_file.parent.mkdir(parents=True, exist_ok=True)
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Log to active trackers
        if 'wandb' in self.active_trackers and self.wandb_run:
            try:
                self.wandb_run.log({name: wandb.Html(text)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log text to Weights & Biases: {str(e)}")
        
        if 'comet' in self.active_trackers and self.comet_experiment:
            try:
                self.comet_experiment.log_text(text, metadata={"name": name, "step": step})
            except Exception as e:
                self.logger.warning(f"Failed to log text to Comet ML: {str(e)}")
    
    def end_experiment(self) -> None:
        """End the current experiment and cleanup resources."""
        if not self.current_experiment_id:
            return
        
        self.logger.info(f"Ending experiment: {self.current_experiment_id}")
        
        # Save local logs
        self._save_local_logs()
        
        # End tracking sessions
        if 'wandb' in self.active_trackers and self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                self.logger.warning(f"Failed to end Weights & Biases run: {str(e)}")
        
        if 'comet' in self.active_trackers and self.comet_experiment:
            try:
                self.comet_experiment.end()
            except Exception as e:
                self.logger.warning(f"Failed to end Comet ML experiment: {str(e)}")
        
        if 'mlflow' in self.active_trackers and self.mlflow_run:
            try:
                mlflow.end_run()
            except Exception as e:
                self.logger.warning(f"Failed to end MLflow run: {str(e)}")
        
        # Reset state
        self.current_experiment_id = None
        self.current_run_id = None
        self.active_trackers = []
        self.wandb_run = None
        self.comet_experiment = None
        self.mlflow_run = None
        
        self.logger.info("Experiment ended successfully")
    
    def _save_local_logs(self) -> None:
        """Save local logs to files."""
        if not self.local_log_dir:
            return
        
        try:
            # Save metrics
            metrics_file = self.local_log_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.local_metrics, f, indent=2)
            
            # Save parameters
            params_file = self.local_log_dir / "parameters.json"
            with open(params_file, 'w') as f:
                json.dump(self.local_parameters, f, indent=2)
            
            # Save experiment summary
            summary = {
                "experiment_id": self.current_experiment_id,
                "run_id": self.current_run_id,
                "end_time": datetime.now().isoformat(),
                "total_metrics": len(self.local_metrics),
                "total_parameters": len(self.local_parameters),
                "active_trackers": self.active_trackers
            }
            
            summary_file = self.local_log_dir / "experiment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Local logs saved to {self.local_log_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save local logs: {str(e)}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow parameter logging.
        
        Args:
            d: Dictionary to flatten.
            parent_key: Parent key for nested items.
            sep: Separator for nested keys.
            
        Returns:
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_experiment_url(self, platform: str = 'wandb') -> Optional[str]:
        """Get the URL for the current experiment on a specific platform.
        
        Args:
            platform: Platform name ('wandb', 'comet', 'mlflow').
            
        Returns:
            Experiment URL or None if not available.
        """
        if platform == 'wandb' and self.wandb_run:
            return self.wandb_run.get_url()
        elif platform == 'comet' and self.comet_experiment:
            return self.comet_experiment.url
        elif platform == 'mlflow' and self.mlflow_run:
            return f"{mlflow.get_tracking_uri()}/#/experiments/{self.mlflow_run.info.experiment_id}/runs/{self.mlflow_run.info.run_id}"
        
        return None
    
    def is_active(self) -> bool:
        """Check if an experiment is currently active.
        
        Returns:
            True if experiment is active, False otherwise.
        """
        return self.current_experiment_id is not None