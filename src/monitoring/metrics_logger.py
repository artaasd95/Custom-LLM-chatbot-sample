"""Metrics logging and monitoring utilities."""

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict, deque
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..core.config import ConfigManager


class MetricsLogger:
    """Advanced metrics logging and system monitoring."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize metrics logger.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_history = defaultdict(deque)
        self.system_metrics = defaultdict(deque)
        self.custom_metrics = defaultdict(deque)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
        
        # Performance tracking
        self.step_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_usage = deque(maxlen=1000)
        
        # Callbacks
        self.metric_callbacks = []
        self.alert_callbacks = []
        
        # Thresholds for alerts
        self.alert_thresholds = {
            'memory_usage_percent': 90.0,
            'gpu_memory_percent': 95.0,
            'training_loss_spike': 2.0,  # multiplier for sudden increases
            'gradient_norm_threshold': 10.0
        }
        
        self.logger.info("MetricsLogger initialized")
    
    def start_monitoring(self, interval: int = 30) -> None:
        """Start system monitoring in a separate thread.
        
        Args:
            interval: Monitoring interval in seconds.
        """
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"System monitoring started with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        timestamp = datetime.now().isoformat()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_sent_mb = network.bytes_sent / (1024**2)
        network_recv_mb = network.bytes_recv / (1024**2)
        
        system_metrics = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'disk_percent': disk_percent,
            'disk_used_gb': disk_used_gb,
            'disk_total_gb': disk_total_gb,
            'network_sent_mb': network_sent_mb,
            'network_recv_mb': network_recv_mb
        }
        
        # GPU metrics (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_metrics = self._collect_gpu_metrics()
            system_metrics.update(gpu_metrics)
        
        # Store metrics
        for key, value in system_metrics.items():
            if key != 'timestamp':
                self.system_metrics[key].append({
                    'value': value,
                    'timestamp': timestamp
                })
        
        # Check for alerts
        self._check_system_alerts(system_metrics)
        
        # Execute callbacks
        for callback in self.metric_callbacks:
            try:
                callback(system_metrics)
            except Exception as e:
                self.logger.error(f"Error in metric callback: {str(e)}")
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU performance metrics.
        
        Returns:
            Dictionary of GPU metrics.
        """
        gpu_metrics = {}
        
        try:
            # Basic GPU info
            gpu_count = torch.cuda.device_count()
            gpu_metrics['gpu_count'] = gpu_count
            
            for i in range(gpu_count):
                device = f'cuda:{i}'
                
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
                max_memory = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
                
                # GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    memory_util = utilization.memory
                    
                    # Temperature
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Power usage
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                    
                    gpu_metrics.update({
                        f'gpu_{i}_utilization': gpu_util,
                        f'gpu_{i}_memory_utilization': memory_util,
                        f'gpu_{i}_temperature': temperature,
                        f'gpu_{i}_power_usage': power_usage
                    })
                    
                except ImportError:
                    self.logger.debug("pynvml not available for detailed GPU metrics")
                except Exception as e:
                    self.logger.debug(f"Error collecting detailed GPU metrics: {str(e)}")
                
                gpu_metrics.update({
                    f'gpu_{i}_memory_allocated_gb': memory_allocated,
                    f'gpu_{i}_memory_reserved_gb': memory_reserved,
                    f'gpu_{i}_max_memory_gb': max_memory
                })
                
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {str(e)}")
        
        return gpu_metrics
    
    def _check_system_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check system metrics against alert thresholds.
        
        Args:
            metrics: Current system metrics.
        """
        alerts = []
        
        # Memory usage alert
        if metrics.get('memory_percent', 0) > self.alert_thresholds['memory_usage_percent']:
            alerts.append({
                'type': 'memory_usage',
                'severity': 'warning',
                'message': f"High memory usage: {metrics['memory_percent']:.1f}%",
                'value': metrics['memory_percent'],
                'threshold': self.alert_thresholds['memory_usage_percent']
            })
        
        # GPU memory alert
        for key, value in metrics.items():
            if 'gpu_' in key and 'memory_utilization' in key:
                if value > self.alert_thresholds['gpu_memory_percent']:
                    alerts.append({
                        'type': 'gpu_memory',
                        'severity': 'warning',
                        'message': f"High GPU memory usage on {key}: {value:.1f}%",
                        'value': value,
                        'threshold': self.alert_thresholds['gpu_memory_percent']
                    })
        
        # Execute alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
    
    def log_training_metric(self, 
                           name: str, 
                           value: float, 
                           step: Optional[int] = None,
                           epoch: Optional[int] = None) -> None:
        """Log a training metric with additional context.
        
        Args:
            name: Metric name.
            value: Metric value.
            step: Training step.
            epoch: Training epoch.
        """
        timestamp = datetime.now().isoformat()
        
        metric_entry = {
            'value': value,
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp
        }
        
        self.metrics_history[name].append(metric_entry)
        
        # Check for training alerts
        self._check_training_alerts(name, value)
        
        self.logger.debug(f"Training metric logged: {name}={value} (step={step}, epoch={epoch})")
    
    def _check_training_alerts(self, name: str, value: float) -> None:
        """Check training metrics for anomalies.
        
        Args:
            name: Metric name.
            value: Current metric value.
        """
        alerts = []
        
        # Loss spike detection
        if 'loss' in name.lower():
            history = [entry['value'] for entry in self.metrics_history[name]]
            if len(history) >= 5:
                recent_avg = sum(history[-5:-1]) / 4  # Average of last 4 values
                if value > recent_avg * self.alert_thresholds['training_loss_spike']:
                    alerts.append({
                        'type': 'loss_spike',
                        'severity': 'warning',
                        'message': f"Loss spike detected in {name}: {value:.4f} (recent avg: {recent_avg:.4f})",
                        'value': value,
                        'recent_average': recent_avg
                    })
        
        # Gradient norm alert
        if 'grad_norm' in name.lower():
            if value > self.alert_thresholds['gradient_norm_threshold']:
                alerts.append({
                    'type': 'high_gradient_norm',
                    'severity': 'warning',
                    'message': f"High gradient norm detected: {value:.4f}",
                    'value': value,
                    'threshold': self.alert_thresholds['gradient_norm_threshold']
                })
        
        # Execute alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
    
    def log_step_time(self, step_time: float) -> None:
        """Log training step time.
        
        Args:
            step_time: Time taken for the step in seconds.
        """
        self.step_times.append({
            'time': step_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_custom_metric(self, 
                         category: str, 
                         name: str, 
                         value: Any,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a custom metric with category and metadata.
        
        Args:
            category: Metric category.
            name: Metric name.
            value: Metric value.
            metadata: Optional metadata.
        """
        metric_key = f"{category}.{name}"
        
        metric_entry = {
            'value': value,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.custom_metrics[metric_key].append(metric_entry)
        
        self.logger.debug(f"Custom metric logged: {metric_key}={value}")
    
    def get_metric_summary(self, name: str, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name.
            last_n: Number of recent values to consider.
            
        Returns:
            Dictionary with summary statistics.
        """
        if name not in self.metrics_history:
            return {}
        
        values = [entry['value'] for entry in self.metrics_history[name]]
        if last_n:
            values = values[-last_n:]
        
        if not values:
            return {}
        
        if NUMPY_AVAILABLE:
            summary = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        else:
            summary = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
        
        return summary
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics.
        
        Returns:
            Dictionary with system metrics summary.
        """
        summary = {}
        
        for metric_name, metric_data in self.system_metrics.items():
            if metric_data:
                values = [entry['value'] for entry in metric_data]
                if NUMPY_AVAILABLE:
                    summary[metric_name] = {
                        'current': values[-1] if values else None,
                        'mean': float(np.mean(values)),
                        'max': float(np.max(values)),
                        'min': float(np.min(values))
                    }
                else:
                    summary[metric_name] = {
                        'current': values[-1] if values else None,
                        'mean': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values)
                    }
        
        return summary
    
    def export_metrics(self, output_dir: str) -> None:
        """Export all metrics to files.
        
        Args:
            output_dir: Directory to save metrics files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export training metrics
        training_metrics = {}
        for name, data in self.metrics_history.items():
            training_metrics[name] = list(data)
        
        with open(output_path / "training_metrics.json", 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        # Export system metrics
        system_metrics = {}
        for name, data in self.system_metrics.items():
            system_metrics[name] = list(data)
        
        with open(output_path / "system_metrics.json", 'w') as f:
            json.dump(system_metrics, f, indent=2)
        
        # Export custom metrics
        custom_metrics = {}
        for name, data in self.custom_metrics.items():
            custom_metrics[name] = list(data)
        
        with open(output_path / "custom_metrics.json", 'w') as f:
            json.dump(custom_metrics, f, indent=2)
        
        # Export summaries
        summaries = {
            'training_summaries': {name: self.get_metric_summary(name) for name in self.metrics_history.keys()},
            'system_summary': self.get_system_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "metrics_summary.json", 'w') as f:
            json.dump(summaries, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")
    
    def add_metric_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback function for metric updates.
        
        Args:
            callback: Function to call with metric updates.
        """
        self.metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback function for alerts.
        
        Args:
            callback: Function to call with alert information.
        """
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric.
        
        Args:
            metric_name: Name of the metric.
            threshold: Alert threshold value.
        """
        self.alert_thresholds[metric_name] = threshold
        self.logger.info(f"Alert threshold set: {metric_name} = {threshold}")
    
    def clear_metrics(self, metric_name: Optional[str] = None) -> None:
        """Clear stored metrics.
        
        Args:
            metric_name: Specific metric to clear, or None to clear all.
        """
        if metric_name:
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].clear()
            if metric_name in self.system_metrics:
                self.system_metrics[metric_name].clear()
            if metric_name in self.custom_metrics:
                self.custom_metrics[metric_name].clear()
        else:
            self.metrics_history.clear()
            self.system_metrics.clear()
            self.custom_metrics.clear()
            self.step_times.clear()
            self.memory_usage.clear()
            self.gpu_usage.clear()
        
        self.logger.info(f"Metrics cleared: {metric_name or 'all'}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance analysis.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'total_metrics': len(self.metrics_history),
            'system_metrics_count': len(self.system_metrics),
            'custom_metrics_count': len(self.custom_metrics)
        }
        
        # Step timing analysis
        if self.step_times:
            step_times = [entry['time'] for entry in self.step_times]
            if NUMPY_AVAILABLE:
                report['step_timing'] = {
                    'mean_time': float(np.mean(step_times)),
                    'std_time': float(np.std(step_times)),
                    'min_time': float(np.min(step_times)),
                    'max_time': float(np.max(step_times)),
                    'total_steps': len(step_times)
                }
            else:
                report['step_timing'] = {
                    'mean_time': sum(step_times) / len(step_times),
                    'min_time': min(step_times),
                    'max_time': max(step_times),
                    'total_steps': len(step_times)
                }
        
        # System performance summary
        report['system_performance'] = self.get_system_summary()
        
        # Training metrics summary
        report['training_metrics'] = {}
        for name in self.metrics_history.keys():
            report['training_metrics'][name] = self.get_metric_summary(name, last_n=100)
        
        return report