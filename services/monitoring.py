"""
Model validation and monitoring service for Customer Churn Prediction System.
Handles model evaluation, performance monitoring, alerting, and prediction storage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import csv
import json
from dataclasses import dataclass, asdict
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from sklearn.calibration import calibration_curve

import config


@dataclass
class EvaluationResult:
    """Result of model evaluation on a dataset."""
    model_version: str
    evaluation_date: datetime
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    n_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['evaluation_date'] = self.evaluation_date.isoformat()
        return result
    
    def meets_threshold(self, recall_threshold: float = config.MIN_RECALL_THRESHOLD) -> bool:
        """Check if recall meets minimum threshold."""
        return self.recall >= recall_threshold


class ModelEvaluator:
    """
    Evaluates model performance on new data.
    Computes metrics, confusion matrix, and calibration curves.
    """
    
    def __init__(self, model_version: Optional[str] = None):
        """
        Initialize ModelEvaluator.
        
        Args:
            model_version: Optional model version identifier for tracking
        """
        self.model_version = model_version
    
    def evaluate(
        self,
        model: XGBClassifier,
        X: np.ndarray,
        y_true: np.ndarray,
        model_version: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate model performance on provided data.
        
        Args:
            model: Trained XGBClassifier model
            X: Feature array
            y_true: True labels array
            model_version: Optional version string to override instance version
            
        Returns:
            EvaluationResult containing all computed metrics
            
        Raises:
            ValueError: If input data is invalid or empty
        """
        # Validate inputs
        if X.shape[0] == 0:
            raise ValueError("Evaluation data is empty")
        if X.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"X and y_true have different lengths: {X.shape[0]} vs {y_true.shape[0]}"
            )
        
        # Generate predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Compute metrics
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))
        
        # Compute ROC AUC (handle case where only one class is present)
        try:
            roc_auc = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            roc_auc = 0.0
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Use provided version or instance version
        version = model_version if model_version is not None else self.model_version
        if version is None:
            version = "unknown"
        
        # Create evaluation result
        result = EvaluationResult(
            model_version=version,
            evaluation_date=datetime.now(),
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            roc_auc=roc_auc,
            confusion_matrix=conf_matrix.tolist(),
            n_samples=int(X.shape[0])
        )
        
        return result
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix for predictions.
        
        Args:
            y_true: True labels array
            y_pred: Predicted labels array
            
        Returns:
            Confusion matrix as 2D numpy array [[TN, FP], [FN, TP]]
            
        Raises:
            ValueError: If arrays have different lengths
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}"
            )
        
        return confusion_matrix(y_true, y_pred)
    
    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = 'uniform'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve to assess probability accuracy.
        
        Args:
            y_true: True labels array
            y_pred_proba: Predicted probabilities array
            n_bins: Number of bins for calibration curve
            strategy: Strategy for binning ('uniform' or 'quantile')
            
        Returns:
            Tuple of (prob_true, prob_pred) arrays representing calibration curve
            
        Raises:
            ValueError: If arrays have different lengths or invalid parameters
        """
        if len(y_true) != len(y_pred_proba):
            raise ValueError(
                f"y_true and y_pred_proba have different lengths: "
                f"{len(y_true)} vs {len(y_pred_proba)}"
            )
        
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")
        
        if strategy not in ['uniform', 'quantile']:
            raise ValueError(f"strategy must be 'uniform' or 'quantile', got '{strategy}'")
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=n_bins,
            strategy=strategy
        )
        
        return prob_true, prob_pred


@dataclass
class Alert:
    """Represents a performance degradation alert."""
    alert_id: str
    model_version: str
    alert_type: str
    message: str
    timestamp: datetime
    severity: str
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class AlertService:
    """
    Monitors model performance and generates alerts for degradation.
    Checks performance metrics against thresholds and creates alerts.
    """
    
    def __init__(
        self,
        recall_threshold: float = config.MIN_RECALL_THRESHOLD,
        alert_log_path: Optional[Path] = None
    ):
        """
        Initialize AlertService.
        
        Args:
            recall_threshold: Minimum acceptable recall value
            alert_log_path: Optional path to log alerts. If None, uses default location
        """
        self.recall_threshold = recall_threshold
        
        # Set up alert log path
        if alert_log_path is None:
            alert_log_path = config.BASE_DIR / "alerts.json"
        self.alert_log_path = alert_log_path
        
        # Initialize alert counter for unique IDs
        self._alert_counter = 0
    
    def check_performance(
        self,
        evaluation_result: EvaluationResult,
        recall_threshold: Optional[float] = None
    ) -> bool:
        """
        Check if model performance meets recall threshold.
        
        Args:
            evaluation_result: EvaluationResult to check
            recall_threshold: Optional custom threshold. If None, uses instance threshold
            
        Returns:
            True if recall meets threshold, False otherwise
        """
        threshold = recall_threshold if recall_threshold is not None else self.recall_threshold
        return evaluation_result.recall >= threshold
    
    def generate_alert(
        self,
        evaluation_result: EvaluationResult,
        alert_type: str = "performance_degradation",
        severity: str = "high"
    ) -> Alert:
        """
        Generate alert for performance degradation.
        
        Args:
            evaluation_result: EvaluationResult that triggered the alert
            alert_type: Type of alert (e.g., 'performance_degradation')
            severity: Alert severity level ('low', 'medium', 'high', 'critical')
            
        Returns:
            Alert object
        """
        # Generate unique alert ID
        self._alert_counter += 1
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._alert_counter}"
        
        # Create alert message
        message = (
            f"Model performance degradation detected for version {evaluation_result.model_version}. "
            f"Recall: {evaluation_result.recall:.4f} (threshold: {self.recall_threshold:.4f}), "
            f"Precision: {evaluation_result.precision:.4f}, "
            f"F1-Score: {evaluation_result.f1_score:.4f}"
        )
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            model_version=evaluation_result.model_version,
            alert_type=alert_type,
            message=message,
            timestamp=datetime.now(),
            severity=severity,
            metrics={
                'recall': evaluation_result.recall,
                'precision': evaluation_result.precision,
                'f1_score': evaluation_result.f1_score,
                'accuracy': evaluation_result.accuracy,
                'roc_auc': evaluation_result.roc_auc
            }
        )
        
        # Log alert
        self._log_alert(alert)
        
        return alert
    
    def _log_alert(self, alert: Alert) -> None:
        """
        Log alert to file.
        
        Args:
            alert: Alert object to log
        """
        # Read existing alerts
        alerts = []
        if self.alert_log_path.exists():
            try:
                with open(self.alert_log_path, 'r') as f:
                    alerts = json.load(f)
            except (json.JSONDecodeError, IOError):
                alerts = []
        
        # Append new alert
        alerts.append(alert.to_dict())
        
        # Write back to file
        with open(self.alert_log_path, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def get_alerts(
        self,
        model_version: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """
        Retrieve logged alerts.
        
        Args:
            model_version: Optional filter by model version
            limit: Optional limit on number of alerts to return (most recent first)
            
        Returns:
            List of Alert objects
        """
        if not self.alert_log_path.exists():
            return []
        
        try:
            with open(self.alert_log_path, 'r') as f:
                alert_dicts = json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
        
        # Convert to Alert objects
        alerts = []
        for alert_dict in alert_dicts:
            # Parse timestamp
            alert_dict['timestamp'] = datetime.fromisoformat(alert_dict['timestamp'])
            alerts.append(Alert(**alert_dict))
        
        # Filter by model version if specified
        if model_version is not None:
            alerts = [a for a in alerts if a.model_version == model_version]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            alerts = alerts[:limit]
        
        return alerts


@dataclass
class StoredPrediction:
    """Represents a stored prediction with actual outcome."""
    prediction_id: str
    customer_id: str
    model_version: str
    prediction: float
    predicted_class: int
    actual_outcome: Optional[int]
    prediction_timestamp: datetime
    outcome_timestamp: Optional[datetime]
    features: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['prediction_timestamp'] = self.prediction_timestamp.isoformat()
        if self.outcome_timestamp is not None:
            result['outcome_timestamp'] = self.outcome_timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredPrediction':
        """Create from dictionary."""
        data['prediction_timestamp'] = datetime.fromisoformat(data['prediction_timestamp'])
        if data.get('outcome_timestamp'):
            data['outcome_timestamp'] = datetime.fromisoformat(data['outcome_timestamp'])
        return cls(**data)


class PredictionStore:
    """
    Stores predictions with actual outcomes for validation and monitoring.
    Uses CSV for simple, portable storage.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize PredictionStore.
        
        Args:
            storage_path: Path to CSV file for storing predictions. If None, uses default
        """
        if storage_path is None:
            storage_path = config.BASE_DIR / "predictions_store.csv"
        
        self.storage_path = storage_path
        
        # Initialize CSV file with headers if it doesn't exist or is empty
        if not self.storage_path.exists() or self.storage_path.stat().st_size == 0:
            self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize CSV storage file with headers."""
        headers = [
            'prediction_id',
            'customer_id',
            'model_version',
            'prediction',
            'predicted_class',
            'actual_outcome',
            'prediction_timestamp',
            'outcome_timestamp',
            'features'
        ]
        
        with open(self.storage_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def store(
        self,
        customer_id: str,
        model_version: str,
        prediction: float,
        predicted_class: int,
        actual_outcome: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
        prediction_id: Optional[str] = None
    ) -> str:
        """
        Store a prediction with optional actual outcome.
        
        Args:
            customer_id: Customer identifier
            model_version: Model version used for prediction
            prediction: Predicted probability
            predicted_class: Predicted class (0 or 1)
            actual_outcome: Optional actual outcome (0 or 1)
            features: Optional dictionary of customer features
            prediction_id: Optional custom prediction ID
            
        Returns:
            Prediction ID
            
        Raises:
            ValueError: If prediction values are invalid
        """
        # Validate inputs
        if not (0.0 <= prediction <= 1.0):
            raise ValueError(f"Prediction must be between 0 and 1, got {prediction}")
        
        if predicted_class not in [0, 1]:
            raise ValueError(f"Predicted class must be 0 or 1, got {predicted_class}")
        
        if actual_outcome is not None and actual_outcome not in [0, 1]:
            raise ValueError(f"Actual outcome must be 0 or 1, got {actual_outcome}")
        
        # Generate prediction ID if not provided
        if prediction_id is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            prediction_id = f"PRED_{customer_id}_{timestamp_str}"
        
        # Create stored prediction
        stored_pred = StoredPrediction(
            prediction_id=prediction_id,
            customer_id=customer_id,
            model_version=model_version,
            prediction=prediction,
            predicted_class=predicted_class,
            actual_outcome=actual_outcome,
            prediction_timestamp=datetime.now(),
            outcome_timestamp=datetime.now() if actual_outcome is not None else None,
            features=features if features is not None else {}
        )
        
        # Append to CSV
        with open(self.storage_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                stored_pred.prediction_id,
                stored_pred.customer_id,
                stored_pred.model_version,
                stored_pred.prediction,
                stored_pred.predicted_class,
                stored_pred.actual_outcome if stored_pred.actual_outcome is not None else '',
                stored_pred.prediction_timestamp.isoformat(),
                stored_pred.outcome_timestamp.isoformat() if stored_pred.outcome_timestamp else '',
                json.dumps(stored_pred.features)
            ])
        
        return prediction_id
    
    def update_outcome(
        self,
        prediction_id: str,
        actual_outcome: int
    ) -> bool:
        """
        Update the actual outcome for a stored prediction.
        
        Args:
            prediction_id: Prediction ID to update
            actual_outcome: Actual outcome value (0 or 1)
            
        Returns:
            True if update successful, False if prediction not found
            
        Raises:
            ValueError: If actual_outcome is invalid
        """
        if actual_outcome not in [0, 1]:
            raise ValueError(f"Actual outcome must be 0 or 1, got {actual_outcome}")
        
        # Read all predictions
        predictions = self._read_all()
        
        # Find and update the prediction
        updated = False
        for pred in predictions:
            if pred.prediction_id == prediction_id:
                pred.actual_outcome = actual_outcome
                pred.outcome_timestamp = datetime.now()
                updated = True
                break
        
        if not updated:
            return False
        
        # Write back all predictions
        self._write_all(predictions)
        
        return True
    
    def retrieve(
        self,
        customer_id: Optional[str] = None,
        model_version: Optional[str] = None,
        has_outcome: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[StoredPrediction]:
        """
        Retrieve historical predictions with optional filters.
        
        Args:
            customer_id: Optional filter by customer ID
            model_version: Optional filter by model version
            has_outcome: Optional filter by whether actual outcome is recorded
            start_date: Optional filter by prediction timestamp >= start_date
            end_date: Optional filter by prediction timestamp <= end_date
            limit: Optional limit on number of results (most recent first)
            
        Returns:
            List of StoredPrediction objects
        """
        predictions = self._read_all()
        
        # Apply filters
        filtered = []
        for pred in predictions:
            # Customer ID filter
            if customer_id is not None and pred.customer_id != customer_id:
                continue
            
            # Model version filter
            if model_version is not None and pred.model_version != model_version:
                continue
            
            # Has outcome filter
            if has_outcome is not None:
                if has_outcome and pred.actual_outcome is None:
                    continue
                if not has_outcome and pred.actual_outcome is not None:
                    continue
            
            # Date range filters
            if start_date is not None and pred.prediction_timestamp < start_date:
                continue
            if end_date is not None and pred.prediction_timestamp > end_date:
                continue
            
            filtered.append(pred)
        
        # Sort by timestamp (most recent first)
        filtered.sort(key=lambda p: p.prediction_timestamp, reverse=True)
        
        # Apply limit
        if limit is not None:
            filtered = filtered[:limit]
        
        return filtered
    
    def _read_all(self) -> List[StoredPrediction]:
        """Read all predictions from storage."""
        if not self.storage_path.exists():
            return []
        
        predictions = []
        
        with open(self.storage_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row or not row.get('prediction_id'):
                    continue
                
                # Parse fields
                prediction_id = row['prediction_id']
                customer_id = row['customer_id']
                model_version = row['model_version']
                prediction = float(row['prediction'])
                predicted_class = int(row['predicted_class'])
                actual_outcome = int(row['actual_outcome']) if row['actual_outcome'] else None
                prediction_timestamp = datetime.fromisoformat(row['prediction_timestamp'])
                outcome_timestamp = (
                    datetime.fromisoformat(row['outcome_timestamp'])
                    if row['outcome_timestamp'] else None
                )
                features = json.loads(row['features']) if row['features'] else {}
                
                pred = StoredPrediction(
                    prediction_id=prediction_id,
                    customer_id=customer_id,
                    model_version=model_version,
                    prediction=prediction,
                    predicted_class=predicted_class,
                    actual_outcome=actual_outcome,
                    prediction_timestamp=prediction_timestamp,
                    outcome_timestamp=outcome_timestamp,
                    features=features
                )
                predictions.append(pred)
        
        return predictions
    
    def _write_all(self, predictions: List[StoredPrediction]) -> None:
        """Write all predictions to storage."""
        # Reinitialize file
        self._initialize_storage()
        
        # Write all predictions
        with open(self.storage_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for pred in predictions:
                writer.writerow([
                    pred.prediction_id,
                    pred.customer_id,
                    pred.model_version,
                    pred.prediction,
                    pred.predicted_class,
                    pred.actual_outcome if pred.actual_outcome is not None else '',
                    pred.prediction_timestamp.isoformat(),
                    pred.outcome_timestamp.isoformat() if pred.outcome_timestamp else '',
                    json.dumps(pred.features)
                ])
    
    def get_statistics(
        self,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about stored predictions.
        
        Args:
            model_version: Optional filter by model version
            
        Returns:
            Dictionary with statistics
        """
        predictions = self.retrieve(model_version=model_version)
        
        total_predictions = len(predictions)
        predictions_with_outcomes = sum(1 for p in predictions if p.actual_outcome is not None)
        
        stats = {
            'total_predictions': total_predictions,
            'predictions_with_outcomes': predictions_with_outcomes,
            'predictions_without_outcomes': total_predictions - predictions_with_outcomes
        }
        
        # Compute accuracy if we have outcomes
        if predictions_with_outcomes > 0:
            correct = sum(
                1 for p in predictions
                if p.actual_outcome is not None and p.predicted_class == p.actual_outcome
            )
            stats['accuracy'] = correct / predictions_with_outcomes
        
        return stats

