"""
Unit tests for model validation and monitoring service.
Tests ModelEvaluator, AlertService, and PredictionStore classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
import csv
from xgboost import XGBClassifier
from sklearn.datasets import make_classification

from services.monitoring import (
    ModelEvaluator,
    EvaluationResult,
    AlertService,
    Alert,
    PredictionStore,
    StoredPrediction
)
import config


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple trained model for testing."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=43)
        return X, y
    
    def test_evaluate_returns_complete_metrics(self, sample_model, sample_data):
        """Test that evaluate returns all required metrics."""
        X, y = sample_data
        evaluator = ModelEvaluator(model_version="v1.0.0")
        
        result = evaluator.evaluate(sample_model, X, y)
        
        assert isinstance(result, EvaluationResult)
        assert result.model_version == "v1.0.0"
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1_score <= 1.0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.roc_auc <= 1.0
        assert isinstance(result.confusion_matrix, list)
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2
        assert result.n_samples == len(y)
    
    def test_evaluate_with_empty_data_raises_error(self, sample_model):
        """Test that evaluate raises error with empty data."""
        evaluator = ModelEvaluator()
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        with pytest.raises(ValueError, match="Evaluation data is empty"):
            evaluator.evaluate(sample_model, X_empty, y_empty)
    
    def test_evaluate_with_mismatched_lengths_raises_error(self, sample_model):
        """Test that evaluate raises error when X and y have different lengths."""
        evaluator = ModelEvaluator()
        X = np.random.rand(10, 5)
        y = np.array([0, 1, 0])  # Different length
        
        with pytest.raises(ValueError, match="different lengths"):
            evaluator.evaluate(sample_model, X, y)
    
    def test_compute_confusion_matrix(self):
        """Test confusion matrix computation."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = evaluator.compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        # TN=2, FP=1, FN=1, TP=2
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 2  # TP
    
    def test_compute_calibration_curve(self):
        """Test calibration curve computation."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.6, 0.15, 0.85, 0.25])
        
        prob_true, prob_pred = evaluator.compute_calibration_curve(
            y_true, y_pred_proba, n_bins=5
        )
        
        assert len(prob_true) <= 5
        assert len(prob_pred) <= 5
        assert all(0.0 <= p <= 1.0 for p in prob_true)
        assert all(0.0 <= p <= 1.0 for p in prob_pred)
    
    def test_compute_calibration_curve_with_invalid_bins_raises_error(self):
        """Test that calibration curve raises error with invalid n_bins."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.2, 0.8, 0.3, 0.7])
        
        with pytest.raises(ValueError, match="n_bins must be at least 2"):
            evaluator.compute_calibration_curve(y_true, y_pred_proba, n_bins=1)
    
    def test_meets_threshold(self, sample_model, sample_data):
        """Test meets_threshold method."""
        X, y = sample_data
        evaluator = ModelEvaluator(model_version="v1.0.0")
        result = evaluator.evaluate(sample_model, X, y)
        
        # Test with different thresholds
        assert result.meets_threshold(0.0) is True
        assert result.meets_threshold(1.0) is False


class TestAlertService:
    """Test AlertService class."""
    
    @pytest.fixture
    def temp_alert_log(self):
        """Create temporary alert log file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def sample_evaluation_result(self):
        """Create sample evaluation result."""
        return EvaluationResult(
            model_version="v1.0.0",
            evaluation_date=datetime.now(),
            precision=0.80,
            recall=0.75,  # Below default threshold of 0.85
            f1_score=0.77,
            accuracy=0.82,
            roc_auc=0.85,
            confusion_matrix=[[40, 10], [5, 45]],
            n_samples=100
        )
    
    def test_check_performance_below_threshold(self, sample_evaluation_result):
        """Test check_performance returns False when below threshold."""
        alert_service = AlertService(recall_threshold=0.85)
        
        result = alert_service.check_performance(sample_evaluation_result)
        
        assert result is False
    
    def test_check_performance_above_threshold(self):
        """Test check_performance returns True when above threshold."""
        alert_service = AlertService(recall_threshold=0.85)
        
        good_result = EvaluationResult(
            model_version="v1.0.0",
            evaluation_date=datetime.now(),
            precision=0.90,
            recall=0.90,  # Above threshold
            f1_score=0.90,
            accuracy=0.90,
            roc_auc=0.92,
            confusion_matrix=[[45, 5], [5, 45]],
            n_samples=100
        )
        
        result = alert_service.check_performance(good_result)
        
        assert result is True
    
    def test_generate_alert(self, temp_alert_log, sample_evaluation_result):
        """Test alert generation."""
        alert_service = AlertService(
            recall_threshold=0.85,
            alert_log_path=temp_alert_log
        )
        
        alert = alert_service.generate_alert(sample_evaluation_result)
        
        assert isinstance(alert, Alert)
        assert alert.model_version == "v1.0.0"
        assert alert.alert_type == "performance_degradation"
        assert alert.severity == "high"
        assert "degradation" in alert.message.lower()
        assert alert.metrics['recall'] == 0.75
        assert alert.metrics['precision'] == 0.80
    
    def test_alert_is_logged(self, temp_alert_log, sample_evaluation_result):
        """Test that alerts are logged to file."""
        alert_service = AlertService(
            recall_threshold=0.85,
            alert_log_path=temp_alert_log
        )
        
        alert = alert_service.generate_alert(sample_evaluation_result)
        
        # Verify file exists and contains alert
        assert temp_alert_log.exists()
        with open(temp_alert_log, 'r') as f:
            alerts = json.load(f)
        
        assert len(alerts) == 1
        assert alerts[0]['alert_id'] == alert.alert_id
        assert alerts[0]['model_version'] == "v1.0.0"
    
    def test_get_alerts(self, temp_alert_log, sample_evaluation_result):
        """Test retrieving alerts."""
        alert_service = AlertService(
            recall_threshold=0.85,
            alert_log_path=temp_alert_log
        )
        
        # Generate multiple alerts
        alert1 = alert_service.generate_alert(sample_evaluation_result)
        alert2 = alert_service.generate_alert(sample_evaluation_result)
        
        # Retrieve all alerts
        alerts = alert_service.get_alerts()
        
        assert len(alerts) == 2
        assert all(isinstance(a, Alert) for a in alerts)
    
    def test_get_alerts_with_version_filter(self, temp_alert_log):
        """Test retrieving alerts filtered by model version."""
        alert_service = AlertService(
            recall_threshold=0.85,
            alert_log_path=temp_alert_log
        )
        
        # Create alerts for different versions
        result_v1 = EvaluationResult(
            model_version="v1.0.0",
            evaluation_date=datetime.now(),
            precision=0.80, recall=0.75, f1_score=0.77,
            accuracy=0.82, roc_auc=0.85,
            confusion_matrix=[[40, 10], [5, 45]],
            n_samples=100
        )
        result_v2 = EvaluationResult(
            model_version="v2.0.0",
            evaluation_date=datetime.now(),
            precision=0.80, recall=0.75, f1_score=0.77,
            accuracy=0.82, roc_auc=0.85,
            confusion_matrix=[[40, 10], [5, 45]],
            n_samples=100
        )
        
        alert_service.generate_alert(result_v1)
        alert_service.generate_alert(result_v2)
        
        # Filter by version
        v1_alerts = alert_service.get_alerts(model_version="v1.0.0")
        
        assert len(v1_alerts) == 1
        assert v1_alerts[0].model_version == "v1.0.0"


class TestPredictionStore:
    """Test PredictionStore class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_store_prediction(self, temp_storage):
        """Test storing a prediction."""
        store = PredictionStore(storage_path=temp_storage)
        
        pred_id = store.store(
            customer_id="CUST001",
            model_version="v1.0.0",
            prediction=0.75,
            predicted_class=1,
            actual_outcome=1,
            features={"tenure_months": 12, "monthly_charges": 50.0}
        )
        
        assert pred_id.startswith("PRED_CUST001_")
    
    def test_store_prediction_with_invalid_probability_raises_error(self, temp_storage):
        """Test that storing prediction with invalid probability raises error."""
        store = PredictionStore(storage_path=temp_storage)
        
        with pytest.raises(ValueError, match="Prediction must be between 0 and 1"):
            store.store(
                customer_id="CUST001",
                model_version="v1.0.0",
                prediction=1.5,  # Invalid
                predicted_class=1
            )
    
    def test_store_prediction_with_invalid_class_raises_error(self, temp_storage):
        """Test that storing prediction with invalid class raises error."""
        store = PredictionStore(storage_path=temp_storage)
        
        with pytest.raises(ValueError, match="Predicted class must be 0 or 1"):
            store.store(
                customer_id="CUST001",
                model_version="v1.0.0",
                prediction=0.75,
                predicted_class=2  # Invalid
            )
    
    def test_retrieve_all_predictions(self, temp_storage):
        """Test retrieving all predictions."""
        store = PredictionStore(storage_path=temp_storage)
        
        # Store multiple predictions
        store.store("CUST001", "v1.0.0", 0.75, 1, 1)
        store.store("CUST002", "v1.0.0", 0.25, 0, 0)
        store.store("CUST003", "v1.0.0", 0.60, 1, None)
        
        predictions = store.retrieve()
        
        assert len(predictions) == 3
        assert all(isinstance(p, StoredPrediction) for p in predictions)
    
    def test_retrieve_with_customer_filter(self, temp_storage):
        """Test retrieving predictions filtered by customer ID."""
        store = PredictionStore(storage_path=temp_storage)
        
        store.store("CUST001", "v1.0.0", 0.75, 1)
        store.store("CUST002", "v1.0.0", 0.25, 0)
        store.store("CUST001", "v1.0.0", 0.80, 1)
        
        predictions = store.retrieve(customer_id="CUST001")
        
        assert len(predictions) == 2
        assert all(p.customer_id == "CUST001" for p in predictions)
    
    def test_retrieve_with_outcome_filter(self, temp_storage):
        """Test retrieving predictions filtered by outcome presence."""
        store = PredictionStore(storage_path=temp_storage)
        
        store.store("CUST001", "v1.0.0", 0.75, 1, actual_outcome=1)
        store.store("CUST002", "v1.0.0", 0.25, 0, actual_outcome=None)
        store.store("CUST003", "v1.0.0", 0.60, 1, actual_outcome=0)
        
        # Get predictions with outcomes
        with_outcomes = store.retrieve(has_outcome=True)
        assert len(with_outcomes) == 2
        
        # Get predictions without outcomes
        without_outcomes = store.retrieve(has_outcome=False)
        assert len(without_outcomes) == 1
    
    def test_update_outcome(self, temp_storage):
        """Test updating actual outcome for a prediction."""
        store = PredictionStore(storage_path=temp_storage)
        
        pred_id = store.store("CUST001", "v1.0.0", 0.75, 1, actual_outcome=None)
        
        # Update outcome
        success = store.update_outcome(pred_id, actual_outcome=1)
        
        assert success is True
        
        # Verify update
        predictions = store.retrieve(customer_id="CUST001")
        assert len(predictions) == 1
        assert predictions[0].actual_outcome == 1
        assert predictions[0].outcome_timestamp is not None
    
    def test_update_outcome_for_nonexistent_prediction(self, temp_storage):
        """Test updating outcome for non-existent prediction returns False."""
        store = PredictionStore(storage_path=temp_storage)
        
        success = store.update_outcome("NONEXISTENT_ID", actual_outcome=1)
        
        assert success is False
    
    def test_get_statistics(self, temp_storage):
        """Test getting statistics about stored predictions."""
        store = PredictionStore(storage_path=temp_storage)
        
        # Store predictions with and without outcomes
        store.store("CUST001", "v1.0.0", 0.75, 1, actual_outcome=1)
        store.store("CUST002", "v1.0.0", 0.25, 0, actual_outcome=0)
        store.store("CUST003", "v1.0.0", 0.60, 1, actual_outcome=None)
        store.store("CUST004", "v1.0.0", 0.80, 1, actual_outcome=1)
        
        stats = store.get_statistics()
        
        assert stats['total_predictions'] == 4
        assert stats['predictions_with_outcomes'] == 3
        assert stats['predictions_without_outcomes'] == 1
        assert 'accuracy' in stats
        assert stats['accuracy'] == 1.0  # All predictions were correct
    
    def test_retrieve_with_limit(self, temp_storage):
        """Test retrieving predictions with limit."""
        store = PredictionStore(storage_path=temp_storage)
        
        # Store multiple predictions
        for i in range(10):
            store.store(f"CUST{i:03d}", "v1.0.0", 0.5, 0)
        
        predictions = store.retrieve(limit=5)
        
        assert len(predictions) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
