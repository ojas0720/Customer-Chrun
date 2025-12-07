"""
Integration tests for model validation and monitoring service.
Tests end-to-end workflows combining ModelEvaluator, AlertService, and PredictionStore.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
from xgboost import XGBClassifier
from sklearn.datasets import make_classification

from services.monitoring import (
    ModelEvaluator,
    AlertService,
    PredictionStore
)
from services.model_repository import ModelRepository
from services.preprocessing import DataTransformer, FeatureEngineer
from services.prediction import ChurnPredictor
import config


class TestMonitoringIntegration:
    """Integration tests for monitoring service."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model and test data."""
        # Create training data
        X_train, y_train = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        # Create test data
        X_test, y_test = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=43
        )
        
        # Train model
        model = XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        alert_log = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        pred_store = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        
        alert_path = Path(alert_log.name)
        store_path = Path(pred_store.name)
        
        alert_log.close()
        pred_store.close()
        
        yield alert_path, store_path
        
        # Cleanup
        if alert_path.exists():
            alert_path.unlink()
        if store_path.exists():
            store_path.unlink()
    
    def test_complete_monitoring_workflow(self, trained_model_and_data, temp_files):
        """Test complete monitoring workflow: evaluate, alert, and store."""
        model, X_test, y_test = trained_model_and_data
        alert_path, store_path = temp_files
        
        # Initialize services
        evaluator = ModelEvaluator(model_version="v1.0.0")
        alert_service = AlertService(
            recall_threshold=0.85,
            alert_log_path=alert_path
        )
        pred_store = PredictionStore(storage_path=store_path)
        
        # Step 1: Evaluate model
        eval_result = evaluator.evaluate(model, X_test, y_test)
        
        assert eval_result.model_version == "v1.0.0"
        assert 0.0 <= eval_result.recall <= 1.0
        
        # Step 2: Check performance and generate alert if needed
        performance_ok = alert_service.check_performance(eval_result)
        
        if not performance_ok:
            alert = alert_service.generate_alert(eval_result)
            assert alert.model_version == "v1.0.0"
            assert "degradation" in alert.message.lower()
        
        # Step 3: Store predictions with outcomes
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        for i in range(min(10, len(y_test))):  # Store first 10 predictions
            pred_store.store(
                customer_id=f"CUST{i:03d}",
                model_version="v1.0.0",
                prediction=float(y_pred_proba[i]),
                predicted_class=int(y_pred[i]),
                actual_outcome=int(y_test[i])
            )
        
        # Step 4: Retrieve and verify stored predictions
        stored_predictions = pred_store.retrieve(model_version="v1.0.0")
        
        assert len(stored_predictions) == 10
        assert all(p.model_version == "v1.0.0" for p in stored_predictions)
        assert all(p.actual_outcome is not None for p in stored_predictions)
        
        # Step 5: Get statistics
        stats = pred_store.get_statistics(model_version="v1.0.0")
        
        assert stats['total_predictions'] == 10
        assert stats['predictions_with_outcomes'] == 10
        assert 'accuracy' in stats
    
    def test_monitoring_with_performance_degradation(self, temp_files):
        """Test monitoring workflow when model performance degrades."""
        alert_path, store_path = temp_files
        
        # Create a poorly performing model
        X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=42)
        X_test = np.random.rand(50, 5)  # Random data - will perform poorly
        y_test = np.random.randint(0, 2, 50)
        
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Initialize services
        evaluator = ModelEvaluator(model_version="v_poor")
        alert_service = AlertService(
            recall_threshold=0.85,
            alert_log_path=alert_path
        )
        
        # Evaluate model
        eval_result = evaluator.evaluate(model, X_test, y_test)
        
        # Check performance - should fail threshold
        performance_ok = alert_service.check_performance(eval_result)
        
        # Generate alert
        if not performance_ok:
            alert = alert_service.generate_alert(
                eval_result,
                severity="critical"
            )
            
            assert alert.severity == "critical"
            assert alert.model_version == "v_poor"
            
            # Verify alert was logged
            alerts = alert_service.get_alerts(model_version="v_poor")
            assert len(alerts) >= 1
            assert alerts[0].model_version == "v_poor"
    
    def test_prediction_store_with_delayed_outcomes(self, temp_files):
        """Test storing predictions and updating outcomes later."""
        _, store_path = temp_files
        
        pred_store = PredictionStore(storage_path=store_path)
        
        # Store predictions without outcomes
        pred_ids = []
        for i in range(5):
            pred_id = pred_store.store(
                customer_id=f"CUST{i:03d}",
                model_version="v1.0.0",
                prediction=0.5 + i * 0.1,
                predicted_class=1 if i >= 2 else 0,
                actual_outcome=None  # No outcome yet
            )
            pred_ids.append(pred_id)
        
        # Verify predictions without outcomes
        preds_without_outcomes = pred_store.retrieve(has_outcome=False)
        assert len(preds_without_outcomes) == 5
        
        # Update outcomes later
        for i, pred_id in enumerate(pred_ids):
            actual = 1 if i >= 2 else 0
            success = pred_store.update_outcome(pred_id, actual)
            assert success is True
        
        # Verify all predictions now have outcomes
        preds_with_outcomes = pred_store.retrieve(has_outcome=True)
        assert len(preds_with_outcomes) == 5
        
        preds_without_outcomes = pred_store.retrieve(has_outcome=False)
        assert len(preds_without_outcomes) == 0
        
        # Check statistics
        stats = pred_store.get_statistics()
        assert stats['predictions_with_outcomes'] == 5
        assert stats['accuracy'] == 1.0  # All predictions were correct
    
    def test_calibration_curve_computation(self, trained_model_and_data):
        """Test calibration curve computation for model validation."""
        model, X_test, y_test = trained_model_and_data
        
        evaluator = ModelEvaluator(model_version="v1.0.0")
        
        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute calibration curve
        prob_true, prob_pred = evaluator.compute_calibration_curve(
            y_test,
            y_pred_proba,
            n_bins=10
        )
        
        # Verify calibration curve properties
        assert len(prob_true) <= 10
        assert len(prob_pred) <= 10
        assert all(0.0 <= p <= 1.0 for p in prob_true)
        assert all(0.0 <= p <= 1.0 for p in prob_pred)
        
        # For a well-calibrated model, prob_true should be close to prob_pred
        # We just verify they're in valid ranges here
        assert len(prob_true) == len(prob_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
