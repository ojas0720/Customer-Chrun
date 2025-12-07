"""
Unit tests for ParameterValidator class.
Tests validation of XGBoost hyperparameters.
"""

import pytest
from services.parameter_validation import ParameterValidator, ParameterValidationError


class TestParameterValidator:
    """Test suite for ParameterValidator class."""
    
    def test_valid_parameters(self):
        """Test validation of valid parameter set."""
        validator = ParameterValidator()
        
        params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        assert validator.validate(params) is True
    
    def test_learning_rate_zero_invalid(self):
        """Test that learning_rate = 0 is invalid."""
        validator = ParameterValidator()
        
        params = {'learning_rate': 0}
        
        with pytest.raises(ParameterValidationError, match="learning_rate"):
            validator.validate(params)
    
    def test_learning_rate_negative_invalid(self):
        """Test that negative learning_rate is invalid."""
        validator = ParameterValidator()
        
        params = {'learning_rate': -0.1}
        
        with pytest.raises(ParameterValidationError, match="learning_rate"):
            validator.validate(params)
    
    def test_learning_rate_above_one_invalid(self):
        """Test that learning_rate > 1 is invalid."""
        validator = ParameterValidator()
        
        params = {'learning_rate': 1.5}
        
        with pytest.raises(ParameterValidationError, match="learning_rate"):
            validator.validate(params)
    
    def test_max_depth_negative_invalid(self):
        """Test that negative max_depth is invalid."""
        validator = ParameterValidator()
        
        params = {'max_depth': -1}
        
        with pytest.raises(ParameterValidationError, match="max_depth"):
            validator.validate(params)
    
    def test_max_depth_zero_valid(self):
        """Test that max_depth = 0 is valid."""
        validator = ParameterValidator()
        
        params = {'max_depth': 0}
        
        assert validator.validate(params) is True
    
    def test_n_estimators_zero_invalid(self):
        """Test that n_estimators = 0 is invalid."""
        validator = ParameterValidator()
        
        params = {'n_estimators': 0}
        
        with pytest.raises(ParameterValidationError, match="n_estimators"):
            validator.validate(params)
    
    def test_n_estimators_negative_invalid(self):
        """Test that negative n_estimators is invalid."""
        validator = ParameterValidator()
        
        params = {'n_estimators': -10}
        
        with pytest.raises(ParameterValidationError, match="n_estimators"):
            validator.validate(params)
    
    def test_subsample_zero_invalid(self):
        """Test that subsample = 0 is invalid."""
        validator = ParameterValidator()
        
        params = {'subsample': 0}
        
        with pytest.raises(ParameterValidationError, match="subsample"):
            validator.validate(params)
    
    def test_subsample_above_one_invalid(self):
        """Test that subsample > 1 is invalid."""
        validator = ParameterValidator()
        
        params = {'subsample': 1.5}
        
        with pytest.raises(ParameterValidationError, match="subsample"):
            validator.validate(params)
    
    def test_colsample_bytree_zero_invalid(self):
        """Test that colsample_bytree = 0 is invalid."""
        validator = ParameterValidator()
        
        params = {'colsample_bytree': 0}
        
        with pytest.raises(ParameterValidationError, match="colsample_bytree"):
            validator.validate(params)
    
    def test_gamma_zero_valid(self):
        """Test that gamma = 0 is valid."""
        validator = ParameterValidator()
        
        params = {'gamma': 0}
        
        assert validator.validate(params) is True
    
    def test_gamma_negative_invalid(self):
        """Test that negative gamma is invalid."""
        validator = ParameterValidator()
        
        params = {'gamma': -0.5}
        
        with pytest.raises(ParameterValidationError, match="gamma"):
            validator.validate(params)
    
    def test_reg_alpha_negative_invalid(self):
        """Test that negative reg_alpha is invalid."""
        validator = ParameterValidator()
        
        params = {'reg_alpha': -0.1}
        
        with pytest.raises(ParameterValidationError, match="reg_alpha"):
            validator.validate(params)
    
    def test_reg_lambda_negative_invalid(self):
        """Test that negative reg_lambda is invalid."""
        validator = ParameterValidator()
        
        params = {'reg_lambda': -1.0}
        
        with pytest.raises(ParameterValidationError, match="reg_lambda"):
            validator.validate(params)
    
    def test_scale_pos_weight_zero_invalid(self):
        """Test that scale_pos_weight = 0 is invalid."""
        validator = ParameterValidator()
        
        params = {'scale_pos_weight': 0}
        
        with pytest.raises(ParameterValidationError, match="scale_pos_weight"):
            validator.validate(params)
    
    def test_scale_pos_weight_negative_invalid(self):
        """Test that negative scale_pos_weight is invalid."""
        validator = ParameterValidator()
        
        params = {'scale_pos_weight': -1.0}
        
        with pytest.raises(ParameterValidationError, match="scale_pos_weight"):
            validator.validate(params)
    
    def test_invalid_objective(self):
        """Test that invalid objective value is rejected."""
        validator = ParameterValidator()
        
        params = {'objective': 'invalid_objective'}
        
        with pytest.raises(ParameterValidationError, match="objective"):
            validator.validate(params)
    
    def test_valid_objective_binary_logistic(self):
        """Test that binary:logistic objective is valid."""
        validator = ParameterValidator()
        
        params = {'objective': 'binary:logistic'}
        
        assert validator.validate(params) is True
    
    def test_invalid_booster(self):
        """Test that invalid booster value is rejected."""
        validator = ParameterValidator()
        
        params = {'booster': 'invalid_booster'}
        
        with pytest.raises(ParameterValidationError, match="booster"):
            validator.validate(params)
    
    def test_valid_booster_gbtree(self):
        """Test that gbtree booster is valid."""
        validator = ParameterValidator()
        
        params = {'booster': 'gbtree'}
        
        assert validator.validate(params) is True
    
    def test_none_values_allowed(self):
        """Test that None values are allowed (use XGBoost defaults)."""
        validator = ParameterValidator()
        
        params = {
            'learning_rate': None,
            'max_depth': None,
            'objective': 'binary:logistic'
        }
        
        assert validator.validate(params) is True
    
    def test_multiple_errors_reported(self):
        """Test that multiple validation errors are reported together."""
        validator = ParameterValidator()
        
        params = {
            'learning_rate': -0.1,
            'max_depth': -5,
            'n_estimators': 0
        }
        
        with pytest.raises(ParameterValidationError) as exc_info:
            validator.validate(params)
        
        error_message = str(exc_info.value)
        assert 'learning_rate' in error_message
        assert 'max_depth' in error_message
        assert 'n_estimators' in error_message
    
    def test_validate_and_get_errors_returns_empty_for_valid(self):
        """Test that validate_and_get_errors returns empty list for valid params."""
        validator = ParameterValidator()
        
        params = {'learning_rate': 0.1, 'max_depth': 6}
        errors = validator.validate_and_get_errors(params)
        
        assert errors == []
    
    def test_validate_and_get_errors_returns_list_for_invalid(self):
        """Test that validate_and_get_errors returns error list for invalid params."""
        validator = ParameterValidator()
        
        params = {'learning_rate': -0.1, 'max_depth': -5}
        errors = validator.validate_and_get_errors(params)
        
        assert len(errors) > 0
        assert any('learning_rate' in e for e in errors)
        assert any('max_depth' in e for e in errors)
    
    def test_get_parameter_info_numerical(self):
        """Test getting parameter info for numerical parameter."""
        validator = ParameterValidator()
        
        info = validator.get_parameter_info('learning_rate')
        
        assert info is not None
        assert 'type' in info
        assert 'min' in info
        assert 'max' in info
        assert 'description' in info
    
    def test_get_parameter_info_categorical(self):
        """Test getting parameter info for categorical parameter."""
        validator = ParameterValidator()
        
        info = validator.get_parameter_info('objective')
        
        assert info is not None
        assert info['type'] == 'categorical'
        assert 'valid_values' in info
        assert 'binary:logistic' in info['valid_values']
    
    def test_get_parameter_info_unknown(self):
        """Test getting parameter info for unknown parameter."""
        validator = ParameterValidator()
        
        info = validator.get_parameter_info('unknown_param')
        
        assert info is None
    
    def test_wrong_type_for_numerical_param(self):
        """Test that wrong type for numerical parameter is rejected."""
        validator = ParameterValidator()
        
        params = {'learning_rate': "0.1"}  # String instead of number
        
        with pytest.raises(ParameterValidationError, match="learning_rate"):
            validator.validate(params)
    
    def test_params_not_dict_raises_error(self):
        """Test that non-dict params raises error."""
        validator = ParameterValidator()
        
        with pytest.raises(ParameterValidationError, match="must be a dictionary"):
            validator.validate("not a dict")
    
    def test_edge_case_learning_rate_exactly_one(self):
        """Test that learning_rate = 1.0 is valid."""
        validator = ParameterValidator()
        
        params = {'learning_rate': 1.0}
        
        assert validator.validate(params) is True
    
    def test_edge_case_subsample_exactly_one(self):
        """Test that subsample = 1.0 is valid."""
        validator = ParameterValidator()
        
        params = {'subsample': 1.0}
        
        assert validator.validate(params) is True
