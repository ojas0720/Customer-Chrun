"""
Parameter validation utility for XGBoost hyperparameters.
Validates hyperparameter configurations before model training.
"""

from typing import Dict, Any, List, Optional


class ParameterValidationError(Exception):
    """Raised when hyperparameter validation fails."""
    pass


class ParameterValidator:
    """
    Validates XGBoost hyperparameters for correctness.
    Ensures parameters are within valid ranges before training.
    """
    
    # Define valid ranges and constraints for XGBoost parameters
    PARAMETER_CONSTRAINTS = {
        'learning_rate': {
            'type': (int, float),
            'min': 0.0,
            'max': 1.0,
            'exclusive_min': True,  # Must be > 0, not >= 0
            'description': 'Learning rate (eta) must be in range (0, 1]'
        },
        'max_depth': {
            'type': int,
            'min': 0,
            'max': None,
            'exclusive_min': False,
            'description': 'Maximum tree depth must be >= 0'
        },
        'n_estimators': {
            'type': int,
            'min': 1,
            'max': None,
            'exclusive_min': False,
            'description': 'Number of estimators must be >= 1'
        },
        'subsample': {
            'type': (int, float),
            'min': 0.0,
            'max': 1.0,
            'exclusive_min': True,
            'description': 'Subsample ratio must be in range (0, 1]'
        },
        'colsample_bytree': {
            'type': (int, float),
            'min': 0.0,
            'max': 1.0,
            'exclusive_min': True,
            'description': 'Column sample by tree must be in range (0, 1]'
        },
        'colsample_bylevel': {
            'type': (int, float),
            'min': 0.0,
            'max': 1.0,
            'exclusive_min': True,
            'description': 'Column sample by level must be in range (0, 1]'
        },
        'colsample_bynode': {
            'type': (int, float),
            'min': 0.0,
            'max': 1.0,
            'exclusive_min': True,
            'description': 'Column sample by node must be in range (0, 1]'
        },
        'min_child_weight': {
            'type': (int, float),
            'min': 0.0,
            'max': None,
            'exclusive_min': False,
            'description': 'Minimum child weight must be >= 0'
        },
        'gamma': {
            'type': (int, float),
            'min': 0.0,
            'max': None,
            'exclusive_min': False,
            'description': 'Gamma (min split loss) must be >= 0'
        },
        'reg_alpha': {
            'type': (int, float),
            'min': 0.0,
            'max': None,
            'exclusive_min': False,
            'description': 'L1 regularization (alpha) must be >= 0'
        },
        'reg_lambda': {
            'type': (int, float),
            'min': 0.0,
            'max': None,
            'exclusive_min': False,
            'description': 'L2 regularization (lambda) must be >= 0'
        },
        'scale_pos_weight': {
            'type': (int, float),
            'min': 0.0,
            'max': None,
            'exclusive_min': True,
            'description': 'Scale positive weight must be > 0'
        },
        'max_delta_step': {
            'type': (int, float),
            'min': 0.0,
            'max': None,
            'exclusive_min': False,
            'description': 'Max delta step must be >= 0'
        },
        'random_state': {
            'type': int,
            'min': None,
            'max': None,
            'exclusive_min': False,
            'description': 'Random state must be an integer'
        }
    }
    
    # Valid categorical parameter values
    CATEGORICAL_CONSTRAINTS = {
        'objective': [
            'reg:squarederror', 'reg:squaredlogerror', 'reg:logistic',
            'reg:pseudohubererror', 'binary:logistic', 'binary:logitraw',
            'binary:hinge', 'count:poisson', 'survival:cox', 'survival:aft',
            'multi:softmax', 'multi:softprob', 'rank:pairwise', 'rank:ndcg',
            'rank:map', 'reg:gamma', 'reg:tweedie'
        ],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist'],
        'grow_policy': ['depthwise', 'lossguide'],
        'eval_metric': [
            'rmse', 'rmsle', 'mae', 'mape', 'mphe', 'logloss', 'error',
            'error@t', 'merror', 'mlogloss', 'auc', 'aucpr', 'ndcg',
            'map', 'ndcg@n', 'map@n', 'poisson-nloglik', 'gamma-nloglik',
            'cox-nloglik', 'gamma-deviance', 'tweedie-nloglik'
        ],
        'sampling_method': ['uniform', 'gradient_based'],
        'multi_strategy': ['one_output_per_tree', 'multi_output_tree']
    }
    
    def __init__(self):
        """Initialize ParameterValidator."""
        pass
    
    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate XGBoost hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters to validate
            
        Returns:
            True if all parameters are valid
            
        Raises:
            ParameterValidationError: If any parameter is invalid
        """
        if not isinstance(params, dict):
            raise ParameterValidationError("Parameters must be a dictionary")
        
        errors = []
        
        # Validate numerical parameters
        for param_name, value in params.items():
            if value is None:
                # None values are allowed (use XGBoost defaults)
                continue
            
            # Check numerical constraints
            if param_name in self.PARAMETER_CONSTRAINTS:
                constraint = self.PARAMETER_CONSTRAINTS[param_name]
                error = self._validate_numerical_param(param_name, value, constraint)
                if error:
                    errors.append(error)
            
            # Check categorical constraints
            elif param_name in self.CATEGORICAL_CONSTRAINTS:
                error = self._validate_categorical_param(
                    param_name, value, self.CATEGORICAL_CONSTRAINTS[param_name]
                )
                if error:
                    errors.append(error)
        
        # Raise error with all validation issues
        if errors:
            error_message = "Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ParameterValidationError(error_message)
        
        return True
    
    def _validate_numerical_param(
        self,
        param_name: str,
        value: Any,
        constraint: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate a numerical parameter against constraints.
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            constraint: Constraint dictionary
            
        Returns:
            Error message if invalid, None if valid
        """
        # Check type
        expected_type = constraint['type']
        if not isinstance(value, expected_type):
            return f"{param_name}: Expected type {expected_type}, got {type(value).__name__}"
        
        # Check minimum value
        if constraint['min'] is not None:
            if constraint['exclusive_min']:
                if value <= constraint['min']:
                    return f"{param_name}: {constraint['description']} (got {value})"
            else:
                if value < constraint['min']:
                    return f"{param_name}: {constraint['description']} (got {value})"
        
        # Check maximum value
        if constraint['max'] is not None:
            if value > constraint['max']:
                return f"{param_name}: {constraint['description']} (got {value})"
        
        return None
    
    def _validate_categorical_param(
        self,
        param_name: str,
        value: Any,
        valid_values: List[str]
    ) -> Optional[str]:
        """
        Validate a categorical parameter against allowed values.
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            valid_values: List of valid values
            
        Returns:
            Error message if invalid, None if valid
        """
        if value not in valid_values:
            return (
                f"{param_name}: Invalid value '{value}'. "
                f"Must be one of: {', '.join(valid_values)}"
            )
        return None
    
    def validate_and_get_errors(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters and return list of errors without raising exception.
        
        Args:
            params: Dictionary of hyperparameters to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        try:
            self.validate(params)
            return []
        except ParameterValidationError as e:
            # Extract individual error messages
            error_msg = str(e)
            if "Parameter validation failed:" in error_msg:
                lines = error_msg.split('\n')[1:]  # Skip first line
                return [line.strip('  - ') for line in lines if line.strip()]
            return [error_msg]
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Get constraint information for a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Constraint dictionary or None if parameter not found
        """
        if param_name in self.PARAMETER_CONSTRAINTS:
            return self.PARAMETER_CONSTRAINTS[param_name].copy()
        elif param_name in self.CATEGORICAL_CONSTRAINTS:
            return {
                'type': 'categorical',
                'valid_values': self.CATEGORICAL_CONSTRAINTS[param_name],
                'description': f'Must be one of: {", ".join(self.CATEGORICAL_CONSTRAINTS[param_name])}'
            }
        return None
