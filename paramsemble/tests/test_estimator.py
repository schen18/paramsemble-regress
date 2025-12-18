"""Tests for ParamsembleRegressor estimator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from paramsemble.estimator import ParamsembleRegressor


def test_elm_regressor_basic_fit_predict():
    """Test basic fit and predict workflow."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples) * 10 + 50
    X_test = np.random.randn(50, n_features)
    y_test = np.random.randn(50) * 10 + 50
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Create and fit estimator
    estimator = ParamsembleRegressor(m=10, f=3, sample='unique', method='elastic', spread=5, random_state=42)
    estimator.fit(X_train_df, y_train, X_test_df, y_test)
    
    # Verify fitted attributes exist
    assert hasattr(estimator, 'feature_names_')
    assert hasattr(estimator, 'n_features_in_')
    assert hasattr(estimator, 'baseline_metrics_')
    assert hasattr(estimator, 'constituent_models_')
    assert hasattr(estimator, 'selected_models_')
    assert hasattr(estimator, 'ensemble_model_')
    
    # Generate predictions
    X_pred = np.random.randn(20, n_features)
    X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
    predictions = estimator.predict(X_pred_df)
    
    # Verify predictions
    assert len(predictions) == 20
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))


def test_elm_regressor_not_fitted_error():
    """Test that NotFittedError is raised when predict is called before fit."""
    estimator = ParamsembleRegressor(m=10, f=3, random_state=42)
    
    X = np.random.randn(20, 5)
    
    with pytest.raises(NotFittedError):
        estimator.predict(X)


def test_elm_regressor_invalid_f_parameter():
    """Test that ValueError is raised when f is not specified."""
    estimator = ParamsembleRegressor(m=10, random_state=42)
    
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    X_test = np.random.randn(50, 5)
    y_test = np.random.randn(50)
    
    with pytest.raises(ValueError, match="Parameter 'f' must be specified"):
        estimator.fit(X_train, y_train, X_test, y_test)


def test_elm_regressor_shape_mismatch():
    """Test that ValueError is raised for shape mismatches."""
    estimator = ParamsembleRegressor(m=10, f=3, random_state=42)
    
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    X_test = np.random.randn(50, 4)  # Different number of features
    y_test = np.random.randn(50)
    
    with pytest.raises(ValueError, match="same number of features"):
        estimator.fit(X_train, y_train, X_test, y_test)
