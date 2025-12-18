"""Tests for ensemble builder."""

import numpy as np
import pytest
from paramsemble.ensemble import EnsembleBuilder


def test_build_elasticnet_ensemble():
    """Test building ElasticNet ensemble."""
    # Create sample constituent predictions
    n_samples = 50
    n_models = 5
    constituent_predictions = np.random.randn(n_samples, n_models) * 10 + 50
    y_true = np.random.randn(n_samples) * 10 + 50
    
    # Build ensemble
    builder = EnsembleBuilder()
    ensemble_model, equation_dict = builder.build_ensemble(
        constituent_predictions, y_true, method='elastic'
    )
    
    # Verify ensemble model is fitted
    assert hasattr(ensemble_model, 'coef_')
    assert hasattr(ensemble_model, 'intercept_')
    
    # Verify equation dictionary structure
    assert isinstance(equation_dict, dict)
    assert 'constant' in equation_dict
    assert len(equation_dict) == n_models + 1  # n_models + constant
    
    # Verify all model keys are present
    for i in range(n_models):
        assert f'model_{i}' in equation_dict
    
    # Verify predictions can be generated
    predictions = ensemble_model.predict(constituent_predictions)
    assert len(predictions) == n_samples


def test_build_mars_ensemble():
    """Test building MARS ensemble."""
    try:
        from pyearth import Earth
    except ImportError:
        pytest.skip("py-earth not installed")
    
    # Create sample constituent predictions
    n_samples = 50
    n_models = 5
    constituent_predictions = np.random.randn(n_samples, n_models) * 10 + 50
    y_true = np.random.randn(n_samples) * 10 + 50
    
    # Build ensemble
    builder = EnsembleBuilder()
    ensemble_model, equation_dict = builder.build_ensemble(
        constituent_predictions, y_true, method='mars'
    )
    
    # Verify ensemble model is fitted
    assert hasattr(ensemble_model, 'predict')
    
    # Verify equation dictionary structure
    assert isinstance(equation_dict, dict)
    assert 'constant' in equation_dict
    
    # Verify predictions can be generated
    predictions = ensemble_model.predict(constituent_predictions)
    assert len(predictions) == n_samples


def test_invalid_method():
    """Test that invalid method raises ValueError."""
    n_samples = 50
    n_models = 5
    constituent_predictions = np.random.randn(n_samples, n_models)
    y_true = np.random.randn(n_samples)
    
    builder = EnsembleBuilder()
    
    with pytest.raises(ValueError, match="Invalid method"):
        builder.build_ensemble(constituent_predictions, y_true, method='invalid')


def test_invalid_shape():
    """Test that invalid input shapes raise ValueError."""
    builder = EnsembleBuilder()
    
    # 1D array instead of 2D
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        builder.build_ensemble(np.array([1, 2, 3]), np.array([1, 2, 3]), method='elastic')
    
    # Mismatched lengths
    with pytest.raises(ValueError, match="same number of samples"):
        builder.build_ensemble(
            np.random.randn(50, 5),
            np.random.randn(40),
            method='elastic'
        )
