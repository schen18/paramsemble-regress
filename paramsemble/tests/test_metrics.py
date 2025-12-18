"""Tests for metrics calculator."""

import numpy as np
import pytest
from paramsemble.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Unit tests for MetricsCalculator class."""
    
    def test_compute_wmape_basic(self):
        """Test wMAPE calculation with simple values."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([12, 18, 32, 38])
        
        wmape = MetricsCalculator.compute_wmape(y_true, y_pred)
        
        # Expected: (|10-12| + |20-18| + |30-32| + |40-38|) / (10 + 20 + 30 + 40)
        # = (2 + 2 + 2 + 2) / 100 = 8 / 100 = 0.08
        assert np.isclose(wmape, 0.08)
    
    def test_compute_wmape_perfect_prediction(self):
        """Test wMAPE when predictions are perfect."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([10, 20, 30, 40])
        
        wmape = MetricsCalculator.compute_wmape(y_true, y_pred)
        
        assert wmape == 0.0
    
    def test_compute_wmape_with_nan_raises_error(self):
        """Test that NaN values raise ValueError."""
        y_true = np.array([10, 20, np.nan, 40])
        y_pred = np.array([12, 18, 32, 38])
        
        with pytest.raises(ValueError, match="NaN values"):
            MetricsCalculator.compute_wmape(y_true, y_pred)
    
    def test_compute_wmape_with_inf_raises_error(self):
        """Test that infinite values raise ValueError."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([12, np.inf, 32, 38])
        
        with pytest.raises(ValueError, match="infinite values"):
            MetricsCalculator.compute_wmape(y_true, y_pred)
    
    def test_compute_wmape_with_zero_denominator_raises_error(self):
        """Test that zero denominator raises ValueError."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 2, 3, 4])
        
        with pytest.raises(ValueError, match="Sum of absolute true values is zero"):
            MetricsCalculator.compute_wmape(y_true, y_pred)
    
    def test_compute_r2_basic(self):
        """Test R2 calculation with simple values."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        r2 = MetricsCalculator.compute_r2(y_true, y_pred)
        
        # R2 should be a value between -inf and 1
        assert isinstance(r2, (float, np.number))
        assert r2 <= 1.0
    
    def test_compute_r2_perfect_prediction(self):
        """Test R2 when predictions are perfect."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([10, 20, 30, 40])
        
        r2 = MetricsCalculator.compute_r2(y_true, y_pred)
        
        assert np.isclose(r2, 1.0)
    
    def test_compute_r2_with_nan_raises_error(self):
        """Test that NaN values raise ValueError."""
        y_true = np.array([10, 20, np.nan, 40])
        y_pred = np.array([12, 18, 32, 38])
        
        with pytest.raises(ValueError, match="NaN values"):
            MetricsCalculator.compute_r2(y_true, y_pred)
    
    def test_compute_r2_with_inf_raises_error(self):
        """Test that infinite values raise ValueError."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([12, np.inf, 32, 38])
        
        with pytest.raises(ValueError, match="infinite values"):
            MetricsCalculator.compute_r2(y_true, y_pred)
