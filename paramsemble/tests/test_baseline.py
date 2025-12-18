"""Tests for baseline model."""

import unittest
import numpy as np
from paramsemble.baseline import BaselineModel


class TestBaselineModel(unittest.TestCase):
    """Test cases for BaselineModel."""
    
    def test_fit_and_evaluate_returns_metrics(self):
        """Test that fit_and_evaluate returns wmape and r2 metrics."""
        # Create simple synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1
        X_test = np.random.randn(50, 5)
        y_test = X_test[:, 0] * 2 + X_test[:, 1] * 3 + np.random.randn(50) * 0.1
        
        # Fit and evaluate baseline model
        baseline = BaselineModel()
        metrics = baseline.fit_and_evaluate(X_train, y_train, X_test, y_test, random_state=42)
        
        # Check that metrics dictionary has correct keys
        self.assertIn('wmape', metrics)
        self.assertIn('r2', metrics)
        
        # Check that metrics are valid numbers
        self.assertIsInstance(metrics['wmape'], (int, float))
        self.assertIsInstance(metrics['r2'], (int, float))
        
        # Check that metrics are in reasonable ranges
        self.assertGreaterEqual(metrics['wmape'], 0)  # wMAPE should be non-negative
        self.assertLessEqual(metrics['r2'], 1.0)  # R2 should be <= 1
    
    def test_fit_and_evaluate_with_random_state(self):
        """Test that random_state produces reproducible results."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1
        X_test = np.random.randn(50, 5)
        y_test = X_test[:, 0] * 2 + X_test[:, 1] * 3 + np.random.randn(50) * 0.1
        
        baseline = BaselineModel()
        
        # Fit twice with same random state
        metrics1 = baseline.fit_and_evaluate(X_train, y_train, X_test, y_test, random_state=42)
        metrics2 = baseline.fit_and_evaluate(X_train, y_train, X_test, y_test, random_state=42)
        
        # Results should be identical
        self.assertEqual(metrics1['wmape'], metrics2['wmape'])
        self.assertEqual(metrics1['r2'], metrics2['r2'])


if __name__ == '__main__':
    unittest.main()
