"""Metrics calculation module."""

import numpy as np
from sklearn.metrics import r2_score


class MetricsCalculator:
    """Calculates regression performance metrics."""
    
    @staticmethod
    def compute_wmape(y_true, y_pred):
        """
        Compute weighted Mean Absolute Percentage Error.
        
        wMAPE = sum(|y_true - y_pred|) / sum(|y_true|)
        
        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
        
        Returns
        -------
        wmape : float
            Weighted MAPE value
            
        Raises
        ------
        ValueError
            If inputs contain NaN or infinite values, or if sum(|y_true|) is zero
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Validate inputs
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input arrays contain NaN values")
        
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Input arrays contain infinite values")
        
        # Calculate wMAPE
        numerator = np.sum(np.abs(y_true - y_pred))
        denominator = np.sum(np.abs(y_true))
        
        if denominator == 0:
            raise ValueError("Sum of absolute true values is zero, cannot compute wMAPE")
        
        return numerator / denominator
    
    @staticmethod
    def compute_r2(y_true, y_pred):
        """
        Compute R-squared coefficient.
        
        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
        
        Returns
        -------
        r2 : float
            R-squared value
            
        Raises
        ------
        ValueError
            If inputs contain NaN or infinite values
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Validate inputs
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input arrays contain NaN values")
        
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Input arrays contain infinite values")
        
        return r2_score(y_true, y_pred)
