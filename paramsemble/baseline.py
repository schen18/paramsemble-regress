"""Baseline model component."""

from sklearn.ensemble import RandomForestRegressor
from .metrics import MetricsCalculator


class BaselineModel:
    """Trains and evaluates baseline Random Forest model."""
    
    def fit_and_evaluate(self, X_train, y_train, X_test, y_test, random_state=None):
        """
        Fit baseline model and compute metrics.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix
        y_train : array-like of shape (n_samples,)
            Training target values
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like of shape (n_samples,)
            Test target values
        random_state : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        baseline_metrics : dict
            Dictionary with 'wmape' and 'r2' keys
        """
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided")
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided")
        
        # Create and fit Random Forest model
        rf_model = RandomForestRegressor(random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Generate predictions on test set
        y_pred = rf_model.predict(X_test)
        
        # Compute metrics
        wmape = MetricsCalculator.compute_wmape(y_test, y_pred)
        r2 = MetricsCalculator.compute_r2(y_test, y_pred)
        
        return {
            'wmape': wmape,
            'r2': r2
        }
