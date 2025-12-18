"""Constituent model trainer."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from .metrics import MetricsCalculator


class ConstituentModelTrainer:
    """Trains multiple Lasso regression models."""
    
    def train_models(self, X_train, y_train, X_test, y_test, feature_combinations):
        """
        Train Lasso models for each feature combination.
        
        Parameters
        ----------
        X_train : array-like or DataFrame of shape (n_samples, n_features)
            Training feature matrix
        y_train : array-like of shape (n_samples,)
            Training target values
        X_test : array-like or DataFrame of shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like of shape (n_samples,)
            Test target values
        feature_combinations : list of lists
            List of feature combinations, where each combination is a list of feature names
        
        Returns
        -------
        models_info : list of dict
            List containing model metrics and coefficients
        """
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided")
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided")
        if not feature_combinations:
            raise ValueError("feature_combinations cannot be empty")
        if not isinstance(feature_combinations, (list, tuple)):
            raise ValueError("feature_combinations must be a list or tuple")
        
        # Convert to DataFrame if needed to support feature indexing
        if not isinstance(X_train, pd.DataFrame):
            # Assume X_train is array-like with feature names in feature_combinations
            # We'll need to handle this by position
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
        else:
            X_train_df = X_train
            X_test_df = X_test
        
        models_info = []
        
        for model_id, features in enumerate(feature_combinations):
            # Select features for this combination
            if isinstance(X_train, pd.DataFrame):
                X_train_subset = X_train_df[features]
                X_test_subset = X_test_df[features]
            else:
                # If features are column indices (integers)
                if all(isinstance(f, int) for f in features):
                    X_train_subset = X_train_df.iloc[:, features]
                    X_test_subset = X_test_df.iloc[:, features]
                else:
                    # Features are names, use them as column names
                    X_train_subset = X_train_df[features]
                    X_test_subset = X_test_df[features]
            
            # Train Lasso model
            lasso_model = Lasso(random_state=42, max_iter=10000)
            lasso_model.fit(X_train_subset, y_train)
            
            # Generate predictions on test set
            y_pred = lasso_model.predict(X_test_subset)
            
            # Compute metrics
            wmape = MetricsCalculator.compute_wmape(y_test, y_pred)
            r2 = MetricsCalculator.compute_r2(y_test, y_pred)
            
            # Extract equation dictionary
            equation_dict = self.extract_equation_dict(lasso_model, features)
            
            # Store model information
            model_info = {
                'model_id': model_id,
                'features': features,
                'wmape': wmape,
                'r2': r2,
                'equation_dict': equation_dict,
                'model_object': lasso_model
            }
            
            models_info.append(model_info)
        
        return models_info
    
    def extract_equation_dict(self, model, feature_names):
        """
        Extract coefficients as equation dictionary.
        
        Parameters
        ----------
        model : sklearn.linear_model.Lasso
            Fitted Lasso model
        feature_names : list
            List of feature names corresponding to model coefficients
        
        Returns
        -------
        equation_dict : dict
            Feature names as keys, coefficients as values, plus 'constant'
        """
        # Validate inputs
        if model is None:
            raise ValueError("model must be provided")
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
        if not hasattr(model, 'coef_'):
            raise ValueError("model must have 'coef_' attribute (must be fitted)")
        if not hasattr(model, 'intercept_'):
            raise ValueError("model must have 'intercept_' attribute (must be fitted)")
        if len(feature_names) != len(model.coef_):
            raise ValueError(
                f"Length of feature_names ({len(feature_names)}) must match "
                f"length of model coefficients ({len(model.coef_)})"
            )
        
        equation_dict = {}
        
        # Add feature coefficients
        for feature_name, coef in zip(feature_names, model.coef_):
            equation_dict[feature_name] = float(coef)
        
        # Add intercept as 'constant'
        equation_dict['constant'] = float(model.intercept_)
        
        return equation_dict
