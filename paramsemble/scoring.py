"""Model scoring module."""

import numpy as np
import pandas as pd


class ModelScorer:
    """Scores new datasets using saved model JSON."""
    
    def score_dataset(self, X, model_data, id_column=None):
        """
        Generate predictions using saved model.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix for scoring
        model_data : dict
            Model configuration loaded from JSON
        id_column : array-like, optional
            Identifiers for samples
            
        Returns
        -------
        predictions_df : DataFrame
            DataFrame with IDs and predictions
        """
        # Validate inputs
        if X is None:
            raise ValueError("X must be provided")
        if model_data is None:
            raise ValueError("model_data must be provided")
        
        # Validate model_data structure
        if not isinstance(model_data, dict):
            raise ValueError("model_data must be a dictionary")
        if 'constituent_models' not in model_data:
            raise ValueError("model_data missing 'constituent_models' key")
        if 'ensemble_equation' not in model_data:
            raise ValueError("model_data missing 'ensemble_equation' key")
        
        # Convert X to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Validate that X is not empty
        if X.shape[0] == 0:
            raise ValueError("X has zero rows")
        
        # Check for NaN and infinite values
        if X.isnull().any().any():
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X.values)):
            raise ValueError("X contains infinite values")
        
        # Get constituent models and ensemble equation
        constituent_models = model_data['constituent_models']
        ensemble_equation = model_data['ensemble_equation']
        
        # Validate constituent models
        if not constituent_models:
            raise ValueError("model_data has no constituent models")
        
        # Step 1: Apply each constituent model equation to generate intermediate predictions
        constituent_predictions = []
        
        for model_info in constituent_models:
            equation_dict = model_info['equation_dict']
            # Apply equation dictionary to get predictions for this constituent model
            predictions = self.apply_equation_dict(X, equation_dict)
            constituent_predictions.append(predictions)
        
        # Convert to array: shape (n_samples, n_constituent_models)
        constituent_predictions = np.column_stack(constituent_predictions)
        
        # Step 2: Create DataFrame with constituent predictions for ensemble
        constituent_df = pd.DataFrame(
            constituent_predictions,
            columns=[f'model_{i}' for i in range(len(constituent_models))]
        )
        
        # Step 3: Apply ensemble equation to constituent predictions
        final_predictions = self.apply_equation_dict(constituent_df, ensemble_equation)
        
        # Step 4: Create output DataFrame with IDs and predictions
        if id_column is not None:
            ids = id_column
        else:
            ids = np.arange(len(final_predictions))
        
        predictions_df = pd.DataFrame({
            'id': ids,
            'predicted': final_predictions
        })
        
        return predictions_df
    
    def apply_equation_dict(self, X, equation_dict):
        """
        Apply equation dictionary to generate predictions.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        equation_dict : dict
            Dictionary with feature names as keys and coefficients as values,
            plus 'constant' key for intercept
            
        Returns
        -------
        predictions : ndarray
            Predicted values
        """
        # Validate inputs
        if X is None:
            raise ValueError("X must be provided")
        if equation_dict is None:
            raise ValueError("equation_dict must be provided")
        if not isinstance(equation_dict, dict):
            raise ValueError("equation_dict must be a dictionary")
        if 'constant' not in equation_dict:
            raise ValueError("equation_dict missing 'constant' key")
        
        # Start with the constant term
        predictions = np.full(len(X), equation_dict['constant'])
        
        # Add contribution from each feature
        for feature, coefficient in equation_dict.items():
            if feature != 'constant':
                # Validate feature exists in X
                if feature not in X.columns:
                    raise ValueError(
                        f"Feature '{feature}' from equation_dict not found in X. "
                        f"Available features: {list(X.columns)}"
                    )
                # Add feature * coefficient to predictions
                predictions += X[feature].values * coefficient
        
        return predictions
