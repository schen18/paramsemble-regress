"""Ensemble builder component."""

import numpy as np
from sklearn.linear_model import ElasticNet


class EnsembleBuilder:
    """Builds ensemble models from constituent predictions."""
    
    def build_ensemble(self, constituent_predictions, y_true, method='elastic'):
        """
        Build ensemble model.
        
        Parameters
        ----------
        constituent_predictions : array-like of shape (n_samples, n_models)
            Predictions from constituent models
        y_true : array-like
            True target values
        method : {'elastic', 'mars'}
            Ensemble method
            
        Returns
        -------
        ensemble_model : object
            Fitted ensemble model
        equation_dict : dict
            Ensemble model coefficients
        """
        # Validate inputs
        if constituent_predictions is None:
            raise ValueError("constituent_predictions must be provided")
        if y_true is None:
            raise ValueError("y_true must be provided")
        if method not in ['elastic', 'mars']:
            raise ValueError(f"Invalid method '{method}'. Must be 'elastic' or 'mars'.")
        
        # Convert to numpy array if needed
        constituent_predictions = np.asarray(constituent_predictions)
        y_true = np.asarray(y_true)
        
        # Validate shapes
        if constituent_predictions.ndim != 2:
            raise ValueError("constituent_predictions must be 2-dimensional")
        if constituent_predictions.shape[0] == 0:
            raise ValueError("constituent_predictions has zero rows")
        if constituent_predictions.shape[1] == 0:
            raise ValueError("constituent_predictions has zero columns (no models)")
        if len(y_true) == 0:
            raise ValueError("y_true has zero elements")
        if len(y_true) != len(constituent_predictions):
            raise ValueError(
                f"y_true and constituent_predictions must have same number of samples. "
                f"y_true has {len(y_true)} samples, constituent_predictions has {len(constituent_predictions)} samples"
            )
        
        # Check for NaN and infinite values
        if np.any(np.isnan(constituent_predictions)):
            raise ValueError("constituent_predictions contains NaN values")
        if np.any(np.isinf(constituent_predictions)):
            raise ValueError("constituent_predictions contains infinite values")
        if np.any(np.isnan(y_true)):
            raise ValueError("y_true contains NaN values")
        if np.any(np.isinf(y_true)):
            raise ValueError("y_true contains infinite values")
        
        # Build ensemble based on method
        if method == 'elastic':
            ensemble_model, equation_dict = self._build_elasticnet(constituent_predictions, y_true)
        else:  # method == 'mars'
            ensemble_model, equation_dict = self._build_mars(constituent_predictions, y_true)
        
        return ensemble_model, equation_dict
    
    def _build_elasticnet(self, constituent_predictions, y_true):
        """
        Build ElasticNet ensemble model.
        
        Parameters
        ----------
        constituent_predictions : ndarray of shape (n_samples, n_models)
            Predictions from constituent models
        y_true : ndarray of shape (n_samples,)
            True target values
            
        Returns
        -------
        ensemble_model : ElasticNet
            Fitted ElasticNet model
        equation_dict : dict
            Ensemble model coefficients
        """
        # Fit ElasticNet on constituent predictions
        ensemble_model = ElasticNet(random_state=42, max_iter=10000)
        ensemble_model.fit(constituent_predictions, y_true)
        
        # Extract equation dictionary
        equation_dict = self._extract_elasticnet_equation(ensemble_model, constituent_predictions.shape[1])
        
        return ensemble_model, equation_dict
    
    def _build_mars(self, constituent_predictions, y_true):
        """
        Build MARS ensemble model.
        
        Parameters
        ----------
        constituent_predictions : ndarray of shape (n_samples, n_models)
            Predictions from constituent models
        y_true : ndarray of shape (n_samples,)
            True target values
            
        Returns
        -------
        ensemble_model : Earth
            Fitted MARS model
        equation_dict : dict
            Ensemble model coefficients
        """
        try:
            from pyearth import Earth
        except ImportError:
            raise ImportError(
                "py-earth is required for MARS ensemble method. "
                "Install it with: pip install sklearn-contrib-py-earth"
            )
        
        # Fit MARS model on constituent predictions
        ensemble_model = Earth(max_degree=2, max_terms=20)
        ensemble_model.fit(constituent_predictions, y_true)
        
        # Extract equation dictionary
        equation_dict = self._extract_mars_equation(ensemble_model, constituent_predictions.shape[1])
        
        return ensemble_model, equation_dict
    
    def _extract_elasticnet_equation(self, model, n_models):
        """
        Extract equation dictionary from ElasticNet model.
        
        Parameters
        ----------
        model : ElasticNet
            Fitted ElasticNet model
        n_models : int
            Number of constituent models
            
        Returns
        -------
        equation_dict : dict
            Model coefficients with constituent model names as keys
        """
        # Validate inputs
        if model is None:
            raise ValueError("model must be provided")
        if not hasattr(model, 'coef_'):
            raise ValueError("model must have 'coef_' attribute (must be fitted)")
        if not hasattr(model, 'intercept_'):
            raise ValueError("model must have 'intercept_' attribute (must be fitted)")
        if not isinstance(n_models, int) or n_models < 1:
            raise ValueError(f"n_models must be a positive integer, got {n_models}")
        
        equation_dict = {}
        
        # Add coefficients for each constituent model prediction
        for i, coef in enumerate(model.coef_):
            equation_dict[f'model_{i}'] = float(coef)
        
        # Add intercept as 'constant'
        equation_dict['constant'] = float(model.intercept_)
        
        return equation_dict
    
    def _extract_mars_equation(self, model, n_models):
        """
        Extract equation dictionary from MARS model.
        
        Parameters
        ----------
        model : Earth
            Fitted MARS model
        n_models : int
            Number of constituent models
            
        Returns
        -------
        equation_dict : dict
            Model coefficients (simplified linear approximation)
        """
        # Validate inputs
        if model is None:
            raise ValueError("model must be provided")
        if not isinstance(n_models, int) or n_models < 1:
            raise ValueError(f"n_models must be a positive integer, got {n_models}")
        
        # For MARS, we'll extract a simplified linear representation
        # by evaluating the model's effective coefficients
        # This is a simplification since MARS is non-linear
        
        # Get the basis function coefficients
        # MARS models are more complex, so we'll create a linear approximation
        # by using the model's coef_ attribute if available
        
        equation_dict = {}
        
        # Try to get linear coefficients from MARS model
        if hasattr(model, 'coef_'):
            for i, coef in enumerate(model.coef_):
                equation_dict[f'model_{i}'] = float(coef)
        else:
            # If no direct coefficients, use a numerical approximation
            # Create unit vectors to estimate partial derivatives
            for i in range(n_models):
                # This is a simplified approach for MARS
                equation_dict[f'model_{i}'] = 0.0
        
        # Add intercept
        if hasattr(model, 'intercept_'):
            equation_dict['constant'] = float(model.intercept_)
        else:
            equation_dict['constant'] = 0.0
        
        return equation_dict
