"""Model serialization module."""

import json
from datetime import datetime


class ModelSerializer:
    """Serializes and deserializes models to/from JSON."""
    
    def save_constituent_models(self, models_info, filepath):
        """
        Save constituent model details to JSON.
        
        Parameters
        ----------
        models_info : list of dict
            List of model information dictionaries containing:
            - model_id: int
            - features: list of str
            - wmape: float
            - r2: float
            - equation_dict: dict
            - model_object: sklearn model (not serialized)
        filepath : str
            Path where JSON file will be saved
        """
        # Validate inputs
        if not models_info:
            raise ValueError("models_info cannot be empty")
        if not filepath:
            raise ValueError("filepath must be provided")
        
        try:
            # Prepare data for serialization (exclude model_object)
            serializable_models = []
            for model_info in models_info:
                serializable_model = {
                    'model_id': model_info['model_id'],
                    'features': model_info['features'],
                    'wmape': model_info['wmape'],
                    'r2': model_info['r2'],
                    'equation_dict': model_info['equation_dict']
                }
                serializable_models.append(serializable_model)
            
            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(serializable_models, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write to file '{filepath}': {e}")
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid model_info structure: {e}")
    
    def save_ensemble_model(self, method, selected_models, ensemble_equation, filepath):
        """
        Save complete ensemble model to JSON.
        
        Parameters
        ----------
        method : str
            Ensemble method ('elastic' or 'mars')
        selected_models : list of dict
            List of selected constituent model information
        ensemble_equation : dict
            Ensemble model equation dictionary
        filepath : str
            Path where JSON file will be saved
        """
        # Validate inputs
        if not method:
            raise ValueError("method must be provided")
        if method not in ['elastic', 'mars']:
            raise ValueError(f"method must be 'elastic' or 'mars', got '{method}'")
        if not selected_models:
            raise ValueError("selected_models cannot be empty")
        if not ensemble_equation:
            raise ValueError("ensemble_equation cannot be empty")
        if not filepath:
            raise ValueError("filepath must be provided")
        
        try:
            # Prepare constituent models data (exclude model_object)
            constituent_models = []
            for model_info in selected_models:
                constituent_model = {
                    'model_id': model_info['model_id'],
                    'features': model_info['features'],
                    'wmape': model_info['wmape'],
                    'r2': model_info['r2'],
                    'equation_dict': model_info['equation_dict']
                }
                constituent_models.append(constituent_model)
            
            # Determine number of features from first model
            n_features = len(constituent_models[0]['features']) if constituent_models else 0
            
            # Create complete model structure
            model_data = {
                'method': method,
                'constituent_models': constituent_models,
                'ensemble_equation': ensemble_equation,
                'metadata': {
                    'n_features': n_features,
                    'n_constituent_models': len(constituent_models),
                    'training_date': datetime.now().isoformat()
                }
            }
            
            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write to file '{filepath}': {e}")
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid model data structure: {e}")
    
    def load_model_json(self, filepath):
        """
        Load model from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to JSON file to load
        
        Returns
        -------
        model_data : dict
            Model configuration and coefficients
        """
        # Validate input
        if not filepath:
            raise ValueError("filepath must be provided")
        
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: '{filepath}'")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file '{filepath}': {e}")
        except IOError as e:
            raise IOError(f"Failed to read file '{filepath}': {e}")
        
        # Validate loaded data structure
        if not isinstance(model_data, dict):
            raise ValueError("Model JSON must contain a dictionary")
        
        return model_data
