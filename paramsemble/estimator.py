"""Main ParamsembleRegressor estimator class."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from .feature_generation import FeatureCombinationGenerator
from .baseline import BaselineModel
from .constituent import ConstituentModelTrainer
from .selection import ModelSelector
from .ensemble import EnsembleBuilder
from .serialization import ModelSerializer
from .scoring import ModelScorer
from .sql_export import SQLExporter


class ParamsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Parametric Ensemble Regression regressor.
    
    Parameters
    ----------
    m : int, default=100
        Number of feature combinations to generate
    f : int
        Number of features per combination
    sample : {'unique', 'replace'}, default='unique'
        Sampling method for feature combinations
    method : {'elastic', 'mars'}, default='elastic'
        Ensemble method to use
    spread : int, default=10
        Number of top models to include in ensemble
    ELM2json : str, optional
        File path to save constituent model details
    modeljson : str, optional
        File path to save final ensemble model
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, m=100, f=None, sample='unique', method='elastic', 
                 spread=10, ELM2json=None, modeljson=None, random_state=None):
        # Validate parameters
        if m is not None and (not isinstance(m, int) or m < 1):
            raise ValueError(f"Parameter 'm' must be a positive integer, got {m}")
        
        if f is not None and (not isinstance(f, int) or f < 1):
            raise ValueError(f"Parameter 'f' must be a positive integer, got {f}")
        
        if sample not in ['unique', 'replace']:
            raise ValueError(f"Parameter 'sample' must be 'unique' or 'replace', got '{sample}'")
        
        if method not in ['elastic', 'mars']:
            raise ValueError(f"Parameter 'method' must be 'elastic' or 'mars', got '{method}'")
        
        if spread is not None and (not isinstance(spread, int) or spread < 1):
            raise ValueError(f"Parameter 'spread' must be a positive integer, got {spread}")
        
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"Parameter 'random_state' must be an integer or None, got {type(random_state)}")
        
        self.m = m
        self.f = f
        self.sample = sample
        self.method = method
        self.spread = spread
        self.ELM2json = ELM2json
        self.modeljson = modeljson
        self.random_state = random_state
    
    def fit(self, X_train, y_train, X_test, y_test, id_column=None):
        """
        Fit the ELM ensemble model.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix
        y_train : array-like of shape (n_samples,)
            Training target values
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix for evaluation
        y_test : array-like of shape (n_samples,)
            Test target values
        id_column : array-like, optional
            Identifiers for test samples
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Input validation
        if self.f is None:
            raise ValueError("Parameter 'f' must be specified")
        
        # Validate that datasets are provided
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided")
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided")
        
        # Convert to numpy arrays for validation
        X_train_array = np.asarray(X_train)
        X_test_array = np.asarray(X_test)
        y_train_array = np.asarray(y_train)
        y_test_array = np.asarray(y_test)
        
        # Validate that datasets are not empty
        if X_train_array.shape[0] == 0:
            raise ValueError("X_train has zero rows")
        if X_test_array.shape[0] == 0:
            raise ValueError("X_test has zero rows")
        if len(y_train_array) == 0:
            raise ValueError("y_train has zero elements")
        if len(y_test_array) == 0:
            raise ValueError("y_test has zero elements")
        
        # Convert to DataFrames if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        # Validate shapes
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"X_train and X_test must have the same number of features. "
                f"X_train has {X_train.shape[1]} features, X_test has {X_test.shape[1]} features"
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train must have the same number of samples. "
                f"X_train has {len(X_train)} samples, y_train has {len(y_train)} samples"
            )
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test and y_test must have the same number of samples. "
                f"X_test has {len(X_test)} samples, y_test has {len(y_test)} samples"
            )
        
        # Check for NaN and infinite values
        if X_train.isnull().any().any():
            raise ValueError("X_train contains NaN values")
        if X_test.isnull().any().any():
            raise ValueError("X_test contains NaN values")
        if np.any(np.isinf(X_train.values)):
            raise ValueError("X_train contains infinite values")
        if np.any(np.isinf(X_test.values)):
            raise ValueError("X_test contains infinite values")
        if np.any(np.isnan(y_train)):
            raise ValueError("y_train contains NaN values")
        if np.any(np.isnan(y_test)):
            raise ValueError("y_test contains NaN values")
        if np.any(np.isinf(y_train)):
            raise ValueError("y_train contains infinite values")
        if np.any(np.isinf(y_test)):
            raise ValueError("y_test contains infinite values")
        
        # Store feature names
        self.feature_names_ = list(X_train.columns)
        self.n_features_in_ = len(self.feature_names_)
        
        # Phase 1: Generate feature combinations
        feature_generator = FeatureCombinationGenerator()
        feature_combinations = feature_generator.generate_combinations(
            self.feature_names_, self.m, self.f, self.sample, self.random_state
        )
        self.feature_combinations_ = feature_combinations
        
        # Phase 2: Establish baseline
        baseline_model = BaselineModel()
        self.baseline_metrics_ = baseline_model.fit_and_evaluate(
            X_train, y_train, X_test, y_test, self.random_state
        )
        
        # Phase 3: Train constituent models
        constituent_trainer = ConstituentModelTrainer()
        self.constituent_models_ = constituent_trainer.train_models(
            X_train, y_train, X_test, y_test, feature_combinations
        )
        
        # Save constituent models if requested
        if self.ELM2json is not None:
            serializer = ModelSerializer()
            serializer.save_constituent_models(self.constituent_models_, self.ELM2json)
        
        # Phase 4: Select top models
        model_selector = ModelSelector()
        self.selected_models_ = model_selector.select_top_models(
            self.constituent_models_, self.baseline_metrics_, self.spread
        )
        
        if len(self.selected_models_) == 0:
            raise ValueError("No models outperformed the baseline. Cannot create ensemble.")
        
        # Phase 5: Build ensemble
        # Generate constituent predictions on test set
        constituent_predictions = []
        for model_info in self.selected_models_:
            model_obj = model_info['model_object']
            features = model_info['features']
            X_test_subset = X_test[features]
            pred = model_obj.predict(X_test_subset)
            constituent_predictions.append(pred)
        
        constituent_predictions = np.column_stack(constituent_predictions)
        
        # Build ensemble model
        ensemble_builder = EnsembleBuilder()
        self.ensemble_model_, self.ensemble_equation_ = ensemble_builder.build_ensemble(
            constituent_predictions, y_test, self.method
        )
        
        # Save ensemble model if requested
        if self.modeljson is not None:
            serializer = ModelSerializer()
            serializer.save_ensemble_model(
                self.method, self.selected_models_, self.ensemble_equation_, self.modeljson
            )
        
        # Store test set ID column if provided
        self.id_column_ = id_column
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the fitted ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        # Check if fitted
        if not hasattr(self, 'ensemble_model_'):
            raise NotFittedError(
                "This ELMRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        
        # Validate input
        if X is None:
            raise ValueError("X must be provided")
        
        # Convert to numpy array for validation
        X_array = np.asarray(X)
        
        # Validate that dataset is not empty
        if X_array.shape[0] == 0:
            raise ValueError("X has zero rows")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        # Validate features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ELMRegressor is expecting "
                f"{self.n_features_in_} features as input."
            )
        
        # Check for NaN and infinite values
        if X.isnull().any().any():
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X.values)):
            raise ValueError("X contains infinite values")
        
        # Generate constituent predictions
        constituent_predictions = []
        for model_info in self.selected_models_:
            model_obj = model_info['model_object']
            features = model_info['features']
            X_subset = X[features]
            pred = model_obj.predict(X_subset)
            constituent_predictions.append(pred)
        
        constituent_predictions = np.column_stack(constituent_predictions)
        
        # Generate ensemble predictions
        y_pred = self.ensemble_model_.predict(constituent_predictions)
        
        return y_pred
    
    def score_from_json(self, X, modeljson_path, id_column=None):
        """
        Score new data using a saved model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for scoring
        modeljson_path : str
            Path to saved model JSON file
        id_column : array-like, optional
            Identifiers for samples
            
        Returns
        -------
        predictions_df : DataFrame
            DataFrame with IDs and predictions
        """
        # Load model data
        serializer = ModelSerializer()
        model_data = serializer.load_model_json(modeljson_path)
        
        # Use ModelScorer to generate predictions
        scorer = ModelScorer()
        predictions_df = scorer.score_dataset(X, model_data, id_column)
        
        return predictions_df
    
    def export_sql(self, modeljson_path, table_name='input_data', id_column='id'):
        """
        Export model as SQL code.
        
        Parameters
        ----------
        modeljson_path : str
            Path to saved model JSON file
        table_name : str, default='input_data'
            Name of the input table in SQL
        id_column : str, default='id'
            Name of the ID column in the input table
            
        Returns
        -------
        sql_code : str
            Complete SQL query implementing the ensemble model
        """
        # Use SQLExporter to generate SQL code
        exporter = SQLExporter()
        sql_code = exporter.export_to_sql(modeljson_path, table_name, id_column)
        
        return sql_code
