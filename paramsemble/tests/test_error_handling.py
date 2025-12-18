"""Tests for comprehensive error handling across all modules."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.linear_model import Lasso

from paramsemble.estimator import ParamsembleRegressor
from paramsemble.feature_generation import FeatureCombinationGenerator
from paramsemble.baseline import BaselineModel
from paramsemble.constituent import ConstituentModelTrainer
from paramsemble.selection import ModelSelector
from paramsemble.ensemble import EnsembleBuilder
from paramsemble.serialization import ModelSerializer
from paramsemble.scoring import ModelScorer
from paramsemble.sql_export import SQLExporter


class TestParamsembleRegressorErrorHandling:
    """Test error handling in ParamsembleRegressor."""
    
    def test_init_invalid_m(self):
        """Test that invalid m parameter raises error."""
        with pytest.raises(ValueError, match="Parameter 'm' must be a positive integer"):
            ParamsembleRegressor(m=-1, f=2)
        
        with pytest.raises(ValueError, match="Parameter 'm' must be a positive integer"):
            ParamsembleRegressor(m=0, f=2)
        
        with pytest.raises(ValueError, match="Parameter 'm' must be a positive integer"):
            ParamsembleRegressor(m="invalid", f=2)
    
    def test_init_invalid_f(self):
        """Test that invalid f parameter raises error."""
        with pytest.raises(ValueError, match="Parameter 'f' must be a positive integer"):
            ParamsembleRegressor(m=10, f=-1)
        
        with pytest.raises(ValueError, match="Parameter 'f' must be a positive integer"):
            ParamsembleRegressor(m=10, f=0)
    
    def test_init_invalid_sample(self):
        """Test that invalid sample parameter raises error."""
        with pytest.raises(ValueError, match="Parameter 'sample' must be 'unique' or 'replace'"):
            ParamsembleRegressor(m=10, f=2, sample='invalid')
    
    def test_init_invalid_method(self):
        """Test that invalid method parameter raises error."""
        with pytest.raises(ValueError, match="Parameter 'method' must be 'elastic' or 'mars'"):
            ParamsembleRegressor(m=10, f=2, method='invalid')
    
    def test_init_invalid_spread(self):
        """Test that invalid spread parameter raises error."""
        with pytest.raises(ValueError, match="Parameter 'spread' must be a positive integer"):
            ParamsembleRegressor(m=10, f=2, spread=-1)
        
        with pytest.raises(ValueError, match="Parameter 'spread' must be a positive integer"):
            ParamsembleRegressor(m=10, f=2, spread=0)
    
    def test_fit_missing_f(self):
        """Test that fit raises error when f is not specified."""
        X_train = pd.DataFrame(np.random.rand(50, 5))
        y_train = np.random.rand(50)
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = np.random.rand(20)
        
        regressor = ParamsembleRegressor(m=5, f=None)
        with pytest.raises(ValueError, match="Parameter 'f' must be specified"):
            regressor.fit(X_train, y_train, X_test, y_test)
    
    def test_fit_missing_data(self):
        """Test that fit raises error when data is missing."""
        regressor = ParamsembleRegressor(m=5, f=2)
        
        with pytest.raises(ValueError, match="X_train and y_train must be provided"):
            regressor.fit(None, None, pd.DataFrame(), np.array([]))
        
        with pytest.raises(ValueError, match="X_test and y_test must be provided"):
            regressor.fit(pd.DataFrame(), np.array([]), None, None)
    
    def test_fit_empty_data(self):
        """Test that fit raises error when data is empty."""
        regressor = ParamsembleRegressor(m=5, f=2)
        
        with pytest.raises(ValueError, match="X_train has zero rows"):
            regressor.fit(pd.DataFrame(), np.array([]), pd.DataFrame(np.random.rand(10, 5)), np.random.rand(10))
    
    def test_fit_shape_mismatch(self):
        """Test that fit raises error when shapes don't match."""
        regressor = ParamsembleRegressor(m=5, f=2)
        
        X_train = pd.DataFrame(np.random.rand(50, 5))
        y_train = np.random.rand(40)  # Wrong size
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = np.random.rand(20)
        
        with pytest.raises(ValueError, match="X_train and y_train must have the same number of samples"):
            regressor.fit(X_train, y_train, X_test, y_test)
    
    def test_fit_nan_values(self):
        """Test that fit raises error when data contains NaN."""
        regressor = ParamsembleRegressor(m=5, f=2)
        
        X_train = pd.DataFrame(np.random.rand(50, 5))
        X_train.iloc[0, 0] = np.nan
        y_train = np.random.rand(50)
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = np.random.rand(20)
        
        with pytest.raises(ValueError, match="X_train contains NaN values"):
            regressor.fit(X_train, y_train, X_test, y_test)
    
    def test_fit_infinite_values(self):
        """Test that fit raises error when data contains infinite values."""
        regressor = ParamsembleRegressor(m=5, f=2)
        
        X_train = pd.DataFrame(np.random.rand(50, 5))
        X_train.iloc[0, 0] = np.inf
        y_train = np.random.rand(50)
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = np.random.rand(20)
        
        with pytest.raises(ValueError, match="X_train contains infinite values"):
            regressor.fit(X_train, y_train, X_test, y_test)
    
    def test_predict_not_fitted(self):
        """Test that predict raises error when not fitted."""
        regressor = ParamsembleRegressor(m=5, f=2)
        X = pd.DataFrame(np.random.rand(10, 5))
        
        with pytest.raises(Exception, match="not fitted"):
            regressor.predict(X)
    
    def test_predict_missing_data(self):
        """Test that predict raises error when data is missing."""
        # Create a fitted regressor by manually setting required attributes
        regressor = ParamsembleRegressor(m=5, f=2, spread=2)
        
        # Manually set fitted attributes to simulate a fitted model
        regressor.feature_names_ = ['f0', 'f1', 'f2', 'f3', 'f4']
        regressor.n_features_in_ = 5
        regressor.ensemble_model_ = "mock_model"  # Just needs to exist
        regressor.selected_models_ = []
        
        with pytest.raises(ValueError, match="X must be provided"):
            regressor.predict(None)
    
    def test_predict_nan_values(self):
        """Test that predict raises error when data contains NaN."""
        # Create a fitted regressor by manually setting required attributes
        regressor = ParamsembleRegressor(m=5, f=2, spread=2)
        
        # Manually set fitted attributes to simulate a fitted model
        regressor.feature_names_ = ['f0', 'f1', 'f2', 'f3', 'f4']
        regressor.n_features_in_ = 5
        regressor.ensemble_model_ = "mock_model"  # Just needs to exist
        regressor.selected_models_ = []
        
        X_pred = pd.DataFrame(np.random.rand(10, 5), columns=[f'f{i}' for i in range(5)])
        X_pred.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="X contains NaN values"):
            regressor.predict(X_pred)


class TestFeatureCombinationGeneratorErrorHandling:
    """Test error handling in FeatureCombinationGenerator."""
    
    def test_generate_combinations_empty_features(self):
        """Test that empty feature list raises error."""
        generator = FeatureCombinationGenerator()
        
        with pytest.raises(ValueError, match="feature_names cannot be empty"):
            generator.generate_combinations([], m=5, f=2)
    
    def test_generate_combinations_invalid_m(self):
        """Test that invalid m raises error."""
        generator = FeatureCombinationGenerator()
        
        with pytest.raises(ValueError, match="m must be a positive integer"):
            generator.generate_combinations(['f1', 'f2', 'f3'], m=-1, f=2)
    
    def test_generate_combinations_invalid_f(self):
        """Test that invalid f raises error."""
        generator = FeatureCombinationGenerator()
        
        with pytest.raises(ValueError, match="f must be a positive integer"):
            generator.generate_combinations(['f1', 'f2', 'f3'], m=5, f=0)
    
    def test_calculate_max_combinations_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        generator = FeatureCombinationGenerator()
        
        with pytest.raises(ValueError, match="n_features must be a positive integer"):
            generator.calculate_max_combinations(n_features=-1, f=2, sample='unique')


class TestModelSerializerErrorHandling:
    """Test error handling in ModelSerializer."""
    
    def test_save_constituent_models_empty_list(self):
        """Test that empty models list raises error."""
        serializer = ModelSerializer()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="models_info cannot be empty"):
                serializer.save_constituent_models([], filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_save_ensemble_model_invalid_method(self):
        """Test that invalid method raises error."""
        serializer = ModelSerializer()
        
        model_info = {
            'model_id': 0,
            'features': ['f1', 'f2'],
            'wmape': 0.5,
            'r2': 0.7,
            'equation_dict': {'f1': 1.0, 'f2': 2.0, 'constant': 0.5}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="method must be 'elastic' or 'mars'"):
                serializer.save_ensemble_model('invalid', [model_info], {'constant': 0.0}, filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_model_json_file_not_found(self):
        """Test that missing file raises error."""
        serializer = ModelSerializer()
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            serializer.load_model_json('nonexistent_file.json')


class TestSQLExporterErrorHandling:
    """Test error handling in SQLExporter."""
    
    def test_export_to_sql_missing_file(self):
        """Test that missing file raises error."""
        exporter = SQLExporter()
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            exporter.export_to_sql('nonexistent_file.json')
    
    def test_export_to_sql_invalid_table_name(self):
        """Test that invalid table name raises error."""
        exporter = SQLExporter()
        
        # Create a temporary model JSON file
        model_data = {
            'constituent_models': [
                {
                    'model_id': 0,
                    'features': ['f1', 'f2'],
                    'equation_dict': {'f1': 1.0, 'f2': 2.0, 'constant': 0.5}
                }
            ],
            'ensemble_equation': {'model_0': 1.0, 'constant': 0.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            json.dump(model_data, f)
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="table_name .* contains invalid characters"):
                exporter.export_to_sql(filepath, table_name='invalid-table', id_column='id')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_export_to_sql_invalid_feature_name(self):
        """Test that invalid feature name raises error."""
        exporter = SQLExporter()
        
        # Create a temporary model JSON file with invalid feature name
        model_data = {
            'constituent_models': [
                {
                    'model_id': 0,
                    'features': ['f-1', 'f2'],  # Invalid feature name with dash
                    'equation_dict': {'f-1': 1.0, 'f2': 2.0, 'constant': 0.5}
                }
            ],
            'ensemble_equation': {'model_0': 1.0, 'constant': 0.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            json.dump(model_data, f)
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Feature name .* contains invalid characters"):
                exporter.export_to_sql(filepath, table_name='input_data', id_column='id')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestModelScorerErrorHandling:
    """Test error handling in ModelScorer."""
    
    def test_score_dataset_missing_data(self):
        """Test that missing data raises error."""
        scorer = ModelScorer()
        model_data = {
            'constituent_models': [],
            'ensemble_equation': {}
        }
        
        with pytest.raises(ValueError, match="X must be provided"):
            scorer.score_dataset(None, model_data)
    
    def test_score_dataset_invalid_model_data(self):
        """Test that invalid model data raises error."""
        scorer = ModelScorer()
        X = pd.DataFrame(np.random.rand(10, 5))
        
        with pytest.raises(ValueError, match="model_data must be a dictionary"):
            scorer.score_dataset(X, "invalid")
    
    def test_score_dataset_missing_keys(self):
        """Test that missing keys in model data raise error."""
        scorer = ModelScorer()
        X = pd.DataFrame(np.random.rand(10, 5))
        
        with pytest.raises(ValueError, match="model_data missing 'constituent_models' key"):
            scorer.score_dataset(X, {})
    
    def test_apply_equation_dict_missing_feature(self):
        """Test that missing feature raises error."""
        scorer = ModelScorer()
        X = pd.DataFrame(np.random.rand(10, 3), columns=['f1', 'f2', 'f3'])
        equation_dict = {'f1': 1.0, 'f4': 2.0, 'constant': 0.5}  # f4 doesn't exist
        
        with pytest.raises(ValueError, match="Feature 'f4' from equation_dict not found in X"):
            scorer.apply_equation_dict(X, equation_dict)


class TestEnsembleBuilderErrorHandling:
    """Test error handling in EnsembleBuilder."""
    
    def test_build_ensemble_missing_data(self):
        """Test that missing data raises error."""
        builder = EnsembleBuilder()
        
        with pytest.raises(ValueError, match="constituent_predictions must be provided"):
            builder.build_ensemble(None, np.array([1, 2, 3]))
    
    def test_build_ensemble_empty_data(self):
        """Test that empty data raises error."""
        builder = EnsembleBuilder()
        
        with pytest.raises(ValueError, match="constituent_predictions has zero rows"):
            builder.build_ensemble(np.array([]).reshape(0, 2), np.array([]))
    
    def test_build_ensemble_nan_values(self):
        """Test that NaN values raise error."""
        builder = EnsembleBuilder()
        
        predictions = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        y_true = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="constituent_predictions contains NaN values"):
            builder.build_ensemble(predictions, y_true)


class TestConstituentModelTrainerErrorHandling:
    """Test error handling in ConstituentModelTrainer."""
    
    def test_train_models_missing_data(self):
        """Test that missing data raises error."""
        trainer = ConstituentModelTrainer()
        
        with pytest.raises(ValueError, match="X_train and y_train must be provided"):
            trainer.train_models(None, None, pd.DataFrame(), np.array([]), [['f1', 'f2']])
    
    def test_train_models_empty_combinations(self):
        """Test that empty combinations raise error."""
        trainer = ConstituentModelTrainer()
        
        X_train = pd.DataFrame(np.random.rand(50, 5))
        y_train = np.random.rand(50)
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = np.random.rand(20)
        
        with pytest.raises(ValueError, match="feature_combinations cannot be empty"):
            trainer.train_models(X_train, y_train, X_test, y_test, [])
    
    def test_extract_equation_dict_unfitted_model(self):
        """Test that unfitted model raises error."""
        trainer = ConstituentModelTrainer()
        model = Lasso()  # Not fitted
        
        with pytest.raises(ValueError, match="model must have 'coef_' attribute"):
            trainer.extract_equation_dict(model, ['f1', 'f2'])


class TestModelSelectorErrorHandling:
    """Test error handling in ModelSelector."""
    
    def test_select_top_models_missing_baseline_keys(self):
        """Test that missing baseline keys raise error."""
        selector = ModelSelector()
        
        models_info = [
            {'model_id': 0, 'wmape': 0.5, 'r2': 0.7, 'features': ['f1', 'f2']}
        ]
        
        # Test empty baseline_metrics
        with pytest.raises(ValueError, match="baseline_metrics cannot be empty"):
            selector.select_top_models(models_info, {}, spread=5)
        
        # Test missing wmape key
        with pytest.raises(ValueError, match="baseline_metrics missing 'wmape' key"):
            selector.select_top_models(models_info, {'r2': 0.5}, spread=5)
        
        # Test missing r2 key
        with pytest.raises(ValueError, match="baseline_metrics missing 'r2' key"):
            selector.select_top_models(models_info, {'wmape': 0.5}, spread=5)


class TestBaselineModelErrorHandling:
    """Test error handling in BaselineModel."""
    
    def test_fit_and_evaluate_missing_data(self):
        """Test that missing data raises error."""
        baseline = BaselineModel()
        
        with pytest.raises(ValueError, match="X_train and y_train must be provided"):
            baseline.fit_and_evaluate(None, None, pd.DataFrame(), np.array([]))
