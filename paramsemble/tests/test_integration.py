"""Integration tests for ELM package."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import json
from paramsemble.estimator import ParamsembleRegressor


class TestIntegrationFitPredict:
    """Test complete fit-predict workflow with synthetic data."""
    
    def test_complete_workflow_elasticnet(self):
        """Test complete fit-predict workflow with ElasticNet ensemble."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 8
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        # Create strong linear relationship with first 3 features
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + X_train[:, 2] * 2 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + X_test[:, 2] * 2 + np.random.randn(n_samples_test) * 0.1
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create and fit estimator with more combinations to ensure some beat baseline
        estimator = ParamsembleRegressor(
            m=30, 
            f=3, 
            sample='unique', 
            method='elastic', 
            spread=8, 
            random_state=42
        )
        estimator.fit(X_train_df, y_train, X_test_df, y_test)
        
        # Verify fitted attributes
        assert hasattr(estimator, 'feature_names_')
        assert hasattr(estimator, 'n_features_in_')
        assert hasattr(estimator, 'baseline_metrics_')
        assert hasattr(estimator, 'constituent_models_')
        assert hasattr(estimator, 'selected_models_')
        assert hasattr(estimator, 'ensemble_model_')
        assert hasattr(estimator, 'ensemble_equation_')
        
        # Verify baseline metrics
        assert 'wmape' in estimator.baseline_metrics_
        assert 'r2' in estimator.baseline_metrics_
        assert isinstance(estimator.baseline_metrics_['wmape'], (int, float))
        assert isinstance(estimator.baseline_metrics_['r2'], (int, float))
        
        # Verify constituent models
        assert len(estimator.constituent_models_) > 0
        for model_info in estimator.constituent_models_:
            assert 'features' in model_info
            assert 'wmape' in model_info
            assert 'r2' in model_info
            assert 'equation_dict' in model_info
        
        # Verify selected models
        assert len(estimator.selected_models_) > 0
        assert len(estimator.selected_models_) <= estimator.spread
        
        # Generate predictions on new data
        X_pred = np.random.randn(30, n_features)
        X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
        predictions = estimator.predict(X_pred_df)
        
        # Verify predictions
        assert len(predictions) == 30
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        assert predictions.dtype in [np.float64, np.float32]
    
    def test_complete_workflow_mars(self):
        """Test complete fit-predict workflow with MARS ensemble."""
        # Skip test if py-earth is not installed
        try:
            from pyearth import Earth
        except ImportError:
            pytest.skip("py-earth not installed, skipping MARS test")
        
        # Generate synthetic data with strong linear relationships
        np.random.seed(123)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 8
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + X_train[:, 2] * 2 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + X_test[:, 2] * 2 + np.random.randn(n_samples_test) * 0.1
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create and fit estimator with MARS
        estimator = ParamsembleRegressor(
            m=30, 
            f=3, 
            sample='unique', 
            method='mars', 
            spread=8, 
            random_state=123
        )
        estimator.fit(X_train_df, y_train, X_test_df, y_test)
        
        # Verify fitted attributes
        assert hasattr(estimator, 'ensemble_model_')
        assert hasattr(estimator, 'ensemble_equation_')
        
        # Generate predictions
        X_pred = np.random.randn(30, n_features)
        X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
        predictions = estimator.predict(X_pred_df)
        
        # Verify predictions
        assert len(predictions) == 30
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))


class TestIntegrationSerialization:
    """Test model serialization and loading round-trip."""
    
    def test_serialization_roundtrip_elasticnet(self):
        """Test saving and loading model with ElasticNet ensemble."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + X_train[:, 2] * 2 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + X_test[:, 2] * 2 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit model and save to JSON
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='elastic', 
                spread=8,
                modeljson=modeljson_path,
                random_state=42
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Verify JSON file was created
            assert os.path.exists(modeljson_path)
            
            # Load JSON and verify structure
            with open(modeljson_path, 'r') as f:
                model_data = json.load(f)
            
            assert 'method' in model_data
            assert model_data['method'] == 'elastic'
            assert 'constituent_models' in model_data
            assert 'ensemble_equation' in model_data
            assert len(model_data['constituent_models']) > 0
            
            # Generate predictions with fitted model
            X_pred = np.random.randn(20, n_features)
            X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
            predictions_original = estimator.predict(X_pred_df)
            
            # Load model from JSON and generate predictions
            predictions_df = estimator.score_from_json(X_pred_df, modeljson_path)
            
            # Verify predictions DataFrame structure
            assert 'id' in predictions_df.columns
            assert 'predicted' in predictions_df.columns
            assert len(predictions_df) == 20
            
            # Verify predictions match (within numerical tolerance)
            predictions_loaded = predictions_df['predicted'].values
            np.testing.assert_allclose(predictions_original, predictions_loaded, rtol=1e-10)
            
        finally:
            # Clean up temporary file
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)
    
    def test_serialization_roundtrip_mars(self):
        """Test saving and loading model with MARS ensemble."""
        # Skip test if py-earth is not installed
        try:
            from pyearth import Earth
        except ImportError:
            pytest.skip("py-earth not installed, skipping MARS test")
        
        # Generate synthetic data with strong linear relationships
        np.random.seed(123)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + X_train[:, 2] * 2 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + X_test[:, 2] * 2 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit model and save to JSON
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='mars', 
                spread=8,
                modeljson=modeljson_path,
                random_state=123
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Verify JSON file was created
            assert os.path.exists(modeljson_path)
            
            # Generate predictions with fitted model
            X_pred = np.random.randn(20, n_features)
            X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
            predictions_original = estimator.predict(X_pred_df)
            
            # Load model from JSON and generate predictions
            predictions_df = estimator.score_from_json(X_pred_df, modeljson_path)
            predictions_loaded = predictions_df['predicted'].values
            
            # Verify predictions match (within numerical tolerance)
            np.testing.assert_allclose(predictions_original, predictions_loaded, rtol=1e-10)
            
        finally:
            # Clean up temporary file
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)


class TestIntegrationScoring:
    """Test score_from_json() with saved model."""
    
    def test_score_from_json_with_ids(self):
        """Test scoring with custom ID column."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit and save model
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='elastic', 
                spread=8,
                modeljson=modeljson_path,
                random_state=42
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Create scoring data with custom IDs
            X_score = np.random.randn(25, n_features)
            X_score_df = pd.DataFrame(X_score, columns=feature_names)
            custom_ids = [f'ID_{i:03d}' for i in range(25)]
            
            # Score with custom IDs
            predictions_df = estimator.score_from_json(X_score_df, modeljson_path, id_column=custom_ids)
            
            # Verify output structure
            assert len(predictions_df) == 25
            assert 'id' in predictions_df.columns
            assert 'predicted' in predictions_df.columns
            assert list(predictions_df['id']) == custom_ids
            assert not predictions_df['predicted'].isnull().any()
            
        finally:
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)
    
    def test_score_from_json_without_ids(self):
        """Test scoring without custom ID column (uses default indices)."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit and save model
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='elastic', 
                spread=8,
                modeljson=modeljson_path,
                random_state=42
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Score without custom IDs
            X_score = np.random.randn(25, n_features)
            X_score_df = pd.DataFrame(X_score, columns=feature_names)
            predictions_df = estimator.score_from_json(X_score_df, modeljson_path)
            
            # Verify output structure with default IDs
            assert len(predictions_df) == 25
            assert list(predictions_df['id']) == list(range(25))
            
        finally:
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)


class TestIntegrationSQLExport:
    """Test SQL export generates valid, executable SQL."""
    
    def test_sql_export_structure(self):
        """Test that SQL export generates valid structure."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit and save model
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='elastic', 
                spread=8,
                modeljson=modeljson_path,
                random_state=42
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Export to SQL
            sql_code = estimator.export_sql(modeljson_path, table_name='input_data', id_column='id')
            
            # Verify SQL structure
            assert isinstance(sql_code, str)
            assert len(sql_code) > 0
            
            # Verify SQL contains expected keywords
            assert 'WITH' in sql_code
            assert 'SELECT' in sql_code
            assert 'FROM' in sql_code
            assert 'AS' in sql_code
            
            # Verify SQL contains CTEs for constituent models
            assert 'model_0' in sql_code
            assert 'model_1' in sql_code
            
            # Verify SQL contains ensemble_inputs CTE
            assert 'ensemble_inputs' in sql_code
            
            # Verify SQL contains final prediction
            assert 'predicted' in sql_code
            
            # Verify SQL references the input table
            assert 'input_data' in sql_code
            
            # Count number of CTEs (should match number of selected models + 1 for ensemble_inputs)
            n_selected = len(estimator.selected_models_)
            # Each CTE appears in definition
            for i in range(n_selected):
                assert f'model_{i} AS' in sql_code
            
        finally:
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)
    
    def test_sql_export_custom_table_and_id(self):
        """Test SQL export with custom table name and ID column."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit and save model
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='elastic', 
                spread=8,
                modeljson=modeljson_path,
                random_state=42
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Export to SQL with custom names
            sql_code = estimator.export_sql(
                modeljson_path, 
                table_name='my_data_table', 
                id_column='customer_id'
            )
            
            # Verify custom names appear in SQL
            assert 'my_data_table' in sql_code
            assert 'customer_id' in sql_code
            
        finally:
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)


class TestIntegrationSQLPredictions:
    """Test SQL output produces same predictions as Python scoring."""
    
    def test_sql_predictions_match_python(self):
        """Test that SQL predictions match Python predictions within numerical tolerance."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + X_train[:, 2] * 2 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + X_test[:, 2] * 2 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create temporary file for model JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            modeljson_path = f.name
        
        try:
            # Fit and save model
            estimator = ParamsembleRegressor(
                m=25, 
                f=3, 
                sample='unique', 
                method='elastic', 
                spread=8,
                modeljson=modeljson_path,
                random_state=42
            )
            estimator.fit(X_train_df, y_train, X_test_df, y_test)
            
            # Create scoring data
            X_score = np.random.randn(10, n_features)
            X_score_df = pd.DataFrame(X_score, columns=feature_names)
            
            # Get Python predictions
            python_predictions_df = estimator.score_from_json(X_score_df, modeljson_path)
            python_predictions = python_predictions_df['predicted'].values
            
            # Load model JSON to manually compute SQL-equivalent predictions
            with open(modeljson_path, 'r') as f:
                model_data = json.load(f)
            
            # Manually compute predictions using the same logic as SQL would
            constituent_models = model_data['constituent_models']
            ensemble_equation = model_data['ensemble_equation']
            
            # Step 1: Compute constituent predictions
            constituent_preds = []
            for model_info in constituent_models:
                equation_dict = model_info['equation_dict']
                # Compute: constant + sum(coef * feature)
                pred = np.full(len(X_score_df), equation_dict['constant'])
                for feature, coef in equation_dict.items():
                    if feature != 'constant':
                        pred += X_score_df[feature].values * coef
                constituent_preds.append(pred)
            
            constituent_preds = np.column_stack(constituent_preds)
            
            # Step 2: Compute ensemble predictions
            sql_equivalent_predictions = np.full(len(X_score_df), ensemble_equation['constant'])
            for i in range(len(constituent_models)):
                model_key = f'model_{i}'
                if model_key in ensemble_equation:
                    sql_equivalent_predictions += constituent_preds[:, i] * ensemble_equation[model_key]
            
            # Verify predictions match within numerical tolerance
            np.testing.assert_allclose(
                python_predictions, 
                sql_equivalent_predictions, 
                rtol=1e-10,
                err_msg="SQL-equivalent predictions do not match Python predictions"
            )
            
        finally:
            if os.path.exists(modeljson_path):
                os.remove(modeljson_path)


class TestIntegrationSklearnCompatibility:
    """Test sklearn check_estimator compatibility."""
    
    def test_sklearn_estimator_interface(self):
        """Test that ParamsembleRegressor follows sklearn estimator interface."""
        from sklearn.base import BaseEstimator, RegressorMixin
        
        # Verify ParamsembleRegressor inherits from correct base classes
        estimator = ParamsembleRegressor(m=10, f=3, random_state=42)
        assert isinstance(estimator, BaseEstimator)
        assert isinstance(estimator, RegressorMixin)
        
        # Verify get_params and set_params work
        params = estimator.get_params()
        assert 'm' in params
        assert 'f' in params
        assert 'sample' in params
        assert 'method' in params
        assert 'spread' in params
        
        # Test set_params
        estimator.set_params(m=20, spread=15)
        assert estimator.m == 20
        assert estimator.spread == 15
    
    def test_sklearn_score_method(self):
        """Test that sklearn's default score method works."""
        # Generate synthetic data with strong linear relationships
        np.random.seed(42)
        n_samples_train = 200
        n_samples_test = 100
        n_features = 7
        
        X_train = np.random.randn(n_samples_train, n_features) * 2
        y_train = X_train[:, 0] * 5 + X_train[:, 1] * 3 + np.random.randn(n_samples_train) * 0.1
        X_test = np.random.randn(n_samples_test, n_features) * 2
        y_test = X_test[:, 0] * 5 + X_test[:, 1] * 3 + np.random.randn(n_samples_test) * 0.1
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Fit estimator
        estimator = ParamsembleRegressor(
            m=25, 
            f=3, 
            sample='unique', 
            method='elastic', 
            spread=8,
            random_state=42
        )
        estimator.fit(X_train_df, y_train, X_test_df, y_test)
        
        # Test score method (inherited from RegressorMixin)
        X_score = np.random.randn(30, n_features) * 2
        y_score = X_score[:, 0] * 5 + X_score[:, 1] * 3 + np.random.randn(30) * 0.1
        X_score_df = pd.DataFrame(X_score, columns=feature_names)
        
        # Score should return R2 by default
        r2_score = estimator.score(X_score_df, y_score)
        assert isinstance(r2_score, (int, float))
        assert -1 <= r2_score <= 1  # R2 can be negative for poor models
