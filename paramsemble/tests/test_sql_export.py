"""Tests for SQL exporter."""

import pytest
import tempfile
import os
import json
from paramsemble.sql_export import SQLExporter
from paramsemble.serialization import ModelSerializer


def test_export_to_sql_basic():
    """Test basic SQL export functionality."""
    # Create a simple model JSON
    constituent_models = [
        {
            'model_id': 0,
            'features': ['feature_0', 'feature_1'],
            'wmape': 0.5,
            'r2': 0.7,
            'equation_dict': {
                'feature_0': 1.5,
                'feature_1': -0.5,
                'constant': 2.0
            }
        }
    ]
    
    ensemble_equation = {
        'model_0': 1.0,
        'constant': 0.0
    }
    
    # Save to temporary file
    serializer = ModelSerializer()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        serializer.save_ensemble_model('elastic', constituent_models, ensemble_equation, temp_file)
        
        # Export to SQL
        exporter = SQLExporter()
        sql_code = exporter.export_to_sql(temp_file)
        
        # Verify SQL structure
        assert 'WITH' in sql_code
        assert 'model_0 AS' in sql_code
        assert 'ensemble_inputs AS' in sql_code
        assert 'SELECT' in sql_code
        assert 'FROM ensemble_inputs' in sql_code
        assert 'AS predicted' in sql_code
        
        # Verify features are in SQL
        assert 'feature_0' in sql_code
        assert 'feature_1' in sql_code
        
        # Verify coefficients are in SQL
        assert '1.5' in sql_code
        assert '-0.5' in sql_code or '(-0.5' in sql_code
        assert '2.0' in sql_code
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_export_to_sql_multiple_models():
    """Test SQL export with multiple constituent models."""
    # Create multiple models
    constituent_models = [
        {
            'model_id': 0,
            'features': ['feature_0', 'feature_1'],
            'wmape': 0.5,
            'r2': 0.7,
            'equation_dict': {
                'feature_0': 1.5,
                'feature_1': -0.5,
                'constant': 2.0
            }
        },
        {
            'model_id': 1,
            'features': ['feature_2', 'feature_3'],
            'wmape': 0.4,
            'r2': 0.8,
            'equation_dict': {
                'feature_2': 0.8,
                'feature_3': 1.2,
                'constant': -1.0
            }
        }
    ]
    
    ensemble_equation = {
        'model_0': 0.6,
        'model_1': 0.4,
        'constant': 0.5
    }
    
    # Save to temporary file
    serializer = ModelSerializer()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        serializer.save_ensemble_model('elastic', constituent_models, ensemble_equation, temp_file)
        
        # Export to SQL
        exporter = SQLExporter()
        sql_code = exporter.export_to_sql(temp_file)
        
        # Verify both models are in SQL
        assert 'model_0 AS' in sql_code
        assert 'model_1 AS' in sql_code
        
        # Verify ensemble_inputs joins both models
        assert 'model_0_pred' in sql_code
        assert 'model_1_pred' in sql_code
        
        # Verify ensemble coefficients
        assert '0.6' in sql_code
        assert '0.4' in sql_code
        assert '0.5' in sql_code
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_create_constituent_cte():
    """Test creating a constituent CTE."""
    exporter = SQLExporter()
    
    model_info = {
        'model_id': 0,
        'features': ['feature_0', 'feature_1'],
        'equation_dict': {
            'feature_0': 1.5,
            'feature_1': -0.5,
            'constant': 2.0
        }
    }
    
    cte_sql = exporter.create_constituent_cte(model_info, 'model_0', 'input_data')
    
    # Verify CTE structure
    assert 'model_0 AS' in cte_sql
    assert 'SELECT' in cte_sql
    assert 'FROM input_data' in cte_sql
    assert 'AS prediction' in cte_sql
    
    # Verify calculation includes all terms
    assert '2.0' in cte_sql  # constant
    assert '1.5' in cte_sql  # feature_0 coefficient
    assert '-0.5' in cte_sql or '(-0.5' in cte_sql  # feature_1 coefficient
    assert 'feature_0' in cte_sql
    assert 'feature_1' in cte_sql


def test_create_ensemble_select():
    """Test creating the ensemble SELECT statement."""
    exporter = SQLExporter()
    
    ensemble_equation = {
        'model_0': 0.6,
        'model_1': 0.4,
        'constant': 0.5
    }
    
    constituent_ctes = ['model_0', 'model_1']
    
    select_sql = exporter.create_ensemble_select(ensemble_equation, constituent_ctes, 'id')
    
    # Verify SELECT structure
    assert 'SELECT' in select_sql
    assert 'FROM ensemble_inputs' in select_sql
    assert 'AS predicted' in select_sql
    assert 'id' in select_sql
    
    # Verify ensemble calculation
    assert '0.5' in select_sql  # constant
    assert '0.6' in select_sql  # model_0 coefficient
    assert '0.4' in select_sql  # model_1 coefficient
    assert 'model_0_pred' in select_sql
    assert 'model_1_pred' in select_sql


def test_export_to_sql_custom_table_and_id():
    """Test SQL export with custom table name and ID column."""
    constituent_models = [
        {
            'model_id': 0,
            'features': ['feature_0'],
            'wmape': 0.5,
            'r2': 0.7,
            'equation_dict': {
                'feature_0': 1.0,
                'constant': 0.0
            }
        }
    ]
    
    ensemble_equation = {
        'model_0': 1.0,
        'constant': 0.0
    }
    
    # Save to temporary file
    serializer = ModelSerializer()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        serializer.save_ensemble_model('elastic', constituent_models, ensemble_equation, temp_file)
        
        # Export to SQL with custom table and ID
        exporter = SQLExporter()
        sql_code = exporter.export_to_sql(temp_file, table_name='my_table', id_column='customer_id')
        
        # Verify custom table name is used
        assert 'FROM my_table' in sql_code
        
        # Verify custom ID column is used
        assert 'customer_id' in sql_code
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
