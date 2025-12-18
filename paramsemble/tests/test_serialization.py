"""Tests for model serialization."""

import json
import tempfile
import os
import pytest
from paramsemble.serialization import ModelSerializer


class TestModelSerializer:
    """Test cases for ModelSerializer class."""
    
    def test_save_constituent_models(self):
        """Test saving constituent models to JSON."""
        # Create sample models_info
        models_info = [
            {
                'model_id': 0,
                'features': ['feature_0', 'feature_1'],
                'wmape': 0.25,
                'r2': 0.85,
                'equation_dict': {
                    'feature_0': 1.5,
                    'feature_1': -0.3,
                    'constant': 2.1
                },
                'model_object': 'should_not_be_serialized'
            },
            {
                'model_id': 1,
                'features': ['feature_2', 'feature_3'],
                'wmape': 0.30,
                'r2': 0.80,
                'equation_dict': {
                    'feature_2': 0.8,
                    'feature_3': 1.2,
                    'constant': -0.5
                },
                'model_object': 'should_not_be_serialized'
            }
        ]
        
        serializer = ModelSerializer()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            # Save models
            serializer.save_constituent_models(models_info, temp_filepath)
            
            # Load and verify
            with open(temp_filepath, 'r') as f:
                loaded_data = json.load(f)
            
            # Verify structure
            assert len(loaded_data) == 2
            
            # Verify first model
            assert loaded_data[0]['model_id'] == 0
            assert loaded_data[0]['features'] == ['feature_0', 'feature_1']
            assert loaded_data[0]['wmape'] == 0.25
            assert loaded_data[0]['r2'] == 0.85
            assert 'equation_dict' in loaded_data[0]
            assert 'model_object' not in loaded_data[0]  # Should be excluded
            
            # Verify second model
            assert loaded_data[1]['model_id'] == 1
            assert loaded_data[1]['features'] == ['feature_2', 'feature_3']
            
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    def test_save_ensemble_model(self):
        """Test saving complete ensemble model to JSON."""
        # Create sample data
        selected_models = [
            {
                'model_id': 0,
                'features': ['feature_0', 'feature_1'],
                'wmape': 0.25,
                'r2': 0.85,
                'equation_dict': {
                    'feature_0': 1.5,
                    'feature_1': -0.3,
                    'constant': 2.1
                },
                'model_object': 'should_not_be_serialized'
            }
        ]
        
        ensemble_equation = {
            'model_0': 0.7,
            'constant': 1.2
        }
        
        method = 'elastic'
        
        serializer = ModelSerializer()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            # Save ensemble model
            serializer.save_ensemble_model(method, selected_models, ensemble_equation, temp_filepath)
            
            # Load and verify
            with open(temp_filepath, 'r') as f:
                loaded_data = json.load(f)
            
            # Verify top-level structure
            assert 'method' in loaded_data
            assert loaded_data['method'] == 'elastic'
            
            assert 'constituent_models' in loaded_data
            assert len(loaded_data['constituent_models']) == 1
            
            assert 'ensemble_equation' in loaded_data
            assert loaded_data['ensemble_equation'] == ensemble_equation
            
            assert 'metadata' in loaded_data
            metadata = loaded_data['metadata']
            assert 'n_features' in metadata
            assert metadata['n_features'] == 2
            assert 'n_constituent_models' in metadata
            assert metadata['n_constituent_models'] == 1
            assert 'training_date' in metadata
            
            # Verify model_object is not serialized
            assert 'model_object' not in loaded_data['constituent_models'][0]
            
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    def test_load_model_json(self):
        """Test loading model from JSON file."""
        # Create sample JSON data
        model_data = {
            'method': 'mars',
            'constituent_models': [
                {
                    'model_id': 0,
                    'features': ['feature_0', 'feature_1'],
                    'wmape': 0.25,
                    'r2': 0.85,
                    'equation_dict': {
                        'feature_0': 1.5,
                        'feature_1': -0.3,
                        'constant': 2.1
                    }
                }
            ],
            'ensemble_equation': {
                'model_0': 0.7,
                'constant': 1.2
            },
            'metadata': {
                'n_features': 2,
                'n_constituent_models': 1,
                'training_date': '2024-01-01T00:00:00'
            }
        }
        
        serializer = ModelSerializer()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
            json.dump(model_data, f)
        
        try:
            # Load model
            loaded_data = serializer.load_model_json(temp_filepath)
            
            # Verify loaded data matches original
            assert loaded_data['method'] == 'mars'
            assert len(loaded_data['constituent_models']) == 1
            assert loaded_data['constituent_models'][0]['model_id'] == 0
            assert loaded_data['ensemble_equation']['model_0'] == 0.7
            assert loaded_data['metadata']['n_features'] == 2
            
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    def test_load_model_json_file_not_found(self):
        """Test loading from non-existent file raises error."""
        serializer = ModelSerializer()
        
        with pytest.raises(FileNotFoundError):
            serializer.load_model_json('nonexistent_file.json')
    
    def test_save_ensemble_model_with_multiple_models(self):
        """Test saving ensemble with multiple constituent models."""
        # Create sample data with multiple models
        selected_models = [
            {
                'model_id': 0,
                'features': ['feature_0', 'feature_1', 'feature_2'],
                'wmape': 0.25,
                'r2': 0.85,
                'equation_dict': {
                    'feature_0': 1.5,
                    'feature_1': -0.3,
                    'feature_2': 0.8,
                    'constant': 2.1
                }
            },
            {
                'model_id': 1,
                'features': ['feature_1', 'feature_3', 'feature_4'],
                'wmape': 0.30,
                'r2': 0.80,
                'equation_dict': {
                    'feature_1': 0.8,
                    'feature_3': 1.2,
                    'feature_4': -0.5,
                    'constant': -0.5
                }
            },
            {
                'model_id': 2,
                'features': ['feature_0', 'feature_2', 'feature_5'],
                'wmape': 0.28,
                'r2': 0.82,
                'equation_dict': {
                    'feature_0': 0.5,
                    'feature_2': 1.1,
                    'feature_5': -0.2,
                    'constant': 1.0
                }
            }
        ]
        
        ensemble_equation = {
            'model_0': 0.4,
            'model_1': 0.3,
            'model_2': 0.3,
            'constant': 0.5
        }
        
        method = 'elastic'
        
        serializer = ModelSerializer()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            # Save ensemble model
            serializer.save_ensemble_model(method, selected_models, ensemble_equation, temp_filepath)
            
            # Load and verify
            loaded_data = serializer.load_model_json(temp_filepath)
            
            # Verify all models are present
            assert len(loaded_data['constituent_models']) == 3
            assert loaded_data['metadata']['n_constituent_models'] == 3
            assert loaded_data['metadata']['n_features'] == 3  # From first model
            
            # Verify ensemble equation has all model coefficients
            assert 'model_0' in loaded_data['ensemble_equation']
            assert 'model_1' in loaded_data['ensemble_equation']
            assert 'model_2' in loaded_data['ensemble_equation']
            assert 'constant' in loaded_data['ensemble_equation']
            
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
