"""Tests for model selector."""

import pytest
from paramsemble.selection import ModelSelector


class TestModelSelector:
    """Unit tests for ModelSelector class."""
    
    def test_select_top_models_basic(self):
        """Test basic model selection functionality."""
        models_info = [
            {'model_id': 0, 'wmape': 0.5, 'r2': 0.7, 'features': ['f1', 'f2']},
            {'model_id': 1, 'wmape': 0.3, 'r2': 0.8, 'features': ['f1', 'f3']},
            {'model_id': 2, 'wmape': 0.4, 'r2': 0.75, 'features': ['f2', 'f3']},
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 2
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # All models beat baseline, so we should get top 2
        assert len(selected) == 2
        # Should be sorted by wMAPE ascending
        assert selected[0]['wmape'] == 0.3
        assert selected[1]['wmape'] == 0.4
    
    def test_select_top_models_filters_poor_performers(self):
        """Test that models not beating baseline are filtered out."""
        models_info = [
            {'model_id': 0, 'wmape': 0.7, 'r2': 0.5, 'features': ['f1', 'f2']},  # Worse on both
            {'model_id': 1, 'wmape': 0.3, 'r2': 0.8, 'features': ['f1', 'f3']},  # Better on both
            {'model_id': 2, 'wmape': 0.8, 'r2': 0.4, 'features': ['f2', 'f3']},  # Worse on both
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 5
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # Only 1 model beats baseline
        assert len(selected) == 1
        assert selected[0]['model_id'] == 1
    
    def test_select_top_models_beats_on_wmape_only(self):
        """Test that models beating baseline on wMAPE only are selected."""
        models_info = [
            {'model_id': 0, 'wmape': 0.5, 'r2': 0.5, 'features': ['f1', 'f2']},  # Better wMAPE, worse R2
            {'model_id': 1, 'wmape': 0.7, 'r2': 0.8, 'features': ['f1', 'f3']},  # Worse wMAPE, better R2
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 5
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # Both models should be selected (each beats on one metric)
        assert len(selected) == 2
    
    def test_select_top_models_beats_on_r2_only(self):
        """Test that models beating baseline on R2 only are selected."""
        models_info = [
            {'model_id': 0, 'wmape': 0.7, 'r2': 0.8, 'features': ['f1', 'f2']},  # Worse wMAPE, better R2
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 5
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # Model beats on R2, should be selected
        assert len(selected) == 1
        assert selected[0]['model_id'] == 0
    
    def test_select_top_models_ranking_order(self):
        """Test that models are ranked by wMAPE ascending, then R2 descending."""
        models_info = [
            {'model_id': 0, 'wmape': 0.4, 'r2': 0.7, 'features': ['f1', 'f2']},
            {'model_id': 1, 'wmape': 0.3, 'r2': 0.8, 'features': ['f1', 'f3']},
            {'model_id': 2, 'wmape': 0.4, 'r2': 0.9, 'features': ['f2', 'f3']},  # Same wMAPE as 0, higher R2
            {'model_id': 3, 'wmape': 0.5, 'r2': 0.85, 'features': ['f1', 'f4']},
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 10
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # All models beat baseline
        assert len(selected) == 4
        
        # Check ordering
        assert selected[0]['model_id'] == 1  # wMAPE 0.3
        assert selected[1]['model_id'] == 2  # wMAPE 0.4, R2 0.9
        assert selected[2]['model_id'] == 0  # wMAPE 0.4, R2 0.7
        assert selected[3]['model_id'] == 3  # wMAPE 0.5
    
    def test_select_top_models_respects_spread(self):
        """Test that spread parameter limits number of selected models."""
        models_info = [
            {'model_id': i, 'wmape': 0.1 + i * 0.05, 'r2': 0.9 - i * 0.05, 'features': [f'f{i}']}
            for i in range(10)
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 3
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # Should only get top 3 models
        assert len(selected) == 3
    
    def test_select_top_models_empty_input(self):
        """Test with empty models list raises error."""
        models_info = []
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 5
        
        selector = ModelSelector()
        with pytest.raises(ValueError, match="models_info cannot be empty"):
            selector.select_top_models(models_info, baseline_metrics, spread)
    
    def test_select_top_models_no_qualifying_models(self):
        """Test when no models beat baseline."""
        models_info = [
            {'model_id': 0, 'wmape': 0.7, 'r2': 0.5, 'features': ['f1', 'f2']},
            {'model_id': 1, 'wmape': 0.8, 'r2': 0.4, 'features': ['f1', 'f3']},
        ]
        
        baseline_metrics = {'wmape': 0.6, 'r2': 0.6}
        spread = 5
        
        selector = ModelSelector()
        selected = selector.select_top_models(models_info, baseline_metrics, spread)
        
        # No models beat baseline
        assert len(selected) == 0
