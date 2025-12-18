"""Model selector component."""


class ModelSelector:
    """Selects best-performing constituent models."""
    
    def select_top_models(self, models_info, baseline_metrics, spread):
        """
        Select top n models based on performance.
        
        Parameters
        ----------
        models_info : list of dict
            Model information including metrics
        baseline_metrics : dict
            Baseline wMAPE and R2
        spread : int
            Number of models to select
            
        Returns
        -------
        selected_models : list of dict
            Top-performing models
        """
        # Validate inputs
        if not models_info:
            raise ValueError("models_info cannot be empty")
        if not isinstance(models_info, (list, tuple)):
            raise ValueError("models_info must be a list or tuple")
        if not baseline_metrics:
            raise ValueError("baseline_metrics cannot be empty")
        if not isinstance(baseline_metrics, dict):
            raise ValueError("baseline_metrics must be a dictionary")
        if 'wmape' not in baseline_metrics:
            raise ValueError("baseline_metrics missing 'wmape' key")
        if 'r2' not in baseline_metrics:
            raise ValueError("baseline_metrics missing 'r2' key")
        if not isinstance(spread, int) or spread < 1:
            raise ValueError(f"spread must be a positive integer, got {spread}")
        
        # Filter models that outperform baseline
        # A model outperforms baseline if it has lower wMAPE OR higher R2
        qualifying_models = []
        
        baseline_wmape = baseline_metrics['wmape']
        baseline_r2 = baseline_metrics['r2']
        
        for model_info in models_info:
            model_wmape = model_info['wmape']
            model_r2 = model_info['r2']
            
            # Model qualifies if it beats baseline on wMAPE or R2
            if model_wmape < baseline_wmape or model_r2 > baseline_r2:
                qualifying_models.append(model_info)
        
        # Rank qualifying models: sort by wMAPE ascending, then R2 descending
        # Primary sort: wMAPE ascending (lower is better)
        # Secondary sort: R2 descending (higher is better)
        ranked_models = sorted(
            qualifying_models,
            key=lambda m: (m['wmape'], -m['r2'])
        )
        
        # Select top n models based on spread parameter
        n_to_select = min(spread, len(ranked_models))
        selected_models = ranked_models[:n_to_select]
        
        return selected_models
