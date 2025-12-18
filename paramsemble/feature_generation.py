"""Feature combination generation module."""

import numpy as np
from scipy.special import comb


class FeatureCombinationGenerator:
    """Generates feature combinations for model training."""
    
    def generate_combinations(self, feature_names, m, f, sample='unique', random_state=None):
        """
        Generate m combinations of f features each.
        
        Parameters
        ----------
        feature_names : list
            List of available feature names
        m : int
            Number of combinations to generate
        f : int
            Number of features per combination
        sample : {'unique', 'replace'}
            Sampling method
        random_state : int, optional
            Random seed
            
        Returns
        -------
        combinations : list of lists
            List of feature combinations
        """
        # Input validation
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
        if not isinstance(feature_names, (list, tuple)):
            raise ValueError("feature_names must be a list or tuple")
        if not isinstance(m, int) or m < 1:
            raise ValueError(f"m must be a positive integer, got {m}")
        if not isinstance(f, int) or f < 1:
            raise ValueError(f"f must be a positive integer, got {f}")
        if sample not in ['unique', 'replace']:
            raise ValueError(f"sample must be 'unique' or 'replace', got '{sample}'")
        
        n_features = len(feature_names)
        
        if sample == 'unique' and f > n_features:
            raise ValueError(
                f"f ({f}) cannot be greater than number of features ({n_features}) "
                f"when sample='unique'"
            )
        
        # Calculate maximum possible combinations and cap m
        max_combinations = self.calculate_max_combinations(n_features, f, sample)
        actual_m = min(m, max_combinations)
        
        # Set random state
        rng = np.random.RandomState(random_state)
        
        # Generate combinations
        combinations = []
        if sample == 'unique':
            # Sample without replacement
            for _ in range(actual_m):
                combo = rng.choice(feature_names, size=f, replace=False).tolist()
                combinations.append(combo)
        else:  # sample == 'replace'
            # Sample with replacement
            for _ in range(actual_m):
                combo = rng.choice(feature_names, size=f, replace=True).tolist()
                combinations.append(combo)
        
        return combinations
    
    def calculate_max_combinations(self, n_features, f, sample):
        """
        Calculate maximum possible combinations.
        
        Parameters
        ----------
        n_features : int
            Total number of available features
        f : int
            Features per combination
        sample : str
            Sampling method
            
        Returns
        -------
        max_combinations : int
            Maximum possible combinations
        """
        # Validate inputs
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError(f"n_features must be a positive integer, got {n_features}")
        if not isinstance(f, int) or f < 1:
            raise ValueError(f"f must be a positive integer, got {f}")
        if sample not in ['unique', 'replace']:
            raise ValueError(f"sample must be 'unique' or 'replace', got '{sample}'")
        
        if sample == 'unique':
            if f > n_features:
                raise ValueError(
                    f"f ({f}) cannot be greater than n_features ({n_features}) "
                    f"when sample='unique'"
                )
            # For unique sampling, use combinations formula: C(n, f) = n! / (f! * (n-f)!)
            return int(comb(n_features, f, exact=True))
        else:  # sample == 'replace'
            # For sampling with replacement, theoretically infinite
            # Return a large number to indicate no practical limit
            return float('inf')
