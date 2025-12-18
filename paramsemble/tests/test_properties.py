"""Property-based tests for Paramsemble package."""

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from scipy.special import comb
from paramsemble.metrics import MetricsCalculator
from paramsemble.feature_generation import FeatureCombinationGenerator


# Feature: paramsemble-package, Property 4: Baseline metrics computation
# Validates: Requirements 2.2, 2.3
@given(
    y_true=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
    y_pred=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=100)
def test_property_4_baseline_metrics_computation(y_true, y_pred):
    """
    Property 4: Baseline metrics computation
    
    For any valid training and test datasets, after fitting the baseline model,
    both wMAPE and R2 metrics should be computed and stored as valid numeric values.
    
    This test verifies that:
    1. wMAPE can be computed for any valid input arrays
    2. R2 can be computed for any valid input arrays
    3. Both metrics return valid numeric values (not NaN, not infinite)
    """
    # Ensure arrays have the same length
    min_len = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    
    # Skip if all y_true values are zero (wMAPE would be undefined)
    if np.sum(np.abs(y_true)) == 0:
        return
    
    # Skip if y_true has zero variance (R2 would be undefined)
    if np.var(y_true) == 0:
        return
    
    # Compute metrics
    wmape = MetricsCalculator.compute_wmape(y_true, y_pred)
    r2 = MetricsCalculator.compute_r2(y_true, y_pred)
    
    # Verify both metrics are valid numeric values
    assert isinstance(wmape, (int, float, np.number)), "wMAPE should be a numeric value"
    assert isinstance(r2, (int, float, np.number)), "R2 should be a numeric value"
    
    # Verify metrics are not NaN
    assert not np.isnan(wmape), "wMAPE should not be NaN"
    assert not np.isnan(r2), "R2 should not be NaN"
    
    # Verify metrics are not infinite
    assert not np.isinf(wmape), "wMAPE should not be infinite"
    assert not np.isinf(r2), "R2 should not be infinite"
    
    # Verify wMAPE is non-negative (by definition)
    assert wmape >= 0, "wMAPE should be non-negative"



# Feature: paramsemble-package, Property 1: Feature set generation count and size
# Validates: Requirements 1.1, 1.4
@given(
    n_features=st.integers(min_value=1, max_value=20),
    m=st.integers(min_value=1, max_value=100),
    f=st.integers(min_value=1, max_value=20),
    sample=st.sampled_from(['unique', 'replace']),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100)
def test_property_1_feature_set_generation_count_and_size(n_features, m, f, sample, random_state):
    """
    Property 1: Feature set generation count and size
    
    For any valid feature list, m value, f value, and sample method,
    the system should generate exactly min(m, max_combinations) feature sets,
    and each feature set should contain exactly f features.
    
    This test verifies that:
    1. The number of generated combinations equals min(m, max_combinations)
    2. Each combination contains exactly f features
    """
    # Skip invalid cases where f > n_features for unique sampling
    if sample == 'unique' and f > n_features:
        return
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create generator
    generator = FeatureCombinationGenerator()
    
    # Calculate expected max combinations
    if sample == 'unique':
        max_combinations = int(comb(n_features, f, exact=True))
    else:
        max_combinations = float('inf')
    
    # Generate combinations
    combinations = generator.generate_combinations(
        feature_names, m, f, sample=sample, random_state=random_state
    )
    
    # Verify the number of combinations
    expected_count = min(m, max_combinations) if max_combinations != float('inf') else m
    assert len(combinations) == expected_count, \
        f"Expected {expected_count} combinations, got {len(combinations)}"
    
    # Verify each combination has exactly f features
    for i, combo in enumerate(combinations):
        assert len(combo) == f, \
            f"Combination {i} has {len(combo)} features, expected {f}"


# Feature: paramsemble-package, Property 2: Unique sampling constraint
# Validates: Requirements 1.2
@given(
    n_features=st.integers(min_value=2, max_value=20),
    m=st.integers(min_value=1, max_value=50),
    f=st.integers(min_value=2, max_value=20),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100)
def test_property_2_unique_sampling_constraint(n_features, m, f, random_state):
    """
    Property 2: Unique sampling constraint
    
    For any feature list and parameters with sample="unique",
    every generated feature set should contain no duplicate features.
    
    This test verifies that:
    1. When sample='unique', no feature appears more than once in any combination
    """
    # Skip invalid cases where f > n_features
    if f > n_features:
        return
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create generator
    generator = FeatureCombinationGenerator()
    
    # Generate combinations with unique sampling
    combinations = generator.generate_combinations(
        feature_names, m, f, sample='unique', random_state=random_state
    )
    
    # Verify each combination has no duplicates
    for i, combo in enumerate(combinations):
        unique_features = set(combo)
        assert len(unique_features) == len(combo), \
            f"Combination {i} has duplicate features: {combo}"


# Feature: paramsemble-package, Property 3: Maximum combinations calculation
# Validates: Requirements 1.5
@given(
    n_features=st.integers(min_value=1, max_value=20),
    f=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=100)
def test_property_3_maximum_combinations_calculation(n_features, f):
    """
    Property 3: Maximum combinations calculation
    
    For any number of features n and features per set f with sample="unique",
    the calculated maximum combinations should equal C(n, f) = n! / (f! * (n-f)!).
    
    This test verifies that:
    1. calculate_max_combinations returns C(n, f) for unique sampling
    """
    # Skip invalid cases where f > n_features
    if f > n_features:
        return
    
    # Create generator
    generator = FeatureCombinationGenerator()
    
    # Calculate max combinations
    max_combinations = generator.calculate_max_combinations(n_features, f, 'unique')
    
    # Calculate expected value using scipy.special.comb
    expected = int(comb(n_features, f, exact=True))
    
    # Verify the calculation
    assert max_combinations == expected, \
        f"Expected {expected} max combinations, got {max_combinations}"


# Feature: paramsemble-package, Property 5: Constituent model count matches feature combinations
# Validates: Requirements 3.1
@given(
    n_samples=st.integers(min_value=20, max_value=100),
    n_features=st.integers(min_value=2, max_value=10),
    n_combinations=st.integers(min_value=1, max_value=20),
    f=st.integers(min_value=1, max_value=5),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_5_constituent_model_count(n_samples, n_features, n_combinations, f, random_state):
    """
    Property 5: Constituent model count matches feature combinations
    
    For any set of generated feature combinations, the number of trained Lasso models
    should equal the number of feature combinations.
    
    This test verifies that:
    1. train_models returns exactly one model per feature combination
    """
    from paramsemble.constituent import ConstituentModelTrainer
    from paramsemble.feature_generation import FeatureCombinationGenerator
    import pandas as pd
    
    # Skip invalid cases where f > n_features
    if f > n_features:
        return
    
    # Generate random data
    np.random.seed(random_state)
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples)
    X_test = np.random.randn(n_samples // 2, n_features)
    y_test = np.random.randn(n_samples // 2)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Generate feature combinations
    generator = FeatureCombinationGenerator()
    combinations = generator.generate_combinations(
        feature_names, n_combinations, f, sample='unique', random_state=random_state
    )
    
    # Train models
    trainer = ConstituentModelTrainer()
    models_info = trainer.train_models(X_train_df, y_train, X_test_df, y_test, combinations)
    
    # Verify the number of models equals the number of combinations
    assert len(models_info) == len(combinations), \
        f"Expected {len(combinations)} models, got {len(models_info)}"


# Feature: paramsemble-package, Property 6: All constituent models have metrics
# Validates: Requirements 3.2
@given(
    n_samples=st.integers(min_value=20, max_value=100),
    n_features=st.integers(min_value=2, max_value=10),
    n_combinations=st.integers(min_value=1, max_value=20),
    f=st.integers(min_value=1, max_value=5),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_6_all_constituent_models_have_metrics(n_samples, n_features, n_combinations, f, random_state):
    """
    Property 6: All constituent models have metrics
    
    For any set of trained constituent models, every model should have valid wMAPE
    and R2 metric values computed from the test dataset.
    
    This test verifies that:
    1. Every model has a 'wmape' key with a valid numeric value
    2. Every model has an 'r2' key with a valid numeric value
    3. Both metrics are not NaN or infinite
    """
    from paramsemble.constituent import ConstituentModelTrainer
    from paramsemble.feature_generation import FeatureCombinationGenerator
    import pandas as pd
    
    # Skip invalid cases where f > n_features
    if f > n_features:
        return
    
    # Generate random data with non-zero variance
    np.random.seed(random_state)
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples) * 10 + 5  # Add scale and offset for non-zero values
    X_test = np.random.randn(n_samples // 2, n_features)
    y_test = np.random.randn(n_samples // 2) * 10 + 5
    
    # Skip if y_test has zero variance or all zeros
    if np.var(y_test) == 0 or np.sum(np.abs(y_test)) == 0:
        return
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Generate feature combinations
    generator = FeatureCombinationGenerator()
    combinations = generator.generate_combinations(
        feature_names, n_combinations, f, sample='unique', random_state=random_state
    )
    
    # Train models
    trainer = ConstituentModelTrainer()
    models_info = trainer.train_models(X_train_df, y_train, X_test_df, y_test, combinations)
    
    # Verify every model has valid metrics
    for i, model_info in enumerate(models_info):
        # Check wmape exists and is valid
        assert 'wmape' in model_info, f"Model {i} missing 'wmape' key"
        wmape = model_info['wmape']
        assert isinstance(wmape, (int, float, np.number)), \
            f"Model {i} wMAPE should be numeric, got {type(wmape)}"
        assert not np.isnan(wmape), f"Model {i} wMAPE is NaN"
        assert not np.isinf(wmape), f"Model {i} wMAPE is infinite"
        assert wmape >= 0, f"Model {i} wMAPE should be non-negative"
        
        # Check r2 exists and is valid
        assert 'r2' in model_info, f"Model {i} missing 'r2' key"
        r2 = model_info['r2']
        assert isinstance(r2, (int, float, np.number)), \
            f"Model {i} R2 should be numeric, got {type(r2)}"
        assert not np.isnan(r2), f"Model {i} R2 is NaN"
        assert not np.isinf(r2), f"Model {i} R2 is infinite"


# Feature: paramsemble-package, Property 7: Equation dictionary structure
# Validates: Requirements 3.3
@given(
    n_samples=st.integers(min_value=20, max_value=100),
    n_features=st.integers(min_value=2, max_value=10),
    n_combinations=st.integers(min_value=1, max_value=20),
    f=st.integers(min_value=1, max_value=5),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_7_equation_dictionary_structure(n_samples, n_features, n_combinations, f, random_state):
    """
    Property 7: Equation dictionary structure
    
    For any trained Lasso model with features F, the equation dictionary should contain
    exactly |F| + 1 keys: one for each feature name and one "constant" key for the intercept,
    with all values being numeric.
    
    This test verifies that:
    1. The equation dictionary has exactly |F| + 1 keys
    2. All feature names are present as keys
    3. The 'constant' key is present
    4. All values are numeric
    """
    from paramsemble.constituent import ConstituentModelTrainer
    from paramsemble.feature_generation import FeatureCombinationGenerator
    import pandas as pd
    
    # Skip invalid cases where f > n_features
    if f > n_features:
        return
    
    # Generate random data
    np.random.seed(random_state)
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples)
    X_test = np.random.randn(n_samples // 2, n_features)
    y_test = np.random.randn(n_samples // 2)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Generate feature combinations
    generator = FeatureCombinationGenerator()
    combinations = generator.generate_combinations(
        feature_names, n_combinations, f, sample='unique', random_state=random_state
    )
    
    # Train models
    trainer = ConstituentModelTrainer()
    models_info = trainer.train_models(X_train_df, y_train, X_test_df, y_test, combinations)
    
    # Verify equation dictionary structure for each model
    for i, model_info in enumerate(models_info):
        assert 'equation_dict' in model_info, f"Model {i} missing 'equation_dict' key"
        
        equation_dict = model_info['equation_dict']
        features = model_info['features']
        
        # Verify the number of keys: |F| + 1 (features + constant)
        expected_keys = len(features) + 1
        assert len(equation_dict) == expected_keys, \
            f"Model {i} equation_dict should have {expected_keys} keys, got {len(equation_dict)}"
        
        # Verify all feature names are present
        for feature in features:
            assert feature in equation_dict, \
                f"Model {i} equation_dict missing feature '{feature}'"
            
            # Verify the value is numeric
            value = equation_dict[feature]
            assert isinstance(value, (int, float, np.number)), \
                f"Model {i} feature '{feature}' coefficient should be numeric, got {type(value)}"
            assert not np.isnan(value), \
                f"Model {i} feature '{feature}' coefficient is NaN"
            assert not np.isinf(value), \
                f"Model {i} feature '{feature}' coefficient is infinite"
        
        # Verify 'constant' key is present
        assert 'constant' in equation_dict, \
            f"Model {i} equation_dict missing 'constant' key"
        
        # Verify constant value is numeric
        constant = equation_dict['constant']
        assert isinstance(constant, (int, float, np.number)), \
            f"Model {i} constant should be numeric, got {type(constant)}"
        assert not np.isnan(constant), \
            f"Model {i} constant is NaN"
        assert not np.isinf(constant), \
            f"Model {i} constant is infinite"



# Feature: paramsemble-package, Property 9: Model selection respects spread parameter
# Validates: Requirements 4.1, 4.3
@given(
    n_models=st.integers(min_value=5, max_value=50),
    spread=st.integers(min_value=1, max_value=30),
    baseline_wmape=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    baseline_r2=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_9_model_selection_respects_spread(n_models, spread, baseline_wmape, baseline_r2, random_state):
    """
    Property 9: Model selection respects spread parameter
    
    For any set of constituent models and spread value n, the number of selected models
    should be min(n, number_of_qualifying_models) where qualifying models outperform the baseline.
    
    This test verifies that:
    1. Only models that outperform baseline (lower wMAPE OR higher R2) are selected
    2. The number of selected models equals min(spread, number_of_qualifying_models)
    """
    from paramsemble.selection import ModelSelector
    
    np.random.seed(random_state)
    
    # Generate models with varying performance
    # Some will beat baseline, some won't
    models_info = []
    
    for i in range(n_models):
        # Generate wMAPE and R2 values around the baseline
        # Some models will be better, some worse
        wmape_offset = np.random.uniform(-0.5, 0.5)
        r2_offset = np.random.uniform(-0.3, 0.3)
        
        model_wmape = max(0.01, baseline_wmape + wmape_offset)
        model_r2 = np.clip(baseline_r2 + r2_offset, -1.0, 1.0)
        
        model_info = {
            'model_id': i,
            'features': [f'feature_{j}' for j in range(3)],
            'wmape': model_wmape,
            'r2': model_r2,
            'equation_dict': {'feature_0': 1.0, 'feature_1': 2.0, 'feature_2': 3.0, 'constant': 0.5}
        }
        models_info.append(model_info)
    
    baseline_metrics = {
        'wmape': baseline_wmape,
        'r2': baseline_r2
    }
    
    # Select top models
    selector = ModelSelector()
    selected_models = selector.select_top_models(models_info, baseline_metrics, spread)
    
    # Count qualifying models (those that beat baseline on wMAPE OR R2)
    qualifying_count = 0
    for model in models_info:
        if model['wmape'] < baseline_wmape or model['r2'] > baseline_r2:
            qualifying_count += 1
    
    # Verify the number of selected models
    expected_count = min(spread, qualifying_count)
    assert len(selected_models) == expected_count, \
        f"Expected {expected_count} selected models, got {len(selected_models)}"
    
    # Verify all selected models actually outperform baseline
    for model in selected_models:
        beats_on_wmape = model['wmape'] < baseline_wmape
        beats_on_r2 = model['r2'] > baseline_r2
        assert beats_on_wmape or beats_on_r2, \
            f"Selected model {model['model_id']} does not outperform baseline"


# Feature: paramsemble-package, Property 10: Model ranking order
# Validates: Requirements 4.2
@given(
    n_models=st.integers(min_value=5, max_value=30),
    baseline_wmape=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    baseline_r2=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
    spread=st.integers(min_value=1, max_value=20),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_10_model_ranking_order(n_models, baseline_wmape, baseline_r2, spread, random_state):
    """
    Property 10: Model ranking order
    
    For any list of constituent models, when ranked for selection, models with lower wMAPE
    should appear before models with higher wMAPE, and among models with equal wMAPE,
    models with higher R2 should appear first.
    
    This test verifies that:
    1. Selected models are sorted by wMAPE in ascending order (primary)
    2. For models with equal wMAPE, they are sorted by R2 in descending order (secondary)
    """
    from paramsemble.selection import ModelSelector
    
    np.random.seed(random_state)
    
    # Generate models with varying performance
    # Ensure some beat the baseline
    models_info = []
    
    for i in range(n_models):
        # Generate wMAPE values that are mostly better than baseline
        model_wmape = max(0.01, baseline_wmape * np.random.uniform(0.3, 1.2))
        model_r2 = np.clip(baseline_r2 * np.random.uniform(0.5, 1.5), -1.0, 1.0)
        
        model_info = {
            'model_id': i,
            'features': [f'feature_{j}' for j in range(3)],
            'wmape': model_wmape,
            'r2': model_r2,
            'equation_dict': {'feature_0': 1.0, 'feature_1': 2.0, 'feature_2': 3.0, 'constant': 0.5}
        }
        models_info.append(model_info)
    
    baseline_metrics = {
        'wmape': baseline_wmape,
        'r2': baseline_r2
    }
    
    # Select top models
    selector = ModelSelector()
    selected_models = selector.select_top_models(models_info, baseline_metrics, spread)
    
    # Skip if no models were selected
    if len(selected_models) == 0:
        return
    
    # Verify ranking order
    for i in range(len(selected_models) - 1):
        current_model = selected_models[i]
        next_model = selected_models[i + 1]
        
        current_wmape = current_model['wmape']
        next_wmape = next_model['wmape']
        current_r2 = current_model['r2']
        next_r2 = next_model['r2']
        
        # Primary sort: wMAPE ascending (current should be <= next)
        # Secondary sort: R2 descending (if wMAPE equal, current R2 should be >= next R2)
        if abs(current_wmape - next_wmape) < 1e-10:  # wMAPE values are equal
            # R2 should be in descending order
            assert current_r2 >= next_r2, \
                f"Models {i} and {i+1} have equal wMAPE but R2 not in descending order: " \
                f"{current_r2} < {next_r2}"
        else:
            # wMAPE should be in ascending order
            assert current_wmape <= next_wmape, \
                f"Models {i} and {i+1} not sorted by wMAPE: {current_wmape} > {next_wmape}"


# Feature: paramsemble-package, Property 11: Ensemble predictions for all test samples
# Validates: Requirements 5.3, 6.3
@given(
    n_samples=st.integers(min_value=10, max_value=100),
    n_models=st.integers(min_value=2, max_value=10),
    method=st.sampled_from(['elastic', 'mars']),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_11_ensemble_predictions_for_all_samples(n_samples, n_models, method, random_state):
    """
    Property 11: Ensemble predictions for all test samples
    
    For any fitted ensemble model (ElasticNet or MARS) and test dataset,
    the ensemble should generate exactly one prediction per test sample.
    
    This test verifies that:
    1. The ensemble model can be fitted successfully
    2. The ensemble generates predictions for all test samples
    3. The number of predictions equals the number of test samples
    4. All predictions are valid numeric values (not NaN, not infinite)
    """
    from paramsemble.ensemble import EnsembleBuilder
    
    # Skip MARS tests if py-earth is not installed
    if method == 'mars':
        try:
            from pyearth import Earth
        except ImportError:
            return
    
    np.random.seed(random_state)
    
    # Generate constituent predictions (simulating predictions from constituent models)
    # Shape: (n_samples, n_models)
    constituent_predictions = np.random.randn(n_samples, n_models) * 10 + 50
    
    # Generate true target values
    y_true = np.random.randn(n_samples) * 10 + 50
    
    # Build ensemble
    builder = EnsembleBuilder()
    ensemble_model, equation_dict = builder.build_ensemble(
        constituent_predictions, y_true, method=method
    )
    
    # Generate predictions using the fitted ensemble
    predictions = ensemble_model.predict(constituent_predictions)
    
    # Verify the number of predictions equals the number of samples
    assert len(predictions) == n_samples, \
        f"Expected {n_samples} predictions, got {len(predictions)}"
    
    # Verify all predictions are valid numeric values
    for i, pred in enumerate(predictions):
        assert isinstance(pred, (int, float, np.number)), \
            f"Prediction {i} should be numeric, got {type(pred)}"
        assert not np.isnan(pred), \
            f"Prediction {i} is NaN"
        assert not np.isinf(pred), \
            f"Prediction {i} is infinite"
    
    # Verify equation dictionary is returned and has correct structure
    assert isinstance(equation_dict, dict), \
        "equation_dict should be a dictionary"
    assert 'constant' in equation_dict, \
        "equation_dict should contain 'constant' key"
    
    # Verify equation dict has entries for each constituent model
    # For ElasticNet and MARS, we expect model_0, model_1, ..., model_{n_models-1}
    for i in range(n_models):
        model_key = f'model_{i}'
        assert model_key in equation_dict, \
            f"equation_dict should contain '{model_key}' key"
        
        # Verify the coefficient is numeric
        coef = equation_dict[model_key]
        assert isinstance(coef, (int, float, np.number)), \
            f"Coefficient for {model_key} should be numeric, got {type(coef)}"
        assert not np.isnan(coef), \
            f"Coefficient for {model_key} is NaN"
        assert not np.isinf(coef), \
            f"Coefficient for {model_key} is infinite"


# Feature: paramsemble-package, Property 8: Model serialization round-trip
# Validates: Requirements 3.4, 5.5, 6.5, 7.1
@given(
    n_models=st.integers(min_value=1, max_value=10),
    n_features=st.integers(min_value=2, max_value=8),
    method=st.sampled_from(['elastic', 'mars']),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_8_model_serialization_round_trip(n_models, n_features, method, random_state):
    """
    Property 8: Model serialization round-trip
    
    For any trained ensemble model, serializing to JSON and then deserializing should
    preserve the method name, all constituent model equation dictionaries, and the
    ensemble equation dictionary with equivalent numeric values.
    
    This test verifies that:
    1. The model can be serialized to JSON
    2. The JSON can be deserialized back to a dictionary
    3. The method name is preserved
    4. All constituent models are preserved with their equation dictionaries
    5. The ensemble equation dictionary is preserved
    6. All numeric values are equivalent (within floating point tolerance)
    """
    from paramsemble.serialization import ModelSerializer
    import tempfile
    import os
    
    np.random.seed(random_state)
    
    # Create mock selected models with equation dictionaries
    selected_models = []
    for i in range(n_models):
        # Generate random feature names for this model
        features = [f'feature_{j}' for j in range(n_features)]
        
        # Create equation dictionary with random coefficients
        equation_dict = {}
        for feature in features:
            equation_dict[feature] = float(np.random.randn())
        equation_dict['constant'] = float(np.random.randn())
        
        model_info = {
            'model_id': i,
            'features': features,
            'wmape': float(np.random.uniform(0.1, 1.0)),
            'r2': float(np.random.uniform(0.0, 0.9)),
            'equation_dict': equation_dict
        }
        selected_models.append(model_info)
    
    # Create ensemble equation dictionary
    ensemble_equation = {}
    for i in range(n_models):
        ensemble_equation[f'model_{i}'] = float(np.random.randn())
    ensemble_equation['constant'] = float(np.random.randn())
    
    # Create serializer
    serializer = ModelSerializer()
    
    # Create temporary file for serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filepath = f.name
    
    try:
        # Serialize the ensemble model
        serializer.save_ensemble_model(method, selected_models, ensemble_equation, temp_filepath)
        
        # Deserialize the model
        loaded_model_data = serializer.load_model_json(temp_filepath)
        
        # Verify method is preserved
        assert 'method' in loaded_model_data, "Loaded model should have 'method' key"
        assert loaded_model_data['method'] == method, \
            f"Method should be '{method}', got '{loaded_model_data['method']}'"
        
        # Verify constituent models are preserved
        assert 'constituent_models' in loaded_model_data, \
            "Loaded model should have 'constituent_models' key"
        loaded_constituent_models = loaded_model_data['constituent_models']
        assert len(loaded_constituent_models) == len(selected_models), \
            f"Expected {len(selected_models)} constituent models, got {len(loaded_constituent_models)}"
        
        # Verify each constituent model
        for i, (original, loaded) in enumerate(zip(selected_models, loaded_constituent_models)):
            # Check model_id
            assert loaded['model_id'] == original['model_id'], \
                f"Model {i} ID mismatch: expected {original['model_id']}, got {loaded['model_id']}"
            
            # Check features
            assert loaded['features'] == original['features'], \
                f"Model {i} features mismatch"
            
            # Check wmape (within tolerance)
            assert abs(loaded['wmape'] - original['wmape']) < 1e-9, \
                f"Model {i} wMAPE mismatch: expected {original['wmape']}, got {loaded['wmape']}"
            
            # Check r2 (within tolerance)
            assert abs(loaded['r2'] - original['r2']) < 1e-9, \
                f"Model {i} R2 mismatch: expected {original['r2']}, got {loaded['r2']}"
            
            # Check equation dictionary
            assert 'equation_dict' in loaded, \
                f"Model {i} should have 'equation_dict' key"
            loaded_eq = loaded['equation_dict']
            original_eq = original['equation_dict']
            
            # Verify same keys
            assert set(loaded_eq.keys()) == set(original_eq.keys()), \
                f"Model {i} equation_dict keys mismatch"
            
            # Verify all coefficients are preserved (within tolerance)
            for key in original_eq.keys():
                assert abs(loaded_eq[key] - original_eq[key]) < 1e-9, \
                    f"Model {i} equation_dict['{key}'] mismatch: " \
                    f"expected {original_eq[key]}, got {loaded_eq[key]}"
        
        # Verify ensemble equation is preserved
        assert 'ensemble_equation' in loaded_model_data, \
            "Loaded model should have 'ensemble_equation' key"
        loaded_ensemble_eq = loaded_model_data['ensemble_equation']
        
        # Verify same keys
        assert set(loaded_ensemble_eq.keys()) == set(ensemble_equation.keys()), \
            "Ensemble equation keys mismatch"
        
        # Verify all ensemble coefficients are preserved (within tolerance)
        for key in ensemble_equation.keys():
            assert abs(loaded_ensemble_eq[key] - ensemble_equation[key]) < 1e-9, \
                f"Ensemble equation['{key}'] mismatch: " \
                f"expected {ensemble_equation[key]}, got {loaded_ensemble_eq[key]}"
        
        # Verify metadata exists
        assert 'metadata' in loaded_model_data, \
            "Loaded model should have 'metadata' key"
        metadata = loaded_model_data['metadata']
        
        # Verify metadata fields
        assert 'n_features' in metadata, "Metadata should have 'n_features'"
        assert 'n_constituent_models' in metadata, "Metadata should have 'n_constituent_models'"
        assert 'training_date' in metadata, "Metadata should have 'training_date'"
        
        # Verify metadata values
        assert metadata['n_features'] == n_features, \
            f"Metadata n_features should be {n_features}, got {metadata['n_features']}"
        assert metadata['n_constituent_models'] == n_models, \
            f"Metadata n_constituent_models should be {n_models}, got {metadata['n_constituent_models']}"
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


# Feature: paramsemble-package, Property 13: Scoring applies constituent then ensemble equations
# Validates: Requirements 7.2, 7.3
@given(
    n_samples=st.integers(min_value=10, max_value=100),
    n_features=st.integers(min_value=2, max_value=8),
    n_models=st.integers(min_value=1, max_value=10),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_13_scoring_applies_constituent_then_ensemble(n_samples, n_features, n_models, random_state):
    """
    Property 13: Scoring applies constituent then ensemble equations
    
    For any scoring dataset and loaded model JSON, the scoring process should first
    apply each constituent equation dictionary to generate intermediate predictions,
    then apply the ensemble equation dictionary to those intermediate predictions
    to generate final predictions.
    
    This test verifies that:
    1. Constituent equations are applied first to generate intermediate predictions
    2. Ensemble equation is applied to intermediate predictions
    3. The final predictions match manual calculation of the two-step process
    """
    from paramsemble.scoring import ModelScorer
    
    np.random.seed(random_state)
    
    # Generate random scoring data
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Create mock constituent models with equation dictionaries
    constituent_models = []
    for i in range(n_models):
        # Create equation dictionary with random coefficients
        equation_dict = {}
        for feature in feature_names:
            equation_dict[feature] = float(np.random.randn())
        equation_dict['constant'] = float(np.random.randn())
        
        model_info = {
            'model_id': i,
            'features': feature_names,
            'wmape': float(np.random.uniform(0.1, 1.0)),
            'r2': float(np.random.uniform(0.0, 0.9)),
            'equation_dict': equation_dict
        }
        constituent_models.append(model_info)
    
    # Create ensemble equation dictionary
    ensemble_equation = {}
    for i in range(n_models):
        ensemble_equation[f'model_{i}'] = float(np.random.randn())
    ensemble_equation['constant'] = float(np.random.randn())
    
    # Create model data structure
    model_data = {
        'method': 'elastic',
        'constituent_models': constituent_models,
        'ensemble_equation': ensemble_equation,
        'metadata': {
            'n_features': n_features,
            'n_constituent_models': n_models,
            'training_date': '2024-01-01T00:00:00'
        }
    }
    
    # Create scorer
    scorer = ModelScorer()
    
    # Score the dataset
    predictions_df = scorer.score_dataset(X_df, model_data)
    
    # Manually calculate predictions using the two-step process
    # Step 1: Apply constituent equations
    manual_constituent_predictions = []
    for model_info in constituent_models:
        equation_dict = model_info['equation_dict']
        # Calculate: constant + sum(coef * feature)
        pred = np.full(n_samples, equation_dict['constant'])
        for feature in feature_names:
            pred += X_df[feature].values * equation_dict[feature]
        manual_constituent_predictions.append(pred)
    
    # Stack into array: shape (n_samples, n_models)
    manual_constituent_predictions = np.column_stack(manual_constituent_predictions)
    
    # Step 2: Apply ensemble equation to constituent predictions
    manual_final_predictions = np.full(n_samples, ensemble_equation['constant'])
    for i in range(n_models):
        model_key = f'model_{i}'
        manual_final_predictions += manual_constituent_predictions[:, i] * ensemble_equation[model_key]
    
    # Verify the predictions match (within numerical tolerance)
    actual_predictions = predictions_df['predicted'].values
    
    assert len(actual_predictions) == n_samples, \
        f"Expected {n_samples} predictions, got {len(actual_predictions)}"
    
    for i in range(n_samples):
        assert abs(actual_predictions[i] - manual_final_predictions[i]) < 1e-9, \
            f"Prediction {i} mismatch: expected {manual_final_predictions[i]}, " \
            f"got {actual_predictions[i]}"


# Feature: paramsemble-package, Property 12: Prediction output structure
# Validates: Requirements 5.4, 6.4, 7.4
@given(
    n_samples=st.integers(min_value=10, max_value=100),
    n_features=st.integers(min_value=2, max_value=8),
    n_models=st.integers(min_value=1, max_value=10),
    has_id_column=st.booleans(),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_12_prediction_output_structure(n_samples, n_features, n_models, has_id_column, random_state):
    """
    Property 12: Prediction output structure
    
    For any prediction operation (training or scoring), the output DataFrame should
    contain exactly two columns: one for identifiers and one named "predicted" for
    prediction values, with row count matching the input dataset.
    
    This test verifies that:
    1. The output is a DataFrame
    2. The DataFrame has exactly 2 columns
    3. One column is named 'predicted'
    4. The other column contains identifiers (either provided or auto-generated)
    5. The number of rows matches the input dataset
    6. All predictions are valid numeric values
    """
    from paramsemble.scoring import ModelScorer
    
    np.random.seed(random_state)
    
    # Generate random scoring data
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Create ID column if requested
    if has_id_column:
        id_column = [f'id_{i}' for i in range(n_samples)]
    else:
        id_column = None
    
    # Create mock constituent models with equation dictionaries
    constituent_models = []
    for i in range(n_models):
        # Create equation dictionary with random coefficients
        equation_dict = {}
        for feature in feature_names:
            equation_dict[feature] = float(np.random.randn())
        equation_dict['constant'] = float(np.random.randn())
        
        model_info = {
            'model_id': i,
            'features': feature_names,
            'wmape': float(np.random.uniform(0.1, 1.0)),
            'r2': float(np.random.uniform(0.0, 0.9)),
            'equation_dict': equation_dict
        }
        constituent_models.append(model_info)
    
    # Create ensemble equation dictionary
    ensemble_equation = {}
    for i in range(n_models):
        ensemble_equation[f'model_{i}'] = float(np.random.randn())
    ensemble_equation['constant'] = float(np.random.randn())
    
    # Create model data structure
    model_data = {
        'method': 'elastic',
        'constituent_models': constituent_models,
        'ensemble_equation': ensemble_equation,
        'metadata': {
            'n_features': n_features,
            'n_constituent_models': n_models,
            'training_date': '2024-01-01T00:00:00'
        }
    }
    
    # Create scorer
    scorer = ModelScorer()
    
    # Score the dataset
    predictions_df = scorer.score_dataset(X_df, model_data, id_column=id_column)
    
    # Verify output is a DataFrame
    assert isinstance(predictions_df, pd.DataFrame), \
        "Output should be a pandas DataFrame"
    
    # Verify DataFrame has exactly 2 columns
    assert len(predictions_df.columns) == 2, \
        f"Output DataFrame should have 2 columns, got {len(predictions_df.columns)}"
    
    # Verify one column is named 'predicted'
    assert 'predicted' in predictions_df.columns, \
        "Output DataFrame should have a 'predicted' column"
    
    # Verify the other column is for identifiers (either 'id' or similar)
    other_columns = [col for col in predictions_df.columns if col != 'predicted']
    assert len(other_columns) == 1, \
        "Output DataFrame should have exactly one ID column besides 'predicted'"
    
    # Verify the number of rows matches input
    assert len(predictions_df) == n_samples, \
        f"Output DataFrame should have {n_samples} rows, got {len(predictions_df)}"
    
    # Verify all predictions are valid numeric values
    predictions = predictions_df['predicted'].values
    for i, pred in enumerate(predictions):
        assert isinstance(pred, (int, float, np.number)), \
            f"Prediction {i} should be numeric, got {type(pred)}"
        assert not np.isnan(pred), \
            f"Prediction {i} is NaN"
        assert not np.isinf(pred), \
            f"Prediction {i} is infinite"
    
    # Verify IDs match if provided
    if has_id_column:
        id_col_name = other_columns[0]
        actual_ids = predictions_df[id_col_name].tolist()
        assert actual_ids == id_column, \
            "IDs in output should match provided id_column"



# Feature: paramsemble-package, Property 15: SQL export generates valid structure
# Validates: Requirements 9.1, 9.2, 9.3, 9.4
@given(
    n_models=st.integers(min_value=1, max_value=10),
    n_features=st.integers(min_value=2, max_value=8),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_15_sql_export_generates_valid_structure(n_models, n_features, random_state):
    """
    Property 15: SQL export generates valid structure
    
    For any valid modeljson file, the exported SQL code should contain one CTE per
    constituent model and one final SELECT statement that references all CTEs.
    
    This test verifies that:
    1. The SQL contains WITH clause
    2. The SQL contains exactly n_models + 1 CTEs (constituent models + ensemble_inputs)
    3. The SQL contains a final SELECT statement
    4. Each constituent model has a corresponding CTE
    5. The final SELECT references the ensemble_inputs CTE
    """
    from paramsemble.sql_export import SQLExporter
    from paramsemble.serialization import ModelSerializer
    import tempfile
    import os
    
    np.random.seed(random_state)
    
    # Create mock constituent models with equation dictionaries
    constituent_models = []
    for i in range(n_models):
        # Generate random feature names for this model
        features = [f'feature_{j}' for j in range(n_features)]
        
        # Create equation dictionary with random coefficients
        equation_dict = {}
        for feature in features:
            equation_dict[feature] = float(np.random.randn())
        equation_dict['constant'] = float(np.random.randn())
        
        model_info = {
            'model_id': i,
            'features': features,
            'wmape': float(np.random.uniform(0.1, 1.0)),
            'r2': float(np.random.uniform(0.0, 0.9)),
            'equation_dict': equation_dict
        }
        constituent_models.append(model_info)
    
    # Create ensemble equation dictionary
    ensemble_equation = {}
    for i in range(n_models):
        ensemble_equation[f'model_{i}'] = float(np.random.randn())
    ensemble_equation['constant'] = float(np.random.randn())
    
    # Create serializer and save model
    serializer = ModelSerializer()
    
    # Create temporary file for model JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_modeljson = f.name
    
    try:
        # Save the ensemble model
        serializer.save_ensemble_model('elastic', constituent_models, ensemble_equation, temp_modeljson)
        
        # Create SQL exporter
        exporter = SQLExporter()
        
        # Export to SQL
        sql_code = exporter.export_to_sql(temp_modeljson)
        
        # Verify SQL contains WITH clause
        assert 'WITH' in sql_code, "SQL should contain WITH clause"
        
        # Verify SQL contains CTEs for each constituent model
        for i in range(n_models):
            cte_name = f'model_{i}'
            assert cte_name in sql_code, f"SQL should contain CTE '{cte_name}'"
        
        # Verify SQL contains ensemble_inputs CTE
        assert 'ensemble_inputs' in sql_code, "SQL should contain 'ensemble_inputs' CTE"
        
        # Verify SQL contains final SELECT statement
        # The final SELECT should come after all CTEs
        assert 'SELECT' in sql_code, "SQL should contain SELECT statement"
        
        # Count the number of CTEs by counting " AS (" patterns
        # Each CTE has the pattern "cte_name AS ("
        cte_count = sql_code.count(' AS (')
        expected_cte_count = n_models + 1  # constituent models + ensemble_inputs
        assert cte_count == expected_cte_count, \
            f"SQL should contain {expected_cte_count} CTEs, found {cte_count}"
        
        # Verify the final SELECT references ensemble_inputs
        # Find the last SELECT statement
        last_select_pos = sql_code.rfind('SELECT')
        final_select_part = sql_code[last_select_pos:]
        assert 'FROM ensemble_inputs' in final_select_part, \
            "Final SELECT should reference 'ensemble_inputs'"
        
        # Verify the final SELECT has 'predicted' column
        assert 'AS predicted' in final_select_part, \
            "Final SELECT should have 'predicted' column"
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_modeljson):
            os.remove(temp_modeljson)



# Feature: paramsemble-package, Property 16: SQL constituent CTE structure
# Validates: Requirements 9.3
@given(
    n_models=st.integers(min_value=1, max_value=10),
    n_features=st.integers(min_value=2, max_value=8),
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_16_sql_constituent_cte_structure(n_models, n_features, random_state):
    """
    Property 16: SQL constituent CTE structure
    
    For any constituent model equation dictionary with features F, the generated CTE
    should contain exactly |F| + 1 terms in its calculation: one multiplication per
    feature and one constant term.
    
    This test verifies that:
    1. Each constituent CTE contains the constant term
    2. Each constituent CTE contains a multiplication term for each feature
    3. The total number of terms equals |F| + 1
    """
    from paramsemble.sql_export import SQLExporter
    from paramsemble.serialization import ModelSerializer
    import tempfile
    import os
    import re
    
    np.random.seed(random_state)
    
    # Create mock constituent models with equation dictionaries
    constituent_models = []
    for i in range(n_models):
        # Generate random feature names for this model
        features = [f'feature_{j}' for j in range(n_features)]
        
        # Create equation dictionary with random coefficients
        equation_dict = {}
        for feature in features:
            equation_dict[feature] = float(np.random.randn())
        equation_dict['constant'] = float(np.random.randn())
        
        model_info = {
            'model_id': i,
            'features': features,
            'wmape': float(np.random.uniform(0.1, 1.0)),
            'r2': float(np.random.uniform(0.0, 0.9)),
            'equation_dict': equation_dict
        }
        constituent_models.append(model_info)
    
    # Create ensemble equation dictionary
    ensemble_equation = {}
    for i in range(n_models):
        ensemble_equation[f'model_{i}'] = float(np.random.randn())
    ensemble_equation['constant'] = float(np.random.randn())
    
    # Create serializer and save model
    serializer = ModelSerializer()
    
    # Create temporary file for model JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_modeljson = f.name
    
    try:
        # Save the ensemble model
        serializer.save_ensemble_model('elastic', constituent_models, ensemble_equation, temp_modeljson)
        
        # Create SQL exporter
        exporter = SQLExporter()
        
        # Export to SQL
        sql_code = exporter.export_to_sql(temp_modeljson)
        
        # For each constituent model, verify the CTE structure
        for i, model_info in enumerate(constituent_models):
            cte_name = f'model_{i}'
            features = model_info['features']
            equation_dict = model_info['equation_dict']
            
            # Extract the CTE for this model
            # Find the CTE definition - look for the pattern from cte_name to the closing )
            # We need to find the matching closing parenthesis
            cte_start = sql_code.find(f'{cte_name} AS (')
            assert cte_start != -1, f"Could not find CTE '{cte_name}' in SQL"
            
            # Find the matching closing parenthesis
            # Start after "AS ("
            paren_start = sql_code.find('(', cte_start)
            paren_count = 1
            pos = paren_start + 1
            while paren_count > 0 and pos < len(sql_code):
                if sql_code[pos] == '(':
                    paren_count += 1
                elif sql_code[pos] == ')':
                    paren_count -= 1
                pos += 1
            
            cte_content = sql_code[paren_start:pos]
            
            # Verify the constant term is present
            constant_value = equation_dict['constant']
            # The constant should appear in the calculation
            assert str(constant_value) in cte_content, \
                f"CTE '{cte_name}' should contain constant term {constant_value}"
            
            # Verify each feature has a multiplication term
            for feature in features:
                coef = equation_dict[feature]
                # Look for pattern: (coefficient * feature_name)
                # The coefficient and feature should both appear
                assert str(coef) in cte_content, \
                    f"CTE '{cte_name}' should contain coefficient {coef} for feature '{feature}'"
                assert feature in cte_content, \
                    f"CTE '{cte_name}' should contain feature '{feature}'"
            
            # Count the number of terms by counting '+' operators
            # The calculation should have the form: term1 + term2 + ... + termN
            # So the number of '+' operators should be |F| (one less than number of terms)
            plus_count = cte_content.count(' + ')
            expected_plus_count = n_features  # |F| features + 1 constant = |F| + 1 terms, so |F| plus signs
            
            assert plus_count == expected_plus_count, \
                f"CTE '{cte_name}' should have {expected_plus_count} '+' operators " \
                f"(for {n_features} features + 1 constant), found {plus_count}"
            
            # Count multiplication operators to verify one per feature
            mult_count = cte_content.count(' * ')
            expected_mult_count = n_features  # One multiplication per feature
            
            assert mult_count == expected_mult_count, \
                f"CTE '{cte_name}' should have {expected_mult_count} '*' operators " \
                f"(one per feature), found {mult_count}"
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_modeljson):
            os.remove(temp_modeljson)



# Feature: paramsemble-package, Property 14: Fitted attributes have trailing underscores
# Validates: Requirements 8.4
@given(
    n_samples=st.integers(min_value=30, max_value=100),
    n_features=st.integers(min_value=3, max_value=10),
    m=st.integers(min_value=5, max_value=20),
    f=st.integers(min_value=2, max_value=5),
    spread=st.integers(min_value=2, max_value=10),
    method=st.sampled_from(['elastic']),  # Only elastic to avoid py-earth dependency
    random_state=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_property_14_fitted_attributes_have_trailing_underscores(
    n_samples, n_features, m, f, spread, method, random_state
):
    """
    Property 14: Fitted attributes have trailing underscores
    
    For any fitted ParamsembleRegressor instance, all attributes storing fitted model state
    should have names ending with an underscore character.
    
    This test verifies that:
    1. After fitting, the estimator has fitted attributes
    2. All fitted attributes end with an underscore (following sklearn convention)
    3. No public attributes without underscores are added during fitting
    """
    from paramsemble.estimator import ParamsembleRegressor
    
    # Skip invalid cases where f > n_features
    if f > n_features:
        return
    
    np.random.seed(random_state)
    
    # Generate random data with non-zero variance
    X_train = np.random.randn(n_samples, n_features) * 10 + 5
    y_train = np.random.randn(n_samples) * 10 + 50
    X_test = np.random.randn(n_samples // 2, n_features) * 10 + 5
    y_test = np.random.randn(n_samples // 2) * 10 + 50
    
    # Skip if y_test has zero variance or all zeros
    if np.var(y_test) == 0 or np.sum(np.abs(y_test)) == 0:
        return
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Create estimator
    estimator = ParamsembleRegressor(
        m=m,
        f=f,
        sample='unique',
        method=method,
        spread=spread,
        random_state=random_state
    )
    
    # Get attributes before fitting (these are constructor parameters)
    attrs_before = set(dir(estimator))
    
    # Fit the estimator
    try:
        estimator.fit(X_train_df, y_train, X_test_df, y_test)
    except ValueError as e:
        # If no models outperform baseline, skip this test case
        if "No models outperformed the baseline" in str(e):
            return
        raise
    
    # Get attributes after fitting
    attrs_after = set(dir(estimator))
    
    # Find new attributes added during fitting
    new_attrs = attrs_after - attrs_before
    
    # Filter to only public attributes (not starting with _)
    # and not methods (not callable)
    fitted_attrs = []
    for attr in new_attrs:
        if not attr.startswith('_'):  # Public attribute
            attr_value = getattr(estimator, attr)
            if not callable(attr_value):  # Not a method
                fitted_attrs.append(attr)
    
    # Verify that all fitted attributes end with underscore
    for attr in fitted_attrs:
        assert attr.endswith('_'), \
            f"Fitted attribute '{attr}' should end with underscore (sklearn convention)"
    
    # Verify that key fitted attributes exist and end with underscore
    expected_fitted_attrs = [
        'feature_names_',
        'n_features_in_',
        'feature_combinations_',
        'baseline_metrics_',
        'constituent_models_',
        'selected_models_',
        'ensemble_model_',
        'ensemble_equation_',
        'id_column_'
    ]
    
    for expected_attr in expected_fitted_attrs:
        assert hasattr(estimator, expected_attr), \
            f"Fitted estimator should have attribute '{expected_attr}'"
        assert expected_attr.endswith('_'), \
            f"Fitted attribute '{expected_attr}' should end with underscore"
