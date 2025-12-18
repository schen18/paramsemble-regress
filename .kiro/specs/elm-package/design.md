# ELM Package Design Document

## Overview

The ELM (Ensemble Linear Models) package is a Python library that implements an automated ensemble learning approach for regression tasks. The system generates multiple feature combinations, trains Lasso regression models on each combination, evaluates performance against a Random Forest baseline, and creates ensemble predictions using either ElasticNet or MARS methods.

The package follows Scikit-Learn design patterns, providing familiar `fit()` and `predict()` interfaces. It supports model serialization for deployment and includes comprehensive metrics tracking throughout the training process.

## Architecture

### High-Level Architecture

The ELM system consists of four main phases:

1. **Feature Combination Generation**: Creates diverse feature subsets based on sampling strategy
2. **Baseline Establishment**: Trains Random Forest on all features to set performance benchmarks
3. **Constituent Model Training**: Fits multiple Lasso models on different feature combinations
4. **Ensemble Creation**: Combines top-performing models using ElasticNet or MARS

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         ELM Estimator                        │
│  (Main Scikit-Learn Compatible Interface)                   │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├──► FeatureCombinationGenerator
                │    - generate_combinations()
                │    - validate_parameters()
                │
                ├──► BaselineModel
                │    - fit_baseline()
                │    - evaluate_baseline()
                │
                ├──► ConstituentModelTrainer
                │    - train_lasso_models()
                │    - evaluate_models()
                │    - extract_coefficients()
                │
                ├──► ModelSelector
                │    - select_top_models()
                │    - rank_by_performance()
                │
                ├──► EnsembleBuilder
                │    - build_elasticnet_ensemble()
                │    - build_mars_ensemble()
                │    - generate_predictions()
                │
                ├──► MetricsCalculator
                │    - compute_wmape()
                │    - compute_r2()
                │
                ├──► ModelSerializer
                │    - save_constituent_models()
                │    - save_ensemble_model()
                │    - load_model_json()
                │
                └──► ModelScorer
                     - score_with_saved_model()
                     - apply_equation_dictionary()
```

## Components and Interfaces

### 1. ELMRegressor (Main Estimator)

The primary user-facing class following Scikit-Learn conventions.

```python
class ELMRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble Linear Models regressor.
    
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
```

### 2. FeatureCombinationGenerator

Generates feature combinations based on sampling strategy.

```python
class FeatureCombinationGenerator:
    """
    Generates feature combinations for model training.
    """
    
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
```

### 3. BaselineModel

Establishes performance baseline using Random Forest.

```python
class BaselineModel:
    """
    Trains and evaluates baseline Random Forest model.
    """
    
    def fit_and_evaluate(self, X_train, y_train, X_test, y_test, random_state=None):
        """
        Fit baseline model and compute metrics.
        
        Returns
        -------
        baseline_metrics : dict
            Dictionary with 'wmape' and 'r2' keys
        """
```

### 4. ConstituentModelTrainer

Trains Lasso models on feature combinations.

```python
class ConstituentModelTrainer:
    """
    Trains multiple Lasso regression models.
    """
    
    def train_models(self, X_train, y_train, X_test, y_test, feature_combinations):
        """
        Train Lasso models for each feature combination.
        
        Returns
        -------
        models_info : list of dict
            List containing model metrics and coefficients
        """
        
    def extract_equation_dict(self, model, feature_names):
        """
        Extract coefficients as equation dictionary.
        
        Returns
        -------
        equation_dict : dict
            Feature names as keys, coefficients as values, plus 'constant'
        """
```

### 5. ModelSelector

Selects top-performing models for ensemble.

```python
class ModelSelector:
    """
    Selects best-performing constituent models.
    """
    
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
```

### 6. EnsembleBuilder

Creates ensemble models using ElasticNet or MARS.

```python
class EnsembleBuilder:
    """
    Builds ensemble models from constituent predictions.
    """
    
    def build_ensemble(self, constituent_predictions, y_true, method='elastic'):
        """
        Build ensemble model.
        
        Parameters
        ----------
        constituent_predictions : array-like of shape (n_samples, n_models)
            Predictions from constituent models
        y_true : array-like
            True target values
        method : {'elastic', 'mars'}
            Ensemble method
            
        Returns
        -------
        ensemble_model : object
            Fitted ensemble model
        equation_dict : dict
            Ensemble model coefficients
        """
```

### 7. MetricsCalculator

Computes performance metrics.

```python
class MetricsCalculator:
    """
    Calculates regression performance metrics.
    """
    
    @staticmethod
    def compute_wmape(y_true, y_pred):
        """
        Compute weighted Mean Absolute Percentage Error.
        
        wMAPE = sum(|y_true - y_pred|) / sum(|y_true|)
        
        Returns
        -------
        wmape : float
            Weighted MAPE value
        """
        
    @staticmethod
    def compute_r2(y_true, y_pred):
        """
        Compute R-squared coefficient.
        
        Returns
        -------
        r2 : float
            R-squared value
        """
```

### 8. ModelSerializer

Handles model persistence and loading.

```python
class ModelSerializer:
    """
    Serializes and deserializes models to/from JSON.
    """
    
    def save_constituent_models(self, models_info, filepath):
        """
        Save constituent model details to JSON.
        """
        
    def save_ensemble_model(self, method, selected_models, ensemble_equation, filepath):
        """
        Save complete ensemble model to JSON.
        """
        
    def load_model_json(self, filepath):
        """
        Load model from JSON file.
        
        Returns
        -------
        model_data : dict
            Model configuration and coefficients
        """
```

### 9. ModelScorer

Scores new data using saved models.

```python
class ModelScorer:
    """
    Scores new datasets using saved model JSON.
    """
    
    def score_dataset(self, X, model_data, id_column=None):
        """
        Generate predictions using saved model.
        
        Returns
        -------
        predictions_df : DataFrame
            IDs and predictions
        """
        
    def apply_equation_dict(self, X, equation_dict):
        """
        Apply equation dictionary to generate predictions.
        
        Returns
        -------
        predictions : ndarray
            Predicted values
        """
```

### 10. SQLExporter

Converts model JSON to executable SQL code.

```python
class SQLExporter:
    """
    Exports ensemble models as SQL code for database deployment.
    """
    
    def export_to_sql(self, modeljson_path, table_name='input_data', id_column='id'):
        """
        Generate SQL code from saved model JSON.
        
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
        
    def create_constituent_cte(self, model_info, cte_name, table_name):
        """
        Create a CTE for a constituent Lasso model.
        
        Parameters
        ----------
        model_info : dict
            Model information including features and equation_dict
        cte_name : str
            Name for the CTE
        table_name : str
            Source table name
            
        Returns
        -------
        cte_sql : str
            SQL CTE definition
        """
        
    def create_ensemble_select(self, ensemble_equation, constituent_ctes, id_column):
        """
        Create final SELECT statement applying ensemble equation.
        
        Parameters
        ----------
        ensemble_equation : dict
            Ensemble model coefficients
        constituent_ctes : list of str
            Names of constituent CTEs
        id_column : str
            ID column name
            
        Returns
        -------
        select_sql : str
            Final SELECT statement
        """
```

## Data Models

### Feature Combination

```python
{
    "combination_id": int,
    "features": List[str]
}
```

### Model Information

```python
{
    "model_id": int,
    "features": List[str],
    "wmape": float,
    "r2": float,
    "equation_dict": {
        "feature_1": float,
        "feature_2": float,
        ...
        "constant": float
    },
    "model_object": sklearn.linear_model.Lasso  # Not serialized
}
```

### Baseline Metrics

```python
{
    "wmape": float,
    "r2": float
}
```

### Ensemble Model JSON Structure

```python
{
    "method": str,  # 'elastic' or 'mars'
    "constituent_models": [
        {
            "model_id": int,
            "features": List[str],
            "wmape": float,
            "r2": float,
            "equation_dict": Dict[str, float]
        },
        ...
    ],
    "ensemble_equation": Dict[str, float],
    "metadata": {
        "n_features": int,
        "n_constituent_models": int,
        "training_date": str
    }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Feature set generation count and size

*For any* valid feature list, m value, f value, and sample method, the system should generate exactly min(m, max_combinations) feature sets, and each feature set should contain exactly f features.

**Validates: Requirements 1.1, 1.4**

### Property 2: Unique sampling constraint

*For any* feature list and parameters with sample="unique", every generated feature set should contain no duplicate features.

**Validates: Requirements 1.2**

### Property 3: Maximum combinations calculation

*For any* number of features n and features per set f with sample="unique", the calculated maximum combinations should equal C(n, f) = n! / (f! * (n-f)!).

**Validates: Requirements 1.5**

### Property 4: Baseline metrics computation

*For any* valid training and test datasets, after fitting the baseline model, both wMAPE and R2 metrics should be computed and stored as valid numeric values.

**Validates: Requirements 2.2, 2.3**

### Property 5: Constituent model count matches feature combinations

*For any* set of generated feature combinations, the number of trained Lasso models should equal the number of feature combinations.

**Validates: Requirements 3.1**

### Property 6: All constituent models have metrics

*For any* set of trained constituent models, every model should have valid wMAPE and R2 metric values computed from the test dataset.

**Validates: Requirements 3.2**

### Property 7: Equation dictionary structure

*For any* trained Lasso model with features F, the equation dictionary should contain exactly |F| + 1 keys: one for each feature name and one "constant" key for the intercept, with all values being numeric.

**Validates: Requirements 3.3**

### Property 8: Model serialization round-trip

*For any* trained ensemble model, serializing to JSON and then deserializing should preserve the method name, all constituent model equation dictionaries, and the ensemble equation dictionary with equivalent numeric values.

**Validates: Requirements 3.4, 5.5, 6.5, 7.1**

### Property 9: Model selection respects spread parameter

*For any* set of constituent models and spread value n, the number of selected models should be min(n, number_of_qualifying_models) where qualifying models outperform the baseline.

**Validates: Requirements 4.1, 4.3**

### Property 10: Model ranking order

*For any* list of constituent models, when ranked for selection, models with lower wMAPE should appear before models with higher wMAPE, and among models with equal wMAPE, models with higher R2 should appear first.

**Validates: Requirements 4.2**

### Property 11: Ensemble predictions for all test samples

*For any* fitted ensemble model (ElasticNet or MARS) and test dataset, the ensemble should generate exactly one prediction per test sample.

**Validates: Requirements 5.3, 6.3**

### Property 12: Prediction output structure

*For any* prediction operation (training or scoring), the output DataFrame should contain exactly two columns: one for identifiers and one named "predicted" for prediction values, with row count matching the input dataset.

**Validates: Requirements 5.4, 6.4, 7.4**

### Property 13: Scoring applies constituent then ensemble equations

*For any* scoring dataset and loaded model JSON, the scoring process should first apply each constituent equation dictionary to generate intermediate predictions, then apply the ensemble equation dictionary to those intermediate predictions to generate final predictions.

**Validates: Requirements 7.2, 7.3**

### Property 14: Fitted attributes have trailing underscores

*For any* fitted ELMRegressor instance, all attributes storing fitted model state should have names ending with an underscore character.

**Validates: Requirements 8.4**

### Property 15: SQL export generates valid structure

*For any* valid modeljson file, the exported SQL code should contain one CTE per constituent model and one final SELECT statement that references all CTEs.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

### Property 16: SQL constituent CTE structure

*For any* constituent model equation dictionary with features F, the generated CTE should contain exactly |F| + 1 terms in its calculation: one multiplication per feature and one constant term.

**Validates: Requirements 9.3**

## Error Handling

### Input Validation Errors

1. **Invalid Feature Count**: Raise `ValueError` when f > number of available features with sample="unique"
2. **Invalid Sample Method**: Raise `ValueError` when sample parameter is not "unique" or "replace"
3. **Invalid Ensemble Method**: Raise `ValueError` when method parameter is not "elastic" or "mars"
4. **Invalid Spread**: Raise `ValueError` when spread < 1
5. **Missing Test Data**: Raise `ValueError` when X_test or y_test is None during fit
6. **Shape Mismatch**: Raise `ValueError` when X and y have incompatible shapes

### Runtime Errors

1. **Model Not Fitted**: Raise `NotFittedError` when predict() is called before fit()
2. **Feature Mismatch**: Raise `ValueError` when prediction data has different features than training data
3. **JSON Load Error**: Raise `FileNotFoundError` or `JSONDecodeError` when model JSON cannot be loaded
4. **Convergence Warning**: Issue warning when Lasso models fail to converge, but continue training

### Data Quality Errors

1. **NaN Values**: Raise `ValueError` when input data contains NaN values
2. **Infinite Values**: Raise `ValueError` when input data contains infinite values
3. **Zero Variance**: Issue warning when features have zero variance
4. **Empty Dataset**: Raise `ValueError` when dataset has zero rows

### File I/O Errors

1. **Invalid Path**: Raise `OSError` when JSON file path is invalid or inaccessible
2. **Write Permission**: Raise `PermissionError` when lacking write permissions for output files
3. **Disk Space**: Raise `OSError` when insufficient disk space for JSON export

## Testing Strategy

### Unit Testing

The ELM package will use pytest for unit testing. Unit tests will cover:

1. **Component Initialization**: Verify each component class can be instantiated with valid parameters
2. **Edge Cases**: Test boundary conditions like m=1, f=1, spread=1
3. **Error Conditions**: Verify appropriate exceptions are raised for invalid inputs
4. **Metrics Calculation**: Test wMAPE and R2 calculations with known inputs/outputs
5. **JSON Structure**: Verify serialized JSON has correct schema
6. **Scikit-Learn Compatibility**: Test that ELMRegressor passes sklearn's `check_estimator` tests

### Property-Based Testing

The package will use Hypothesis for property-based testing in Python. Property-based tests will:

- Run a minimum of 100 iterations per property test
- Use Hypothesis strategies to generate random but valid test data
- Each property test must include a comment tag in the format: `# Feature: elm-package, Property {number}: {property_text}`
- Each correctness property from the design document must be implemented as a single property-based test

**Property Test Implementation Requirements:**

1. **Generators**: Create Hypothesis strategies for:
   - Random feature lists (varying sizes and names)
   - Random datasets (varying shapes, value ranges)
   - Random parameter combinations (m, f, sample, method, spread)
   - Valid train/test splits

2. **Property Test Coverage**: Each of the 14 correctness properties must have a corresponding property-based test

3. **Invariant Testing**: Properties should test invariants that hold across all valid inputs, not specific examples

4. **Shrinking**: Leverage Hypothesis's shrinking to find minimal failing examples when properties fail

### Integration Testing

Integration tests will verify:

1. **End-to-End Workflow**: Complete fit-predict cycle with real-world-like data
2. **Serialization Round-Trip**: Save and load models, verify predictions match
3. **Multiple Ensemble Methods**: Test both ElasticNet and MARS paths produce valid results
4. **Large-Scale Combinations**: Test with realistic m values (e.g., m=100) and multiple features

### Dependencies

- **Testing Framework**: pytest >= 7.0
- **Property Testing**: hypothesis >= 6.0
- **Test Data**: numpy, pandas for generating test datasets
- **Coverage**: pytest-cov for code coverage reporting

## Implementation Notes

### MARS Implementation

For MARS regression, the package will use the `py-earth` library, which provides a Python implementation of Multivariate Adaptive Regression Splines. Alternative: `sklearn.ensemble.GradientBoostingRegressor` with appropriate parameters can approximate MARS behavior if py-earth is unavailable.

### Performance Considerations

1. **Parallel Training**: Consider using `joblib` to parallelize Lasso model training across feature combinations
2. **Memory Management**: For large m values, consider batch processing to avoid memory issues
3. **Caching**: Cache baseline model results to avoid recomputation during experimentation

### Scikit-Learn Compatibility

The ELMRegressor will inherit from:
- `sklearn.base.BaseEstimator`: Provides `get_params()` and `set_params()`
- `sklearn.base.RegressorMixin`: Provides default `score()` method using R2

### SQL Export Implementation

The SQL export feature converts trained ensemble models into executable SQL code for database deployment. The generated SQL follows this structure:

```sql
WITH 
  -- Constituent Model CTEs
  model_1 AS (
    SELECT 
      id,
      (constant + coef1 * feature1 + coef2 * feature2 + ...) AS prediction
    FROM input_data
  ),
  model_2 AS (
    SELECT 
      id,
      (constant + coef1 * feature1 + coef2 * feature2 + ...) AS prediction
    FROM input_data
  ),
  ...
  -- Ensemble CTE combining constituent predictions
  ensemble_inputs AS (
    SELECT
      m1.id,
      m1.prediction AS model_1_pred,
      m2.prediction AS model_2_pred,
      ...
    FROM model_1 m1
    JOIN model_2 m2 ON m1.id = m2.id
    ...
  )
-- Final ensemble prediction
SELECT
  id,
  (ensemble_constant + 
   ensemble_coef1 * model_1_pred + 
   ensemble_coef2 * model_2_pred + 
   ...) AS predicted
FROM ensemble_inputs;
```

**SQL Compatibility Considerations:**
- Use standard SQL syntax compatible with PostgreSQL, MySQL, and SQL Server
- Feature names must be SQL-safe (alphanumeric and underscores only)
- Handle NULL values appropriately in calculations
- Use explicit type casting where needed for cross-database compatibility

### Package Structure

```
elm/
├── __init__.py
├── _version.py
├── estimator.py          # ELMRegressor main class
├── feature_generation.py # FeatureCombinationGenerator
├── baseline.py           # BaselineModel
├── constituent.py        # ConstituentModelTrainer
├── selection.py          # ModelSelector
├── ensemble.py           # EnsembleBuilder
├── metrics.py            # MetricsCalculator
├── serialization.py      # ModelSerializer
├── scoring.py            # ModelScorer
├── sql_export.py         # SQLExporter
├── utils.py              # Utility functions
└── tests/
    ├── __init__.py
    ├── test_estimator.py
    ├── test_feature_generation.py
    ├── test_baseline.py
    ├── test_constituent.py
    ├── test_selection.py
    ├── test_ensemble.py
    ├── test_metrics.py
    ├── test_serialization.py
    ├── test_scoring.py
    ├── test_sql_export.py
    ├── test_properties.py      # Property-based tests
    └── test_integration.py
```

### PyPI Distribution

The package will include:
- `pyproject.toml`: Modern Python packaging configuration
- `README.md`: Installation instructions, quick start, examples
- `LICENSE`: Open source license (e.g., MIT)
- `CHANGELOG.md`: Version history
- `docs/`: Sphinx documentation with API reference and tutorials
- `examples/`: Jupyter notebooks demonstrating usage

### Version Management

- Initial release: 0.1.0
- Follow semantic versioning: MAJOR.MINOR.PATCH
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible
