# Paramsemble-Regress (Parametric Ensemble Regression)

A Python machine learning library for regression tasks that combines automated feature selection, baseline comparison, and ensemble modeling.

## Overview

The Paramsemble-regress package generates multiple feature combinations, trains Lasso regression models on each combination, evaluates their performance against a Random Forest baseline, and creates ensemble predictions using either ElasticNet or MARS (Multivariate Adaptive Regression Splines) methods.

This apprach can outperform XGBoost and other ensemble methods for datasets that are highly heterogenous, or have high variability of noise.


## Features

- **Automated Feature Selection**: Generate diverse feature combinations automatically
- **Baseline Comparison**: Establish performance benchmarks using Random Forest
- **Multiple Ensemble Methods**: Choose between ElasticNet and MARS for ensemble creation
- **Scikit-Learn Compatible**: Familiar `fit()` and `predict()` interfaces
- **Model Serialization**: Save and load models for deployment
- **SQL Export**: Deploy models directly in database environments without Python dependencies

## Installation

```bash
pip install paramsemble-regress
```

For MARS ensemble support (optional):

```bash
pip install paramsemble-regress[mars]
```

For development:

```bash
pip install paramsemble-regress[dev]
```

## Requirements

- Python >= 3.8
- scikit-learn >= 1.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- py-earth >= 0.1.0 (optional, for MARS ensemble method)

## Quick Start

```python
from paramsemble import ParamsembleRegressor
import numpy as np
import pandas as pd

# Create sample data
X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
y_train = np.random.rand(100)
X_test = pd.DataFrame(np.random.rand(30, 10), columns=[f'feature_{i}' for i in range(10)])
y_test = np.random.rand(30)

# Initialize and fit the model
regressor = ParamsembleRegressor(
    m=50,           # Generate 50 feature combinations
    f=5,            # Each combination has 5 features
    method='elastic',  # Use ElasticNet for ensemble
    spread=10,      # Select top 10 models for ensemble
    random_state=42
)

regressor.fit(X_train, y_train, X_test, y_test)

# Make predictions
predictions = regressor.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
```

## Usage Examples

### Basic Workflow: Fit and Predict

```python
from paramsemble import ParamsembleRegressor
import pandas as pd
import numpy as np

# Load your data
X_train = pd.read_csv('train_features.csv')
y_train = pd.read_csv('train_target.csv').values.ravel()
X_test = pd.read_csv('test_features.csv')
y_test = pd.read_csv('test_target.csv').values.ravel()

# Initialize the Paramsemble regressor
regressor = ParamsembleRegressor(
    m=100,              # Number of feature combinations to generate
    f=8,                # Number of features per combination
    sample='unique',    # Sampling method: 'unique' or 'replace'
    method='elastic',   # Ensemble method: 'elastic' or 'mars'
    spread=15,          # Number of top models to include in ensemble
    random_state=42
)

# Fit the model
regressor.fit(X_train, y_train, X_test, y_test)

# Make predictions on new data
X_new = pd.read_csv('new_data.csv')
predictions = regressor.predict(X_new)

# Access fitted attributes
print(f"Baseline wMAPE: {regressor.baseline_metrics_['wmape']:.4f}")
print(f"Baseline R²: {regressor.baseline_metrics_['r2']:.4f}")
print(f"Number of selected models: {len(regressor.selected_models_)}")
```

### Model Serialization and Scoring

```python
from paramsemble import ParamsembleRegressor
import pandas as pd

# Train and save the model
regressor = ParamsembleRegressor(
    m=50, 
    f=5, 
    method='elastic', 
    spread=10,
    modeljson='my_model.json'  # Save ensemble model to JSON
)

regressor.fit(X_train, y_train, X_test, y_test)

# Later, score new data using the saved model
X_new = pd.read_csv('new_data.csv')
id_column = X_new.index  # Or use a specific ID column

predictions_df = regressor.score_from_json(
    X_new, 
    modeljson_path='my_model.json',
    id_column=id_column
)

print(predictions_df.head())
# Output:
#    id  predicted
# 0   0   1.234567
# 1   1   2.345678
# 2   2   3.456789
```

### Using MARS Ensemble Method

```python
from paramsemble import ParamsembleRegressor

# Use MARS (Multivariate Adaptive Regression Splines) for ensemble
regressor = ParamsembleRegressor(
    m=100,
    f=6,
    method='mars',  # Use MARS instead of ElasticNet
    spread=12,
    random_state=42
)

regressor.fit(X_train, y_train, X_test, y_test)
predictions = regressor.predict(X_test)
```

### Saving Constituent Models

```python
from paramsemble import ParamsembleRegressor

# Save both constituent models and final ensemble
regressor = ParamsembleRegressor(
    m=50,
    f=5,
    method='elastic',
    spread=10,
    ELM2json='constituent_models.json',  # Save all constituent models
    modeljson='ensemble_model.json'       # Save final ensemble
)

regressor.fit(X_train, y_train, X_test, y_test)

# The JSON files now contain:
# - constituent_models.json: All Lasso models with their metrics and coefficients
# - ensemble_model.json: Selected models + ensemble equation
```

### SQL Export for Database Deployment

```python
from paramsemble import ParamsembleRegressor

# Train and save model
regressor = ParamsembleRegressor(m=50, f=5, method='elastic', spread=10, modeljson='model.json')
regressor.fit(X_train, y_train, X_test, y_test)

# Export model as SQL code
sql_code = regressor.export_sql(
    modeljson_path='model.json',
    table_name='customer_features',  # Your database table name
    id_column='customer_id'           # Your ID column name
)

# Save SQL to file
with open('model_deployment.sql', 'w') as f:
    f.write(sql_code)

# The generated SQL can be executed in PostgreSQL, MySQL, or SQL Server
# Example SQL structure:
"""
WITH 
  model_1 AS (
    SELECT 
      customer_id,
      (0.5 + 0.3 * feature_1 + 0.2 * feature_2 + ...) AS prediction
    FROM customer_features
  ),
  model_2 AS (
    SELECT 
      customer_id,
      (0.4 + 0.1 * feature_3 + 0.5 * feature_4 + ...) AS prediction
    FROM customer_features
  ),
  ...
  ensemble_inputs AS (
    SELECT
      m1.customer_id,
      m1.prediction AS model_1_pred,
      m2.prediction AS model_2_pred,
      ...
    FROM model_1 m1
    JOIN model_2 m2 ON m1.customer_id = m2.customer_id
    ...
  )
SELECT
  customer_id,
  (0.6 + 0.4 * model_1_pred + 0.3 * model_2_pred + ...) AS predicted
FROM ensemble_inputs;
"""
```

### Complete End-to-End Example

```python
from paramsemble import ParamsembleRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate synthetic dataset
X, y = make_regression(n_samples=500, n_features=20, n_informative=15, 
                       noise=10, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Paramsemble with ElasticNet ensemble
regressor = ParamsembleRegressor(
    m=100,              # Generate 100 feature combinations
    f=10,               # Each combination has 10 features
    sample='unique',    # No duplicate features in a combination
    method='elastic',   # Use ElasticNet for ensemble
    spread=15,          # Select top 15 models
    ELM2json='constituent_models.json',
    modeljson='ensemble_model.json',
    random_state=42
)

# Fit the model
print("Training Paramsemble model...")
regressor.fit(X_train, y_train, X_test, y_test)

# Display baseline metrics
print(f"\nBaseline Model Performance:")
print(f"  wMAPE: {regressor.baseline_metrics_['wmape']:.4f}")
print(f"  R²: {regressor.baseline_metrics_['r2']:.4f}")

# Display ensemble info
print(f"\nEnsemble Information:")
print(f"  Total feature combinations: {len(regressor.feature_combinations_)}")
print(f"  Models selected for ensemble: {len(regressor.selected_models_)}")

# Make predictions
predictions = regressor.predict(X_test)
print(f"\nPredictions generated: {len(predictions)}")

# Score new data using saved model
X_new = X_test.iloc[:10]  # Use first 10 test samples as "new" data
predictions_df = regressor.score_from_json(
    X_new,
    modeljson_path='ensemble_model.json',
    id_column=X_new.index
)
print(f"\nScoring from JSON:")
print(predictions_df)

# Export to SQL
sql_code = regressor.export_sql(
    modeljson_path='ensemble_model.json',
    table_name='features',
    id_column='id'
)
print(f"\nSQL code generated: {len(sql_code)} characters")
with open('model.sql', 'w') as f:
    f.write(sql_code)
print("SQL code saved to 'model.sql'")
```

## API Reference

### ParamsembleRegressor

The main estimator class for Parametric Ensemble Regression.

#### Parameters

- **m** : `int`, default=100
  - Number of feature combinations to generate
  - Must be a positive integer
  - Will be capped at maximum possible combinations if too large

- **f** : `int`, required
  - Number of features per combination
  - Must be a positive integer
  - Must be ≤ total number of features when `sample='unique'`

- **sample** : `{'unique', 'replace'}`, default='unique'
  - Sampling method for feature combinations
  - `'unique'`: Each feature appears at most once per combination
  - `'replace'`: Features can appear multiple times per combination

- **method** : `{'elastic', 'mars'}`, default='elastic'
  - Ensemble method to use
  - `'elastic'`: Use ElasticNet regression for ensemble
  - `'mars'`: Use MARS (Multivariate Adaptive Regression Splines) for ensemble

- **spread** : `int`, default=10
  - Number of top-performing models to include in ensemble
  - Must be a positive integer
  - Only models that outperform baseline are considered

- **ELM2json** : `str`, optional
  - File path to save constituent model details
  - Saves all Lasso models with metrics and coefficients
  - JSON format for easy inspection and debugging

- **modeljson** : `str`, optional
  - File path to save final ensemble model
  - Required for `score_from_json()` and `export_sql()` methods
  - Contains selected models and ensemble equation

- **random_state** : `int`, optional
  - Random seed for reproducibility
  - Controls feature combination generation and model training

#### Methods

##### `fit(X_train, y_train, X_test, y_test, id_column=None)`

Fit the Paramsemble ensemble model.

**Parameters:**
- **X_train** : array-like of shape (n_samples, n_features)
  - Training feature matrix
  - Can be numpy array or pandas DataFrame
  
- **y_train** : array-like of shape (n_samples,)
  - Training target values
  
- **X_test** : array-like of shape (n_samples, n_features)
  - Test feature matrix for evaluation
  - Must have same features as X_train
  
- **y_test** : array-like of shape (n_samples,)
  - Test target values
  
- **id_column** : array-like, optional
  - Identifiers for test samples

**Returns:**
- **self** : object
  - Fitted estimator

**Fitted Attributes:**
- `feature_names_` : list - Names of features used in training
- `n_features_in_` : int - Number of features
- `feature_combinations_` : list - Generated feature combinations
- `baseline_metrics_` : dict - Baseline model wMAPE and R²
- `constituent_models_` : list - All trained Lasso models
- `selected_models_` : list - Top-performing models for ensemble
- `ensemble_model_` : object - Fitted ensemble model
- `ensemble_equation_` : dict - Ensemble coefficients

##### `predict(X)`

Generate predictions using the fitted ensemble.

**Parameters:**
- **X** : array-like of shape (n_samples, n_features)
  - Feature matrix for prediction
  - Must have same features as training data

**Returns:**
- **y_pred** : ndarray of shape (n_samples,)
  - Predicted values

**Raises:**
- `NotFittedError` : If called before `fit()`
- `ValueError` : If X has wrong number of features or contains invalid values

##### `score_from_json(X, modeljson_path, id_column=None)`

Score new data using a saved model.

**Parameters:**
- **X** : array-like of shape (n_samples, n_features)
  - Feature matrix for scoring
  
- **modeljson_path** : str
  - Path to saved model JSON file
  
- **id_column** : array-like, optional
  - Identifiers for samples

**Returns:**
- **predictions_df** : DataFrame
  - DataFrame with two columns: IDs and 'predicted' values

##### `export_sql(modeljson_path, table_name='input_data', id_column='id')`

Export model as SQL code for database deployment.

**Parameters:**
- **modeljson_path** : str
  - Path to saved model JSON file
  
- **table_name** : str, default='input_data'
  - Name of the input table in SQL
  
- **id_column** : str, default='id'
  - Name of the ID column in the input table

**Returns:**
- **sql_code** : str
  - Complete SQL query implementing the ensemble model
  - Compatible with PostgreSQL, MySQL, and SQL Server

## Model Persistence

Paramsemble supports two types of model serialization:

1. **Constituent Models JSON** (`ELM2json` parameter)
   - Contains all trained Lasso models
   - Includes features, coefficients, and performance metrics
   - Useful for model inspection and debugging

2. **Ensemble Model JSON** (`modeljson` parameter)
   - Contains selected top-performing models
   - Includes ensemble equation and method
   - Required for scoring and SQL export

JSON files can be loaded and used for scoring without retraining:

```python
# Train once
regressor = ParamsembleRegressor(m=50, f=5, modeljson='model.json')
regressor.fit(X_train, y_train, X_test, y_test)

# Score many times
predictions = regressor.score_from_json(X_new, 'model.json')
```

## Example Notebooks

The `examples/` directory contains comprehensive Jupyter notebooks demonstrating Paramsemble usage:

### 1. Basic Usage Guide (`paramsemble_basic_usage.ipynb`)

Perfect for getting started with Paramsemble. Covers:
- Training with ElasticNet and MARS ensemble methods
- Model serialization and loading
- Scoring new data with saved models
- SQL export for database deployment
- Performance visualization and comparison

**Run the notebook:**
```bash
cd examples
jupyter notebook paramsemble_basic_usage.ipynb
```

### 2. Advanced Examples (`paramsemble_advanced_examples.ipynb`)

Explores advanced features and real-world scenarios:
- Working with real datasets (California Housing, Diabetes)
- Hyperparameter tuning and comparison
- Feature importance analysis
- Model inspection and interpretation
- Complete production deployment workflow
- Sampling method comparison

**Test notebook dependencies:**
```bash
python examples/test_notebooks.py
```

See `examples/README.md` for detailed information about the notebooks.

## Performance Tips

1. **Start with smaller `m` values** (e.g., 50-100) for faster experimentation
2. **Use `sample='unique'`** for better feature diversity
3. **Adjust `spread`** based on your dataset size (10-20 is typical)
4. **Try both ensemble methods** - ElasticNet is faster, MARS can capture non-linearity
5. **Monitor baseline metrics** - if constituent models don't beat baseline, adjust parameters

## Error Handling

Paramsemble provides clear error messages for common issues:

- **Invalid parameters**: Validates all parameters at initialization
- **Shape mismatches**: Checks that train/test data have compatible shapes
- **Missing values**: Raises errors for NaN or infinite values
- **Not fitted**: Prevents prediction before fitting
- **Feature mismatches**: Ensures prediction data matches training features

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use Paramsemble in your research, please cite:

```bibtex
@software{paramsemble_regressor,
  title={Paramsemble: Parametric Ensemble Regression},
  author={Stephen Chen},
  year={2024},
  url={https://github.com/schen18/paramsemble-regress}
}
```
