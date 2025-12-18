# Paramsemble Package Examples

This directory contains Jupyter notebooks demonstrating the usage of the Paramsemble (Parametric Ensemble Regression) package.

## Notebooks

### 1. `elm_basic_usage.ipynb` - Basic Usage Guide

**Recommended for:** First-time users and quick start

**Contents:**
- Setup and synthetic data generation
- Training with ElasticNet ensemble method
- Training with MARS ensemble method
- Model serialization and loading workflow
- Scoring new data with saved models
- SQL export for database deployment
- Performance visualization and comparison

**Key Features Demonstrated:**
- Complete fit-predict workflow
- Both ensemble methods (ElasticNet and MARS)
- JSON serialization for model persistence
- SQL code generation for database deployment
- Multiple visualization types (scatter plots, residuals, bar charts, histograms)

**Estimated Runtime:** 5-10 minutes

---

### 2. `elm_advanced_examples.ipynb` - Advanced Usage Patterns

**Recommended for:** Users familiar with Paramsemble basics who want to explore advanced features

**Contents:**
- Working with real-world datasets (California Housing, Diabetes)
- Hyperparameter tuning and comparison
- Feature importance analysis
- Model inspection and interpretation
- Complete production deployment workflow
- Sampling method comparison ('unique' vs 'replace')

**Key Features Demonstrated:**
- Data preprocessing and standardization
- Systematic hyperparameter exploration
- Feature contribution analysis
- JSON model structure inspection
- Batch scoring pipeline
- Model monitoring and statistics
- Performance optimization strategies

**Estimated Runtime:** 10-15 minutes

---

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install paramsemble-regress jupyter matplotlib seaborn scikit-learn
```

### Running the Notebooks

1. Navigate to the examples directory:
   ```bash
   cd examples
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open either notebook and run cells sequentially

### Directory Structure

After running the notebooks, the following files will be created:

```
examples/
├── elm_basic_usage.ipynb
├── elm_advanced_examples.ipynb
├── README.md
└── models/                              # Created by notebooks
    ├── constituent_models_elastic.json
    ├── constituent_models_mars.json
    ├── ensemble_model_elastic.json
    ├── ensemble_model_mars.json
    ├── ensemble_model_elastic.sql
    ├── housing_model.json
    ├── production_model.json
    ├── production_model.sql
    └── batch_predictions.csv
```

## Example Workflows

### Quick Start (5 minutes)

```python
from paramsemble import ParamsembleRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Paramsemble
regressor = ParamsembleRegressor(m=50, f=5, method='elastic', spread=10, random_state=42)
regressor.fit(X_train, y_train, X_test, y_test)

# Predict
predictions = regressor.predict(X_test)
```

### Model Persistence

```python
# Save model during training
regressor = ParamsembleRegressor(
    m=50, f=5, method='elastic', spread=10,
    modeljson='my_model.json',
    random_state=42
)
regressor.fit(X_train, y_train, X_test, y_test)

# Score new data with saved model
predictions = regressor.score_from_json(X_new, 'my_model.json', id_column=ids)
```

### SQL Export

```python
# Export trained model to SQL
sql_code = regressor.export_sql(
    'my_model.json',
    table_name='input_data',
    id_column='id'
)

# Save to file
with open('model.sql', 'w') as f:
    f.write(sql_code)

# Execute in your database
# PostgreSQL: psql -d mydb -f model.sql
# MySQL: mysql -u user -p mydb < model.sql
```

## Common Use Cases

### 1. Feature Selection
Use Paramsemble to identify important feature combinations:
```python
regressor = ParamsembleRegressor(m=100, f=5, spread=15)
regressor.fit(X_train, y_train, X_test, y_test)

# Analyze selected models
for model in regressor.selected_models_:
    print(f"Features: {model['features']}, R²: {model['r2']:.4f}")
```

### 2. Model Comparison
Compare ElasticNet vs MARS ensembles:
```python
# ElasticNet
regressor_elastic = ParamsembleRegressor(method='elastic', ...)
regressor_elastic.fit(X_train, y_train, X_test, y_test)

# MARS
regressor_mars = ParamsembleRegressor(method='mars', ...)
regressor_mars.fit(X_train, y_train, X_test, y_test)

# Compare predictions
```

### 3. Production Deployment
Complete workflow from training to deployment:
```python
# 1. Train
regressor = ParamsembleRegressor(modeljson='prod_model.json', ...)
regressor.fit(X_train, y_train, X_test, y_test)

# 2. Export SQL
sql = regressor.export_sql('prod_model.json')

# 3. Deploy to database
# 4. Score new data
predictions = regressor.score_from_json(X_new, 'prod_model.json')
```

## Tips and Best Practices

### Performance Optimization

1. **Feature Scaling**: Always standardize features before training
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Hyperparameter Tuning**:
   - Start with `m=50-100` for initial experiments
   - Use `f=3-5` for most datasets
   - Adjust `spread` based on how many models beat baseline

3. **Sampling Strategy**:
   - Use `sample='unique'` for most cases (no duplicate features)
   - Use `sample='replace'` when you want to explore feature interactions

### Model Selection

- **ElasticNet**: Faster, works well for linear relationships
- **MARS**: Captures non-linear patterns, more complex

### Debugging

If you encounter issues:

1. Check feature names are consistent between train and test
2. Ensure no NaN or infinite values in data
3. Verify `f` is not larger than number of features (with `sample='unique'`)
4. Check that test set is provided during `fit()`

## Additional Resources

- **Package Documentation**: See main README.md
- **API Reference**: Check docstrings in source code
- **Issue Tracker**: Report bugs on GitHub
- **Examples**: These notebooks!

## Questions?

If you have questions or need help:
1. Review the notebooks in order (basic → advanced)
2. Check the main package README
3. Examine the docstrings in the source code
4. Open an issue on GitHub

---

**Happy modeling with Paramsemble!** 🚀
