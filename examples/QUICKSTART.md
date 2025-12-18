# Paramsemble Examples - Quick Start

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Core dependencies (required)
pip install paramsemble-regress scikit-learn numpy pandas

# Visualization dependencies (for notebooks)
pip install matplotlib seaborn jupyter
```

### Step 2: Test Your Setup

```bash
python examples/test_notebooks.py
```

You should see:
```
✓ All tests passed! You're ready to run the notebooks.
```

### Step 3: Launch Jupyter

```bash
cd examples
jupyter notebook
```

### Step 4: Open a Notebook

Start with **`elm_basic_usage.ipynb`** for a comprehensive introduction.

---

## 📚 What's Included

### `elm_basic_usage.ipynb` (Recommended First)
- ⏱️ **Time:** 10-15 minutes
- 🎯 **Level:** Beginner
- 📖 **Topics:**
  - Basic ELM workflow
  - ElasticNet vs MARS ensembles
  - Model saving and loading
  - SQL export
  - Visualization

### `elm_advanced_examples.ipynb`
- ⏱️ **Time:** 15-20 minutes
- 🎯 **Level:** Intermediate
- 📖 **Topics:**
  - Real-world datasets
  - Hyperparameter tuning
  - Feature importance
  - Production deployment
  - Performance optimization

---

## 💡 Quick Code Snippets

### Train a Model (30 seconds)

```python
from paramsemble import ParamsembleRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Paramsemble
regressor = ParamsembleRegressor(m=50, f=5, method='elastic', spread=10, random_state=42)
regressor.fit(X_train, y_train, X_test, y_test)

# Predict
predictions = regressor.predict(X_test)
print(f"Generated {len(predictions)} predictions!")
```

### Save and Load Model

```python
# Save during training
regressor = ParamsembleRegressor(m=50, f=5, modeljson='my_model.json')
regressor.fit(X_train, y_train, X_test, y_test)

# Load and score later
predictions = regressor.score_from_json(X_new, 'my_model.json')
```

### Export to SQL

```python
# Generate SQL code
sql = regressor.export_sql('my_model.json', table_name='features', id_column='id')

# Save to file
with open('model.sql', 'w') as f:
    f.write(sql)

# Deploy in your database!
```

---

## 🎓 Learning Path

1. **Complete Beginner?**
   - Start with `elm_basic_usage.ipynb`
   - Run all cells sequentially
   - Experiment with different parameters

2. **Have ML Experience?**
   - Skim `elm_basic_usage.ipynb`
   - Focus on `elm_advanced_examples.ipynb`
   - Try with your own datasets

3. **Ready for Production?**
   - Review the production workflow in advanced notebook
   - Study the SQL export section
   - Check model monitoring examples

---

## 🔧 Troubleshooting

### Import Error: No module named 'paramsemble'

```bash
pip install paramsemble-regress
```

### Visualization Not Working

```bash
pip install matplotlib seaborn
```

### Jupyter Not Found

```bash
pip install jupyter
```

### "No models outperformed baseline" Error

This can happen with small or noisy datasets. Try:
- Increase dataset size
- Reduce noise in data
- Adjust `m` (more combinations)
- Change `f` (different feature count)

---

## 📊 What You'll Learn

By completing both notebooks, you'll understand:

✅ How to train Paramsemble models with different configurations  
✅ When to use ElasticNet vs MARS ensembles  
✅ How to save and load models for production  
✅ How to deploy models in databases using SQL  
✅ How to analyze feature importance  
✅ How to tune hyperparameters  
✅ How to monitor model performance  

---

## 🎯 Next Steps

After completing the notebooks:

1. **Try Your Own Data**
   - Load your dataset
   - Preprocess features
   - Train an ELM model
   - Compare with your current approach

2. **Experiment with Parameters**
   - Try different `m` values (50, 100, 200)
   - Test different `f` values (3, 5, 8)
   - Compare `spread` settings (5, 10, 20)

3. **Deploy to Production**
   - Export your best model to SQL
   - Set up batch scoring pipeline
   - Monitor prediction quality

---

## 📖 Additional Resources

- **Main README:** `../README.md` - Complete API reference
- **Examples README:** `README.md` - Detailed notebook descriptions
- **Test Script:** `test_notebooks.py` - Verify your setup

---

**Happy Learning! 🎉**

Questions? Check the main README or open an issue on GitHub.
