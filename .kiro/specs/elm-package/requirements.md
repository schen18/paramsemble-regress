# Requirements Document

## Introduction

The ELM (Ensemble Linear Models) package is a Python machine learning library designed for regression tasks that combines automated feature selection, baseline comparison, and ensemble modeling. The system generates multiple feature combinations, trains Lasso regression models on each combination, evaluates their performance against a Random Forest baseline, and creates ensemble predictions using either ElasticNet or MARS (Multivariate Adaptive Regression Splines) methods. The package provides Scikit-Learn compatible interfaces and supports model serialization for deployment and scoring.

## Glossary

- **ELM System**: The Ensemble Linear Models package that performs feature combination generation, model training, ensemble creation, and scoring
- **Feature Set**: A collection of features selected from the available feature list
- **Baseline Model**: A Random Forest regression model trained on all available features to establish performance benchmarks
- **Constituent Model**: An individual Lasso regression model trained on a specific feature set
- **Ensemble Model**: A meta-model (ElasticNet or MARS) that combines predictions from multiple constituent models
- **wMAPE**: Weighted Mean Absolute Percentage Error, a performance metric where lower values indicate better performance
- **R2**: R-squared coefficient of determination, a performance metric where higher values indicate better performance
- **Equation Dictionary**: A JSON-serializable dictionary containing feature names as keys and coefficients as values, plus an intercept under the "constant" key
- **Training Dataset**: The dataset used to fit regression models
- **Test Dataset**: The dataset used to evaluate model performance and generate metrics
- **Scoring Dataset**: A new dataset for which predictions are generated using a trained ensemble model

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to generate diverse feature combinations automatically, so that I can explore different feature subsets without manual enumeration.

#### Acceptance Criteria

1. WHEN the ELM System receives a feature list and parameters m, f, and sample method, THEN the ELM System SHALL generate m feature sets each containing f features
2. WHERE the sample parameter equals "unique", WHEN generating feature sets, THEN the ELM System SHALL ensure each feature appears at most once within any single feature set
3. WHERE the sample parameter equals "replace", WHEN generating feature sets, THEN the ELM System SHALL allow features to appear multiple times within a single feature set
4. WHEN the requested m value exceeds the maximum possible combinations for the given f and sample method, THEN the ELM System SHALL override m with the calculated maximum possible combinations
5. WHEN calculating maximum combinations with sample method "unique", THEN the ELM System SHALL compute the value as C(n, f) where n is the total number of available features

### Requirement 2

**User Story:** As a data scientist, I want to establish a baseline performance using all features, so that I can compare the performance of feature-reduced models against a comprehensive model.

#### Acceptance Criteria

1. WHEN the ELM System begins training, THEN the ELM System SHALL fit a Random Forest regression model using all available features on the Training Dataset
2. WHEN the Baseline Model completes training, THEN the ELM System SHALL evaluate the Baseline Model on the Test Dataset and compute wMAPE and R2 metrics
3. WHEN baseline metrics are computed, THEN the ELM System SHALL store the baseline wMAPE and R2 values for subsequent comparison with Constituent Models

### Requirement 3

**User Story:** As a data scientist, I want to train multiple Lasso regression models on different feature combinations, so that I can identify which feature subsets produce strong predictive models.

#### Acceptance Criteria

1. WHEN feature sets are generated, THEN the ELM System SHALL fit one Lasso regression model per feature set using the Training Dataset
2. WHEN each Constituent Model completes training, THEN the ELM System SHALL evaluate the model on the Test Dataset and compute wMAPE and R2 metrics
3. WHEN each Constituent Model is evaluated, THEN the ELM System SHALL extract an Equation Dictionary containing feature names as keys, coefficient values as values, and the intercept value under the "constant" key
4. WHERE the ELM2json parameter is not null, WHEN all Constituent Models complete training, THEN the ELM System SHALL export all wMAPE values, R2 values, and Equation Dictionaries to a JSON file at the path specified by ELM2json

### Requirement 4

**User Story:** As a data scientist, I want to select the best-performing constituent models, so that I can create a high-quality ensemble without including poor-performing models.

#### Acceptance Criteria

1. WHEN all Constituent Models are evaluated, THEN the ELM System SHALL identify models that outperform the Baseline Model on wMAPE or R2 metrics
2. WHEN identifying top-performing models, THEN the ELM System SHALL rank Constituent Models by wMAPE in ascending order and R2 in descending order
3. WHEN the spread parameter is provided, THEN the ELM System SHALL select the top n Constituent Models where n equals the spread parameter value
4. WHEN selected models are identified, THEN the ELM System SHALL generate predictions for the Test Dataset using each selected Constituent Model

### Requirement 5

**User Story:** As a data scientist, I want to create an ElasticNet ensemble of top models, so that I can combine their predictions into a robust final prediction.

#### Acceptance Criteria

1. WHERE the method parameter equals "elastic", WHEN top Constituent Models are selected, THEN the ELM System SHALL use the predictions from selected models as input features for an ElasticNet regression model
2. WHEN the ElasticNet Ensemble Model is fitted, THEN the ELM System SHALL train it using the predictions from selected Constituent Models on the Test Dataset
3. WHEN the ElasticNet Ensemble Model completes training, THEN the ELM System SHALL generate final predictions for the Test Dataset
4. WHEN final predictions are generated, THEN the ELM System SHALL return a DataFrame containing test set identifiers in one column and predicted values in a "predicted" column
5. WHERE the modeljson parameter is not null, WHEN the ElasticNet Ensemble Model completes training, THEN the ELM System SHALL export the method name, Equation Dictionaries of selected Constituent Models, and the Ensemble Model Equation Dictionary to a JSON file at the path specified by modeljson

### Requirement 6

**User Story:** As a data scientist, I want to create a MARS ensemble of top models, so that I can capture non-linear relationships in the ensemble combination.

#### Acceptance Criteria

1. WHERE the method parameter equals "mars", WHEN top Constituent Models are selected, THEN the ELM System SHALL use the predictions from selected models as input features for a MARS regression model
2. WHEN the MARS Ensemble Model is fitted, THEN the ELM System SHALL train it using the predictions from selected Constituent Models on the Test Dataset
3. WHEN the MARS Ensemble Model completes training, THEN the ELM System SHALL generate final predictions for the Test Dataset
4. WHEN final predictions are generated, THEN the ELM System SHALL return a DataFrame containing test set identifiers in one column and predicted values in a "predicted" column
5. WHERE the modeljson parameter is not null, WHEN the MARS Ensemble Model completes training, THEN the ELM System SHALL export the method name, Equation Dictionaries of selected Constituent Models, and the Ensemble Model Equation Dictionary to a JSON file at the path specified by modeljson

### Requirement 7

**User Story:** As a data scientist, I want to score new data using a trained ensemble model, so that I can generate predictions for production or validation datasets.

#### Acceptance Criteria

1. WHEN the ELM System receives a Scoring Dataset and a modeljson file path, THEN the ELM System SHALL load the Equation Dictionaries and method from the modeljson file
2. WHEN Equation Dictionaries are loaded, THEN the ELM System SHALL apply each Constituent Model Equation Dictionary to the Scoring Dataset to generate intermediate predictions
3. WHEN intermediate predictions are generated, THEN the ELM System SHALL apply the Ensemble Model Equation Dictionary to the intermediate predictions to generate final predictions
4. WHEN final predictions are computed, THEN the ELM System SHALL return a DataFrame containing scoring dataset identifiers and predicted values

### Requirement 8

**User Story:** As a data scientist, I want the ELM package to follow Scikit-Learn conventions, so that I can integrate it seamlessly into existing ML pipelines.

#### Acceptance Criteria

1. WHEN the ELM System is instantiated, THEN the ELM System SHALL provide a fit method that accepts training and test datasets
2. WHEN the ELM System is fitted, THEN the ELM System SHALL provide a predict method that accepts a dataset and returns predictions
3. WHEN the ELM System is configured, THEN the ELM System SHALL accept hyperparameters through constructor arguments following Scikit-Learn naming conventions
4. WHEN the ELM System completes fitting, THEN the ELM System SHALL store fitted model attributes with trailing underscores following Scikit-Learn conventions

### Requirement 9

**User Story:** As a data engineer, I want to export the trained ensemble model as SQL code, so that I can deploy predictions directly in database environments without Python dependencies.

#### Acceptance Criteria

1. WHEN the ELM System receives a modeljson file path, THEN the ELM System SHALL generate SQL code that implements the complete ensemble prediction logic
2. WHEN generating SQL code, THEN the ELM System SHALL create Common Table Expressions (CTEs) for each Constituent Model equation dictionary
3. WHEN creating constituent CTEs, THEN the ELM System SHALL generate SQL expressions that compute predictions using the feature coefficients and constant from each equation dictionary
4. WHEN all constituent CTEs are created, THEN the ELM System SHALL generate a final SELECT statement that applies the Ensemble Model equation dictionary to the constituent predictions
5. WHEN the SQL code is generated, THEN the ELM System SHALL return valid SQL that can be executed in standard SQL databases (PostgreSQL, MySQL, SQL Server)

### Requirement 10

**User Story:** As a developer, I want to package ELM for PyPI distribution, so that users can install it easily using pip.

#### Acceptance Criteria

1. WHEN the ELM package is built, THEN the ELM System SHALL include a setup.py or pyproject.toml file with package metadata
2. WHEN the ELM package is installed, THEN the ELM System SHALL declare dependencies on scikit-learn, numpy, pandas, and any MARS implementation library
3. WHEN the ELM package is distributed, THEN the ELM System SHALL include documentation, examples, and a README file
4. WHEN the ELM package is versioned, THEN the ELM System SHALL follow semantic versioning conventions
