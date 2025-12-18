# Implementation Plan

- [x] 1. Set up package structure and configuration





  - Create package directory structure with all module files
  - Create pyproject.toml with package metadata and dependencies (scikit-learn, numpy, pandas, py-earth, hypothesis, pytest)
  - Create __init__.py files to expose main ELMRegressor class
  - Set up version management in _version.py
  - _Requirements: 9.1, 9.2, 9.4_

- [x] 2. Implement metrics calculation module





  - Create MetricsCalculator class with compute_wmape() and compute_r2() static methods
  - Implement wMAPE calculation: sum(|y_true - y_pred|) / sum(|y_true|)
  - Implement R2 calculation using sklearn.metrics.r2_score
  - Add input validation for NaN and infinite values
  - _Requirements: 2.2, 3.2_

- [x] 2.1 Write property test for metrics calculation


  - **Property 4: Baseline metrics computation**
  - **Validates: Requirements 2.2, 2.3**

- [x] 3. Implement feature combination generator





  - Create FeatureCombinationGenerator class
  - Implement calculate_max_combinations() for unique sampling using scipy.special.comb
  - Implement generate_combinations() with support for "unique" and "replace" sampling methods
  - Add logic to cap m at maximum possible combinations
  - Add input validation for invalid sample methods and f > n_features
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3.1 Write property test for feature set generation count


  - **Property 1: Feature set generation count and size**
  - **Validates: Requirements 1.1, 1.4**

- [x] 3.2 Write property test for unique sampling constraint


  - **Property 2: Unique sampling constraint**
  - **Validates: Requirements 1.2**

- [x] 3.3 Write property test for maximum combinations calculation


  - **Property 3: Maximum combinations calculation**
  - **Validates: Requirements 1.5**

- [x] 4. Implement baseline model component




  - Create BaselineModel class
  - Implement fit_and_evaluate() method using sklearn RandomForestRegressor
  - Train on X_train, y_train and evaluate on X_test, y_test
  - Return dictionary with baseline wMAPE and R2 metrics
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 5. Implement constituent model trainer





  - Create ConstituentModelTrainer class
  - Implement train_models() to fit Lasso regression for each feature combination
  - Implement extract_equation_dict() to extract coefficients and intercept
  - Store model metrics (wMAPE, R2) and equation dictionary for each model
  - Return list of model information dictionaries
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 5.1 Write property test for constituent model count


  - **Property 5: Constituent model count matches feature combinations**
  - **Validates: Requirements 3.1**

- [x] 5.2 Write property test for constituent model metrics


  - **Property 6: All constituent models have metrics**
  - **Validates: Requirements 3.2**

- [x] 5.3 Write property test for equation dictionary structure


  - **Property 7: Equation dictionary structure**
  - **Validates: Requirements 3.3**

- [x] 6. Implement model selector





  - Create ModelSelector class
  - Implement select_top_models() to filter models that outperform baseline
  - Implement ranking logic: sort by wMAPE ascending, then R2 descending
  - Select top n models based on spread parameter
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 6.1 Write property test for model selection count


  - **Property 9: Model selection respects spread parameter**
  - **Validates: Requirements 4.1, 4.3**

- [x] 6.2 Write property test for model ranking order


  - **Property 10: Model ranking order**
  - **Validates: Requirements 4.2**

- [x] 7. Implement ensemble builder





  - Create EnsembleBuilder class
  - Implement build_ensemble() with method parameter for "elastic" or "mars"
  - For "elastic": fit sklearn ElasticNet on constituent predictions
  - For "mars": fit py_earth.Earth on constituent predictions
  - Extract ensemble equation dictionary from fitted ensemble model
  - Return fitted ensemble model and equation dictionary
  - _Requirements: 5.1, 5.2, 6.1, 6.2_

- [x] 7.1 Write property test for ensemble predictions


  - **Property 11: Ensemble predictions for all test samples**
  - **Validates: Requirements 5.3, 6.3**

- [x] 8. Implement model serialization





  - Create ModelSerializer class
  - Implement save_constituent_models() to export model details to JSON
  - Implement save_ensemble_model() to export complete ensemble configuration
  - Implement load_model_json() to deserialize model from JSON file
  - Ensure JSON structure matches design specification
  - _Requirements: 3.4, 5.5, 6.5, 7.1_

- [x] 8.1 Write property test for serialization round-trip


  - **Property 8: Model serialization round-trip**
  - **Validates: Requirements 3.4, 5.5, 6.5, 7.1**

- [x] 9. Implement model scorer



  - Create ModelScorer class
  - Implement apply_equation_dict() to compute predictions from equation dictionary
  - Implement score_dataset() to load model JSON and generate predictions
  - Apply constituent equations first, then ensemble equation
  - Return DataFrame with IDs and predictions
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 9.1 Write property test for scoring equation application


  - **Property 13: Scoring applies constituent then ensemble equations**
  - **Validates: Requirements 7.2, 7.3**

- [x] 9.2 Write property test for prediction output structure


  - **Property 12: Prediction output structure**
  - **Validates: Requirements 5.4, 6.4, 7.4**

- [x] 10. Implement SQL exporter





  - Create SQLExporter class
  - Implement create_constituent_cte() to generate CTE for each Lasso model equation
  - Each CTE should compute: constant + (coef1 * feature1) + (coef2 * feature2) + ...
  - Implement create_ensemble_select() to generate final SELECT applying ensemble equation
  - Implement export_to_sql() to orchestrate full SQL generation from modeljson
  - Generate valid SQL compatible with PostgreSQL, MySQL, and SQL Server
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 10.1 Write property test for SQL structure validation


  - **Property 15: SQL export generates valid structure**
  - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

- [x] 10.2 Write property test for SQL CTE term count


  - **Property 16: SQL constituent CTE structure**
  - **Validates: Requirements 9.3**

- [x] 11. Implement main ELMRegressor estimator





  - Create ELMRegressor class inheriting from BaseEstimator and RegressorMixin
  - Implement __init__() with parameters: m, f, sample, method, spread, ELM2json, modeljson, random_state
  - Implement fit() method that orchestrates all components
  - Implement predict() method for generating predictions
  - Implement score_from_json() for scoring with saved models
  - Implement export_sql() method that delegates to SQLExporter
  - Store fitted attributes with trailing underscores (e.g., baseline_metrics_, selected_models_)
  - Add NotFittedError check in predict()
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 11.1 Write property test for fitted attributes naming


  - **Property 14: Fitted attributes have trailing underscores**
  - **Validates: Requirements 8.4**

- [x] 12. Add comprehensive error handling





  - Add input validation in ELMRegressor.__init__() for invalid parameters
  - Add shape validation in fit() for X_train, y_train, X_test, y_test
  - Add NaN and infinite value checks in fit()
  - Add feature mismatch validation in predict()
  - Add NotFittedError in predict() when called before fit()
  - Add file I/O error handling in serialization and SQL export methods
  - Add validation for SQL-safe feature names in SQLExporter
  - _Requirements: All requirements (error handling)_

- [x] 13. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Create package documentation





  - Write README.md with installation instructions, quick start guide, and basic examples
  - Create API documentation strings for all public methods
  - Add usage examples showing fit, predict, score_from_json, and export_sql workflows
  - Document all parameters and return values
  - Include SQL export example in documentation
  - _Requirements: 10.3_

- [x] 15. Create example notebooks





  - Create example Jupyter notebook demonstrating basic ELM usage
  - Show example with ElasticNet ensemble method
  - Show example with MARS ensemble method
  - Demonstrate model serialization and scoring workflow
  - Demonstrate SQL export and show how to use generated SQL in a database
  - Include visualization of model performance comparison
  - _Requirements: 10.3_

- [x] 16. Write integration tests




  - Test complete fit-predict workflow with synthetic data
  - Test both ElasticNet and MARS ensemble methods end-to-end
  - Test model serialization and loading round-trip
  - Test score_from_json() with saved model
  - Test SQL export generates valid, executable SQL
  - Test SQL output produces same predictions as Python scoring (within numerical tolerance)
  - Verify sklearn check_estimator compatibility
  - _Requirements: All requirements (integration testing)_

- [x] 17. Final checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.
