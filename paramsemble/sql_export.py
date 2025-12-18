"""SQL export module."""

import json
import re


class SQLExporter:
    """Exports ensemble models as SQL code for database deployment."""
    
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
        # Validate inputs
        if not modeljson_path:
            raise ValueError("modeljson_path must be provided")
        if not table_name:
            raise ValueError("table_name must be provided")
        if not id_column:
            raise ValueError("id_column must be provided")
        
        # Validate SQL-safe names
        if not self._is_sql_safe(table_name):
            raise ValueError(
                f"table_name '{table_name}' contains invalid characters. "
                f"Only alphanumeric characters and underscores are allowed."
            )
        if not self._is_sql_safe(id_column):
            raise ValueError(
                f"id_column '{id_column}' contains invalid characters. "
                f"Only alphanumeric characters and underscores are allowed."
            )
        
        # Load model JSON
        try:
            with open(modeljson_path, 'r') as f:
                model_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: '{modeljson_path}'")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file '{modeljson_path}': {e}")
        except IOError as e:
            raise IOError(f"Failed to read file '{modeljson_path}': {e}")
        
        # Validate model data structure
        if 'constituent_models' not in model_data:
            raise ValueError("Model JSON missing 'constituent_models' key")
        if 'ensemble_equation' not in model_data:
            raise ValueError("Model JSON missing 'ensemble_equation' key")
        
        constituent_models = model_data['constituent_models']
        ensemble_equation = model_data['ensemble_equation']
        
        # Validate constituent models
        if not constituent_models:
            raise ValueError("Model JSON has no constituent models")
        
        # Validate feature names are SQL-safe
        for model_info in constituent_models:
            if 'features' not in model_info:
                raise ValueError("Constituent model missing 'features' key")
            for feature in model_info['features']:
                if not self._is_sql_safe(feature):
                    raise ValueError(
                        f"Feature name '{feature}' contains invalid characters. "
                        f"Only alphanumeric characters and underscores are allowed."
                    )
        
        # Generate CTEs for each constituent model
        cte_list = []
        cte_names = []
        
        for i, model_info in enumerate(constituent_models):
            cte_name = f'model_{i}'
            cte_names.append(cte_name)
            cte_sql = self.create_constituent_cte(model_info, cte_name, table_name)
            cte_list.append(cte_sql)
        
        # Create ensemble inputs CTE that joins all constituent predictions
        ensemble_inputs_cte = self._create_ensemble_inputs_cte(cte_names, id_column)
        cte_list.append(ensemble_inputs_cte)
        
        # Create final SELECT statement
        final_select = self.create_ensemble_select(ensemble_equation, cte_names, id_column)
        
        # Combine all parts into complete SQL
        sql_code = "WITH\n"
        sql_code += ",\n".join(cte_list)
        sql_code += "\n" + final_select
        
        return sql_code
    
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
        equation_dict = model_info['equation_dict']
        
        # Start with the constant term
        constant = equation_dict['constant']
        
        # Build the calculation expression: constant + (coef1 * feature1) + ...
        terms = [str(constant)]
        
        # Add terms for each feature
        for key, value in equation_dict.items():
            if key != 'constant':
                # Format: (coefficient * feature_name)
                terms.append(f"({value} * {key})")
        
        # Combine all terms with addition
        calculation = " + ".join(terms)
        
        # Create the CTE
        cte_sql = f"  {cte_name} AS (\n"
        cte_sql += f"    SELECT\n"
        cte_sql += f"      id,\n"
        cte_sql += f"      {calculation} AS prediction\n"
        cte_sql += f"    FROM {table_name}\n"
        cte_sql += f"  )"
        
        return cte_sql
    
    def _create_ensemble_inputs_cte(self, cte_names, id_column):
        """
        Create CTE that joins all constituent predictions.
        
        Parameters
        ----------
        cte_names : list of str
            Names of constituent CTEs
        id_column : str
            ID column name
            
        Returns
        -------
        cte_sql : str
            Ensemble inputs CTE definition
        """
        # Build SELECT clause with all model predictions
        select_parts = [f"      m0.{id_column}"]
        for i, cte_name in enumerate(cte_names):
            select_parts.append(f"      m{i}.prediction AS {cte_name}_pred")
        
        # Build FROM and JOIN clauses
        from_clause = f"    FROM {cte_names[0]} m0"
        join_clauses = []
        for i in range(1, len(cte_names)):
            join_clauses.append(
                f"    JOIN {cte_names[i]} m{i} ON m0.{id_column} = m{i}.{id_column}"
            )
        
        # Combine into CTE
        cte_sql = "  ensemble_inputs AS (\n"
        cte_sql += "    SELECT\n"
        cte_sql += ",\n".join(select_parts) + "\n"
        cte_sql += from_clause
        if join_clauses:
            cte_sql += "\n" + "\n".join(join_clauses)
        cte_sql += "\n  )"
        
        return cte_sql
    
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
        # Start with the constant term
        constant = ensemble_equation['constant']
        
        # Build the calculation expression
        terms = [str(constant)]
        
        # Add terms for each constituent model prediction
        for cte_name in constituent_ctes:
            coef = ensemble_equation[cte_name]
            # Format: (coefficient * model_X_pred)
            terms.append(f"({coef} * {cte_name}_pred)")
        
        # Combine all terms with addition
        calculation = " + ".join(terms)
        
        # Create the final SELECT
        select_sql = "SELECT\n"
        select_sql += f"  {id_column},\n"
        select_sql += f"  {calculation} AS predicted\n"
        select_sql += "FROM ensemble_inputs"
        
        return select_sql
    
    def _is_sql_safe(self, name):
        """
        Check if a name is SQL-safe (alphanumeric and underscores only).
        
        Parameters
        ----------
        name : str
            Name to validate
            
        Returns
        -------
        is_safe : bool
            True if name is SQL-safe, False otherwise
        """
        if not name:
            return False
        # Allow alphanumeric characters and underscores only
        # Must not start with a number
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, name))
