import pandas as pd
import numpy as np
from autora.variable import VariableCollection
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

condition_names = ['cur_task_strength', 'alt_task_strength']

def regressor_experimentalist(
    variables: VariableCollection,
    conditions: pd.DataFrame,
    experiment_data: pd.DataFrame,
    fixed_var_name: str="is_switch"
) -> pd.DataFrame:
    if len(experiment_data) == 0:
        return conditions
    switch_costs_df = experiment_data.groupby(condition_names).apply(
        lambda x: x.loc[x['is_switch'] == 0, 'cur_task_performance'].values[0] - 
                x.loc[x['is_switch'] == 1, 'cur_task_performance'].values[0],
                include_groups=False
        ).reset_index(name='switch_cost')
    # Step 1: Fit Random Forest Regressor
    X = switch_costs_df[condition_names].values
    y = switch_costs_df['switch_cost'].values
    model = RandomForestRegressor(n_estimators=20)
    model.fit(X, y)

    # Step 2: Define Objective Function for Optimization
    def objective_function(x, model):
        x = np.clip(x, 0, 1)  # Ensure x is within bounds
        return -model.predict(x.reshape(1, -1))[0]  # Negative because we minimize

    # Step 3: Optimize to Find Conditions that Maximize Switch Cost
    bounds = [(0, 1), (0, 1)]
    res = minimize(objective_function, x0=[0.5, 0.5], bounds=bounds, args=(model,), method="Powell")
    optimized_conditions = res.x  # Proposed cur_task_strength and alt_task_strength

    # Get unique values for the fixed variable
    fixed_var = next(iv for iv in variables.independent_variables if iv.name == fixed_var_name)
    if fixed_var.allowed_values is None:
        raise ValueError(f"allowed_values must be set for the fixed variable: {fixed_var_name}")
    
    # Create rows by repeating the random samples for the fixed variable's values
    results = []
    fixed_values = fixed_var.allowed_values  # List of possible values for the fixed variable
    for fixed_value in fixed_values:
        row = {condition_names[i] : optimized_conditions[i] for i in range(len(condition_names))}
        row[fixed_var_name] = fixed_value  # Set the fixed variable value
        results.append(row)
    
    return pd.DataFrame(results)