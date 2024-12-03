import pandas as pd
import numpy as np
from autora.variable import VariableCollection
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize, LinearConstraint
from utils import calculate_switch_costs, get_asymmetric_switch_costs_from_switch_costs

condition_names = ['cur_task_strength', 'alt_task_strength']

def regressor_experimentalist(
    variables: VariableCollection,
    conditions: pd.DataFrame,
    experiment_data: pd.DataFrame,
    fixed_var_name: str="is_switch",
    mode: str="switch_cost",  # or "asymmetric_cost"
) -> pd.DataFrame:
    if len(experiment_data) == 0:
        return conditions
    if mode == "switch_cost":
        # Original switch cost calculation
        costs_df = calculate_switch_costs(experiment_data)
        condition_cols = ['cur_task_strength', 'alt_task_strength']
        cost_col = 'switch_cost'
    else:  # mode == "asymmetric_cost"
        # Calculate asymmetric costs
        switch_costs = calculate_switch_costs(experiment_data)
        costs_df = get_asymmetric_switch_costs_from_switch_costs(switch_costs)
        condition_cols = ['strong_task_strength', 'weak_task_strength']
        cost_col = 'asymmetric_switch_cost'
    
    # Fit Random Forest Regressor
    X = costs_df[condition_cols].values
    y = costs_df[cost_col].values
    model = RandomForestRegressor(n_estimators=20)
    model.fit(X, y)

    # Step 2: Define Objective Function for Optimization
    def objective_function(x, model):
        x = np.clip(x, 0, 1)  # Ensure x is within bounds
        return -model.predict(x.reshape(1, -1))[0]  # Negative because we minimize

    # Step 3: Optimize to Find Conditions that Maximize Switch Cost
    bounds = [(0, 1), (0, 1)]
    # Original optimization for switch costs
    res = minimize(objective_function, 
                x0=[0.5, 0.5], 
                bounds=bounds, 
                args=(model,),
                method="Powell")

    optimized_conditions = res.x
    # perturb the conditions if they are the same and we're in asymmetric mode
    if optimized_conditions[0] == optimized_conditions[1]:
        if mode == "asymmetric_cost":
            perturbation = np.random.uniform(low=-0.1, high=0.1)
            index_to_perturb = np.random.choice([0, 1])
            optimized_conditions[index_to_perturb] += perturbation
            optimized_conditions[index_to_perturb] = np.clip(optimized_conditions[index_to_perturb], 0.0, 1.0)

    # Get unique values for the fixed variable
    fixed_var = next(iv for iv in variables.independent_variables if iv.name == fixed_var_name)
    fixed_values = fixed_var.allowed_values
    
    # Create rows by repeating the random samples for the fixed variable's values
    results = []
    fixed_values = fixed_var.allowed_values  # List of possible values for the fixed variable
    if mode == "switch_cost":
        # Original condition generation
        for fixed_value in fixed_values:
            row = {
                'cur_task_strength': optimized_conditions[0],
                'alt_task_strength': optimized_conditions[1],
                fixed_var_name: fixed_value
            }
            results.append(row)
    else:  # mode == "asymmetric_cost"
        # Generate both original and flipped conditions
        strong = optimized_conditions[0]
        weak = optimized_conditions[1]
        assert strong != weak, f"Strong and weak task strengths must be different, got {strong} and {weak}"
        
        # Create all four conditions in the correct order
        for strengths in [(strong, weak), (weak, strong)]:  # Two orientations
            for fixed_value in fixed_values:  # Both switch values for each orientation
                results.append({
                    'strong_task_strength': strengths[0],
                    'weak_task_strength': strengths[1],
                    fixed_var_name: fixed_value
                })
    return pd.DataFrame(results)