import pandas as pd
import numpy as np
from typing import Optional
from autora.variable import VariableCollection, ValueType

def random_pool_with_fixed_variable(
    variables: VariableCollection,
    fixed_var_name: str="is_switch",
    num_sample_groups: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    
    # Sample the independent variables randomly, except for the fixed variable
    raw_conditions = {}
    for iv in variables.independent_variables:
        if iv.name == fixed_var_name:
            # Skip sampling for the fixed variable for now
            continue
        if iv.allowed_values is not None:
            raw_conditions[iv.name] = rng.choice(iv.allowed_values, size=num_sample_groups, replace=replace)
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(*iv.value_range, size=num_sample_groups)
        else:
            raise ValueError(f"allowed_values or [value_range and type==REAL] needs to be set for {iv}")

    # Get unique values for the fixed variable
    fixed_var = next(iv for iv in variables.independent_variables if iv.name == fixed_var_name)
    if fixed_var.allowed_values is None:
        raise ValueError(f"allowed_values must be set for the fixed variable: {fixed_var_name}")
    
    # Create rows by repeating the random samples for the fixed variable's values
    results = []
    for i in range(num_sample_groups):
        fixed_values = fixed_var.allowed_values  # List of possible values for the fixed variable
        for fixed_value in fixed_values:
            row = {key: raw_conditions[key][i] for key in raw_conditions}
            row[fixed_var_name] = fixed_value  # Set the fixed variable value
            results.append(row)
    
    return pd.DataFrame(results)