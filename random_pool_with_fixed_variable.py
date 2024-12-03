import pandas as pd
import numpy as np
from typing import Optional
from autora.variable import VariableCollection, ValueType

def random_pool_with_fixed_variable(
    variables: VariableCollection,
    fixed_var_name: str="is_switch",
    mode: str="switch_cost",  # or "asymmetric_cost"
    num_sample_groups: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
) -> pd.DataFrame:
    """
    Generate random experimental conditions, either for switch costs or asymmetric switch costs.
    
    Args:
        variables: Collection of experimental variables
        fixed_var_name: Name of the variable to fix (typically "is_switch")
        mode: "switch_cost" or "asymmetric_cost"
        num_sample_groups: Number of base condition groups to generate
        random_state: Random seed
        replace: Whether to sample with replacement
        
    Returns:
        DataFrame with experimental conditions
    """
    rng = np.random.default_rng(random_state)
    
    # Get non-fixed independent variables
    strength_variables = []
    raw_conditions = {}
    
    for iv in variables.independent_variables:
        if iv.name == fixed_var_name:
            continue
            
        strength_variables.append(iv.name)
        
        if iv.allowed_values is not None:
            raw_conditions[iv.name] = rng.choice(
                iv.allowed_values, 
                size=num_sample_groups, 
                replace=replace
            )
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(
                *iv.value_range, 
                size=num_sample_groups
            )
        else:
            raise ValueError(
                f"allowed_values or [value_range and type==REAL] needs to be set for {iv}"
            )
    
    # Verify we have exactly two strength variables
    assert len(strength_variables) == 2, "This function assumes exactly two strength variables"
    
    # For asymmetric costs, add flipped conditions
    if mode == "asymmetric_cost":
        # Store original conditions before modifications
        original_conditions = {
            name: values.copy() for name, values in raw_conditions.items()
        }
        
        for iv_name in strength_variables:
            # Get the other variable's ORIGINAL conditions
            other_iv = [v for v in strength_variables if v != iv_name][0]
            # Concatenate original and flipped conditions
            raw_conditions[iv_name] = np.concatenate([
                original_conditions[iv_name],
                original_conditions[other_iv]
            ])
    
    # Get fixed variable values
    fixed_var = next(iv for iv in variables.independent_variables 
                    if iv.name == fixed_var_name)
    if fixed_var.allowed_values is None:
        raise ValueError(
            f"allowed_values must be set for the fixed variable: {fixed_var_name}"
        )
    
    # Create final conditions
    results = []
    num_iterations = num_sample_groups * (2 if mode == "asymmetric_cost" else 1)
    
    for i in range(num_iterations):
        # For each base condition, create variants with all fixed variable values
        for fixed_value in fixed_var.allowed_values:
            row = {key: raw_conditions[key][i] for key in raw_conditions}
            row[fixed_var_name] = fixed_value
            results.append(row)
    
    return pd.DataFrame(results)