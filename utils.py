import pandas as pd
import numpy as np

def get_switch_costs_from_ordered_results(experiment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate switch costs from data where every two consecutive rows form a group
    that differs only by is_switch value.
    
    Args:
        experiment_data: DataFrame with rows ordered in switch/non-switch pairs
        
    Returns:
        DataFrame with columns ['switch_cost', 'cur_task_strength', 'alt_task_strength']
    """
    assert len(experiment_data) % 2 == 0, "Data must contain even number of rows"
    results = []
    
    for i in range(0, len(experiment_data), 2):
        pair = experiment_data.iloc[i:i+2]
        assert len(pair['is_switch'].unique()) == 2, f"Rows {i} and {i+1} must differ in is_switch"
        
        # Calculate switch cost
        switch_cost = (pair.loc[pair['is_switch'] == 0, 'cur_task_performance'].iloc[0] -
                      pair.loc[pair['is_switch'] == 1, 'cur_task_performance'].iloc[0])
        
        # Get task strengths from first row of pair (they're the same in both rows)
        results.append({
            'switch_cost': switch_cost,
            'cur_task_strength': pair.iloc[0]['cur_task_strength'],
            'alt_task_strength': pair.iloc[0]['alt_task_strength']
        })
    
    return pd.DataFrame(results)

def calculate_switch_costs(experiment_data: pd.DataFrame, 
                         group_columns=['cur_task_strength', 'alt_task_strength']) -> pd.DataFrame:
    """
    Calculate switch costs for each unique combination of conditions in group_columns.
    
    Args:
        experiment_data: DataFrame with columns for conditions and 'cur_task_performance'
        group_columns: List of column names to group by (default: task strength columns)
    
    Returns:
        DataFrame with group_columns and 'switch_cost' column
    """
    switch_costs_df = (experiment_data
        .groupby(group_columns)
        .apply(lambda x: x.loc[x['is_switch'] == 0, 'cur_task_performance'].mean() -
                        x.loc[x['is_switch'] == 1, 'cur_task_performance'].mean())
        .reset_index(name='switch_cost'))
    
    return switch_costs_df

def get_asymmetric_switch_costs_from_switch_costs(switch_costs_df):
    results = []
    for _, row in switch_costs_df.iterrows():
        # Find the flipped condition
        flipped_condition = switch_costs_df[
            (switch_costs_df['cur_task_strength'] == row['alt_task_strength']) & 
            (switch_costs_df['alt_task_strength'] == row['cur_task_strength'])
        ]
        
        if not flipped_condition.empty and row['cur_task_strength'] > row['alt_task_strength']:
            strong_to_weak = row['switch_cost']
            weak_to_strong = flipped_condition.iloc[0]['switch_cost']
            
            results.append({
                'strong_task_strength': row['cur_task_strength'],
                'weak_task_strength': row['alt_task_strength'],
                'asymmetric_switch_cost': strong_to_weak - weak_to_strong
            })
    
    return pd.DataFrame(results)