import pandas as pd
import numpy as np

def get_switch_costs_from_results(experiment_data: pd.DataFrame) -> np.ndarray:
    switch_costs = []
    for loop in range(len(experiment_data)//2):
        experiment_group = experiment_data.iloc[2*loop:2*loop+2]
        grouped = experiment_group.groupby(['cur_task_strength', 'alt_task_strength'])
        # Calculate switch cost: performance difference between switch and non-switch conditions
        switch_cost = grouped.apply(lambda x: x.loc[x['is_switch'] == 0, 'cur_task_performance'].values[0] - 
                                            x.loc[x['is_switch'] == 1, 'cur_task_performance'].values[0],
                                            include_groups=False)
        switch_costs.append(switch_cost.values[0])
    return np.array(switch_costs)