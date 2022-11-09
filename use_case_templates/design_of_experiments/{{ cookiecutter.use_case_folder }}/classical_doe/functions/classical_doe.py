import os

from datetime import datetime

import numpy as np
import pandas as pd

from classical_designs import get_design
from classical_designs import get_d_utility_for_polynomial_model

def get_doe_design(metrics, parameters, design_type, n_trials, sort_values, sort_ascending, data_dir, experiment_name):
    result_columns = [metric + suffix for metric in metrics for suffix in ("_mean", "_SEM")]

    design = get_design(parameters=parameters,
                        design_type=design_type, 
                        n_trials=n_trials,
                        sort_values=sort_values,
                        sort_ascending=sort_ascending,
                        metrics=result_columns)

    print("design:")
    display(design)
    print(f"type = {design_type}")
    print(f"n_trials = {len(design)}")

    data_file_name = os.path.join(data_dir,  f"data_{experiment_name}_running_trials.csv")
    print(f"the data will be stored in: {data_file_name}")

    if os.path.exists(data_file_name):
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.rename(data_file_name, os.path.join(data_dir,  f"data_{experiment_name}_running_trials_{dt}.csv"))

    design.to_csv(data_file_name)

    return design

def judge_design(design, design_type, mixed, order, parameters, metrics):
    if all(p["type"] == "range" for p in parameters):
        random_design = get_design(parameters, design_type="random", n_trials=len(design), metrics=metrics)
        d_utility = get_d_utility_for_polynomial_model(parameters, design, order=order, mixed=mixed)
        random_d_utility = get_d_utility_for_polynomial_model(parameters, random_design, order=order, mixed=mixed)

        print(f"d utility for a {'mixed' if mixed else 'simple'} polynomial model of order = {order}:")
        print(f"for this {design_type} design: {d_utility}")
        print(f"for a random design: {random_d_utility}")