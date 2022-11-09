import os

from datetime import datetime

import numpy as np
import pandas as pd

import classical_designs

def get_design_data(observables, parameters, design_names):
    observable_names = [variable.name for variable in observables]

    design_datas = [classical_designs.get_design(parameters, design_name, metrics=observable_names)
                    for design_name in design_names]

    n_trials = [len(table) for table in design_datas]

    designs = [{parameter["variable"]: design_data[parameter["name"]].values for parameter in parameters}
            for design_data in design_datas]

    return observable_names, design_datas, n_trials, designs

def show_eig(design_names, n_trials, eigs):
    design_stats = pd.DataFrame(index=pd.Index((), name="index"))
    design_stats["name"] = design_names
    design_stats["n_trials"] = n_trials
    design_stats["EIG"] = eigs
    design_stats["EIG_per_trial"] = design_stats["EIG"] / design_stats["n_trials"]

    display(design_stats)

    i_largest_eig = design_stats["EIG"].argmax()
    i_largest_eig_per_trial = design_stats["EIG_per_trial"].argmax()

    largest_eig_stats = pd.Series((i_largest_eig, ), index=("index",)).append(design_stats.loc[i_largest_eig])
    largest_eig_per_trial_stats = pd.Series((i_largest_eig_per_trial, ), index=("index",)).append(design_stats.loc[i_largest_eig_per_trial])

    print("\ndesign with largest EIG:")
    display(largest_eig_stats)
    
    print("\ndesign with largest EIG per trial:")
    display(largest_eig_per_trial_stats)

def show_design(data_dir, design_data, experiment_name):
    display(design_data)

    data_file_name = os.path.join(data_dir,  f"data_{experiment_name}_running_trials.csv")
    print(f"the data will be stored in: {data_file_name}")

    if os.path.exists(data_file_name):
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.rename(data_file_name, os.path.join(data_dir,  f"data_{experiment_name}_running_trials_{dt}.csv"))

    design_data.to_csv(data_file_name)

def show_experiment_results(data_file_name, design_variables, observables, parameters, observable_names):
    data = pd.read_csv(data_file_name, index_col="index")

    display(data)

    variables = design_variables + observables
    data_for_variables = [data[parameter["name"]] for parameter in parameters] + [data[name] for name in observable_names]
    data_for_fit = {variable: data_for_variable.values for variable, data_for_variable in zip(variables, data_for_variables)}

    return data_for_fit

# <halerium id="4e65ea3a-27fe-45fa-aa19-32a3eb814c54">
def show_model_results(means, stds):
# </halerium id="4e65ea3a-27fe-45fa-aa19-32a3eb814c54">
    print("target means:")
    for name, value in means.items():
        print(f"\n{name}:\n{value}")
        
    print("\ntarget standard deviations:\n(note that some models don't compute standard deviations)")
    for name, value in stds.items():
        print(f"\n{name}:\n{value}")   