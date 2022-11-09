import os

from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ax.service.ax_client import AxClient
from ax import RangeParameter, ChoiceParameter
from ax.exceptions.core import DataRequiredError, SearchSpaceExhausted
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.core.base_trial import TrialStatus
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models, ModelRegistryBase

import ax.plot as ax_plot

plt.style.use("dark_background")

def read_existing_data(parameters, metrics, data_file_name):
    parameter_columns = [parameter["name"] for parameter in parameters ] 
    result_columns    = [metric + suffix for metric in metrics for suffix in ("_mean", "_SEM")]
    data_columns      = parameter_columns + result_columns

    n_trials = 0
    n_completed_trials = 0
    n_outstanding_trials = 0
    prior_data = None

    if os.path.exists(data_file_name):
        print(f"reading prior data from {data_file_name}...")
        prior_data = pd.read_csv(data_file_name, index_col="index")

        missing_colums = set(data_columns) - set(prior_data.columns)
        if missing_colums:
            raise ValueError(f"data file missing colums: {missing_colums}.")
        prior_data = prior_data[data_columns]   

        n_trials = len(prior_data[parameter_columns].dropna(axis='index', how='any'))
        n_completed_trials = len(prior_data.dropna(axis='index', how='any'))
        n_outstanding_trials = n_trials - n_completed_trials

    else:
        print("no prior data.")

    return n_trials, n_completed_trials, n_outstanding_trials, prior_data, parameter_columns, result_columns, data_columns

def set_up_client(experiment_name, parameters, objective_name, minimize, parameter_constraints, outcome_constraints, max_batch_size, always_max_batch_size, n_trials, initial_n_trials):
    generation_strategy_steps=[
            # 1. Initialization step (does not require pre-existing data and is well-suited for 
            # initial sampling of the search space)
            GenerationStep(
                model=Models.SOBOL,
                num_trials=max(max_batch_size, initial_n_trials) if always_max_batch_size else initial_n_trials,  # How many trials should be produced from this generation step
                min_trials_observed=3, # How many trials need to be completed to move to next model
                max_parallelism=max(max_batch_size, 5) if always_max_batch_size else 5,  # Max parallelism for this step
                model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
                model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
            ),
            # 2. Bayesian optimization step (requires data obtained from previous phase and learns
            # from all data available at the time of each new candidate generation call)
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                max_parallelism=max(3, max_batch_size) if always_max_batch_size else 3,  # Parallelism limit for this step, often lower than for Sobol
            ),
        ]


    if n_trials >= initial_n_trials:
        generation_strategy = GenerationStrategy(generation_strategy_steps[1:])
    else:
        generation_strategy = GenerationStrategy(generation_strategy_steps)


    ax_client = AxClient(
        generation_strategy=generation_strategy,
        enforce_sequential_optimization=not always_max_batch_size, )

    ax_client.create_experiment(
        name=experiment_name,
        parameters=parameters,
        objective_name=objective_name,
        minimize=minimize,
        parameter_constraints=parameter_constraints,
        outcome_constraints=outcome_constraints,
    )

    return generation_strategy, ax_client

def feed_data_to_client(prior_data, parameter_columns, result_columns, ax_client, metrics):
    prior_trials = dict()
    if prior_data is not None:
        for index, trial_data in prior_data.iterrows():

            trial_parameters = trial_data[parameter_columns]
            if any(trial_parameters.isna()):
                missing_trial_parameters = ", ".join(trial_parameters[trial_parameters.isna()].index)
                print(f"row {index}: missing parameter values for: {missing_trial_parameters}.")
                continue

            trial_parameters = trial_parameters.to_dict()
            trial_parameters, trial_index = ax_client.attach_trial(parameters=trial_parameters)

            trial_results = trial_data[result_columns]
            if any(trial_results.isna()):
                missing_results = ", ".join(trial_results[trial_results.isna()].index)
                print(f"row {index}: outstanding results for: {missing_results}.")
            else:
                raw_data = dict()
                for metric in metrics:
                    metric_mean = trial_results[metric + "_mean"]
                    metric_SEM  = trial_results[metric + "_SEM"]
                    raw_data[metric] = (metric_mean, metric_SEM)
                ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

            trial_results = trial_results.to_dict()
            prior_trials[trial_index] = {**trial_parameters, **trial_results}

    return prior_trials

def suggest_new_trials(experiment_name, n_outstanding_trials, suggest_new_trials, suggest_when_outstanding, max_batch_size, always_max_batch_size, ax_client, result_columns, data_columns, prior_trials, data_file_name, data_dir):
    if (n_outstanding_trials > 0) and not suggest_when_outstanding:
        print(f"There are {n_outstanding_trials} outstanding trials. Will not suggest new trials.")
        suggest_new_trials = False


    if suggest_new_trials and max_batch_size > 0:

        new_trials = dict()
        exhausted = False

        try:
            if always_max_batch_size:
                for _ in range(max_batch_size):
                    trial_parameters, trial_index = ax_client.get_next_trial()
                    trial_results = {c: None for c in result_columns}
                    trial_data = {**trial_parameters, **trial_results}
                    new_trials[trial_index] = trial_data
            else:
                # workaround (get_next_trials won't generate new trials sometimes if loaded from disk unless get_next_trial was called)
                trial_parameters, trial_index = ax_client.get_next_trial()
                trial_results = {c: None for c in result_columns}
                trial_data = {**trial_parameters, **trial_results}
                new_trials[trial_index] = trial_data
                if max_batch_size > 1:
                    more_trials, exhausted = ax_client.get_next_trials(max_trials=max_batch_size - 1)
                    new_trials.update(more_trials)
        except (DataRequiredError, SearchSpaceExhausted, MaxParallelismReachedException) as exception:
            print(f"no more trials because {type(exception).__name__}: {exception}")
            pass
            
        _, exhausted = ax_client.get_current_trial_generation_limit()

        batch_size = len(new_trials)

        if (batch_size <= 0) and exhausted:
            print("exhausted the search. no more trials to suggest.")

        elif batch_size <= 0:
            print("no new trials to suggest. maybe you have to complete outstanding trials first.")

        else:
            print(f"got {batch_size} new trials.")

            if os.path.exists(data_file_name):
                dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                os.rename(data_file_name, os.path.join(data_dir,  f"data_{experiment_name}_running_trials_{dt}.csv"))

            data = {**prior_trials, **new_trials}
            data = pd.DataFrame.from_dict(data, orient='index')
            data = data[data_columns]
            data.index.name = "index"
            data.to_csv(data_file_name)

def estimate_best_params(ax_client, best_parameters_file_name, data_dir, experiment_name, parameter_columns, metrics, n_completed_trials):
    if os.path.exists(best_parameters_file_name):
        prior_best_parameters_data = pd.read_csv(best_parameters_file_name) 
    else:
        prior_best_parameters_data = pd.DataFrame(columns=["n_completed_trials"] + parameter_columns + metrics)


    best_parameters_result = ax_client.get_best_parameters()
    if best_parameters_result is None:
        best_parameters = None
        means = None
        covariances = None
        new_best_parameters_data = pd.DataFrame(columns=["n_completed_trials"] + parameter_columns + metrics)
    else:
        best_parameters, (means, covariances) = best_parameters_result
        new_best_parameters_data = pd.DataFrame.from_records(({
            "n_completed_trials": n_completed_trials,
            **best_parameters, **means
        },))


    best_parameters_data = prior_best_parameters_data.append(new_best_parameters_data)
    if os.path.exists(best_parameters_file_name):
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.rename(best_parameters_file_name, os.path.join(data_dir,  f"data_{experiment_name}_best_parameters_{dt}.csv"))
    best_parameters_data.to_csv(best_parameters_file_name, index=False)

    if len(best_parameters_data) > 0:
        print("\nbest parameters so far (from oldest to most recent):")
        display(best_parameters_data)
    else:
        print("no best parameters yet.")

    return best_parameters_data

def show_results(ax_client, minimize, objective_name, best_parameters_data):
    experiment = ax_client.experiment

    ob_trials =  {i: t.objective_mean for i,t in experiment.trials.items() if t.completed_successfully}
    ob_trials = [ob_trials[i] for i in sorted(ob_trials.keys())]

    if minimize:
        best_ob_trials = np.minimum.accumulate(ob_trials)
    else:
        best_ob_trials = np.maximum.accumulate(ob_trials)

    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))


    ax = axs[0]
    ax.plot(ob_trials, '.r');
    ax.set_xlabel('trial number')
    ax.set_ylabel(objective_name)


    ax = axs[1]
    ax.plot(best_parameters_data["n_completed_trials"], best_parameters_data[objective_name], '.r');
    ax.set_xlabel('number of completed trials');
    ax.set_ylabel('best estimated ' + objective_name);
    xmax = best_parameters_data["n_completed_trials"].max() if (len(best_parameters_data["n_completed_trials"]) > 0) else 0.
    ax.set_xlim(-0.5, xmax + 0.5);
    plt.show()