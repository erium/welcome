from copy import deepcopy

import numpy as np
import pandas as pd


def _values_from_bounds(lower_bound, upper_bound, n_values):
    if n_values < 1:
        return []
    elif n_values == 1:
        return [0.5 * (lower_bound + upper_bound)]
    else:
        return np.linspace(lower_bound, upper_bound, n_values)


def _values_for_parameter(p):
    """Get values for parameter.

    Parameters
    ----------
    p :
        The parameter.

    Returns
    -------
    values :
        The values.

    """
    recognized_types = {"range", "choice"}

    ptype = p["type"]
    if ptype == "choice":
        values = p["values"]
    elif ptype == "range":
        lower_bound, upper_bound = p["bounds"]
        n_values = p["n_values"]
        values = _values_from_bounds(lower_bound, upper_bound, n_values)
    else:
        raise ValueError(f"Unknown type '{ptype}' for parameter {p}. "
                         f"Recognized types are: {', '.join(recognized_types)}.")

    return values


def _design_from_matrix(design_matrix, names, values,
                        sort_values=False, sort_ascending=True,
                        metrics=None):
    """Make design from matrix.

    Parameters
    ----------
    design_matrix :
        The design matrix.
    names :
        The names of parameters.
    values :
        The values for the parameters.
    sort_values : bool, list, optional
        Whether to sort the trials by (a subset) of their parameter values.
    sort_ascending : bool, optional
        When sorting, whether to sort in ascending order.
    metrics :
        A list of metrics to add as empty columns to the output
        (for you to fill in the outcomes of the trials).

    Returns
    -------
    design :
        The design.

    """
    n_parameters = len(names)
    design = [[values[c][row[c]] for c in range(n_parameters)] for row in design_matrix]
    design = pd.DataFrame(design, columns=names)
    design.index.name = "index"

    if sort_values is True:
        sort_values = names
    if sort_values:
        design.sort_values(by=sort_values, ascending=sort_ascending,
                           ignore_index=True, inplace=True)

    if metrics is not None:
        metrics = pd.DataFrame(columns=metrics)
        design = design.join(metrics)

    return design


def get_central_design_matrix(n_parameters, ):
    """Get the design matrix for a central design.

    Parameters
    ----------
    n_parameters : int
        The number of parameters.

    Returns
    -------
    design_matrix : np.array
        The design matrix.

    """
    n_trials = 2 * n_parameters + 1
    design_matrix = np.ones((n_trials, n_parameters), dtype=int)
    for i_parameter in range(n_parameters):
        i_trial = 2 * i_parameter + 1
        design_matrix[i_trial, i_parameter] = 0
        i_trial = 2 * i_parameter + 2
        design_matrix[i_trial, i_parameter] = 2

    return design_matrix


def get_mixed_second_order_design_matrix(n_parameters, ):
    """Get the design matrix for a mixed second order design.

    Parameters
    ----------
    n_parameters : int
        The number of parameters.

    Returns
    -------
    design_matrix : np.array
        The design matrix.

    """
    n_trials = 2 * n_parameters * (n_parameters - 1)
    design_matrix = np.ones((n_trials, n_parameters), dtype=int)
    i_trial = 0
    for i_parameter in range(n_parameters):
        for j_parameter in range(i_parameter):
            design_matrix[i_trial, i_parameter] = 0
            design_matrix[i_trial, j_parameter] = 0
            i_trial += 1
            design_matrix[i_trial, i_parameter] = 0
            design_matrix[i_trial, j_parameter] = 2
            i_trial += 1
            design_matrix[i_trial, i_parameter] = 2
            design_matrix[i_trial, j_parameter] = 0
            i_trial += 1
            design_matrix[i_trial, i_parameter] = 2
            design_matrix[i_trial, j_parameter] = 2
            i_trial += 1

    return design_matrix


def get_central_composite_design_matrix(n_parameters,
                                        design_type="simple_central"):
    """Get the design matrix for a central composite design.

    Parameters
    ----------
    n_parameters : int
        The number of parameters.
    design_type : str
        The type of design:
         - "simple_central": just a central design
         - "mixed_central": central + mixed second order design
         - "full_central": central + full 2-level factorial design

    Returns
    -------
    design_matrix : np.array
        The design matrix.

    """
    recognized_design_types = {"simple_central", "mixed_central", "full_central"}

    if design_type == "simple_central":
        design_matrix = get_central_design_matrix(n_parameters)
    elif design_type == "mixed_central":
        scd_matrix = get_central_design_matrix(n_parameters)
        mso_matrix = get_mixed_second_order_design_matrix(n_parameters)
        design_matrix = np.concatenate([scd_matrix, mso_matrix])
    elif design_type == "full_central":
        scd_matrix = get_central_design_matrix(n_parameters)
        n_values = np.full((n_parameters,), fill_value=2)
        ffd_matrix = get_full_factorial_design_matrix(n_values) * 2
        design_matrix = np.concatenate([scd_matrix, ffd_matrix])
    else:
        raise ValueError(f"Unrecognized design_type={design_type}. Note, "
                         f"recognized values are: {', '.join(recognized_design_types)}.")

    return design_matrix


def get_full_factorial_design_matrix(n_values):
    """Get the design matrix for a full factorial design.

    Parameters
    ----------
    n_values :
        The list of number of values for each parameter.

    Returns
    -------
    design_matrix : np.array
        The design matrix.

    """
    n_trials = np.prod(n_values)
    stride = n_trials // np.cumprod(n_values)
    design_matrix = np.stack(tuple((i // stride) % n_values for i in range(n_trials)))
    return design_matrix


def get_random_factorial_design_matrix(n_values, n_trials):
    """Get the design matrix for a random factorial design.

    Parameters
    ----------
    n_values :
        The list of number of values for each parameter.
    n_trials : int
        The number of trials.

    Returns
    -------
    design_matrix : np.array
        The design matrix.

    """
    ffd_matrix = get_full_factorial_design_matrix(n_values)
    n_different_trials = len(ffd_matrix)

    if n_trials < n_different_trials:
        picked_trials = np.random.choice(n_different_trials, size=n_trials, replace=False)
        design_matrix = ffd_matrix[picked_trials]
    elif n_trials == n_different_trials:
        design_matrix = ffd_matrix
    else:
        picked_trials = np.random.choice(n_different_trials, size=n_trials - n_different_trials, replace=True)
        design_matrix = np.concatenate([ffd_matrix, ffd_matrix[picked_trials]])

    return design_matrix


def get_central_composite_design(parameters, design_type="simple",
                                 sort_values=False, sort_ascending=True,
                                 metrics=None):
    """Get central composite design.

    Get a table of parameter values for the trials in a central composite design.

    Parameters
    ----------
    parameters :
        A list of parameters, where each list item is a dictionary with keys
         - "name": the name of the parameter,
         - "type": the type, either "range" (continuous parameter), or "choice",
        If the type is "choice",
         - "values" : a list of values to try.
        If type is "range":
         - "bounds": the lower and upper bound as tuple
         - "n_values": the number of value to try.
    design_type : str
        The type of design:
         - "simple": just a central design
         - "mixed": central + mixed second order design
         - "full": central + full factorial design
    sort_values : bool, list, optional
        Whether to sort the trials by their parameter values.
        If a list of parameter names is given, the trials are sorted by the
        values of these parameters.
    sort_ascending : bool, optional
        When sorting, whether to sort in ascending order.
    metrics :
        A list of metrics to add as empty columns to the output
        (for you to fill in the outcomes of the trials).

    Returns
    -------
    design : pd.DataFrame
        The table with the parameter values for the trials.

    """
    n_parameters = len(parameters)
    names = [p["name"] for p in parameters]
    values = [_values_for_parameter(p) for p in parameters]

    for name, v in zip(names, values):
        if len(v) != 3:
            raise ValueError(
                f"Parameter {name} has number of values = {len(v)} != 3. "
                f"Note: the central composite design requires 3 values "
                f"for each parameter.")

    design_matrix = get_central_composite_design_matrix(n_parameters, design_type)
    design = _design_from_matrix(design_matrix=design_matrix,
                                 names=names,
                                 values=values,
                                 sort_values=sort_values,
                                 sort_ascending=sort_ascending,
                                 metrics=metrics)
    return design


def get_full_factorial_design(parameters,
                              sort_values=False, sort_ascending=True,
                              metrics=None):
    """Get full factorial design.

    Get a table of parameter values for the trials in a full factorial design.

    Parameters
    ----------
    parameters :
        A list of parameters, where each list item is a dictionary with keys
         - "name": the name of the parameter,
         - "type": the type, either "range" (continuous parameter), or "choice",
        If the type is "choice",
         - "values" : a list of values to try.
        If type is "range":
         - "bounds": the lower and upper bound as tuple
         - "n_values": the number of value to try.
    sort_values : bool, list, optional
        Whether to sort the trials by their parameter values.
        If a list of parameter names is given, the trials are sorted by the
        values of these parameters.
    sort_ascending : bool, optional
        When sorting, whether to sort in ascending order.
    metrics :
        A list of metrics to add as empty columns to the output
        (for you to fill in the outcomes of the trials).

    Returns
    -------
    design : pd.DataFrame
        The table with the parameter values for the trials.

    """
    names = [p["name"] for p in parameters]
    values = [_values_for_parameter(p) for p in parameters]
    n_values = [len(v) for v in values]

    design_matrix = get_full_factorial_design_matrix(n_values)
    design = _design_from_matrix(design_matrix=design_matrix,
                                 names=names,
                                 values=values,
                                 sort_values=sort_values,
                                 sort_ascending=sort_ascending,
                                 metrics=metrics)
    return design


def get_random_factorial_design(parameters, n_trials,
                                sort_values=False, sort_ascending=True,
                                metrics=None):
    """Get a random factorial design.

    Get a table of parameter values for the trials in a random factorial design.

    Parameters
    ----------
    parameters :
        A list of parameters, where each list item is a dictionary with keys
         - "name": the name of the parameter,
         - "type": the type, either "range" (continuous parameter), or "choice",
        If the type is "choice",
         - "values" : a list of values to try.
        If type is "range":
         - "bounds": the lower and upper bound as tuple
         - "n_values": the number of value to try.
    n_trials : int
        The number of trials.
    sort_values : bool, list, optional
        Whether to sort the trials by their parameter values.
        If a list of parameter names is given, the trials are sorted by the
        values of these parameters.
    sort_ascending : bool, optional
        When sorting, whether to sort in ascending order.
    metrics :
        A list of metrics to add as empty columns to the output
        (for you to fill in the outcomes of the trials).

    Returns
    -------
    design : pd.DataFrame
        The table with the parameter values for the trials.

    """
    names = [p["name"] for p in parameters]
    values = [_values_for_parameter(p) for p in parameters]
    n_values = [len(v) for v in values]

    design_matrix = get_random_factorial_design_matrix(n_values, n_trials)
    design = _design_from_matrix(design_matrix=design_matrix,
                                 names=names,
                                 values=values,
                                 sort_values=sort_values,
                                 sort_ascending=sort_ascending,
                                 metrics=metrics)
    return design


def get_random_design(parameters, n_trials,
                      sort_values=False, sort_ascending=True,
                      metrics=None):
    """Get a random design.

    Get a table of parameter values for the trials in a random design.

    Parameters
    ----------
    parameters :
        A list of parameters, where each list item is a dictionary with keys
         - "name": the name of the parameter,
         - "type": the type, either "range" (continuous parameter), or "choice",
        If the type is "choice",
         - "values" : a list of values to try.
        If type is "range":
         - "bounds": the lower and upper bound as tuple
         - "n_values" will be ignored,
           since the values will be drawn from the whole interval
           (this is the main difference to `get_random_factorial_design`).
    n_trials : int
        The number of trials.
    sort_values : bool, list, optional
        Whether to sort the trials by their parameter values.
        If a list of parameter names is given, the trials are sorted by the
        values of these parameters.
    sort_ascending : bool, optional
        When sorting, whether to sort in ascending order.
    metrics :
        A list of metrics to add as empty columns to the output
        (for you to fill in the outcomes of the trials).

    Returns
    -------
    design : pd.DataFrame
        The table with the parameter values for the trials.

    """
    parameters = deepcopy(parameters)
    for p in parameters:
        if p["type"] == "range":
            p["n_values"] = 1

    design = get_random_factorial_design(parameters=parameters,
                                         n_trials=n_trials,
                                         metrics=metrics)

    for p in parameters:
        if p["type"] == "range":
            values = np.random.uniform(low=p["bounds"][0], high=p["bounds"][1],
                                       size=(n_trials,))
            design[p["name"]] = values

    if sort_values is True:
        sort_values = [p["name"] for p in parameters]
    if sort_values:
        design.sort_values(by=sort_values, ascending=sort_ascending,
                           ignore_index=True, inplace=True)

    return design


def get_design(parameters, design_type, n_trials=None,
               sort_values=False, sort_ascending=True,
               metrics=None):
    """Get central composite design.

    Get a table of parameter values for the trials in a central composite design.

    Parameters
    ----------
    parameters :
        A list of parameters, where each list item is a dictionary with keys
         - "name": the name of the parameter,
         - "type": the type, either "range" (continuous parameter), or "choice",
        If the type is "choice",
         - "values" : a list of values to try.
        If type is "range":
         - "bounds": the lower and upper bound as tuple
         - "n_values": the number of value to try.
    design_type : str
        The type of design:
         - "simple_central": just a central design
         - "mixed_central": central + mixed second order design
         - "full_central": central + full factorial design
         - "full_factorial": full factorial design
         - "random_factorial": random factorial design
         - "random": random design
    n_trials :
        The number of trials (unless determined by the specific design).
    sort_values : bool, list, optional
        Whether to sort the trials by their parameter values.
        If a list of parameter names is given, the trials are sorted by the
        values of these parameters.
    sort_ascending : bool, optional
        When sorting, whether to sort in ascending order.
    metrics :
        A list of metrics to add as empty columns to the output
        (for you to fill in the outcomes of the trials).

    Returns
    -------
    design : pd.DataFrame
        The table with the parameter values for the trials.

    """
    recognized_design_types = {"simple_central", "mixed_central", "full_central",
                               "full_factorial", "random_factorial", "random"}

    if design_type in {"simple_central", "mixed_central", "full_central"}:
        design = get_central_composite_design(parameters=parameters,
                                              design_type=design_type,
                                              sort_values=sort_values,
                                              sort_ascending=sort_ascending,
                                              metrics=metrics)
    elif design_type == "full_factorial":
        design = get_full_factorial_design(parameters=parameters,
                                           sort_values=sort_values,
                                           sort_ascending=sort_ascending,
                                           metrics=metrics)
    elif design_type == "random_factorial":
        design = get_random_factorial_design(parameters=parameters,
                                             n_trials=n_trials,
                                             sort_values=sort_values,
                                             sort_ascending=sort_ascending,
                                             metrics=metrics)
    elif design_type == "random":
        design = get_random_design(parameters=parameters,
                                   n_trials=n_trials,
                                   sort_values=sort_values,
                                   sort_ascending=sort_ascending,
                                   metrics=metrics)
    else:
        raise ValueError(f"Unrecognized design_type={design_type}. Note, "
                         f"recognized values are: {', '.join(recognized_design_types)}.")
    return design


def get_d_utility_for_polynomial_model(parameters, design, order, mixed):
    for p in parameters:
        if p["type"] != "range":
            raise ValueError(f"Parameter {p['name']} with "
                             f"type = {p['type']} != 'range' encountered.")

    n_parameters = len(parameters)
    names = [p["name"] for p in parameters]
    lower_bounds = np.array([p["bounds"][0] for p in parameters])
    upper_bounds = np.array([p["bounds"][1] for p in parameters])
    scale_factors = 2 / (upper_bounds - lower_bounds)

    design_matrix = design[names].values
    design_matrix = design_matrix - lower_bounds
    design_matrix = np.einsum("ij,j -> ij", design_matrix, scale_factors)
    design_matrix = design_matrix - 1

    order = int(order)
    mixed = bool(mixed)

    r_0 = np.ones((len(design_matrix), 1))
    r_1 = design_matrix

    if order < 0:
        raise ValueError("Polynomial order {order} < 0 encountered. "
                         "Only non-negative orders are supported.")
    elif order == 0:
        r = r_0
    elif mixed is False:
        r = np.concatenate([r_0] + [r_1 ** k for k in range(1, order + 1)], axis=1)
    elif order == 2:
        n_trials, n_parameters = design_matrix.shape

        r_2 = np.zeros((n_trials, n_parameters * (n_parameters + 1) // 2))
        for r in range(n_trials):
            for i in range(n_parameters):
                for j in range(i + 1):
                    r_2[r, i * (i + 1) // 2 + j] = r_1[r, i] * r_1[r, j]

        r = np.concatenate([r_0, r_1, r_2], axis=1)
    else:
        raise ValueError("Polynomial with mixed terms not supported for "
                         "order {order} > 2.")

    _, u = np.linalg.slogdet(r.transpose() @ r)
    u = u / n_parameters

    return u
