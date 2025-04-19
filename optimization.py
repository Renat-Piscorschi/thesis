import time
from itertools import combinations
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from graphs import create_graph, get_connection_degree_metric, get_symmetric_metric, \
    get_beam_continuity_metric

def total_objective(trial,
                    choices_complexity,
                    n_required,
                    nodes_1,
                    nodes_2,
                    nodes_1_coord,
                    nodes_2_coord,
                    mass):
    """
    This function calculates multiple metrics for a given set of choices (e.g., truss types) based on the trial
    and returns a tuple of graph mass, connection degree metric, symmetry metric, and beam continuity metric.

    Parameters:
    trial (optuna.trial.Trial): The current trial object used for hyperparameter optimization.
    choices_complexity (list): A list of complexity choices for each parameter.

    Returns:
    tuple: A tuple containing four metrics:
        - graph_mass: The total mass of the graph edges.
        - connection_metric: The connection degree metric.
        - symmetry_metric: The symmetry metric.
        - beam_continuity_metric: The beam continuity metric.
    """

    # Get the parameter names for the edges (mass information) and convert them to a list of strings
    param_names = mass.iloc[:, 0].values.astype(str).flatten().tolist()
    truss_types_var = []  # This list will hold the suggested truss types for each parameter
    # Iterate over parameter names and their corresponding complexity choices
    for param_name, choice in zip(param_names, choices_complexity):
        # Filter out any invalid choices (999 values)
        filtered_choice = [choice_el for choice_el in choice if choice_el != 999]

        # If there are valid choices, suggest a categorical value for each parameter
        if len(filtered_choice) > 0:
            # `trial.suggest_categorical` suggests a truss type for each parameter
            suggested = trial.suggest_categorical(f"s_{param_name}", filtered_choice)
            truss_types_var.append(suggested)  # Append the suggested truss type to the list

    # Create a DataFrame with the selected truss types for each parameter
    truss_types_var_df = pd.DataFrame(truss_types_var, columns=["truss_type"])

    # Create a graph using the selected truss types and the provided node data
    G, data_nodes = create_graph(nodes_1, nodes_2, nodes_1_coord, nodes_2_coord, mass, truss_types_var_df)

    # Calculate the various metrics for the generated graph
    connection_metric = get_connection_degree_metric(G)
    symmetry_metric = get_symmetric_metric(G)
    beam_continuity_metric = get_beam_continuity_metric(G)

    # Calculate the total mass of the graph edges
    graph_mass = sum([data["mass"] for _, _, data in G.edges(data=True)])

    unique_truss_types = set(truss_types_var)
    n_distinct = len(unique_truss_types)


    if n_distinct < n_required:
        PENALTY_MULTIPLIER = 1e9
        penalty = (n_required - n_distinct) * PENALTY_MULTIPLIER

        graph_mass += penalty
        connection_metric += penalty
        symmetry_metric += penalty
        beam_continuity_metric += penalty

    # Return the calculated metrics as a tuple
    return graph_mass, connection_metric, symmetry_metric, beam_continuity_metric

def select_column_combinations(df, n):
    """
    Generate all possible combinations of 'n' columns from the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n (int): The number of columns to include in each combination.

    Returns:
    tuple: A tuple containing:
        - result_values (list of lists): A list where each element is a list of row values
          corresponding to a specific column combination.
        - result_columns (list of lists): A list where each element is a list of column names
          corresponding to a specific column combination.
    """
    # Generate all possible column combinations of size 'n'
    column_combinations = list(combinations(df.columns, n))

    # Extract values for each column combination
    result_values = [df[list(comb)].values.tolist() for comb in column_combinations]

    # Extract column names for each combination
    result_columns = [df[list(comb)].columns.tolist() for comb in column_combinations]

    return result_values, result_columns

def run_total_tpe_optimizer(n,
                            data_complexity,
                            n_trials=100,
                            nodes_1=None,
                            nodes_2=None,
                            nodes_1_coord=None,
                            nodes_2_coord=None,
                            mass=None):

    # Select possible column combinations from the data complexity based on the number of columns (n)
    choices_list_complexity, columns_list_complexity = select_column_combinations(data_complexity.iloc[:, 1:], n)

    # Lists to store best trial values and parameters for each set of choices
    best_trial_values = []
    best_trial_params = []
    # Iterate over each set of complexity choices
    for i in range(len(choices_list_complexity)):

        sampler = TPESampler(consider_prior=False,
                             consider_magic_clip=False,
                             multivariate=True,
                             group=True,
                             consider_endpoints=False,
                             constant_liar=False,
                             n_startup_trials=100,
                             n_ei_candidates=100,
                             seed=42,
                             )
        study = optuna.create_study(sampler=sampler, directions=["minimize"] * 4)  # We are minimizing 4 objectives

        # Define the objective function to optimize, which takes the current trial and complexity choices as inputs
        def objective_with_choice(trial):
            return total_objective(trial,
                                   choices_list_complexity[i],
                                   n,
                                   nodes_1,
                                   nodes_2,
                                   nodes_1_coord,
                                   nodes_2_coord,
                                   mass)

        # Optimize the objective function for the current set of choices
        study.optimize(objective_with_choice, n_trials=n_trials)

        # Store the best trial values and parameters from the optimization process
        best_trial_values += [trial.values for trial in study.best_trials]
        best_trial_params += [trial.params for trial in study.best_trials]

    # Return the best trial values and parameters for all sets of complexity choices
    return best_trial_values, best_trial_params

def minmax_scale(column_name):

    col_min = column_name.min()
    col_max = column_name.max()

    if col_min == col_max:
        return pd.Series([0.5] * len(column_name), index=column_name.index)  # Avoid division by zero

    return ((column_name - col_min) / (col_max - col_min))

def compute_post_optimization_score(optimized_data_column, weights):
    scaled_values = optimized_data_column[["mass", "connect_deg", "symmetry", "beam_cont"]].apply(minmax_scale)
    score = (scaled_values * weights).sum(axis=1)
    optimized_data_column["weighted_score"] = score
    return optimized_data_column.loc[[score.idxmin()]]

def run_algorithm(N=1,
                  N_TRIALS=10,
                  weights=None,
                  N_PROFILES=0,
                  structural_complexity_data=None,
                  nodes_1=None,
                  nodes_2=None,
                  nodes_1_coord=None,
                  nodes_2_coord=None,
                  mass=None):
    start_time = time.perf_counter()

    best_trial_values, best_trial_params = run_total_tpe_optimizer(N,
                                                               structural_complexity_data,
                                                               n_trials=N_TRIALS,
                                                               nodes_1=nodes_1,
                                                               nodes_2=nodes_2,
                                                               nodes_1_coord=nodes_1_coord,
                                                               nodes_2_coord=nodes_2_coord,
                                                               mass=mass)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    values = []
    params = []
    for i in range(len(best_trial_values)):
        if len(best_trial_params[i]) == N_PROFILES:
            values.append(best_trial_values[i])
            params.append(best_trial_params[i])
    data = list(zip(values, params))

    params_df = pd.DataFrame(params)

    distinct_s_count = params_df.nunique(axis=1)
    params_df['N'] = distinct_s_count

    values_df = pd.DataFrame([val if isinstance(val, (list, tuple)) else [val] for val, _ in data])

    values_df.columns = ['mass', 'connect_deg', 'symmetry', 'beam_cont']

    # Concatenate both DataFrames along columns.
    final_df = pd.concat([values_df, params_df], axis=1)
    final_df = final_df.drop_duplicates(keep="first").reset_index(drop=True)

    result_df = pd.DataFrame(columns=list(final_df.columns) + [f'w_{i}' for i in range(1, len(weights[0]) + 1)])
    for weight_set in weights:
        post_optimization_score = compute_post_optimization_score(final_df, weight_set)
        weight_df = pd.DataFrame([weight_set], columns=[f'w_{i}' for i in range(1, len(weights[0]) + 1)])
        post_optimization_score = pd.concat([post_optimization_score.reset_index(drop=True), weight_df], axis=1)
        result_df = pd.concat([result_df, post_optimization_score], axis=0, ignore_index=True)

    print(f"Execution time of algorithm: {execution_time:.6f} seconds", sep='\n')
    return result_df