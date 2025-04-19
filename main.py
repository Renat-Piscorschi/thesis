import argparse
import json
import os.path

import pandas as pd
from matplotlib import pyplot as plt, image as mpimg

from optimization import run_algorithm
from plots import save_plots, create_validation_plot

pd.options.display.float_format = '{:.6f}'.format

def check_uc(data, uc, mask_value=999):
    """
    Apply a masking operation on the DataFrame 'data' based on unity_check (uc). Any value in 'data' where the corresponding
    value in 'uc' (excluding the first column) is greater than 1 is replaced with 999.

    Parameters:
    data (pd.DataFrame): The input DataFrame to be modified.

    Returns:
    pd.DataFrame: The modified DataFrame with values replaced where applicable.
    """
    # Ensure 'uc' is defined before use (assuming 'uc' is a global variable)
    mask = uc.iloc[:, 1:] <= 1  # Create a boolean mask where 'uc' values are <= 1

    # Apply the mask to 'data', replacing values where 'mask' is False with 999
    data.iloc[:, 1:] = data.iloc[:, 1:][mask].fillna(mask_value)

    return data

def load_weights(weights_path):
    try:
        with open(weights_path, "r") as f:
            data = json.load(f)

        # Ensure "weights" key exists
        if "weights" not in data:
            raise ValueError("JSON file does not contain a 'weights' key.")

        weights_matrix = data["weights"]

        # Validate matrix structure (list of lists)
        if not isinstance(weights_matrix, list) or not all(isinstance(row, list) for row in weights_matrix):
            raise ValueError("Invalid matrix format in JSON file: Expected list of lists.")

        return weights_matrix

    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Error reading JSON file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Define arguments
    parser.add_argument("--file_name", type=str, required=True, default="MY_data", help="Path to input file")
    parser.add_argument("--experiment_name", type=str, required=True, default="test", help="Name of the experiment")
    parser.add_argument("--weights_name", type=str, required=True, default="weights_4", help="A JSON-formatted weights file")
    parser.add_argument("--n_trials", type=int, required=True, help="N_trials argument")
    parser.add_argument("--is_title", action="store_true", help="Show title")
    parser.add_argument("--is_validation", action="store_true", help="compute validation")

    # Parse arguments
    args = parser.parse_args()

    file_path = os.path.join("data", f"{args.file_name}.xlsx")
    weights_path = os.path.join("data", f"{args.weights_name}.json")

    # read element U.C sheet
    uc = pd.read_excel(file_path, sheet_name="Element U.C.")

    # read element mass U.C sheet
    mass = pd.read_excel(file_path, sheet_name="Element Mass")

    # read #node_1 & #node_2
    nodes_1 = pd.read_excel(file_path, sheet_name="Element # NODE_1")
    nodes_2 = pd.read_excel(file_path, sheet_name="Element # NODE_2")

    # read #node_1 & #node_2 coordinates
    nodes_1_coord = pd.read_excel(file_path, sheet_name="Element # NODE_1 coord.")
    nodes_2_coord = pd.read_excel(file_path, sheet_name="Element # NODE_2 coord.")

    N_PROFILES = len(mass)

    truss_types = mass.columns[1:].tolist()

    # Create a DataFrame where each column represents a truss type,
    # and each column contains 9 repeated values of its name.
    structural_complexity_data = pd.DataFrame({col: [col] * N_PROFILES for col in truss_types})

    # Add an 'edge' column, numbering rows from 1 to the length of the DataFrame.
    structural_complexity_data["edge"] = [i for i in range(1, len(structural_complexity_data) + 1)]

    # Reorder columns to ensure 'edge' is the first column
    columns_order = ["edge"] + [col for col in structural_complexity_data.columns if col != "edge"]
    structural_complexity_data = structural_complexity_data[columns_order]

    # Apply `check_uc()` function to structural complexity data
    structural_complexity_data = check_uc(structural_complexity_data, uc)

    # Apply `check_uc()` function to `mass`
    mass = check_uc(mass, uc)

    WEIGHTS = load_weights(weights_path)

    column_names = (["mass", "connect_deg", "symmetry", "beam_cont", "weighted_score"] +
                    [f"s_{i}" for i in range(1, N_PROFILES + 1)] +
                    ["w_1", "w_2", "w_3", "w_4", "N"])

    parallel_coordinates_data = pd.DataFrame(columns=column_names)

    for n in range(1, len(truss_types) + 1):
        data = run_algorithm(N=n,
                             N_TRIALS=args.n_trials,
                             weights=WEIGHTS,
                             N_PROFILES=N_PROFILES,
                             structural_complexity_data=structural_complexity_data,
                             nodes_1=nodes_1,
                             nodes_2=nodes_2,
                             nodes_1_coord=nodes_1_coord,
                             nodes_2_coord=nodes_2_coord,
                             mass=mass)
        parallel_coordinates_data = pd.concat([parallel_coordinates_data, data], axis=0, ignore_index=True)

    save_plots(parallel_coordinates_data,
               WEIGHTS,
               nodes_1,
               nodes_2,
               nodes_1_coord,
               nodes_2_coord,
               mass,
               is_title=args.is_title,
               N_PROFILES=N_PROFILES,
               N_TRIALS=args.n_trials,
               experiment_name=args.experiment_name)
    parallel_coordinates_data.to_csv(f'plots/{args.experiment_name}/data_{args.n_trials}.csv', index=False)

    objectives = ["mass", "connect_deg", "symmetry", "beam_cont"]

    if args.is_validation:
        for objective in objectives:
            for n_idx in range(1, len(truss_types) + 1):
                for dist_n_idx in range(1, len(WEIGHTS) + 1):
                    create_validation_plot(args.experiment_name, objective, dist_n_idx, n_idx)

        for dist_n_idx in range(1, len(WEIGHTS) + 1):
            for n_idx in range(1, len(truss_types) + 1):
                fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 20))  # Adjust height as needed

                mass_plot_path = os.path.join("plots", args.experiment_name, f"validation_plot_mass_distribution_{dist_n_idx}_n_{n_idx}.png")
                connectivity_plot_path = os.path.join("plots", args.experiment_name, f"validation_plot_connect_deg_distribution_{dist_n_idx}_n_{n_idx}.png")
                symmetry_plot_path = os.path.join("plots", args.experiment_name, f"validation_plot_symmetry_distribution_{dist_n_idx}_n_{n_idx}.png")
                beam_cont_plot_path = os.path.join("plots", args.experiment_name, f"validation_plot_beam_cont_distribution_{dist_n_idx}_n_{n_idx}.png")

                axs[0].imshow(mpimg.imread(mass_plot_path))
                axs[0].axis("off")  # Hide axes for the line plot
                axs[0].set_title(f"Distribution {dist_n_idx} n = {n_idx}", fontsize=20, fontweight='bold', color='black')

                axs[1].imshow(mpimg.imread(connectivity_plot_path))
                axs[1].axis("off")

                axs[2].imshow(mpimg.imread(symmetry_plot_path))
                axs[2].axis("off")

                axs[3].imshow(mpimg.imread(beam_cont_plot_path))
                axs[3].axis("off")

                plt.tight_layout()

                dest_to_save = os.path.join("plots",
                                            args.experiment_name,
                                            f"validation_plot_stacked_{dist_n_idx}_n_{n_idx}.png")
                fig.savefig(os.path.join(dest_to_save), dpi=300, bbox_inches='tight')
                plt.close(fig)  # Close figure to prevent memory issues


if __name__ == "__main__":
    main()