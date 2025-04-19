import glob
import os
import re
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import pyplot as plt, image as mpimg
import io
import seaborn as sns
from PIL import Image
import matplotlib.patches as mpatches

from graphs import create_graph


def plot_lines(weights, data, feature_to_plot='weighted_score'):
    w_len = len(weights[0])
    num_rows = len(data)

    # Create an empty figure
    fig = go.Figure()
    line_x = pd.Series(data['N'].unique())
    for k in range(min(num_rows // w_len - 1, len(weights))):
        line_y = pd.Series()
        for i in range(k, num_rows, w_len):
            line_y = pd.concat([line_y, pd.Series([data.iloc[i][feature_to_plot]])])

        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name=f"{weights[k]}"))

    # Customize layout
    fig.update_layout(
        height=700,
        xaxis_title="N",
        yaxis_title=feature_to_plot,
        legend_title="Distributions",
        xaxis=dict(
            tickmode='linear',  # This ensures we show ticks at regular intervals
            tick0=1,  # Starting value of x-axis
            dtick=1  # Set the step size for ticks (1 for integers)
        )
    )

    # Show the plot
    fig.show()


def plot_parallel_coordinates_plot(full_data):
    full_data['N'] = pd.to_numeric(full_data['N'], errors='coerce')
    full_data['w_1'] = pd.to_numeric(full_data['w_1'], errors='coerce')
    full_data['w_2'] = pd.to_numeric(full_data['w_2'], errors='coerce')
    full_data['w_3'] = pd.to_numeric(full_data['w_3'], errors='coerce')
    full_data['w_4'] = pd.to_numeric(full_data['w_4'], errors='coerce')

    fig = px.parallel_coordinates(data_frame=full_data,
                                  dimensions=['mass', 'connect_deg', 'symmetry',
                                              'beam_cont', 'w_1', 'w_2', 'w_3', 'w_4', 'N'],
                                  color_continuous_scale=px.colors.diverging.RdBu,
                                  ############################################### schimba gradientu
                                  color_continuous_midpoint=3)

    fig.update_layout(
        height=700,
    )
    fig.show()


# Function to extract number from truss type labels
def extract_number(truss_label):
    match = re.search(r'HEA(\d+)', truss_label)  # Extract digits after "HEA"
    return int(match.group(1)) if match else 2  # Default to 2 if no match

def plot_graph(G,
               N_TRIALS=0,
               save=True,
               edge_colors_map=None,
               n_dist=0,
               n_id=0,
               is_title=False,
               truss_types_new=None):
    # Create figure and axis with larger size for better visibility
    fig, ax = plt.subplots(figsize=(16, 4))

    # Extract node positions from the 'pos' attribute
    pos = nx.get_node_attributes(G, 'pos')

    # Extract edge attributes
    elements_id_labels = nx.get_edge_attributes(G, 'element_id')
    truss_types_labels = nx.get_edge_attributes(G, 'truss_type')


    # Access colormap correctly
    colormap = plt.colormaps['tab10']  # Access 'tab10' colormap
    unique_truss_types = sorted(set(truss_types_labels.values()))


    truss_width_map = {truss: extract_number(truss) // 25 + i * 2 for i, truss in
                       enumerate(truss_types_new)}  ### Edge thickness

    print(truss_width_map)

    # Assign thickness using the mapping
    edge_thickness = [truss_width_map[truss_types_labels[edge]] for edge in G.edges()]

    # If no external edge_colors_map is provided, create it
    if edge_colors_map is None:
        edge_colors = [colormap(i % len(unique_truss_types)) for i in range(len(unique_truss_types))]
        edge_colors_map = {truss: edge_colors[i] for i, truss in enumerate(unique_truss_types)}

    # Assign colors to edges based on 'truss_type'
    edge_colors = [edge_colors_map[truss_types_labels[edge]] for edge in G.edges()]

    # Concatenate 'element_id' and 'truss_type' for edge labeling
    edge_labels = {
        key: f"{str(elements_id_labels[key])}_{str(truss_types_labels[key])}"
        for key in elements_id_labels
    }

    # Make grid and axes visible BEFORE drawing the graph
    ax.grid(True, linestyle="--", alpha=1, color='#000')
    ax.set_facecolor("none")  # White background
    ax.set_axis_on()

    # Ensure axes are visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    # Draw the graph (without node labels)
    nx.draw_networkx(G,
                     pos,
                     ax=ax,
                     with_labels=False,
                     node_color="black",
                     edge_color=edge_colors,
                     width=edge_thickness,
                     node_size=300)  ### NODE SIZE

    # Manually place the edge labels (both element_id and truss_type)
    for (start, end), label in edge_labels.items():
        x_start, y_start = pos[start]
        x_end, y_end = pos[end]
        x = (x_start + x_end) / 2  # Position in the middle of the edge
        y = (y_start + y_end) / 2

        # Split label into element_id and truss_type
        element_label, truss_label = label.split('_')

        edge_color = edge_colors_map[truss_types_labels[(start, end)]]

        # Style for element_id label
        ax.text(x, y + 0.08, element_label, ha='center', va='bottom', fontsize=12, fontweight="bold", color="black",
                bbox=dict(facecolor='skyblue', edgecolor='black', boxstyle="round,pad=0.4"))

        # Style for truss_type label (more space below the element_id)
        ax.text(x, y - 0.12, truss_label, ha='center', va='top', fontsize=14, fontweight="bold", color=edge_color,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.2"))

    ax.set_xlim(min(x for x, y in pos.values()) - 0.5, max(x for x, y in pos.values()) + 0.5)
    ax.set_ylim(min(y for x, y in pos.values()) - 0.5, max(y for x, y in pos.values()) + 0.5)

    if is_title:
        ax.set_title(f"Distribution {n_dist + 1} N_trials = {N_TRIALS} n = {n_id}")

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    fig.patch.set_alpha(0)
    if not save:
        # Show the plot
        plt.show()
    else:
        return ax

def plot_pie_chart(pie_data, save=True):
    pie_data['label'] = ['Mass', 'Connection degree', 'Symmetry', 'Beam continuity']

    colors = sns.color_palette('pastel')[0:4]  # Adjusting the number of colors to match the number of slices
    fig, ax = plt.subplots()

    # Create the pie chart without labels near the slices
    wedges, texts, autotexts = ax.pie(pie_data['values'], labels=[""] * 4, autopct='%.0f%%', colors=colors)

    # Add a legend outside the pie chart
    ax.legend(wedges, pie_data['label'], title="Categories", loc="center left", bbox_to_anchor=(1, 0.5))

    if not save:
        plt.show()
    else:
        # Save the figure to a temporary buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')  # Added bbox_inches='tight' to prevent cutoffs
        buf.seek(0)
        img = Image.open(buf)
        return img  # Return the image object

def plot_line(weights, data, feature_to_plot='weighted_score'):
    labels = ['Mass [kg]', 'Connection degree [-]', 'Symmetry [-]', 'Beam continuity [-]']
    w_len = len(weights[0])
    num_rows = len(data)

    line_x = pd.Series(data['N'].unique())

    plots = []  # Store figures if you need multiple plots

    for k in range(min(num_rows // w_len - 1, len(weights))):
        line_y = pd.Series()
        for i in range(k, num_rows, w_len):
            line_y = pd.concat([line_y, pd.Series([data.iloc[i][feature_to_plot]])])

        # Create a new figure for each line
        fig = go.Figure()
        formatted_values = line_y.round(2)

        # Add the trace with values on the line
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines+text',  # 'lines+text' adds the text on the line
            name=f"{weights[k]}",
            text=formatted_values,  # The values to display at each data point
            textposition='top center',  # Position the text above the data points
            line=dict(color='blue', width=2),  # Optional: Customize the line color and width
            textfont=dict(color='black', size=20, weight='bold'),  # Set the text color to black
            # Adjust the text position slightly above the line to avoid overlap
            showlegend=False,  # Hide the legend for this trace
        ))

        label = ""
        if feature_to_plot == 'mass':
            label = labels[0]
        elif feature_to_plot == 'connect_deg':
            label = labels[1]
        elif feature_to_plot == 'symmetry':
            label = labels[2]
        elif feature_to_plot == 'beam_cont':
            label = labels[3]

        # Customize layout with larger width, white background, and no grid
        fig.update_layout(
            height=700,  # Height of the plot
            width=1200,  # Width of the plot
            xaxis_title="Number of configurations (n)",
            yaxis_title=label,
            legend_title="Distributions",
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1,
                mirror=True,
                showline=True,  # Show the x-axis line
                linecolor='black',  # Color of the x-axis line
                linewidth=2,  # Line width of the x-axis line
                title=dict(text="Number of configurations (n)", font=dict(size=30, weight='bold')),
                griddash='dash',
            ),
            yaxis=dict(
                mirror=True,
                showline=True,  # Show the y-axis line
                linecolor='black',  # Color of the y-axis line
                linewidth=2,  # Line width of the y-axis line
                title=dict(text=label, font=dict(size=30, weight='bold')),
                griddash='dash',
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to white
            paper_bgcolor='rgba(0,0,0,0)',  # Set the background of the entire paper to white
            font=dict(color='black', size=25)  # Set the font color for axis titles and other text to black
        )

        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",  # Use "paper" to cover the full area
            x0=0, x1=1,  # Covers the entire width
            y0=0, y1=1,  # Covers the entire height
            line=dict(color="black", width=2))  # Black border

        # Show the plot immediately if required
        # fig.show()

        # Store the figure in case you want to return it
        plots.append(fig)

    return plots

def save_plots(parallel_coordinates_data,
               weights,
               nodes_1,
               nodes_2,
               nodes_1_coord,
               nodes_2_coord,
               mass,
               is_title=True,
               experiment_name="",
               N_PROFILES=0,
               N_TRIALS=0):
    # Define the directory to save images
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    truss_types = parallel_coordinates_data[[f"s_{i}" for i in range(1, N_PROFILES + 1)]]
    weights_data = parallel_coordinates_data[[f"w_{i}" for i in range(1, 5)]]

    graphs = []
    pie_charts = []

    mass_plots = plot_line(weights, parallel_coordinates_data, feature_to_plot='mass')
    connectivity_plots = plot_line(weights, parallel_coordinates_data, feature_to_plot='connect_deg')
    symmetry_plots = plot_line(weights, parallel_coordinates_data, feature_to_plot='symmetry')
    beam_cont_plots = plot_line(weights, parallel_coordinates_data, feature_to_plot='beam_cont')
    unique_weights_conf = weights_data.drop_duplicates()

    unique_truss_types = sorted(list(set(truss_types.values.flatten().tolist())))
    colormap = plt.colormaps['tab10']
    edge_colors = [colormap(i % len(unique_truss_types)) for i in range(len(unique_truss_types))]
    edge_colors_map = {truss: edge_colors[i] for i, truss in enumerate(unique_truss_types)}

    counter = 1
    for i in range(len(truss_types)):
        truss_types_conf = pd.DataFrame(truss_types.iloc[i].values.flatten(), columns=['values'])

        G, data_nodes = create_graph(nodes_1, nodes_2, nodes_1_coord, nodes_2_coord, mass, truss_types_conf)

        graph_to_be_plotted = plot_graph(G,
                                         save=True,
                                         edge_colors_map=edge_colors_map,
                                         is_title=is_title,  ############################### Deconecteaza titlul
                                         n_dist=i % len(weights),
                                         n_id=counter,
                                         N_TRIALS=N_TRIALS,
                                         truss_types_new=unique_truss_types)

        graphs.append(graph_to_be_plotted)

        if i % 5 == 0 and i != 0:  # Every 5 iterations
            counter += 1

    for i in range(len(unique_weights_conf)):
        unique_weight_conf = pd.DataFrame(unique_weights_conf.iloc[i].values.flatten(), columns=['values'])
        pie_chart = plot_pie_chart(unique_weight_conf)
        pie_charts.append(pie_chart)

    for j in range(len(mass_plots)):
        fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(16, 20))  # Adjust height as needed

        mass_plot = mass_plots[j].to_image(format="png")
        connectivity_plot = connectivity_plots[j].to_image(format="png")
        symmetry_plot = symmetry_plots[j].to_image(format="png")
        beam_cont_plot = beam_cont_plots[j].to_image(format="png")
        pie_chart = pie_charts[j]

        # Save images to temporary files
        mass_plot_path = os.path.join(save_dir, f"mass_plot_{i}_{N_TRIALS}.png")
        connectivity_plot_path = os.path.join(save_dir, f"connectivity_plot_{i}_{N_TRIALS}.png")
        symmetry_plot_path = os.path.join(save_dir, f"symmetry_plot_{i}_{N_TRIALS}.png")
        beam_cont_plot_path = os.path.join(save_dir, f"beam_cont_plot_{i}_{N_TRIALS}.png")

        with open(mass_plot_path, "wb") as f:
            f.write(mass_plot)
        with open(connectivity_plot_path, "wb") as f:
            f.write(connectivity_plot)
        with open(symmetry_plot_path, "wb") as f:
            f.write(symmetry_plot)
        with open(beam_cont_plot_path, "wb") as f:
            f.write(beam_cont_plot)

        # Plot the Pie Chart Image in Matplotlib
        axs[0].imshow(pie_chart)
        axs[0].axis("off")  # Hide axes for the pie chart
        axs[0].set_title(f"Distribution {j + 1} N_trials = {N_TRIALS}")

        # Load the line plot images with Matplotlib
        axs[1].imshow(mpimg.imread(mass_plot_path))
        axs[1].axis("off")  # Hide axes for the line plot

        axs[2].imshow(mpimg.imread(connectivity_plot_path))
        axs[2].axis("off")

        axs[3].imshow(mpimg.imread(symmetry_plot_path))
        axs[3].axis("off")

        axs[4].imshow(mpimg.imread(beam_cont_plot_path))
        axs[4].axis("off")

        # Adjust spacing for better layout
        plt.tight_layout()

        dest_to_save = os.path.join(save_dir, experiment_name,
                                    f'n_profiles_{N_PROFILES}_distribution_{j + 1}_n_trials_{N_TRIALS}')
        os.makedirs(dest_to_save, exist_ok=True)
        # Save the stacked figure
        fig.savefig(os.path.join(dest_to_save, f"pie_line_plot.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to prevent memory issues

        # Delete the intermediate plot files
        os.remove(mass_plot_path)
        os.remove(connectivity_plot_path)
        os.remove(symmetry_plot_path)
        os.remove(beam_cont_plot_path)

    for n_idx in range(len(weights)):
        figure, axs = plt.subplots(nrows=5, ncols=1, figsize=(16, 20))  # Adjust height as needed

        dest_to_save = os.path.join(save_dir, experiment_name,
                                    f'n_profiles_{N_PROFILES}_distribution_{n_idx + 1}_n_trials_{N_TRIALS}')
        plt.subplots_adjust(bottom=0.1)
        for j, distribution_idx in enumerate(range(n_idx, len(graphs), len(weights))):
            fig = graphs[distribution_idx].get_figure() if hasattr(graphs[distribution_idx], 'get_figure') else graphs[
                distribution_idx]

            # Load the line plot images with Matplotlib
            fig.savefig(os.path.join(dest_to_save, f"graph_n_{j + 1}.png"), dpi=300, bbox_inches='tight')
            axs[j].imshow(mpimg.imread(os.path.join(dest_to_save, f"graph_n_{j + 1}.png")))

            axs[j].axis("off")

            plt.close(fig)
        legend_entries = [mpatches.Patch(color=color, label=truss) for truss, color in edge_colors_map.items()]

        figure.legend(handles=legend_entries, loc='lower center', ncol=5)
        plt.tight_layout()
        figure.savefig(os.path.join(dest_to_save, f"stacked_graph.png"), dpi=300, bbox_inches='tight', transparent=True)


def visualize_graph(G):
    """
    Visualize a graph with node positions and labeled edges.

    Parameters:
    G (networkx.Graph): The input graph, where nodes have a 'pos' attribute
                        for positioning, and edges have 'element_id' and 'truss_type' attributes.

    Displays:
    A Matplotlib plot of the graph with labeled edges.
    """
    # Extract node positions from the 'pos' attribute
    pos = nx.get_node_attributes(G, 'pos')

    # Extract edge labels from 'element_id' and 'truss_type' attributes
    elements_id_labels = nx.get_edge_attributes(G, 'element_id')
    truss_types_labels = nx.get_edge_attributes(G, 'truss_type')

    # Concatenate 'element_id' and 'truss_type' for edge labeling
    concatenated_labels = {
        key: f'{str(value)}_{truss_types_labels[key]}'
        for key, value in elements_id_labels.items()
    }

    # Draw the graph with node positions and labels
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")

    # Draw edge labels with concatenated information
    nx.draw_networkx_edge_labels(G, pos, edge_labels=concatenated_labels)

    # Display the plot
    # plt.show()


def get_nth_occurrence_index(lst, value, n):
    count = 0
    for i, elem in enumerate(lst):
        if elem == value:
            count += 1
            if count == n:
                return i  # Return the index of the nth occurrence
    return None


def create_validation_plot(experiment_name, objective, dist_n, n):
    labels = ['Mass [kg]', 'Connection degree [-]', 'Symmetry [-]', 'Beam continuity [-]']

    csv_files = glob.glob(f"plots/{experiment_name}/*.csv")
    exhaustive_data = pd.read_excel("data/exhaustive_data.xlsx", sheet_name="ExhS_1234")

    x = []
    y = []
    y_exhaustive = []
    csv_files = sorted(csv_files, key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[0]))

    for file in csv_files:
        filename = os.path.basename(file)  # Get only the file name, not the full path
        match = re.findall(r'\d+', filename)  # Find all numbers in the filename

        temp_df = pd.read_csv(file)
        column_values = temp_df["N"].tolist()
        n_index = get_nth_occurrence_index(column_values, n, dist_n)
        value = temp_df[objective].iloc[n_index]
        y.append(value)

        for number in match:
            x.append(int(number))
    x = sorted(x)

    exhaustive_data_column_values = exhaustive_data["N"].tolist()
    n_index_exhaustive = get_nth_occurrence_index(exhaustive_data_column_values, n, dist_n)
    value_exhaustive = exhaustive_data[objective].iloc[n_index_exhaustive]

    label = ""
    if objective == 'mass':
        label = labels[0]
    elif objective == 'connect_deg':
        label = labels[1]
    elif objective == 'symmetry':
        label = labels[2]
    elif objective == 'beam_cont':
        label = labels[3]

    for _ in range(len(x)):
        y_exhaustive.append(value_exhaustive)

    y_rounded = [round(num, 2) for num in y]
    y_exhaustive_rounded = [round(num, 2) for num in y_exhaustive]

    overlapping_points = set(zip(x, y)) & set(zip(x, y_exhaustive))

    # Keep text for TPE sampler and remove from Exhaustive
    y_exhaustive_text_filtered = ["" if (x[i], y_exhaustive[i]) in overlapping_points else str(val) for i, val in
                                  enumerate(y_exhaustive)]

    fig = go.Figure()

    # Add first line
    fig.add_trace(go.Scatter(x=x,
                             y=y_rounded,
                             mode='lines+text',
                             name='TPE sampler',
                             text=y_rounded,
                             textposition='top center',
                             textfont=dict(color='black', size=20, weight='bold')
                             ))
    # Add second line
    fig.add_trace(go.Scatter(x=x,
                             y=y_exhaustive_rounded,
                             mode='lines+text',
                             name='Exhaustive',
                             text=y_exhaustive_text_filtered,
                             textposition='top center',
                             textfont=dict(color='black', size=20, weight='bold')
                             ))

    fig.update_layout(
        height=700,  # Height of the plot
        width=1200,  # Width of the plot
        xaxis_title="Number of trials (m)",
        yaxis_title=label,
        legend_title="Distributions",
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            mirror=True,
            showline=True,  # Show the x-axis line
            linecolor='black',  # Color of the x-axis line
            linewidth=2,  # Line width of the x-axis line
            title=dict(text="Number of trials (m)", font=dict(size=30, weight='bold')),
            griddash='dash',
            showgrid=True,  # Ensure grid is visible
            gridcolor='black',
        ),
        yaxis=dict(
            mirror=True,
            showline=True,  # Show the y-axis line
            linecolor='black',  # Color of the y-axis line
            linewidth=2,  # Line width of the y-axis line
            title=dict(text=label, font=dict(size=30, weight='bold')),
            griddash='dash',
            showgrid=True,  # Ensure grid is visible
            gridcolor='black',
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to white
        paper_bgcolor='rgba(0,0,0,0)',  # Set the background of the entire paper to white
        font=dict(color='black', size=25)
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    x_ticks = sorted(set(x))  # Combine and get unique x values
    y_ticks = sorted(set(y_rounded + y_exhaustive_rounded))  # Combine and get unique y values

    # Update the figure axes to only show these values
    fig.update_xaxes(tickmode='array', tickvals=x_ticks)
    fig.update_yaxes(tickmode='array', tickvals=y_ticks)  # Combine and get unique y values

    fig.write_image(f"plots/{experiment_name}/validation_plot_{objective}_distribution_{dist_n}_n_{n}.png")
