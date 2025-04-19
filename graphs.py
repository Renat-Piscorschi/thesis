from collections import defaultdict
import networkx as nx
import pandas as pd

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

def get_connection_degree_metric(G):
    """
    Computes the connection degree metric for a given graph G.

    The metric is based on the number of unique "truss_type" values
    associated with the edges connected to each node.

    Parameters:
    G (networkx.Graph): The input graph where edges have a "truss_type" attribute.

    Returns:
    int: The total connection degree metric, computed as the sum of unique truss types per node.
    """

    # Dictionary to store the count of unique truss types for each node
    connection_degree_dict = defaultdict(int)

    # Iterate over each node in the graph
    for node in G.nodes():
        # Retrieve edges connected to the node, including edge attributes
        temp = G.edges(node, data=True)

        # Extract unique truss types from connected edges (if the attribute exists)
        connection_degree_dict[node] = len({edge[2]["truss_type"] for edge in temp if "truss_type" in edge[2]})

    # Compute the total connection degree metric as the sum of all unique truss type counts per node
    return sum(connection_degree_dict.values())

def find_symmetric_nodes(G, center=(4, 1.25)):
    """
    Identifies symmetric node pairs in a graph based on reflection across a vertical axis.

    A node (x, y) is considered symmetric with another node (x', y) if:
        x' = 2 * center_x - x
        y' = y
    where (center_x, center_y) defines the vertical reflection axis.

    Parameters:
    G (networkx.Graph): The input graph with nodes having a "pos" attribute containing (x, y) coordinates.
    center (tuple): The (x, y) coordinates of the vertical axis of symmetry.

    Returns:
    dict: A dictionary where keys are nodes and values are their symmetric counterparts.
    """

    symmetric_pairs = {}  # Dictionary to store symmetric node pairs
    center_x, center_y = center  # Extract center coordinates

    # Iterate over all nodes and their positions
    for node, data in G.nodes(data=True):
        x, y = data["pos"]  # Get node coordinates

        # Compute mirrored x-coordinate across the vertical axis at center_x
        mirrored_x = 2 * center_x - x

        # Find the node with mirrored coordinates (mirrored_x, y)
        mirrored_node = next(
            (id for id, coord in G.nodes(data=True)
             if coord["pos"][0] == mirrored_x and coord["pos"][1] == y),
            None
        )

        # Store the symmetric pair if it hasn't been recorded already
        if mirrored_node and mirrored_node not in symmetric_pairs:
            symmetric_pairs[node] = mirrored_node

    return symmetric_pairs

def find_symmetric_elements(graph, symmetric_nodes):
    """
    Identifies symmetric edges in a graph based on node symmetry.

    An edge (start, end) is considered symmetric if its mirrored counterpart
    (mirrored_start, mirrored_end) exists in the graph.

    Parameters:
    graph (networkx.Graph): The input graph where edges have a "truss_type" attribute.
    symmetric_nodes (dict): A dictionary mapping nodes to their symmetric counterparts.

    Returns:
    list: A list of tuples where each tuple contains the truss types of symmetric edges.
    """

    symmetric_elements = []  # List to store pairs of symmetric truss types
    edges = set(graph.edges())  # Convert edges to a set for faster lookups
    edges_data = list(graph.edges(data=True))  # Get edges along with their attributes

    # Iterate over each edge and check for a symmetric counterpart
    for start, end, data in edges_data:
        # Find the mirrored nodes for the current edge's nodes
        mirrored_start = symmetric_nodes.get(start)
        mirrored_end = symmetric_nodes.get(end)

        # Check if the mirrored edge exists in the graph
        if mirrored_start and mirrored_end:
            mirrored_edge = (mirrored_start, mirrored_end)
            reversed_mirrored_edge = (mirrored_end, mirrored_start)

            if mirrored_edge in edges or reversed_mirrored_edge in edges:
                # Retrieve the truss type of the mirrored edge
                mirrored_edge_truss_type = graph.get_edge_data(mirrored_start, mirrored_end)["truss_type"]

                # Store the truss type pair
                symmetric_elements.append((data["truss_type"], mirrored_edge_truss_type))

    return symmetric_elements

def get_symmetric_metric(G):
    """
    Computes a symmetric metric for the graph G based on the symmetry of nodes and edges.

    The metric counts the number of edges whose truss types are different between the node
    and its mirrored counterpart. This can be used to assess the symmetry of a structural graph.

    Parameters:
    G (networkx.Graph): The input graph where nodes and edges have position and truss_type attributes.

    Returns:
    int: The symmetric metric value, which is the count of edges with differing truss types between
         mirrored node pairs.
    """

    # Find symmetric nodes in the graph (nodes with mirrored counterparts)
    symmetric_nodes = find_symmetric_nodes(G)

    # Find symmetric edges based on the symmetric nodes
    symmetric_elements = find_symmetric_elements(G, symmetric_nodes)

    # Compute the symmetric metric by counting edges with different truss types for mirrored nodes
    symmetric_metric = sum([1 if symmetric_element[0] != symmetric_element[1] else 0
                            for symmetric_element in symmetric_elements])

    return symmetric_metric

def iterate_over_beam(graph, start_node):
    """
    Iterates over the edges connected to the starting node in the graph, and identifies edges
    that are part of the same horizontal beam (i.e., share the same y-coordinate).

    The function considers edges whose start node and end node lie on the same horizontal line
    as the start node and ensures the nodes are ordered from left to right.

    Parameters:
    graph (networkx.Graph): The input graph with node positions and edge attributes.
    start_node (tuple): The starting node, which is a tuple (node_id, node_data) where node_data
                        contains the 'pos' attribute with (x, y) coordinates.

    Returns:
    list: A list of truss types for all edges in the same horizontal beam, ordered from left to right.
    """

    # Get the coordinates of the starting node
    start_x, start_y = start_node[1]["pos"]

    # Initialize a list to store subsequent edges that are on the same horizontal beam
    subsequent_edges = []

    # Iterate over all edges in the graph
    for u, v in graph.edges():
        # Get the coordinates of the nodes connected by this edge
        u_x, u_y = graph.nodes[u]['pos']
        v_x, v_y = graph.nodes[v]['pos']

        # Check if both nodes (u, v) are on the same horizontal line (y-coordinate)
        # and if the edge is ordered correctly (from left to right relative to the start node)
        if start_y == u_y == v_y and start_x <= u_x and start_x <= v_x:
            # Add the truss type of the edge to the subsequent_edges list
            subsequent_edges.append(graph.get_edge_data(u, v)["truss_type"])

    return subsequent_edges

def get_beam_continuity_metric(G):
    """
    Calculates a beam continuity metric for the graph G by analyzing the horizontal beams
    at the top-left and bottom-left corners of the graph.

    The metric rewards continuity in truss types along the horizontal beams. If the truss type
    changes between consecutive edges in the beam, the metric is incremented.

    Parameters:
    G (networkx.Graph): The input graph where nodes have positions and edges have truss type attributes.

    Returns:
    int: The beam continuity metric, which measures the consistency of truss types in horizontal beams.
    """

    # Identify the top-left node (smallest x-coordinate, largest y-coordinate)
    top_left_node = min(G.nodes(data=True), key=lambda x: (x[1]['pos'][0], -x[1]['pos'][1]))

    # Identify the bottom-left node (smallest x-coordinate, smallest y-coordinate)
    bottom_left_node = min(G.nodes(data=True), key=lambda x: (x[1]['pos'][0], x[1]['pos'][1]))

    # Iterate over the top horizontal beam starting from the top-left node
    top_beam = iterate_over_beam(G, top_left_node)

    # Iterate over the bottom horizontal beam starting from the bottom-left node
    bottom_beam = iterate_over_beam(G, bottom_left_node)

    # Initialize the beam continuity metric, starting with a value of 2 (representing the first pair)
    beam_continuity_metric = 2

    # Iterate through the top beam and bottom beam to check for continuity of truss types
    for i in range(1, len(top_beam)):
        # Increment the metric if the truss type changes in the top beam
        if top_beam[i-1] != top_beam[i]:
            beam_continuity_metric += 1
        # Increment the metric if the truss type changes in the bottom beam
        if bottom_beam[i-1] != bottom_beam[i]:
            beam_continuity_metric += 1

    return beam_continuity_metric

def create_graph(data_nodes_1, data_nodes_2,
                 data_nodes_1_coord, data_nodes_2_coord,
                 mass, truss_types_data=None):
    """
    Creates a NetworkX graph from node and edge data, incorporating node positions
    and edge attributes such as element ID, mass, and truss type.

    Parameters:
    data_nodes_1 (pd.DataFrame): DataFrame containing edges and first node identifiers.
    data_nodes_2 (pd.DataFrame): DataFrame containing second node identifiers.
    data_nodes_1_coord (pd.DataFrame): Coordinates for the first set of nodes.
    data_nodes_2_coord (pd.DataFrame): Coordinates for the second set of nodes.
    truss_types_data (pd.DataFrame, optional): DataFrame containing truss type information for edges.

    Returns:
    tuple:
        - G (networkx.Graph): The constructed graph with nodes and edges.
        - data_nodes (pd.DataFrame): Processed DataFrame containing edge and node information.
    """
    G = nx.Graph()

    # Merge node and coordinate data into a single DataFrame
    data_nodes = pd.concat([
        data_nodes_1.iloc[:, :2],  # Edge and first node
        data_nodes_1_coord.iloc[:, 1],  # First node coordinates
        data_nodes_2.iloc[:, 1],  # Second node
        data_nodes_2_coord.iloc[:, 1]  # Second node coordinates
    ], axis=1)
    # Append truss type data if provided
    data_nodes = pd.concat([data_nodes, truss_types_data], axis=1)

    # Assign column names for clarity
    data_nodes.columns = ["edge", "node_1", "coord_node_1", "node_2", "coord_node_2", "truss_type"]

    # Get unique nodes from node_1 and node_2 columns
    unique_nodes = pd.concat([data_nodes["node_1"], data_nodes["node_2"]]).drop_duplicates().values

    # Get unique coordinate values, removing duplicates
    unique_coordinates = pd.concat([data_nodes["coord_node_1"], data_nodes["coord_node_2"]]).drop_duplicates().values

    # Clean coordinate data and convert to tuples
    unique_coordinates = [
        coordinates_tuple.replace(',', '.')
                         .replace('{', '').replace('}', '')
                         .split("_")
        for coordinates_tuple in unique_coordinates
    ]
    unique_coordinates = [(float(coord[0]), float(coord[2])) for coord in unique_coordinates]

    # Add nodes with position attributes
    for node, coordinates in zip(unique_nodes, unique_coordinates):
        G.add_node(node, pos=coordinates)

    # Add edges with attributes (element_id, mass, truss_type)
    for _, row in data_nodes.iterrows():
        if pd.notna(row["truss_type"]):  # Ensure truss type exists before adding edge
            G.add_edge(
                row["node_1"],
                row["node_2"],
                element_id=row["edge"],
                mass=mass.loc[int(row["edge"]) - 1, row["truss_type"]],  # Assuming 'mass' is defined globally
                truss_type=row["truss_type"]
            )
    return G, data_nodes

