import pandas as pd

from graphs import create_graph, visualize_graph

if __name__ == '__main__':
    file_data_name = "data/MY_data.xlsx"
    # read element U.C sheet
    uc = pd.read_excel(file_data_name, sheet_name="Element U.C.")

    # read element mass U.C sheet
    mass = pd.read_excel(file_data_name, sheet_name="Element Mass")
    # read #node_1 & #node_2
    nodes_1 = pd.read_excel(file_data_name, sheet_name="Element # NODE_1")
    nodes_2 = pd.read_excel(file_data_name, sheet_name="Element # NODE_2")

    # read #node_1 & #node_2 coordinates
    nodes_1_coord = pd.read_excel(file_data_name, sheet_name="Element # NODE_1 coord.")
    nodes_2_coord = pd.read_excel(file_data_name, sheet_name="Element # NODE_2 coord.")

    df = pd.DataFrame(["HEA100"] * 9, columns=["truss_type"])

    G, data_nodes = create_graph(nodes_1,
                                 nodes_2,
                                 nodes_1_coord,
                                 nodes_2_coord,
                                 mass, truss_types_data=df)
    visualize_graph(G)