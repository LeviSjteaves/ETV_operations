# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:59:56 2023

@author: levi1
"""

import osmnx as ox
import pandas as pd
import networkx as nx

airport_name = 'Schiphol, The Netherlands'                  #Name of airport for usecase
runways = ['18R/36L','18C/36C','06/24','09/27','18L/36R']   #Names of the runways of that airport
operational_gates = ['H1']
  
# Load the graph for taxiways and serviceways
G_taxi = ox.graph_from_place(airport_name,custom_filter = '["aeroway"~"runway|taxiway"]', simplify=True)

#Create taxi network
G_data = ox.utils_graph.graph_to_gdfs(G_taxi)
G_data_nodes , G_data_edges = G_data

node_nr_id = {node: i for i, node in enumerate(G_taxi.nodes())}


nodes_AMS_x = G_data_nodes.get('x').to_list()
nodes_AMS_y = G_data_nodes.get('y').to_list()

nodes_AMS_data = {'x': nodes_AMS_x, 'y': nodes_AMS_y }
nodes_AMS_a_df = pd.DataFrame(nodes_AMS_data)

edges_AMS_1 = G_data_edges.index.get_level_values('u').to_list()
edges_AMS_2 = G_data_edges.index.get_level_values('v').to_list()

for i in range(len(edges_AMS_1)):
    edges_AMS_1[i] = node_nr_id.get(edges_AMS_1[i])
    edges_AMS_2[i] = node_nr_id.get(edges_AMS_2[i])

edges_AMS_data = {'Endnodes1': edges_AMS_1, 'Endnodes2':edges_AMS_2, 'Weight':G_data_edges.get('length').to_list()}
edges_AMS_a_df = pd.DataFrame(edges_AMS_data)

G_AMS_a = nx.from_pandas_edgelist(edges_AMS_a_df, 'Endnodes1', 'Endnodes2', edge_attr='Weight', create_using=nx.Graph)

# Remove nodes
node_to_remove = [74]


# Add edges between all pairs of neighbors
for k in range(len(node_to_remove)):
    neighbors = list(G_AMS_a.neighbors(node_to_remove[k]))
    for i in range(len(neighbors)):
        for j in range(len(neighbors)):
            if i!=j:
                sum_of_weights = G_AMS_a[neighbors[i]][node_to_remove[k]]['Weight']+G_AMS_a[neighbors[j]][node_to_remove[k]]['Weight']
                G_AMS_a.add_edge(neighbors[i], neighbors[j], weight=sum_of_weights)
    G_AMS_a.remove_node(node_to_remove[k])

# Remove the node


#nodes_AMS_a_mat = pd.read_csv('EHEH_nodes_a.csv')
node_positions = {node_id: [float(x), float(y)] for node_id, (x, y) in nodes_AMS_a_df[['x', 'y']].iterrows()}
nx.draw(G_AMS_a, pos=node_positions, with_labels=True, node_size= 25, font_size= 7)


fig, ax = ox.plot_graph(G_taxi, node_color='b', bgcolor='w', edge_color = 'k' ,show=True)






















