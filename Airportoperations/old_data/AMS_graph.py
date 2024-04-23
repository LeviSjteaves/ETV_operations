# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:59:56 2023

@author: levi1
"""

import osmnx as ox
import pandas as pd
import networkx as nx
from geopy.distance import geodesic

airport_name = 'Schiphol, The Netherlands'                  #Name of airport for usecase
runways = ['18R/36L','18C/36C','06/24','09/27','18L/36R']   #Names of the runways of that airport
operational_gates = ['H1']
  
# Load the graph for taxiways and serviceways
G_taxi_D = ox.graph_from_place(airport_name,custom_filter = '["highway"~"service"]', simplify=False)

G_taxi = ox.projection.project_graph(G_taxi_D, to_crs='EPSG:3395')

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

edges_AMS_data = {'Endnodes1': edges_AMS_1, 'Endnodes2':edges_AMS_2, 'weight':G_data_edges.get('length').to_list()}
edges_AMS_a_df = pd.DataFrame(edges_AMS_data)

G_AMS_a = nx.from_pandas_edgelist(edges_AMS_a_df, 'Endnodes1', 'Endnodes2', edge_attr='weight', create_using=nx.Graph)

'''
# Remove nodes
nodes_to_remove = [28,22, 201, 185, 205, 204, 479]
nodes_to_merge = []
nodes_to_merge.append([189, 192, 190, 478])
nodes_to_merge.append([195, 485, 29,  166])
nodes_to_merge.append([21,20,497,493,30])
nodes_to_merge.append([492,533,509,26,496])
nodes_to_merge.append([460,462,455,456,91,37,457,458,90,459])
nodes_to_merge.append([23,159,150,157,465,489,463,464,158])
nodes_to_merge.append([469,472,468,96,466])
nodes_to_merge.append([480,470,467,471,481,461,94])
nodes_to_merge.append([512,216,228,513,227])
nodes_to_merge.append([95,215,207,211,214,209,210,212])
nodes_to_merge.append([220,217,222,218,221,219,610,611,213])
nodes_to_merge.append([155,231,230,233,229])


for i in range(len(nodes_to_merge)):
    while 0 < len(nodes_to_merge[i])-1:
        G_AMS_a = nx.contracted_nodes(G_AMS_a, nodes_to_merge[i][-1], nodes_to_merge[i][0], self_loops=False)
        nodes_to_merge[i].pop(0) 

  
for k in range(len(nodes_to_remove)):  
    neighbors = list(G_AMS_a.neighbors(nodes_to_remove[k]))
    for i in range(len(neighbors)):
        for j in range(len(neighbors)):
            if i!=j:
                sum_of_weights = G_AMS_a[neighbors[i]][nodes_to_remove[k]]['weight']+G_AMS_a[neighbors[j]][nodes_to_remove[k]]['weight']
                G_AMS_a.add_edge(neighbors[i], neighbors[j], weight=sum_of_weights)
    G_AMS_a.remove_node(nodes_to_remove[k])
'''    
  
node_positions = {node_id: [float(x), float(y)] for node_id, (x, y) in nodes_AMS_a_df[['x', 'y']].iterrows()}
nx.draw(G_AMS_a, pos=node_positions, with_labels=True, node_size= 25, font_size= 12)
























