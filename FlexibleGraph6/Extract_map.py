# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:44:14 2023

@author: levi1
"""

# Import packages
import matplotlib.pyplot as plt
import osmnx as ox
import math
import networkx as nx
import pandas as pd
import geopandas as gpd
import geopy.distance

#Airport inputs
airport_name = 'Schiphol, The Netherlands'
runways = ['18R/36L','18C/36C','06/24']

# Load the graph 
G = ox.graph_from_place(airport_name,custom_filter = '["aeroway"~"runway|taxiway|service_.*"]', simplify=True)
G_data = ox.utils_graph.graph_to_gdfs(G)
G_data_nodes , G_data_edges = G_data
ox.plot_graph(G)

# Initialize 
node_coordinates = []

# Iterate through the nodes and extract their coordinates
for node, data in G.nodes(data=True):
    x = data['x']  # Longitude (x-coordinate)
    y = data['y']  # Latitude (y-coordinate)
    
    # Append the coordinates to the list
    node_coordinates.append((x, y))
    
# Initialize
transformed_coordinates = []

for node, data in G.nodes(data=True):
    x = data['x']  # Longitude (x-coordinate)
    y = data['y']  # Latitude (y-coordinate)
    
    latitude0 = math.radians(52.308933);
    longitude0 = math.radians(4.762055);
    
    RADIUS_OF_THE_EARTH = 6371000.0;
    
    transformed_x = RADIUS_OF_THE_EARTH * (math.radians(x) - longitude0) * math.cos(latitude0);
    transformed_y = RADIUS_OF_THE_EARTH * (math.radians(y) - latitude0);
        
    # Append the transformed coordinates to the list
    transformed_coordinates.append((transformed_x, transformed_y))

# Now, transformed_coordinates contains a list of (x, y) coordinates in meters for all nodes
# print(transformed_coordinates)

orig = list(G)[600]
dest = list(G)[150]

# calculate two routes by minimizing travel distance vs travel time
route1 = nx.shortest_path(G, orig, dest, weight='length')
route1_length = nx.shortest_path_length(G, orig, dest, weight='length')
route1_data = ox.utils_graph.route_to_gdf(G, route1, weight='length')

# Find the second shortest path
route2 = None
route2_length = float('inf')

# Iterate through the edges in the first shortest path
for i in range(len(route1) - 1):
    u = route1[i]
    v = route1[i + 1]
    
    # Create a copy of the graph with the edge (u, v) removed
    modified_G = G.copy()
    modified_G.remove_edge(u, v)
    
    # Calculate the shortest path in the modified graph
    try:
        path = nx.shortest_path(modified_G, orig, dest, weight='length')
        length = nx.shortest_path_length(modified_G, orig, dest, weight='length')
        
        # Check if this path is shorter than the current second shortest path
        if length < route2_length:
            route2 = path
            route2_length = length
    except nx.NetworkXNoPath:
        # No path found in the modified graph
        continue

# Print the second shortest path and its length
fig, ax = ox.plot_graph_route(G, route1,  route_color='r' ,route_linewidth=6, node_size=10, bgcolor='k')
fig, ax = ox.plot_graph_route(G, route2,  route_color='b' ,route_linewidth=6, node_size=10, bgcolor='k')

# compare the two routes
print('Route 1 is', route1_length, 'meters')
print("Route 2 is", route2_length, 'meters')

# find runway nodes
if len(runways) <= 1:
    filter_condition1 = G_data_edges['ref'] == runways[0]
    filter_condition = filter_condition1
elif len(runways) <= 2:
    filter_condition1 = G_data_edges['ref'] == runways[0]
    filter_condition2 = G_data_edges['ref'] == runways[1]
    filter_condition = filter_condition1 | filter_condition2
else:
    filter_condition1 = G_data_edges['ref'] == runways[0]
    filter_condition2 = G_data_edges['ref'] == runways[1]
    filter_condition3 = G_data_edges['ref'] == runways[2]
    filter_condition = filter_condition1 | filter_condition2 | filter_condition3



# Apply the filter to the GeoDataFrame
runway_polderbaan_nodes = G_data_edges[filter_condition]
list_runway_polderbaan_nodes = runway_polderbaan_nodes.index.get_level_values('u').to_list()

# Plot the graph with highlighted nodes
node_colors = {node: 'r' if node in list_runway_polderbaan_nodes else 'w' for node in G.nodes()}
fig, ax = ox.plot_graph(G, node_color=list(node_colors.values()), bgcolor='k', show=True)
