# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:08:12 2024

@author: levi1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import networkx as nx
from matplotlib.colors import ListedColormap

global iterations,Runtime,taxi_delay,total_delay,Kg_kerosene,Costs_etvs,ETVs_cat1,ETVs_cat2,allowed_delay,N_aircraft,obj,gap,bounds,times,gap_per,Kg_frac,gap_percentage

# Define global variables
iterations = []
Runtime = []
taxi_delay = []
total_delay = []
Kg_kerosene = []
Costs_etvs = []
ETVs_cat1 = []
ETVs_cat2 = []
allowed_delay = []
N_aircraft = []
obj = []
gap = []
bounds = []
times = []
gap_per = []
Kg_frac = []
gap_percentage = []

def Load_new_file(folder_name):
    global iterations,Runtime,taxi_delay,total_delay,Kg_kerosene,Costs_etvs,ETVs_cat1,ETVs_cat2,allowed_delay,N_aircraft,obj,gap,bounds,times,gap_per,Kg_frac,gap_percentage
    # Initialize an empty list to store the loaded dictionaries
    iterations.clear()
    Runtime.clear()
    taxi_delay.clear()
    total_delay.clear()
    Kg_kerosene.clear()
    Costs_etvs.clear()
    ETVs_cat1.clear()
    ETVs_cat2.clear()
    allowed_delay.clear()
    N_aircraft.clear()
    obj.clear()
    gap.clear()
    bounds.clear()
    times.clear()
    gap_per.clear()
    Kg_frac.clear()
    gap_percentage.clear()
    
    files = os.listdir(folder_name)
    # Load each file from the folder
    for file in files:
        if file.endswith('.npy'):  # Check if the file is a numpy file
            loaded_dict = np.load(os.path.join(folder_name, file), allow_pickle=True).item()  # Load the dictionary
            iterations.append(loaded_dict)  # Append the loaded dictionary to the list
    # Define global variables
    # Clear previous data

    Runtime = [iteration['Runtime'] for iteration in iterations] 
    taxi_delay = [iteration['Taxi_delay'] for iteration in iterations]
    total_delay = [iteration['Total_delay'] for iteration in iterations]
    Kg_kerosene = [iteration['Kg_kerosene'] for iteration in iterations]  
    Costs_etvs = [iteration['Costs_etvs'] for iteration in iterations]  
    ETVs_cat1 = [iteration['p']['N_etvs_cat1'] for iteration in iterations]
    ETVs_cat2 = [iteration['p']['N_etvs_cat2'] for iteration in iterations]
    allowed_delay = [iteration['p']['start_delay'] for iteration in iterations]
    N_aircraft = [iteration['p']['N_aircraft'] for iteration in iterations]
    obj = [iteration['obj_values'] for iteration in iterations]
    gap = [iteration['gap_values'] for iteration in iterations]
    bounds = [iteration['bound_values'] for iteration in iterations]
    times = [iteration['times'] for iteration in iterations]
    gap_per = []
        
    gap_percentage = []
    for i in range(len(iterations)):
        gap_per = [abs(bounds[i][-1] - obj[i][-1]) / abs(obj[i][-1])*100]
        gap_percentage.append(gap_per)
        
folder_name = "2024-03-23_00-09-53"
Load_new_file(folder_name)

i=0  
I_up = iterations[i]['I_up'] 
I_do = iterations[i]['I_do']
t_min = iterations[i]['t_min']
appear_times = iterations[i]['appear_times'] 
G_a = iterations[i]['G_a']
cat = iterations[i]['cat']
dep = iterations[i]['dep'] 
E_a_dis =iterations[i]['E_a_dis']
E_e_return = iterations[i]['E_e_return']
variable_values = iterations[i]['variable_values']
P = iterations[i]['P']
p = iterations[i]['p']
      
towed=[]   
for a in range(N_aircraft[0]):
    towed.append(1) 
    
# Initialize node_visits with all nodes and zero visits
node_visits = {node: 0 for node in G_a.nodes}
node_visits_tot = {node: 0 for node in G_a.nodes}

# Iterate over each aircraft's path
for i, path in enumerate(P):
    # Increment the count of visits for nodes visited by any aircraft
    for node in path:
        node_visits_tot[node] = node_visits_tot.get(node, 0) + 1
    # Check if the aircraft is towed
    if towed[i] == 1:
        # Increment the count of visits for nodes visited by towed aircraft
        for node in path:
            node_visits[node] += 1

edge_tuples = [(edge[0], edge[1]) for edge in G_a.edges]        
edge_visits = {edge: 0 for edge in edge_tuples}
edge_visits_tot = {edge: 0 for edge in edge_tuples}
# Iterate over each aircraft's path
for i, path in enumerate(P):
    # Check if the aircraft is towed
    for j in range(len(path)-1):
        # Define the current edge
        edge = (path[j], path[j+1])
        # Increment the count of visits for the edge
        edge_visits_tot[edge] = edge_visits_tot.get(edge, 0) + 1
    if towed[i] == 1:
        # Iterate over the nodes in the path
        for j in range(len(path)-1):
            # Define the current edge
            edge = (path[j], path[j+1])
            # Increment the count of visits for the edge
            edge_visits[edge] = edge_visits.get(edge, 0) + 1


per_node_visit = [int(node_visits[i]/node_visits_tot[i] *100) if node_visits_tot[i] > 0 else 0 for i in node_visits]
per_edge_visit = [int(edge_visits[i]/edge_visits_tot[i] *100) if edge_visits_tot[i] > 0 else 0 for i in edge_visits]

# Define a list of edge colors and widths based on visit frequency
blue_cmap_tot = plt.get_cmap('rainbow')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.2,0.9, max(edge_visits.values()))))
widthmap = np.linspace(0.5,10, max(edge_visits.values())+1)


edge_widths = []

for edge in edge_tuples:
    if edge in edge_visits:
        visits = edge_visits[edge]
        edge_widths.append(widthmap[visits]) 
    else:
        edge_widths.append(1.5)  

node_colors = []
node_widths = []
sizemap = np.linspace(10,25, max(node_visits.values()))

for node in G_a.nodes:
    if node in node_visits:
        visits = node_visits[node]
        node_widths.append(3)  
        node_colors.append('grey')
    else:
        node_widths.append(1)  
        node_colors.append('grey')


towed=[]   
for a in range(N_aircraft[0]):
    towed.append(sum(variable_values['X'][a][i] + sum(variable_values['O'][a][I_up[a][b]][i] for b in range(len(I_up[a]))) for i in range(ETVs_cat1[0]+ETVs_cat2[0])))    

# Initialize node_visits with all nodes and zero visits
node_visits = {node: 0 for node in G_a.nodes}
node_visits_tot = {node: 0 for node in G_a.nodes}

# Iterate over each aircraft's path
for i, path in enumerate(P):
    # Increment the count of visits for nodes visited by any aircraft
    for node in path:
        node_visits_tot[node] = node_visits_tot.get(node, 0) + 1
    # Check if the aircraft is towed
    if towed[i] == 1:
        # Increment the count of visits for nodes visited by towed aircraft
        for node in path:
            node_visits[node] += 1
            

edge_tuples = [(edge[0], edge[1]) for edge in G_a.edges]        
edge_visits = {edge: 0 for edge in edge_tuples}
edge_visits_tot = {edge: 0 for edge in edge_tuples}
# Iterate over each aircraft's path
for i, path in enumerate(P):
    # Check if the aircraft is towed
    for j in range(len(path)-1):
        # Define the current edge
        edge = (path[j], path[j+1])
        # Increment the count of visits for the edge
        edge_visits_tot[edge] = edge_visits_tot.get(edge, 0) + 1
    if towed[i] == 1:
        # Iterate over the nodes in the path
        for j in range(len(path)-1):
            # Define the current edge
            edge = (path[j], path[j+1])
            # Increment the count of visits for the edge
            edge_visits[edge] = edge_visits.get(edge, 0) + 1

per_node_visit = [int(node_visits[i]/node_visits_tot[i] *100) if node_visits_tot[i] > 0 else 0 for i in node_visits]
#per_edge_visit = [int(edge_visits[i]/edge_visits_tot[i] *100) if edge_visits_tot[i] > 0 else 0 for i in edge_visits]
per_edge_visit = {(edge[0], edge[1]): int(edge_visits[edge]/edge_visits_tot[edge] * 100) if edge_visits_tot[edge] > 0 else 0 for edge in edge_visits}

max_scale = 35
blue_cmap_tot = plt.get_cmap('rainbow')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.2,0.9, max_scale)))

edge_colors = {}

for edge in per_edge_visit:
    visits = per_edge_visit[edge]
    if visits > 0:      
        edge_colors[edge] = blue_cmap(visits)
    else:
        edge_colors[edge] = 'grey'

fig, ax = plt.subplots(figsize=(18/2.54, 12/2.54))
node_positions_a = nx.get_node_attributes(G_a, 'pos')
# Add edges with custom colors and widths

nx.draw_networkx_edges(G_a, pos=node_positions_a, ax=ax, edge_color=[edge_colors[edge] for edge in edge_tuples], width=edge_widths, style='solid', arrows=False)
nx.draw_networkx_nodes(G_a, pos=node_positions_a, ax=ax, node_color=node_colors, node_size=node_widths)    

sm = ScalarMappable(cmap=blue_cmap, norm=plt.Normalize(vmin=0, vmax=max_scale))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label(r'Percentage towed $[\%]$')

# Sort and get unique edge widths
unique_widths = sorted(set(edge_widths))
unique_visits = sorted(set(edge_visits_tot.values()))

'''
# Create a custom legend for edge widths
for width in unique_widths:
    plt.plot([], [], linestyle='-', linewidth=width, label=f'Edge Width: {width}', color='grey')
# Add the legend
plt.legend(title='Edge Widths', loc='upper left')
'''
ax.set_aspect('equal')
ax.set_aspect('equal')

#edge_per_filtered = {edge: per_edge_visit[i] for i, edge in enumerate(edge_visits) if per_edge_visit[i] != 0}
# Draw edge labels for filtered edges
#nx.draw_networkx_edge_labels(G_a,horizontalalignment='center', pos=node_positions_a, edge_labels=edge_per_filtered, rotate=False, font_color='black',bbox=dict(facecolor='none',edgecolor='none'))

plt.show()