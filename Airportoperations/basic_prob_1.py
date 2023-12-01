# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:17:48 2023

@author: levi1
"""

from gurobipy import GRB
import networkx as nx
import pandas as pd
import mplcursors
import scipy.io
import matplotlib.pyplot as plt
from Functions_basicprob import Create_model
from Functions_basicprob import Short_path_dist
from Functions_basicprob import Plotting

# Load CSV file with EHEH edges
edges_EHEH_a_df = pd.read_csv('EHEH_a.csv')
G_EHEH_a = nx.from_pandas_edgelist(edges_EHEH_a_df, 'EndNodes_1', 'EndNodes_2', edge_attr='Weight', create_using=nx.Graph)
edges_EHEH_e_df = pd.read_csv('EHEH_e.csv')
G_EHEH_e = nx.from_pandas_edgelist(edges_EHEH_e_df, 'EndNodes_1', 'EndNodes_2', create_using=nx.Graph)

nodes_EHEH_a_mat = pd.read_csv('EHEH_nodes_a.csv')
node_positions = {node_id+1: [float(x), float(y)] for node_id, (x, y) in nodes_EHEH_a_mat[['x', 'y']].iterrows()}
nx.draw(G_EHEH_a, pos=node_positions, with_labels=True, node_size= 25, font_size= 7)

# Show the plot inline
plt.show()

# Airport info
G_basic_a = nx.Graph()
G_basic_a.add_edges_from([(0, 2,{'weight': 10}), (1, 2,{'weight': 10}), (2, 3,{'weight': 10}), (3, 4,{'weight': 10}), (3, 5,{'weight': 10})])
G_basic_e = nx.Graph()
G_basic_e.add_edges_from([(0, 1,{'weight': 20}), (1, 5,{'weight': 20}), (5, 4,{'weight': 20}), (4, 0,{'weight': 20})])
p = {}

# Graph
G_a = G_basic_a
G_e = G_basic_e

g=9.81

# Aircraft info
max_speed_a = 1      # Max aircraft velocity
min_speed_a = 0.1    # Min aircraft velocity
O_a = [4, 4, 4, 4, 4, 4, 9, 13]   # Origins
D_a = [0, 0, 0, 0, 0, 0, 57, 57]   # Destinations
tO_a = [0, 0, 0, 0, 0, 0, 0, 0]  # Appearing times
N_aircraft = 6# Number of aircraft
#N_aircraft = len(O_a)# Number of aircraft
mu = 0.02            # Rolling resistance
m_a = 40000          # Airplane mass
eta = 0.3            # Turbine efficiency

# ETV info
N_etvs = 2      # Number of ETVs
speed_e = 3         # ETV velocity
bat_e = 500000      #battery capacity

#pack p
p['N_aircraft'] = N_aircraft
p['max_speed_a'] = max_speed_a      # Max aircraft velocity
p['min_speed_a'] = min_speed_a 
p['N_etvs'] = N_etvs         # Number of ETVs
p['speed_e'] = speed_e
p['bat_e'] = bat_e 
p['g'] = g
p['mu'] = mu          # Rolling resistance
p['m_a'] = m_a          # Airplane mass
p['eta'] = eta 

# Aircraft routing
N_Va = []
P = []
d_a = []

for a in range(N_aircraft):
    sp = nx.shortest_path(G_a, source=O_a[a], target=D_a[a])
    d_a.append(nx.shortest_path_length(G_a, source=O_a[a], target=D_a[a], weight='weight'))
    N_Va.append(len(sp))
    P.append(sp)


# Create the Gurobi model
model = Create_model(G_a, G_e, p, P, tO_a, O_a, D_a, d_a)

# Optimize the model
model.optimize()

# Display results
variable_values = {}

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for var in model.getVars():
        print(f"{var.varName}: {var.x}")

        # Split variable name into parts based on underscores and convert indices to integers
        parts = [int(part) if part.isdigit() else part for part in var.varName.split('_')]

        # Create nested dictionaries for each part
        current_dict = variable_values
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Assign the variable value to the nested dictionary
        current_dict[parts[-1]] = var.x

    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")

for a in range(N_aircraft):
   for i in range(N_etvs): 
       for b in range(N_aircraft):    
            if variable_values['O'][a][b][i] == 1:
                print(f"{'O'}_{a}_{b}_{i}")
    
Plotting(variable_values, N_aircraft, N_etvs, P)                 
         
    
    
