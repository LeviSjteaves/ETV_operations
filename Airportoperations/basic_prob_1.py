# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:17:48 2023

@author: levi1
"""

from gurobipy import GRB
import networkx as nx
from Functions_basicprob import Create_model
from Functions_basicprob import Plotting
from Functions_basicprob import Load_Graph
from Functions_basicprob import Load_Aircraft_Info
import numpy as np

#Choose from: basic:'basic', Eindhoven:'EHEH', Schiphol: 'EHAM'
airport = 'EHAM'
#Choose from: Manually insert flight info:'manual', use Schiphol flight API:'API', import saved document 'saved' 
setting = 'API'
    
 
# Parameters
g=9.81
# Aircraft info
max_speed_a = 10      # Max aircraft velocity
min_speed_a = 1    # Min aircraft velocity
start_delay = 120    # Max allowed start delay ('start taxi time' - 'appear time') in seconds
mu = 0.02            # Rolling resistance
m_a = 40000          # Airplane mass
eta = 0.3            # Turbine efficiency
dock = [44]           # Node corresponding to charging dock
# ETV info
N_etvs = 3     # Number of ETVs
speed_e = 15         # ETV velocity
bat_e = 50000000      #battery capacity

# Load aircraft info

G_a, G_e, gate_runway_locs = Load_Graph(airport)

O_a = []
D_a = []

if setting == 'manual':
    O_a = [57, 57, 57, 9, 57, 57, 57, 57]   # Origins
    D_a = [9, 9, 9, 57, 9, 9, 9, 9]   # Destinations
    tO_a = [0, 0, 60, 30, 0, 0, 30, 30]  # Appearing times
    N_aircraft = len(O_a)# Number of aircraft
    
elif setting == 'API':
    date_of_interest = '2023-11-01'     #Pick the date for which you want the API data
    pagelimit = [20,21]                   #Specify the amount of pages of data
    
    Flight_orig, Flight_dest, appear_times = Load_Aircraft_Info(date_of_interest, pagelimit)
    
    tO_a = appear_times
    
    for i in range(len(Flight_orig)):
        O_a.append(gate_runway_locs.get(Flight_orig[i]))   # Origins
        D_a.append(gate_runway_locs.get(Flight_dest[i]))   # Destinations
    
    N_aircraft = 10# Number of aircraft
    #N_aircraft = len(O_a)# Number of aircraft
    
elif setting == 'saved':
    gate_runway_locs = {'A':80,'B':82, 'C':83, 'D':84, 'E':56, 'F':55, 'G':54, 'H':53, 'J':52, 'P':52,
                        'K':99, 'M':102, 'R':79, 'S':107,'18R':3, '18L':92, '18C':22, '24':103, '22':98}
    
    tO_a = np.load('Flight_t.npy')
    Flight_orig = np.load('Flight_O.npy')
    Flight_dest = np.load('Flight_D.npy')
    
    for i in range(len(Flight_orig)):
        O_a.append(gate_runway_locs.get(Flight_orig[i]))   # Origins
        D_a.append(gate_runway_locs.get(Flight_dest[i]))   # Destinations
    
    N_aircraft = 10# Number of aircraft
    #N_aircraft = len(O_a)# Number of aircraft
    
else:
     print("Please specify the method to get flight info")   

p = {}
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

print("DATA EXTRACTED")
# Create the Gurobi model
model, I_up, I_do = Create_model(G_a, G_e, p, P, tO_a, O_a, D_a, d_a, dock, start_delay)

print("MODEL CREATED")
# Optimize the model
model.optimize()

print("MODEL SOLVED")
# Display results
variable_values = {}

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for var in model.getVars():
        #print(f"{var.varName}: {var.x}")
        # Split variable name into parts based on underscores and convert indices to integers
        parts = [int(part) if part.isdigit() else part for part in var.varName.split('_')]

        # Create nested dictionaries for each part
        current_dict = variable_values
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Assign the variable value to the nested dictionary
        current_dict[parts[-1]] = round(var.x)

    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")

for a in range(N_aircraft):
   for i in range(N_etvs): 
       if variable_values['X'][a][i] == 1:
           print(f"{'X'}_{a}_{i}")
       for b in range(len(I_up[a])):    
            if variable_values['O'][a][I_up[a][b]][i] == 1:
                print(f"{'O'}_{a}_{I_up[a][b]}_{i}")
                
    
Plotting(variable_values, N_aircraft, N_etvs, P, bat_e, I_up, p, d_a,  appear_times)                 
         
    
    
