# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:17:48 2023

@author: levi1
"""



from gurobipy import GRB
import networkx as nx
from importlib import import_module
import sys

#model 1 for time-dep energycons. 
#model 2 for dist-dep energycons.
M = 2
module = import_module(f'AirportOp_Func_{M}')
Load_Aircraft_Info = module.Load_Aircraft_Info
Create_model = module.Create_model
Plotting = module.Plotting
Load_Graph = module.Load_Graph
Short_path_dist = module.Short_path_dist

#from f'AirportOp_Func_{M}' import Load_Aircraft_Info
import numpy as np
import matplotlib.pyplot as plt

#close all open plots
plt.close('all')

#Choose from: basic:'basic', Eindhoven:'EHEH', Schiphol: 'EHAM'
airport = 'EHAM'
#Choose from: Manually insert flight info:'manual', use Schiphol flight API:'API', import saved document 'saved' 
setting = 'API'
    
# Parameters
g=9.81
# Aircraft parameters
max_speed_a = 996      # Max aircraft velocity m/min
min_speed_a = 60         # Min aircraft velocity m/min
max_speed_e = 534           #m/min
min_speed_e = 60            #m/min
free_speed_e = 534          #m/min
start_delay = 30   # Max allowed start delay ('start taxi time' - 'appear time') in minutes
mu = 0.02            # Rolling resistance
m_a = [10000, 20000, 45000, 70000, 120000, 180000, 240000, 300000, 380000, 450000]        # Airplane mass in tons
fuelmass = [2000, 4000, 10000, 26000, 42000, 65000, 100000, 130000, 250000, 280000]       # fuel mass in tons
eta = 0.3            # Turbine efficiency
dock = [143,110,119]           # Node corresponding to charging dock  
N_etvs_cat1 = 1    # Number of ETVs of category 1
N_etvs_cat2 = 0   # Number of ETVs of category 2
bat_e1 = 576      #battery capacity \MJoule
bat_e2 = 864      #battery capacity \MJoule
eta_e = 0.9
I_ch = 1.32 # MJoule/min     
E_e = 0.0039 # etvs energy consumption Mjoule per unit distance[m]
E_a = 0.0001048 # aircrafts energy consumption Mjoule per kg per unit time[min]
d_sep = [22, 28, 37, 45, 49, 55, 72, 76, 77, 84]
v_avg = 534 #m/min
t_pushback = 1 #pushback time in minutes
T_charge_min = 1 #min charging time in minutes
e_KE = 43.1 # MJ/Kg
F_delay = 0.0001

# Load aircraft info
N_etvs = N_etvs_cat1+N_etvs_cat2
bat_e = []
bat_e_min =[]
bat_e_max =[]
for i in range(N_etvs_cat1):
    bat_e.append(bat_e1)
    bat_e_min.append( 0.2*bat_e1)
    bat_e_max.append( 0.9*bat_e1)
for i in range(N_etvs_cat2):
    bat_e.append(bat_e2) 
    bat_e_min.append(0.2*bat_e2)
    bat_e_max.append(0.9*bat_e2)


G_a, G_e = Load_Graph(airport)

O_a = []
D_a = []

if setting == 'manual':
    O_a = [57, 57, 9, 9]   # Origins
    D_a = [9, 9, 57, 57]   # Destinations
    tO_a = [100, 100, 100, 100]  # Appearing times
    dep = [0, 0, 1, 1] 
    cat = [3, 3, 3, 3] 
    N_aircraft = len(O_a)# Number of aircraft
    appear_times = tO_a
   
elif setting == 'API':
    date_of_interest = '2023-11-01'     #Pick the date for which you want the API data
    pagelimit = [20,20]                   #Specify the amount of pages of data [-1] for last page
    
    Flight_orig, Flight_dest, appear_times, dep, cat, flightdata = Load_Aircraft_Info(date_of_interest, pagelimit)
    
    tO_a = appear_times
    gate_runway_locs = {'A':80,'B':82, 'C':83, 'D':84, 'E':56, 'F':55, 'G':54, 'H':53, 'J':52, 'P':52,
                        'K':99, 'M':102, 'R':79, 'S':107,'18R':3, '18L':92, '18C':22, '24':108, '22':98, 'HG':98}
    
    for i in range(len(Flight_orig)):
        O_a.append(gate_runway_locs.get(Flight_orig[i]))   # Origins
        D_a.append(gate_runway_locs.get(Flight_dest[i]))   # Destinations
    
    if None in O_a or None in D_a:
        print('Check if all used runways are saved in gate_runway_locs')
        sys.exit()
        
    #N_aircraft = 6# Number of aircraft
    N_aircraft = len(O_a)# Number of aircraft
    
elif setting == 'saved':
    gate_runway_locs = {'A':80,'B':82, 'C':83, 'D':84, 'E':56, 'F':55, 'G':54, 'H':53, 'J':52, 'P':52,
                        'K':99, 'M':102, 'R':79, 'S':107,'18R':3, '18L':92, '18C':22, '24':108, '22':98, 'HG':98}
    
    tO_a = np.load('Flight_t.npy')
    appear_times = tO_a
    Flight_orig = np.load('Flight_O.npy')
    Flight_dest = np.load('Flight_D.npy')
    dep = np.load('Flight_dep.npy')
    cat = np.load('Flight_info.npy')
    
    for i in range(len(Flight_orig)):
        O_a.append(gate_runway_locs.get(Flight_orig[i]))   # Origins
        D_a.append(gate_runway_locs.get(Flight_dest[i]))   # Destinations
    
    N_aircraft = 1# Number of aircraft
    #N_aircraft = len(O_a)# Number of aircraft
else:
    print("Please specify the method to get flight info")   

p = {}
#pack p
p['N_aircraft'] = N_aircraft
p['max_speed_a'] = max_speed_a      # Max aircraft velocity
p['min_speed_a'] = min_speed_a 
p['N_etvs'] = N_etvs                # Number of ETVs
p['N_etvs_cat1'] = N_etvs_cat1
p['N_etvs_cat2'] = N_etvs_cat2
p['max_speed_e'] = max_speed_e  
p['min_speed_e'] = min_speed_e 
p['free_speed_e'] = free_speed_e  
p['bat_e'] = bat_e 
p['bat_e_min'] = bat_e_min 
p['bat_e_max'] = bat_e_max
p['g'] = g
p['mu'] = mu                        # Rolling resistance
p['m_a'] = m_a                      # Airplane mass
p['eta'] = eta 
p['eta_e'] = eta_e
p['I_ch'] = I_ch 
p['E_e'] = E_e
p['E_a'] = E_a
p['d_sep'] = d_sep
p['v_avg'] = v_avg 
p['t_pushback'] = t_pushback
p['T_charge_min'] = T_charge_min
p['start_delay'] = start_delay
p['e_KE'] = e_KE
p['fuelmass'] = fuelmass
p['F_delay'] = F_delay

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
model, I_up, I_do, E_a_time, E_e_return  = Create_model(G_a, G_e, p, P, tO_a, O_a, D_a, d_a, dock, dep, cat)

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
        current_dict[parts[-1]] = round(var.x,2)

    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")
    
    
Kg_kerosene = ((model.objVal-F_delay*sum((variable_values['t'][a][len(P[a])-1]) for a in range(N_aircraft)))/e_KE)*100/76
Costs_etvs = int(1*10**6*N_etvs_cat1+1.5*10**6*N_etvs_cat2)


for a in range(N_aircraft):
   for i in range(N_etvs): 
       if variable_values['X'][a][i] == 1:
           print(f"{'X'}_{a}_{i}")
       for b in range(len(I_up[a])):    
            if variable_values['O'][a][I_up[a][b]][i] == 1:
                print(f"{'O'}_{a}_{I_up[a][b]}_{i}")
                
    
Plotting(variable_values, N_aircraft, N_etvs, P, I_up, p, d_a,  appear_times, G_a, cat, dep,  E_a_time, E_e_return )                 
         
print(f"KG Kerosene: {Kg_kerosene}")
print(f"ETV costs: \N{euro sign}{Costs_etvs},-")

    
