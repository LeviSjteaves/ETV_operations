# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:10:55 2023

@author: levi1
"""

import gurobipy as grp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def Create_model(G_a, G_e, p, P, tO_a, O_a, D_a, d_a, dock):
    model = grp.Model("Aircraft_Taxiing")
    
    # unpack P
    N_aircraft = p['N_aircraft']
    N_etvs = p['N_etvs']
    max_speed_a = p['max_speed_a']
    min_speed_a = p['min_speed_a']
    speed_e = p['speed_e']
    bat_e = p['bat_e']
    g = p['g']
    mu = p['mu']
    m_a = p['m_a']
    eta = p['eta']
    
    # Time definitions
    t_min = []
    t_max = []

    for a in range(N_aircraft):
        t_a_min = [] 
        t_a_max = []
        for n in range(len(P[a])):
            t_a_min.append(tO_a[a] + Short_path_dist(G_a, O_a[a], P[a][n]) / max_speed_a)
            t_a_max.append(tO_a[a] + Short_path_dist(G_a, O_a[a], P[a][n]) / min_speed_a)
        t_min.append(t_a_min)
        t_max.append(t_a_max)
    
    # Definitions
    I_col = [] #set of aircraft with possible collisions
    I_col_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(len(P[a])):
                for m in range(len(P[b])):
                    e_col = []
                    #possibility to be at the same node at the same time
                    if a < b and P[a][n] == P[b][m] and (t_min[a][n] <= t_max[b][m] and t_min[b][m] <= t_max[a][n]):
                        e_col = [a,b]
                        I_col_nodes.append(P[a][n])
                        I_col.append(e_col)
                        
    I_col_ot = [] #set of aircraft with possible overtake collisions
    I_col_ot_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(1, len(P[a])):
                for m in range(1, len(P[b])):
                    if a < b and P[a][n] == P[b][m] and P[a][n-1] == P[b][m-1] and t_min[a][n] <= t_max[b][m] and t_min[b][m-1] <= t_max[a][n-1]:
                       e_col_ot = [a,b]
                       I_col_ot_nodes.append([P[a][n-1],P[a][n]])
                       I_col_ot.append(e_col_ot)
                       
    I_col_ho = [] #set of aircraft with possible head on collisions
    I_col_ho_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(1, len(P[a])):
                for m in range(len(P[b])-1):
                    if a < b and P[a][n] == P[b][m] and P[a][n-1] == P[b][m+1]  and t_min[a][n] <= t_max[b][m] and t_min[b][m+1] <= t_max[a][n-1]:
                       e_col_ho = [a,b]
                       I_col_ho_nodes.append([P[a][n-1],P[a][n]])
                       I_col_ho.append(e_col_ho)   
                
    # Decision variables
    t = {}  # Arrival times nodes
    Z = {}  # Order of visiting nodes
    X = {}  # Last towed aircraft of an ETV
    O = {}  # Order for towing tasks
    C = {}  # Choose to charge
    E = {}  # State of charge before every task
    
    for a in range(N_aircraft):
        for n in range(len(P[a])):
            t[a, n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{a}_{n}")

    for a in range(N_aircraft):
        for i in range(N_etvs):
            X[a, i] = model.addVar(vtype=GRB.BINARY, name=f"X_{a}_{i}")
    
    for n in range(len(I_col_nodes)):
            Z[n] = model.addVar(vtype=GRB.BINARY, name=f"Z_{n}")
               
    for a in range(N_aircraft):
       for i in range(N_etvs): 
           for b in range(N_aircraft):
              O[a,b,i] = model.addVar(lb=0, vtype=GRB.BINARY, name=f"O_{a}_{b}_{i}") 
               
    for a in range(N_aircraft):
       for i in range(N_etvs):           
              E[i, a] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"E_{i}_{a}") 
              
    for a in range(N_aircraft):
       for i in range(N_etvs):          
              C[i,a] = model.addVar(vtype=GRB.BINARY, name=f"C_{i}_{a}")
           
    # Objective function: minimize emissions (only rolling resistance) + total taxitime
    model.setObjective(grp.quicksum(mu*m_a*g*d_a[a]*eta*(1-grp.quicksum(X[a,i] + grp.quicksum(O[a,b,i] for b in range(N_aircraft)) for i in range(N_etvs))) for a in range(N_aircraft)) 
                       +grp.quicksum(t[a, n] for a in range(N_aircraft) for n in range(1, len(P[a])))
                       , sense=GRB.MINIMIZE)
                   #-grp.quicksum(E[i,a] for i in range(N_etvs) for a in range(N_aircraft))   
     
    # Constraints////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for a in range(N_aircraft):
        #Min start time
        model.addConstr(t[a, 0] >= tO_a[a], f"appear_constraint_{a}")
        # max one etv per aircraft
        model.addConstr(grp.quicksum(X[a, i] +grp.quicksum(O[a,b,i] for b in range(N_aircraft)) for i in range(N_etvs)) <= 1, f"one_towing_task_{a}")
        for n in range(1, len(P[a])):
            #Min/max time to arrive at each node (from min/max speed)
            model.addConstr(t[a, n] >= t[a, n-1] + Short_path_dist(G_a, P[a][n-1], P[a][n]) / max_speed_a, f"arrival_constraint_min_{a}_{n}")
            model.addConstr(t[a, n] <= t[a, n-1] + Short_path_dist(G_a, P[a][n-1], P[a][n]) / min_speed_a, f"arrival_constraint_max_{a}_{n}")
    
    # Collision
    for n in range(len(I_col_nodes)):
            model.addConstr((Z[n] == 1) >> (t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] <= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])]), f"auxiliary_order1_node_{n}")   
            model.addConstr((Z[n] == 0) >> (t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] >= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])]), f"auxiliary_order2_node_{n}")   
            model.addConstr(t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] >= t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] +5 - (1-Z[n])*10000, f"collision_conflict_node_{n}") 
            model.addConstr(t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] >= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] +5 - (Z[n])*10000, f"collision_conflict_node_{n}") 
    
    for n in range(len(I_col_ho_nodes)):   
        idx_aircraftpairs = [i for i, sublist in enumerate(I_col) if sublist == I_col_ho[n]]
        model.addConstr(Z[idx_aircraftpairs[0]]-Z[idx_aircraftpairs[1]] == 0, f"collision_headon_node_{n}")
            
    for n in range(len(I_col_ot_nodes)):  
        idx_aircraftpairs = [i for i, sublist in enumerate(I_col) if sublist == I_col_ot[n]]
        model.addConstr(Z[idx_aircraftpairs[0]]-Z[idx_aircraftpairs[1]] == 0, f"collision_overtake_node_{n}")
        #model.addConstr(Z[idx_aircraftpairs[0]]+Z[idx_aircraftpairs[1]] == 1, f"collision_overtake_node_{n}")
                
    #Ordering of tasks
    for i in range(N_etvs):
        model.addConstr(grp.quicksum(X[a,i] for a in range(N_aircraft)) <= 1)
        for a in range(N_aircraft):
            model.addConstr(grp.quicksum(O[k, a, i] for k in range(N_aircraft)) <= 1)
            model.addConstr(grp.quicksum(O[a, k, i] for k in range(N_aircraft)) <= 1)
            model.addConstr(X[a,i] + grp.quicksum(O[a,l,i] for l in range(N_aircraft)) <= 1, f"available_etv_constraint_{a}_{b}_{i}")
            for b in range(N_aircraft):
                if a!=b:
                    model.addConstr((O[a,b,i] == 1) >> (t[b,0] >= t[a,len(P[a])-1] + Short_path_dist(G_e, D_a[a], O_a[b]) / speed_e), f"auxiliary_task_{a}_{b}_{i}")   
                    model.addConstr((O[a,b,i] == 1) >> (grp.quicksum(O[b,k,i] for k in range(N_aircraft))+X[b,i] >=1 ))
                if a == b:  
                    model.addConstr(O[a,b,i] == 0)
                                          
    # ETV energy consumption
    for i in range(N_etvs): 
        for a in range(N_aircraft):
            for b in range(N_aircraft): 
                if a!=b:
                    model.addConstr((C[i,a] == 1) >> (E[i, b] <= E[i, a]-(mu*m_a*g*d_a[a]*eta+(Short_path_dist(G_e, D_a[a], dock[0])+Short_path_dist(G_e, dock[0], O_a[b]))*100-((t[b,0]-t[a,len(P[a])-1])*1000))+(1-(O[a,b,i]))*bat_e*10), f"available_etv_constraint1_{a}_{b}_{i}")   
                    model.addConstr((C[i,a] == 0) >> (E[i, b] <= E[i, a]-(mu*m_a*g*d_a[a]*eta+Short_path_dist(G_e, D_a[a], O_a[b])*(100))+(1-(O[a,b,i]))*bat_e*10), f"available_etv_constraint2_{a}_{b}_{i}")   
                     
     # ETV energy availability
    for i in range(N_etvs): 
        for a in range(N_aircraft):
            model.addConstr((X[a,i] == 1) >> (E[i, a] >= (mu*m_a*g*d_a[a]*eta+Short_path_dist(G_e, D_a[a], dock[0])*(100))))#return to C (charge dock)
            model.addConstr(E[i, a] <=  bat_e,  f"max_cap_{i}_{a}")
            for b in range(N_aircraft):
                model.addConstr((O[a,b,i] == 1) >> (E[i, a]-(mu*m_a*g*d_a[a]*eta+Short_path_dist(G_e, D_a[a], O_a[b])*(100)) >= 0))       
    
    model.update()
    return model

def Short_path_dist(G, n1, n2):
    dist = nx.shortest_path_length(G, source=n1, target=n2, weight='weight')
    return dist

def Plotting(variable_values, N_aircraft, N_etvs, P):
    colors = ['grey','blue','green','red','yellow']
    etv_color = []
    for a in range(N_aircraft):
        towed = sum(variable_values['X'][a][i] + sum(variable_values['O'][a][b][i] for b in range(N_aircraft)) for i in range(N_etvs)) 
        if towed == 1:
            for i in range(N_etvs):
                tow = variable_values['X'][a][i] + sum(variable_values['O'][a][b][i] for b in range(N_aircraft))
                if tow == 1:
                    etv_color.append(colors[i+1])
        else:
            etv_color.append(colors[0])
               
               
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Iterate through aircraft and tasks to plot horizontal bars
    for a in range(N_aircraft): 
            start_time = variable_values['t'][a][0]
            duration = variable_values['t'][a][len(P[a])-1] - start_time
            
            ax.barh(a, duration, left=start_time, color=etv_color[a])

            for n in range(len(variable_values['t'][a])):
                node_number= P[a][n]
                ax.text(variable_values['t'][a][n], a, str(node_number), color='black',
                    ha='center', va='center', fontweight='bold', fontsize= 7)
                for i in range(len(variable_values['C'])):
                    if variable_values['C'][i][a] == 1:
                        ax.text(variable_values['t'][a][len(P[a])-1], a, str('C'), color='green',
                            ha='center', va='center', fontweight='bold', fontsize= 12)
                    if variable_values['X'][a][i] == 1:
                        ax.text(variable_values['t'][a][0], a+0.5, variable_values['E'][i][a], color='black',
                            ha='center', va='center', fontweight='normal', fontsize= 7)
                    for b in range(N_aircraft): 
                        if variable_values['O'][a][b][i] == 1: 
                            ax.text(variable_values['t'][b][0], b+0.5, variable_values['E'][i][b], color='black',
                                ha='center', va='center', fontweight='normal', fontsize= 7)  
                            
                            
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Aircraft Task')
    ax.set_yticks(range(N_aircraft))
    ax.set_yticklabels([f'Aircraft {i}' for i in range(N_aircraft)])
 
    '''
    # Extract nodes and
    Dvars_time = Dvars_name[0:sum(N_Va)]
    Dvars_value_time = Dvars_value[0:sum(N_Va)]
    time_nodes_aircraft_inv = []
    time_nodes_aircraft = []
    time_aircraft = []
    time_nodes = []
    
    for s in Dvars_time:
        time_aircraft.append(int(s.split('_')[1])) 
        time_nodes.append(int(s.split('_')[2]))
        
    time_nodes_aircraft_inv.append(Dvars_value_time)
    time_nodes_aircraft_inv.append(time_aircraft)
    time_nodes_aircraft_inv.append(time_nodes)
    
    time_nodes_aircraft = [[time_nodes_aircraft_inv[0][i], time_nodes_aircraft_inv[1][i], time_nodes_aircraft_inv[2][i]] for i in range(len(Dvars_time))]
    sorted_time_nodes_aircraft = sorted(time_nodes_aircraft, key=lambda x: x[0]) 
    
    pos = nx.spring_layout(G_a)
    #times = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    times = np.linspace(0, 70, 25).tolist()
    
    
    for i in range(1,len(times)):  
        nx.draw(G_a, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8)
        plt.title(f'Time: {times[i]}')
        for k in range(len(sorted_time_nodes_aircraft)):
            if times[i-1]<= sorted_time_nodes_aircraft[k][0]  < times[i]:
                #nx.draw_networkx_nodes(G_a, pos, nodelist=[P[sorted_time_nodes_aircraft[k][1]][sorted_time_nodes_aircraft[k][2]]], node_color='red', node_size=500, label=f'aircraft: {sorted_time_nodes_aircraft[k][1]}')
                nx.draw_networkx_labels(G_a, pos, labels={P[sorted_time_nodes_aircraft[k][1]][sorted_time_nodes_aircraft[k][2]]: f'\n aircraft: {sorted_time_nodes_aircraft[k][1]} Time: {times[i]}'}, font_size=8, font_color='black')
                
    plt.show() 
    '''           
