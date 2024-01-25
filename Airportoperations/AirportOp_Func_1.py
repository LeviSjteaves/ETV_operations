# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:10:55 2023

@author: levi1
"""
import gurobipy as grp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import sys
import numpy as np
import matplotlib.patches as mpatches 

#from Functions_basicprob import appeartimes
from datetime import datetime

def Load_Graph(airport):
    if airport == 'EHEH':
        # Load CSV file with EHEH edges
        edges_EHEH_a_df = pd.read_csv('EHEH_a.csv')
        G_EHEH_a = nx.from_pandas_edgelist(edges_EHEH_a_df, 'EndNodes_1', 'EndNodes_2', edge_attr='weight', create_using=nx.Graph)
        #edges_EHEH_e_df = pd.read_csv('EHEH_e.csv')
        #G_EHEH_e = nx.from_pandas_edgelist(edges_EHEH_e_df, 'EndNodes_1', 'EndNodes_2', create_using=nx.Graph)
    
        nodes_EHEH_a_mat = pd.read_csv('EHEH_nodes_a.csv')
        node_positions = {node_id+1: [float(x), float(y)] for node_id, (x, y) in nodes_EHEH_a_mat[['x', 'y']].iterrows()}
        nx.draw(G_EHEH_a, pos=node_positions, with_labels=True, node_size= 25, font_size= 7)
        
        G_a = G_EHEH_a
        G_e = G_EHEH_a
        
        # Show the plot inline
        plt.show()
    
    elif airport == 'EHAM':

        file_path1 = "EHAM_graph_hand_a.gpickle"
        file_path2 = "EHAM_graph_hand_e.gpickle"
        
        G_AMS_a = nx.read_gpickle(file_path1)
        G_AMS_e = nx.read_gpickle(file_path2)
        
        G_a = G_AMS_a
        G_e = G_AMS_e
    
        # Set labels and title
        fig, ax = plt.subplots()
        node_positions_a = nx.get_node_attributes(G_a, 'pos')
        node_positions_e = nx.get_node_attributes(G_e, 'pos')

        img = plt.imread("EHAM_chart.png")  # Replace with the actual path to your screenshot
        
        ax.imshow(img, extent=[107000, 108000, 483000, 484000], alpha=0.8)  # Adjust the extent based on your graph size
        
        # Add nodes on top of the image
        nx.draw(G_a, pos=node_positions_a, with_labels=True, node_size=15, font_size=8, ax=ax)
        nx.draw(G_e, pos=node_positions_e, with_labels=True, node_color='red', node_size=15, font_size=8, ax=ax, edge_color='grey')

        
        ax.set_title('Schiphol airport (EHAM)')

        
    elif airport == 'basic':
        # Airport info
        G_basic_a = nx.Graph()
        G_basic_a.add_edges_from([(0, 2,{'weight': 10}), (1, 2,{'weight': 10}), (2, 3,{'weight': 10}), (3, 4,{'weight': 10}), (3, 5,{'weight': 10})])
        G_basic_e = nx.Graph()
        G_basic_e.add_edges_from([(0, 1,{'weight': 20}), (1, 5,{'weight': 20}), (5, 4,{'weight': 20}), (4, 0,{'weight': 20})])
        G_a = G_basic_a
        G_e = G_basic_e

    else:
        print("Select correct airport!")
    return G_a, G_e

def appeartimes(appear_times_T, date_of_interest):
    timestamp_format1 = "%Y-%m-%dT%H:%M:%S.%f%z"
    timestamp_format2 = "%Y-%m-%d"

    # Parse the timestamp string into a datetime object
    appear_times = []
    for i in range(len(appear_times_T)):
        appear_times_C = datetime.strptime(appear_times_T[i], timestamp_format1)
        timestamp_reference = datetime.strptime(date_of_interest, timestamp_format2)
        appear_times.append((int(appear_times_C.timestamp())-int(timestamp_reference.timestamp()))/60)
    return appear_times

def Load_Aircraft_Info(date_of_interest, pagelimit):
    #Import API data from schiphhol

    page = pagelimit[0]
    page_end = pagelimit[1]
    rawflightdata = []
    base_url = 'https://api.schiphol.nl/operational-flight/flights'
    headers = {
        'Accept': 'application/json',
        'ResourceVersion': 'v2',
        'app_id': '719fa989',
        'app_key': '54a53ddf37aeb5d990ba2b384c23e722'
    }
    
    if page_end == -1:
        url = f'{base_url}?scheduleDate={date_of_interest}&page={page}'
        try:
            response = requests.get(url, headers=headers)
        except requests.exceptions.ConnectionError as error:
            print(error)
            sys.exit()
            
        last_page_link = response.links.get('last').get('url')
        page_end = int(last_page_link.split('page=')[1])-1
            
    while page != page_end+1:
        url = f'{base_url}?scheduleDate={date_of_interest}&page={page}'
        
        try:
            response = requests.get(url, headers=headers)
        except requests.exceptions.ConnectionError as error:
            print(error)
            sys.exit()
    
        if response.status_code == 200:
            flightList = response.json()
            rawflightdata.extend(flightList['flights'])
            # Check if there is a next page in the response
            next_page_link = response.links.get('next').get('url')
            if next_page_link:
                # Extract the page number from the next page link and update the 'page' variable
                page = int(next_page_link.split('page=')[1])
            else:
                # If there is no next page, break out of the loop
                break  
                  
        else:
            print('''Something went wrong
                  Http response code: {}
                  {}'''.format(response.status_code,
                  response.text))
            break
    
   #Remove duplicates
    seen_main_flights = set()
   
   # Filter out dictionary elements with duplicate 'mainFlight' values
    flightdata = []
    for flight in rawflightdata:
        main_flight = flight['aircraftRegistration']
        if main_flight not in seen_main_flights and flight['cdmFlightState'] != 'CNX' and flight['aircraftType']['aircraftCategory']>1 and (flight['actualOffBlockTime'] is not None or flight['actualLandingTime'] is not None):
            seen_main_flights.add(main_flight)
            flightdata.append(flight)

    operational_runways = []   #Names of the runways of that airport
    operational_gates = []
    appear_times_T = []
    dep = []
    cat = []
    
    for flight in flightdata:
        operational_gates.append(flight['ramp']['current'])
        operational_runways.append(flight['runway'])
        cat.append(flight['aircraftType']['aircraftCategory']-1)
        if flight['flightDirection'] == 'D':
            appear_times_T.append(flight['actualOffBlockTime'])
            dep.append(1)
        else:
            appear_times_T.append(flight['actualLandingTime'])
            dep.append(0)

    appear_times = appeartimes(appear_times_T, date_of_interest)
    
    Flight_orig=[]
    Flight_dest=[]

    for flight in flightdata:
        if flight['flightDirection'] == 'D':
            Flight_orig.append(flight['ramp']['current'][:-2])
            Flight_dest.append(flight['runway'])
        else:
            Flight_dest.append(flight['ramp']['current'][:-2])
            Flight_orig.append(flight['runway'])
            
    day_before = [index for index, num in enumerate(appear_times) if num <= 0]
    
    # Remove the specified index
    for i in range(len(day_before)):
        if day_before[i] in range(len(appear_times)):
            appear_times.pop(day_before[i])   
            Flight_orig.pop(day_before[i]) 
            Flight_dest.pop(day_before[i]) 
            dep.pop(day_before[i]) 
            cat.pop(day_before[i]) 
            
    np.save('Flight_O.npy', Flight_orig)
    np.save('Flight_D.npy', Flight_dest)
    np.save('Flight_t.npy', appear_times)
    np.save('Flight_dep.npy', dep)
    np.save('Flight_info.npy', cat)
    
    return Flight_orig, Flight_dest, appear_times, dep, cat, flightdata
    
def Create_model(G_a, G_e, p, P, tO_a, O_a, D_a, d_a, dock, dep, cat):
    model = grp.Model("Aircraft_Taxiing")
    #model.Params.NonConvex = 2
    # unpack P
    N_aircraft = p['N_aircraft']
    N_etvs = p['N_etvs']
    N_etvs_cat1 = p['N_etvs_cat1']
    #N_etvs_cat2 = p['N_etvs_cat2']
    max_speed_a = p['max_speed_a']
    min_speed_a = p['min_speed_a']
    max_speed_e = p['max_speed_e']
    min_speed_e = p['min_speed_e']
    free_speed_e = p['free_speed_e']
    bat_e = p['bat_e']
    bat_e_min = p['bat_e_min'] 
    bat_e_max = p['bat_e_max'] 
    m_a = p['m_a']
    eta = p['eta']
    eta_e =p['eta_e']
    I_ch = p['I_ch']
    E_e = p['E_e']
    E_a = p['E_a']
    d_sep = p['d_sep']
    v_avg = p['v_avg']
    t_pushback = p['t_pushback']
    T_charge_min = p['T_charge_min']
    start_delay = p['start_delay']
    fuelmass = p['fuelmass']
    
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
    I_up = []
    for a in range(N_aircraft):
        I_up.append([])
        for b in range(N_aircraft):
            if tO_a[b]+start_delay >= tO_a[a]:
                I_up[a].append(b)
                
    I_do = []
    for a in range(N_aircraft):
        I_do.append([])
        for b in range(N_aircraft):
            if tO_a[b] <= tO_a[a]+start_delay:
                I_do[a].append(b)
    
    
    I_col = [] #set of aircraft with possible collisions
    I_col_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(len(P[a])):
                for m in range(len(P[b])):
                    e_col = []
                    #possibility to be at the same node at the same time
                    if a < b and P[a][n] == P[b][m] and (t_min[a][n] <= t_max[b][m]+start_delay and t_min[b][m] <= t_max[a][n]+start_delay):
                        e_col = [a,b]
                        I_col_nodes.append(P[a][n])
                        I_col.append(e_col)
                        
    I_col_ot = [] #set of aircraft with possible overtake collisions
    I_col_ot_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(1, len(P[a])):
                for m in range(1, len(P[b])):
                    if a < b and P[a][n] == P[b][m] and P[a][n-1] == P[b][m-1] and t_min[a][n] <= t_max[b][m]+start_delay and t_min[b][m-1] <= t_max[a][n-1]+start_delay:
                       e_col_ot = [a,b]
                       I_col_ot_nodes.append([P[a][n-1],P[a][n]])
                       I_col_ot.append(e_col_ot)
                    
                       
    I_col_ho = [] #set of aircraft with possible head on collisions
    I_col_ho_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(1, len(P[a])):
                for m in range(len(P[b])-1):
                    if a < b and n != 0 and m != len(P[b]) and P[a][n] == P[b][m] and P[a][n-1] == P[b][m+1]  and t_min[a][n] <= t_max[b][m]+start_delay and t_min[b][m+1] <= t_max[a][n-1]+start_delay:
                       e_col_ho = [a,b]
                       I_col_ho_nodes.append([P[a][n-1],P[a][n]])
                       I_col_ho.append(e_col_ho) 
                       
    E_a_time = []
    E_e_return = []
    for a in range(N_aircraft):
        E_a_time.append(60*E_a*(m_a[cat[a]]-fuelmass[cat[a]]*(1-dep[a])*(1/eta_e)))
        E_e_return.append(min(Short_path_dist(G_e, D_a[a], dock[c])for c in range(len(dock)))*(E_e))
        
    # Decision variables
    t = {}  # Arrival times nodes
    Z = {}  # Order of visiting nodes
    X = {}  # Last towed aircraft of an ETV
    O = {}  # Order for towing tasks
    C = {}  # Choose to charge
    E = {}  # State of charge before every task
    
    for a in range(N_aircraft):
        for n in range(len(P[a])):
            t[a, n] = model.addVar(lb=t_min[a][n]+t_pushback*dep[a],ub=t_max[a][n]+start_delay+t_pushback*dep[a],vtype=GRB.CONTINUOUS, name=f"t_{a}_{n}")
            
    for a in range(N_aircraft):
        for i in range(N_etvs):
            X[a, i] = model.addVar(vtype=GRB.BINARY, name=f"X_{a}_{i}")
    
    for n in range(len(I_col_nodes)):
            Z[n] = model.addVar(vtype=GRB.BINARY, name=f"Z_{n}")
               
    for a in range(N_aircraft):
       for i in range(N_etvs): 
           for b in range(len(I_up[a])):
              O[a,I_up[a][b],i] = model.addVar(vtype=GRB.BINARY, name=f"O_{a}_{I_up[a][b]}_{i}") 
               
    for a in range(N_aircraft):
       for i in range(N_etvs):           
              E[i, a] = model.addVar(lb=bat_e_min[i], ub=bat_e_max[i], vtype=GRB.CONTINUOUS, name=f"E_{i}_{a}") 
              
    for a in range(N_aircraft):
       for i in range(N_etvs):          
              C[i,a] = model.addVar(vtype=GRB.BINARY, name=f"C_{i}_{a}")
           
    # Objective function: minimize emissions (only rolling resistance) + total taxitime
    model.setObjective(grp.quicksum(E_a_time[a]*eta_e*(1/eta)*(t[a,len(P[a])-1]-(t[a,0]))*(1-grp.quicksum(X[a,i] + grp.quicksum(O[a,I_up[a][b],i] for b in range(len(I_up[a])))for i in range(N_etvs))) for a in range(N_aircraft))
                       +0.0001*grp.quicksum((t[a,len(P[a])-1]) for a in range(N_aircraft))
                       , sense=GRB.MINIMIZE)
    
    # Constraints////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for a in range(N_aircraft):
        #Min start time
        model.addConstr(t[a, 0] >= tO_a[a]+t_pushback*dep[a], f"appear_constraint_{a}")
        model.addConstr(t[a, 0] <= tO_a[a]+start_delay+t_pushback*dep[a], f"appear_constraint_{a}")
        #if arriving, start to taxi directly
        model.addConstr(t[a, 0] <= tO_a[a]+t_pushback*dep[a]+(dep[a])*10000, f"appear_constraint_{a}")
        
        # max one etv per aircraft
        model.addConstr(grp.quicksum(X[a, i] +grp.quicksum(O[a,I_up[a][b],i] for b in range(len(I_up[a]))) for i in range(N_etvs)) <= 1, f"one_towing_task_{a}")
        for n in range(1, len(P[a])):
            #Min/max time to arrive at each node (from min/max speed)
            model.addConstr(t[a, n] >= t[a, n-1] + (Short_path_dist(G_a, P[a][n-1], P[a][n]) / max_speed_a) - (grp.quicksum(X[a, i] +grp.quicksum(O[a,I_up[a][b],i] for b in range(len(I_up[a]))) for i in range(N_etvs))*10000), f"arrival_constraint_min_{a}_{n}")
            model.addConstr(t[a, n] <= t[a, n-1] + (Short_path_dist(G_a, P[a][n-1], P[a][n]) / min_speed_a) + (grp.quicksum(X[a, i] +grp.quicksum(O[a,I_up[a][b],i] for b in range(len(I_up[a]))) for i in range(N_etvs))*10000), f"arrival_constraint_max_{a}_{n}")            
            #Min/max time to arrive at each node while towed (from min/max speed)
            model.addConstr(t[a, n] >= t[a, n-1] + (Short_path_dist(G_a, P[a][n-1], P[a][n]) / max_speed_e) - (1 - (grp.quicksum(X[a, i] +grp.quicksum(O[a,I_up[a][b],i] for b in range(len(I_up[a]))) for i in range(N_etvs))))*10000, f"arrival_constraint_min_tow_{a}_{n}")
            model.addConstr(t[a, n] <= t[a, n-1] + (Short_path_dist(G_a, P[a][n-1], P[a][n]) / min_speed_e) + (1 - (grp.quicksum(X[a, i] +grp.quicksum(O[a,I_up[a][b],i] for b in range(len(I_up[a]))) for i in range(N_etvs))))*10000, f"arrival_constraint_max_tow_{a}_{n}")            
    
    # Weight categories
    for a in range(N_aircraft):        
        for i in range(N_etvs):
            if cat[a]>=5 and i<=N_etvs_cat1-1:
                model.addConstr(X[a, i] == 0)
                for b in range(len(I_up[a])):
                    model.addConstr(O[a,I_up[a][b],i] == 0)
               
    # Collision
    for n in range(len(I_col_nodes)):
            model.addConstr((Z[n] == 1) >> (t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] <= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])]), f"auxiliary_order1_node_{n}")   
            model.addConstr((Z[n] == 0) >> (t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] >= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])]), f"auxiliary_order2_node_{n}")   
            model.addConstr(t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] >= t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] +(d_sep[cat[a]]/v_avg) - (1-Z[n])*10000, f"collision_conflict_node_{n}") 
            model.addConstr(t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] >= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] +(d_sep[cat[a]]/v_avg) - (Z[n])*10000, f"collision_conflict_node_{n}") 
    
    for n in range(len(I_col_ho_nodes)):   
        idx_aircraftpairs = [i for i, sublist in enumerate(I_col) if sublist == I_col_ho[n]]
        for i in range(len(idx_aircraftpairs)-1):
            model.addConstr(Z[idx_aircraftpairs[i]]-Z[idx_aircraftpairs[i+1]] == 0, f"collision_headon_node_{n}")
                
    for n in range(len(I_col_ot_nodes)):  
        idx_aircraftpairs = [i for i, sublist in enumerate(I_col) if sublist == I_col_ot[n]]
        for i in range(len(idx_aircraftpairs)-1):
            model.addConstr(Z[idx_aircraftpairs[i]]-Z[idx_aircraftpairs[i+1]] == 0, f"collision_headon_node_{n}")
            
    #Ordering of tasks
    for i in range(N_etvs):
        model.addConstr(grp.quicksum(X[a,i] for a in range(N_aircraft)) <= 1)
        for a in range(N_aircraft):
            model.addConstr(grp.quicksum(O[I_do[a][k], a, i] for k in range(len(I_do[a])))<= 1)
            model.addConstr(grp.quicksum(O[a, I_up[a][k], i] for k in range(len(I_up[a]))) <= 1)
            model.addConstr(X[a,i] + grp.quicksum(O[a,I_up[a][l],i] for l in range(len(I_up[a]))) <= 1, f"available_etv_constraint_{a}_{b}_{i}")
            for b in range(len(I_up[a])):
                if a!=I_up[a][b]:
                    model.addConstr((O[a,I_up[a][b],i] == 1) >> (t[I_up[a][b],0] >= t[a,len(P[a])-1] + Short_path_dist(G_e, D_a[a], O_a[I_up[a][b]]) / free_speed_e +t_pushback*dep[I_up[a][b]]), f"auxiliary_task_{a}_{b}_{i}")   
                    model.addConstr((O[a,I_up[a][b],i] == 1) >> (t[I_up[a][b],0] >= t[a,len(P[a])-1] + (min(Short_path_dist(G_e, D_a[a], dock[c])+Short_path_dist(G_e, dock[c], O_a[I_up[a][b]])for c in range(len(dock)))) / free_speed_e  +T_charge_min +t_pushback*dep[I_up[a][b]] -(1-C[i,a])*10000) , f"auxiliary_task_{a}_{b}_{i}")   
                    model.addConstr((O[a,I_up[a][b],i] == 1) >> (grp.quicksum(O[I_up[a][b],I_up[I_up[a][b]][k] ,i] for k in range(len(I_up[I_up[a][b]])))+X[I_up[a][b],i] >=1 ))
                if a == I_up[a][b]:  
                    model.addConstr(O[a,I_up[a][b],i] == 0)
                                    
    # ETV energy consumption
    for i in range(N_etvs): 
        for a in range(N_aircraft):
            for b in range(len(I_up[a])): 
                if a!=I_up[a][b]:
                    model.addConstr((C[i,a] == 1) >> (E[i, I_up[a][b]] <= E[i, a]-(E_a_time[a]*(t[a,len(P[a])-1]-(t[a,0]))+ (min(Short_path_dist(G_e, D_a[a], dock[c])+Short_path_dist(G_e, dock[c], O_a[I_up[a][b]])for c in range(len(dock))) )*E_e-(((t[I_up[a][b],0]-(t[a,len(P[a])-1]+Short_path_dist(G_e, D_a[a], O_a[I_up[a][b]]))/free_speed_e))*I_ch))+(1-(O[a,I_up[a][b],i]))*bat_e[i]*10), f"available_etv_constraint1_{a}_{b}_{i}")   
                    model.addConstr((C[i,a] == 0) >> (E[i, I_up[a][b]] <= E[i, a]-(E_a_time[a]*(t[a,len(P[a])-1]-(t[a,0]))+ Short_path_dist(G_e, D_a[a], O_a[I_up[a][b]])*E_e)+(1-(O[a,I_up[a][b],i]))*bat_e[i]*10), f"available_etv_constraint2_{a}_{b}_{i}")   
                    
    # ETV energy availability
    for i in range(N_etvs): 
        for a in range(N_aircraft):
            model.addConstr((X[a,i] == 1) >> (E[i, a] >= (E_a_time[a]*(t[a,len(P[a])-1]-(t[a,0]))+E_e_return[a]+bat_e_min[i])))#return to C (charge dock)
            model.addConstr(E[i, a] <=  bat_e_max[i],  f"max_cap_{i}_{a}")
            for b in range(len(I_up[a])): 
                model.addConstr((O[a,I_up[a][b],i] == 1) >> (E[i, a] >= (E_a_time[a]*(t[a,len(P[a])-1]-(t[a,0]))+Short_path_dist(G_e, D_a[a], O_a[I_up[a][b]])*(E_e))+bat_e_min[i]))
    
    model.update()
    return model, I_up, I_do,  E_a_time, E_e_return 


def Short_path_dist(G, n1, n2):
    dist = nx.shortest_path_length(G, source=n1, target=n2, weight='weight')
    return dist

def Plotting(variable_values, N_aircraft, N_etvs, P, I_up, p, d_a, appear_times, G_a, cat, dep, E_a_time, E_e_return ):
    # unpack P
    N_aircraft = p['N_aircraft']
    N_etvs = p['N_etvs']
    bat_e = p['bat_e']
    max_speed_a = p['max_speed_a']
    start_delay = p['start_delay']
    
    colors = sns.color_palette('husl', n_colors=N_etvs)
    etv_color = []
    etv_number = []
    for a in range(N_aircraft):
        towed = sum(variable_values['X'][a][i] + sum(variable_values['O'][a][I_up[a][b]][i] for b in range(len(I_up[a]))) for i in range(N_etvs)) 
        if towed == 1:
            for i in range(N_etvs):
                tow = variable_values['X'][a][i] + sum(variable_values['O'][a][I_up[a][b]][i] for b in range(len(I_up[a])))
                if tow == 1:
                    etv_color.append(colors[i])
                    etv_number.append(i)
        else:
            etv_color.append('grey')
            etv_number.append(N_etvs)   
               
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    durations = []
    # Iterate through aircraft and tasks to plot horizontal bars
    for a in range(N_aircraft): 
            start_time = variable_values['t'][a][0]
            duration = variable_values['t'][a][len(P[a])-1] - start_time
            ax.barh(a, duration, left=start_time, color=etv_color[a])
            ax.plot(appear_times[a], a, 'go', markersize=10)
            durations.append(duration)
            for n in range(len(variable_values['t'][a])):
                node_number= P[a][n]
                ax.text(variable_values['t'][a][n], a, str(node_number), color='black',
                    ha='center', va='center', fontweight='bold', fontsize= 7)
                if n >= 1:
                    ax.plot(variable_values['t'][a][n], a-1+(Short_path_dist(G_a, P[a][n-1], P[a][n])/(variable_values['t'][a][n]-variable_values['t'][a][n-1]))/max_speed_a, 'go', markersize=5)
    appear_times_min = []
    
    for i in range(len(appear_times)):
        appear_times_min.append(int(appear_times[i]))
        

    
    # Set a reasonable number of tick locations based on the range of total minutes
    tick_locations = np.linspace(min(appear_times_min), max(appear_times_min)+max(durations)/60+start_delay/60,50)
    hours, remainder = (divmod(tick_locations, 60))
    timestamps = [f'{int(h):02d}:{int(r):02d}' for h, r in zip(hours, remainder)]
    
    # Set labels and title
    ax.set_xlabel('Time [h]')
    plt.xticks(tick_locations, timestamps, rotation=45, ha='right')
    ax.set_ylabel('Aircraft Task')
    ax.set_yticks(range(N_aircraft))
    ax.set_yticklabels([f'Aircraft {i+1} Cat.{cat[i]}' for i in range(N_aircraft)])

    patch = []
    for i in range(N_etvs):
        patch.append(mpatches.Patch(color=colors[i], label=f'ETV {i+1}'))
    patch.append(mpatches.Patch(color='grey', label='Aircraft not towed'))
    patch.append(mpatches.Patch(color='Green', label='Appear time'))
    ax.legend(handles=patch, loc='upper left')
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Iterate through aircraft and tasks to plot horizontal bars
    for a in range(N_aircraft): 
            if etv_number[a] != N_etvs:
                start_time = variable_values['t'][a][0]
                duration = variable_values['t'][a][len(P[a])-1] - start_time
                ax.barh(etv_number[a], duration, left=start_time, color='lightblue')
                ax.text((start_time+0.5*duration), etv_number[a], f'A:{a+1}', color='black',
                    ha='center', va='center', fontsize= 8)
                for i in range(len(variable_values['C'])):
                    if variable_values['C'][i][a] == 1:
                        ax.text(variable_values['t'][a][len(P[a])-1], etv_number[a], str('C'), color='green',
                            ha='center', va='center', fontweight='bold', fontsize= 12)
            else:
                start_time = variable_values['t'][a][0]
                duration = variable_values['t'][a][len(P[a])-1] - start_time
                ax.barh(etv_number[a], duration, left=start_time, color='grey')
                ax.text((start_time+0.5*duration), etv_number[a], f'{a+1}', color='black',
                    ha='center', va='center', fontsize= 8)
                        
    # Set labels and title
    ax.set_xlabel('Time [h]')
    plt.xticks(tick_locations, timestamps, rotation=45, ha='right')
    ax.set_ylabel('ETV')
    ax.set_yticks(range(N_etvs+1))
    labels = [f'ETV {i+1}' for i in range(N_etvs)]
    labels.append('Not towed')
    ax.set_yticklabels(labels)
    blue_patch = mpatches.Patch(color='lightblue', label='Aircraft a towed by ETV')
    red_patch = mpatches.Patch(color='red', label='SOC')
    grey_patch = mpatches.Patch(color='grey', label=' Aircraft not towed')
    green_patch = mpatches.Patch(color='green', label='Charge-Indicator')
    ax.legend(handles=[blue_patch, red_patch, grey_patch, green_patch], loc='upper left')


        #etv_SOC.append(G_SOC)
    for i in range(N_etvs):
        for a in range(N_aircraft):
                if variable_values['X'][a][i] == 1:
                    ax.bar(variable_values['t'][a][0], variable_values['E'][i][a]/bat_e[i], bottom=etv_number[a]-0.5, width=2, color='red')
                    ax.text(variable_values['t'][a][0], etv_number[a]-0.5+variable_values['E'][i][a]/bat_e[i], round(variable_values['E'][i][a]/bat_e[i],2), color='black',
                        ha='center', va='center', fontweight='normal', fontsize= 7)
                    
                    ax.bar(variable_values['t'][a][len(P[a])-1], (variable_values['E'][i][a]-(E_a_time[a]*(variable_values['t'][a][len(P[a])-1]-variable_values['t'][a][0])))/(bat_e[i]), bottom=etv_number[a]-0.5,width=2,  color='red')
                    ax.text(variable_values['t'][a][len(P[a])-1], etv_number[a]-0.5+(variable_values['E'][i][a]-(E_a_time[a]*(variable_values['t'][a][len(P[a])-1]-(variable_values['t'][a][0]))))/(bat_e[i]), round((variable_values['E'][i][a]-(E_a_time[a]*(variable_values['t'][a][len(P[a])-1]-(variable_values['t'][a][0]))))/(bat_e[i]),2), color='black',
                            ha='center', va='center', fontweight='normal', fontsize= 7)
                for b in range(len(I_up[a])): 
                    if variable_values['O'][a][I_up[a][b]][i] == 1: 
                        ax.bar(variable_values['t'][a][0], variable_values['E'][i][a]/bat_e[i], bottom=etv_number[a]-0.5, width=2, color='red')
                        ax.text(variable_values['t'][a][0], etv_number[a]-0.5+variable_values['E'][i][a]/bat_e[i], round(variable_values['E'][i][a]/bat_e[i],2), color='black',
                            ha='center', va='center', fontweight='normal', fontsize= 7)
                        
                        ax.bar(variable_values['t'][a][len(P[a])-1], (variable_values['E'][i][a]-(E_a_time[a]*(variable_values['t'][a][len(P[a])-1]-variable_values['t'][a][0])))/(bat_e[i]), bottom=etv_number[a]-0.5,width=2,  color='red')
                        ax.text(variable_values['t'][a][len(P[a])-1], etv_number[a]-0.5+(variable_values['E'][i][a]-(E_a_time[a]*(variable_values['t'][a][len(P[a])-1]-(variable_values['t'][a][0]))))/(bat_e[i]), round((variable_values['E'][i][a]-(E_a_time[a]*(variable_values['t'][a][len(P[a])-1]-(variable_values['t'][a][0]))))/(bat_e[i]),2), color='black',
                                ha='center', va='center', fontweight='normal', fontsize= 7)
    
    ''' 
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    colors_air = sns.color_palette('husl', n_colors=N_aircraft)
    for a in range(N_aircraft): 
        plane_path_x = []
        plane_path_y = []
        for n in range(1, len(variable_values['t'][a])):
            node_number= P[a][n]
            plane_path_y.append(node_number)
            plane_path_x.append(variable_values['t'][a][n]/60)
        ax.plot(plane_path_x, plane_path_y, color=colors_air[a], markersize=5)
    '''
    return
