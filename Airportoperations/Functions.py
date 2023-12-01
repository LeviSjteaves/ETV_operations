"""
Created on Mon Oct  2 12:26:59 2023

@author: levi1
"""

# Import packages
import requests
import sys
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gp
import gurobipy as grp
from datetime import datetime
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def LoadAPIdata(date_of_interest, pagelimit):

    page = pagelimit[0]
    page_end = pagelimit[1]
    rawflightdata = []
    base_url = 'https://api.schiphol.nl/public-flights/flights'
    headers = {
        'Accept': 'application/json',
        'ResourceVersion': 'v4',
        'app_id': '1511baff',
        'app_key': '396dfa82415ffcd90dd22d2fe41383ee'
    }
        
    while page != page_end+1:
        url = f'{base_url}?scheduleDate={date_of_interest}&includedelays=false&page={page}&sort=%2BscheduleTime'
        
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
        main_flight = flight['mainFlight']
        if main_flight not in seen_main_flights and flight['publicFlightState'] != {'flightStates': ['CNX']}:
            seen_main_flights.add(main_flight)
            flightdata.append(flight)
            
    return flightdata

def LoadOSMdata(airport_name, runways, operational_gates):
    
    # Load the graph for taxiways and serviceways
    G_taxi = ox.graph_from_place(airport_name,custom_filter = '["aeroway"~"runway|taxiway"]', simplify=True)
    
    #G_taxi = ox.consolidate_intersections(G_taxi_t, rebuild_graph=True, tolerance=0.0002, dead_ends=True)
    
    G_ser = ox.graph_from_place(airport_name,custom_filter = '["highway"~"service"]', simplify=True)
    
    #Create taxi network
    G_data = ox.utils_graph.graph_to_gdfs(G_taxi)
    G_data_nodes , G_data_edges = G_data
    
    #Create service network
    G_ser_data = ox.utils_graph.graph_to_gdfs(G_ser)
    G_ser_data_nodes , G_ser_data_edges = G_ser_data
    #ox.plot_graph(G_ser)
    
    # Load gate nodes
    all_gates = ox.features_from_place(airport_name,tags={'aeroway': 'gate'})
    gates = []
    for idx in enumerate(operational_gates):
        gates.append(all_gates[all_gates['ref']==idx[1]].copy())
    gates = pd.concat(gates, ignore_index=True)
    gate_nodes = ox.distance.nearest_nodes(G_taxi, gates.geometry.x, gates.geometry.y, return_dist=False)
    
    # Find runway nodes
    if len(runways) <= 1:
        filter_condition1 = G_data_edges['ref'] == runways[0]
        filter_condition = filter_condition1
    elif len(runways) <= 2:
        filter_condition1 = G_data_edges['ref'] == runways[0]
        filter_condition2 = G_data_edges['ref'] == runways[1]
        filter_condition = filter_condition1 | filter_condition2
    elif len(runways) <= 3:
        filter_condition1 = G_data_edges['ref'] == runways[0]
        filter_condition2 = G_data_edges['ref'] == runways[1]
        filter_condition3 = G_data_edges['ref'] == runways[2]
        filter_condition = filter_condition1 | filter_condition2
    elif len(runways) <= 4:
        filter_condition1 = G_data_edges['ref'] == runways[0]
        filter_condition2 = G_data_edges['ref'] == runways[1]
        filter_condition3 = G_data_edges['ref'] == runways[2]
        filter_condition4 = G_data_edges['ref'] == runways[3]
        filter_condition = filter_condition1 | filter_condition2
    else:
        filter_condition1 = G_data_edges['ref'] == runways[0]
        filter_condition2 = G_data_edges['ref'] == runways[1]
        filter_condition3 = G_data_edges['ref'] == runways[2]
        filter_condition4 = G_data_edges['ref'] == runways[3]
        filter_condition5 = G_data_edges['ref'] == runways[4]
        filter_condition = filter_condition1 | filter_condition2 | filter_condition3 | filter_condition4 | filter_condition5
    
    
    # Apply the filter to the GeoDataFrame
    runway_nodes = G_data_edges[filter_condition]
    list_runway_nodes = runway_nodes.index.get_level_values('u').to_list()
    # Plot the graph with highlighted nodes
    node_colors = {node: 'r' if node in list_runway_nodes else 'g' if node in gate_nodes else 'k' for node in G_taxi.nodes()}


    node_nr_id = {node: i for i, node in enumerate(G_taxi.nodes())}
    node_x = G_taxi.nodes('x')
    node_y = G_taxi.nodes('y')
    


    fig, ax = ox.plot_graph(G_taxi, node_color=list(node_colors.values()), bgcolor='w', edge_color = 'k' ,show=True)
    return G_taxi, gates, list_runway_nodes, node_mapping
    
def Routing(origins, destinations, graph):
    route = {}
    origins = [origins]
    x=0
    for orig in origins:
        for dest in destinations:
        
            # calculate two routes by minimizing travel distance vs travel time
            route1 = nx.shortest_path(graph, orig, dest, weight='length')
            route1_length = nx.shortest_path_length(graph, orig, dest, weight='length')
            route1_data = ox.utils_graph.route_to_gdf(graph, route1, weight='length')
            '''
            # Find the second shortest path
            route2 = None
            route2_length = float('inf')
            
            # Iterate through the edges in the first shortest path
            for i in range(len(route1) - 1):
                u = route1[i]
                v = route1[i + 1]
                
                # Create a copy of the graph with the edge (u, v) removed
                modified_G = graph.copy()
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
            '''
            # Print the second shortest path and its length
            #fig, ax = ox.plot_graph_route(graph, route1,  route_color='r' ,route_linewidth=6, node_size=10, bgcolor='k')
            #fig, ax = ox.plot_graph_route(graph, route2,  route_color='b' ,route_linewidth=6, node_size=10, bgcolor='k')

            # compare the two routes
            print('Route 1 is', route1_length, 'meters')
            #print("Route 2 is", route2_length, 'meters')

            # Append the items to the dictionary using a specific key
            route[x] = route1
            x=x+1
    route = [v for v in route.values()]        
    return route

def Timeplanning(routes, graph):
    velocity_mps = 10.0  # adjust this according to the agent's speed
    start_time =0 # 8:00 AM in seconds
    
    # Calculate arrival time at each node in the route
    arrival_times_list = []
    route_times = {}
    current_time = start_time
    
    for i in range(len(routes)):
        route = routes[i]
        arrival_times_list = []
        for j in range(len(route) - 1):
            node1, node2 = route[j], route[j + 1]
            
            # Calculate edge length in meters
            edge_length = graph[node1][node2][0]['length']
            
            # Calculate travel time in seconds based on edge length and agent's velocity
            travel_time = edge_length / velocity_mps
            
            # Update current time and add it to arrival_times list
            current_time += travel_time
            arrival_times_list.append(current_time)
        route_times[i] = arrival_times_list
        current_time = 0
    # arrival_times list now contains the arrival time at each node in the route
    route_times = [v for v in route_times.values()] 
    
    return route_times

def init_routes(flightdata, routes, orig, destinations, G_taxi):
    from Functions import Routing
    #for orig, dest in zip(origins, destinations):
    routes = Routing(orig, destinations, G_taxi)

    #Add routes to flightdata list
    for i in range(len(flightdata)):
        flightdata[i]['Plane_route'] = routes[i]

    fig, ax = ox.plot_graph_route(G_taxi, route=routes[0] ,  route_color='b' ,route_linewidth=6, node_size=10, bgcolor='k')

    from Functions import Timeplanning
    route_times = Timeplanning(routes, G_taxi)
    return route_times

def appeartimes(appear_times_T, date_of_interest):
    timestamp_format1 = "%Y-%m-%dT%H:%M:%S.%f%z"
    timestamp_format2 = "%Y-%m-%d"

    # Parse the timestamp string into a datetime object
    appear_times = []
    for i in range(len(appear_times_T)):
        appear_times_C = datetime.strptime(appear_times_T[i], timestamp_format1)
        timestamp_reference = datetime.strptime(date_of_interest, timestamp_format2)
        appear_times.append(int(appear_times_C.timestamp())-int(timestamp_reference.timestamp()))
    return appear_times

def create_model(G_a, G_e, N_aircraft, P, d_a, N_etvs, max_speed_a, min_speed_a, speed_e, t_min, t_max, O_a, D_a, tO_a, mu, m_a, eta):
    model = grp.Model("Aircraft_Taxiing")

    # Definitions
    g=9.81
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
    X = {}  # Towing tasks
    Z = {}  # Order of visiting nodes
    Y = {}  # Order for towing tasks
    
    for a in range(N_aircraft):
        for n in range(len(P[a])):
            t[a, n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{a}_{n}")

    for a in range(N_aircraft):
        for i in range(N_etvs):
            X[a, i] = model.addVar(vtype=GRB.BINARY, name=f"X_{a}_{i}")
    
    for n in range(len(I_col_nodes)):
            Z[n] = model.addVar(vtype=GRB.BINARY, name=f"Z_{n}")
            
    for a in range(N_aircraft):
       for b in range(N_aircraft):
           for i in range(N_etvs):
               Y[a,b,i] = model.addVar(vtype=GRB.BINARY, name=f"Y_{a}_{b}_{i}")
               
    # Objective function: minimize emissions (only rolling resistance) + total taxitime
    model.setObjective(grp.quicksum(mu*m_a*g*d_a[a]*eta*(1-sum(X[a,i] for i in range(N_etvs))) for a in range(N_aircraft))+grp.quicksum(t[a, n] for a in range(N_aircraft) for n in range(1, len(P[a]))), sense=GRB.MINIMIZE)
                        
    # Constraints
    for a in range(N_aircraft):
        #Min start time
        model.addConstr(t[a, 0] >= tO_a[a], f"appear_constraint_{a}")
        # max one etv per aircraft
        model.addConstr(grp.quicksum(X[a, i] for i in range(N_etvs)) <= 1, f"one_towing_task_{a}")
        for n in range(1, len(P[a])):
            #Min/max time to arrive at each node (from min/max speed)
            model.addConstr(t[a, n] >= t[a, n-1] + short_path_dist(G_a, P[a][n-1], P[a][n]) / max_speed_a, f"arrival_constraint_min_{a}_{n}")
            model.addConstr(t[a, n] <= t[a, n-1] + short_path_dist(G_a, P[a][n-1], P[a][n]) / min_speed_a, f"arrival_constraint_max_{a}_{n}")
    
    # Collision   
    #for n in range(len(I_col_nodes)):
            #model.addConstr(t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] >= t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])]+5, f"collision_constraint_node_{a}_{n}")
    for n in range(len(I_col_nodes)):
            model.addConstr((Z[n] == 1) >> (t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] <= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])]), f"auxiliary_order1_node_{n}")   
            model.addConstr((Z[n] == 0) >> (t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] >= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])]), f"auxiliary_order2_node_{n}")   
            model.addConstr(t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] >= t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] +5 - (1-Z[n])*100, f"collision_conflict_node_{n}") 
            model.addConstr(t[I_col[n][0],P[I_col[n][0]].index(I_col_nodes[n])] >= t[I_col[n][1],P[I_col[n][1]].index(I_col_nodes[n])] +5 - (Z[n])*100, f"collision_conflict_node_{n}") 
    
    for n in range(len(I_col_ho_nodes)):   
        idx_aircraftpairs = [i for i, sublist in enumerate(I_col) if sublist == I_col_ho[n]]
        model.addConstr(Z[idx_aircraftpairs[0]]-Z[idx_aircraftpairs[1]] == 0, f"collision_headon_node_{n}")
            
    for n in range(len(I_col_ot_nodes)):  
        idx_aircraftpairs = [i for i, sublist in enumerate(I_col) if sublist == I_col_ot[n]]
        if len(idx_aircraftpairs) > 2:
            model.addConstr(Z[idx_aircraftpairs[0]]-Z[idx_aircraftpairs[1]] == 0, f"collision_overtake_node_{n}")
              
    # ETV constraints availability due to towing other tasks
    for a in range(N_aircraft):
       for b in range(N_aircraft):
           for i in range(N_etvs):
               if a!=b:
                   model.addConstr((Y[a,b,i] == 1) >> (t[b,0] >= t[a,len(P[a])-1] + short_path_dist(G_e, D_a[a], O_a[b]) / speed_e), f"auxiliary_task_{a}_{b}_{i}")   
                   model.addConstr(X[a,i]+X[b,i] <=1 + Y[a,b,i] + Y[b,a,i], f"available_etv_constraint_{a}_{b}_{i}")
                   
   # ETV energy consumption
   
                   
    model.update()
    return model

def short_path_dist(G, n1, n2):
    dist = nx.shortest_path_length(G, source=n1, target=n2, weight='weight')
    return dist

def Opt(G_taxi,appear_times,origins,destinations):
    
    # Airport info
    #G_a = nx.Graph()
    #G_a.add_edges_from([(0, 2,{'weight': 10}), (1, 2,{'weight': 10}), (2, 3,{'weight': 10}), (3, 4,{'weight': 10}), (3, 5,{'weight': 10})])
    G_a = G_taxi
    #G_e = nx.Graph()
    #G_e.add_edges_from([(0, 1,{'weight': 20}), (1, 5,{'weight': 20}), (5, 4,{'weight': 20}), (4, 0,{'weight': 20})])
    G_e = G_taxi

    # Aircraft info
    
    max_speed_a = 1
    min_speed_a = 0.1
    O_a = origins   # Origins
    D_a = destinations   # Destinations
    tO_a = appear_times  # Appearing times
    N_aircraft = len(O_a)
    mu = 0.02            # Rolling resistance
    m_a = 40000          # Airplane mass
    eta = 0.3            # Turbine efficiency

    # ETV info
    N_etvs = 5
    speed_e = 3

    # Aircraft routing
    N_Va = []
    P = []
    d_a = []

    for a in range(N_aircraft):
        sp = nx.shortest_path(G_a, source=O_a[a], target=D_a[a])
        d_a.append(nx.shortest_path_length(G_a, source=O_a[a], target=D_a[a], weight='weight'))
        N_Va.append(len(sp))
        P.append(sp)

    # Time definitions
    t_min = []
    t_max = []
    
    for a in range(N_aircraft):
        t_a_min = [] 
        t_a_max = []
        for n in range(len(P[a])):
            t_a_min.append(tO_a[a] + short_path_dist(G_a, O_a[a], P[a][n]) / max_speed_a)
            t_a_max.append(tO_a[a] + short_path_dist(G_a, O_a[a], P[a][n]) / min_speed_a)
        t_min.append(t_a_min)
        t_max.append(t_a_max)
    
    # Create the Gurobi model
    model = create_model(G_a, G_e, N_aircraft, P, d_a, N_etvs, max_speed_a, min_speed_a, speed_e, t_min, t_max, O_a, D_a, tO_a, mu, m_a, eta)

    # Optimize the model
    model.optimize()

    # Display results
    Dvars_name = []
    Dvars_value = []
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for var in model.getVars():
            Dvars_name.append(var.varName)
            Dvars_value.append(var.x)
            print(f"{var.varName}: {var.x}")
            
        print("Optimal objective value:", model.objVal)
    else:
        print("No optimal solution found.")
    return model

