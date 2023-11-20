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
from collections import defaultdict

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
    list_runway_gate_nodes = list_runway_nodes+gate_nodes
    
    '''
    # Remove nodes from the graph
    G_remove = ox.graph_from_bbox(52.3123, 52.3068, 4.7722, 4.7438,custom_filter = '["aeroway"~"runway|taxiway"]', simplify=False)
    remove_nodes = set(G_remove.nodes())
    remove_edges = set(G_remove.edges())
    G_taxi.remove_nodes_from(remove_nodes)
    G_taxi.remove_edges_from(remove_edges)
    ox.plot_graph(G_taxi)
    '''
    
    # Plot the graph with highlighted nodes
    node_colors = {node: 'r' if node in list_runway_nodes else 'g' if node in gate_nodes else 'k' for node in G_taxi.nodes()}
    fig, ax = ox.plot_graph(G_taxi, node_color=list(node_colors.values()), bgcolor='w', edge_color = 'k' ,show=True)
    return G_taxi, gates, list_runway_nodes
    
def Routing(origins, destinations, graph):
    route = {}
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


