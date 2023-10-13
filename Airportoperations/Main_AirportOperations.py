"""
Created on Wed Oct 11 11:51:41 2023

@author: levi1
"""
# Import packages
import osmnx as ox

#Import API data from schiphhol
date_of_interest = '2023-10-01'     #Pick the date for which you want the API data
pagelimit = [0,3]                   #Specify the amount of pages of data

from Functions import LoadAPIdata   #Returns a list of dicts with flight info
flightdata = LoadAPIdata(date_of_interest, pagelimit)

#Airport inputs
airport_name = 'Schiphol, The Netherlands'                  #Name of airport for usecase
operational_runways = ['18R/36L','18C/36C','06/24','09/27','18L/36R']   #Names of the runways of that airport
operational_gates = []
for flight in flightdata:
    operational_gates.append(flight['gate'])
    
operational_gates = ['H1' if x is None else x for x in operational_gates]


from Functions import LoadOSMdata
[G_taxi, gates, list_runway_nodes ]= LoadOSMdata(airport_name, operational_runways, operational_gates)

origins = [list(G_taxi)[600]]
destination_gate = gates['ref']
dest_list = ox.distance.nearest_nodes(G_taxi, gates.geometry.x , gates.geometry.y , return_dist=False)
destinations = dest_list

from Functions import Routing
#for orig, dest in zip(origins, destinations):
routes = Routing(origins, destinations, G_taxi)

#Add routes to flightdata list
for i in range(len(flightdata)):
    flightdata[i]['Plane_route'] = routes[i]

fig, ax = ox.plot_graph_route(G_taxi, route=routes[0] ,  route_color='b' ,route_linewidth=6, node_size=10, bgcolor='k')

from Functions import Timeplanning
arrival_times = Timeplanning(routes, G_taxi)










