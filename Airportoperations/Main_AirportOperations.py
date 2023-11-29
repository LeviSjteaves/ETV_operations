"""
Created on Wed Oct 11 11:51:41 2023

@author: levi1
"""
# Import packages
import osmnx as ox
from datetime import datetime
from gurobipy import GRB

#Import API data from schiphhol
date_of_interest = '2023-11-01'     #Pick the date for which you want the API data
pagelimit = [0,3]                   #Specify the amount of pages of data

from Functions import LoadAPIdata   #Returns a list of dicts with flight info
flightdata = LoadAPIdata(date_of_interest, pagelimit)

#Airport inputs
airport_name = 'Schiphol, The Netherlands'                  #Name of airport for usecase
operational_runways = ['18R/36L','18C/36C','06/24','09/27','18L/36R']   #Names of the runways of that airport
operational_gates = []
appear_times_T = []

for flight in flightdata:
    operational_gates.append(flight['gate'])
    if flight['flightDirection'] == 'D':
        appear_times_T.append(flight['actualOffBlockTime'])
    else:
        appear_times_T.append(flight['actualLandingTime'])
    
operational_gates = ['H1' if x is None else x for x in operational_gates]


from Functions import LoadOSMdata
[G_taxi, gates, list_runway_nodes ]= LoadOSMdata(airport_name, operational_runways, operational_gates)


destination_gate = gates['ref']
dest_list = ox.distance.nearest_nodes(G_taxi, gates.geometry.x , gates.geometry.y , return_dist=False)
destinations = dest_list
orig = list(G_taxi)[300]
origins = []
for i in range(len(destinations)):
    origins.append(orig)

from Functions import Routing
#for orig, dest in zip(origins, destinations):
routes = Routing(orig, destinations, G_taxi)

#Add routes to flightdata list
for i in range(len(flightdata)):
    flightdata[i]['Plane_route'] = routes[i]

fig, ax = ox.plot_graph_route(G_taxi, route=routes[0] ,  route_color='b' ,route_linewidth=6, node_size=10, bgcolor='k')

from Functions import Timeplanning
route_times = Timeplanning(routes, G_taxi)


timestamp_format1 = "%Y-%m-%dT%H:%M:%S.%f%z"
timestamp_format2 = "%Y-%m-%d"

# Parse the timestamp string into a datetime object
appear_times = []
for i in range(len(appear_times_T)):
    appear_times_C = datetime.strptime(appear_times_T[i], timestamp_format1)
    timestamp_reference = datetime.strptime(date_of_interest, timestamp_format2)
    appear_times.append(int(appear_times_C.timestamp())-int(timestamp_reference.timestamp()))

#optimization
from Functions import Opt
model = Opt(G_taxi,appear_times,origins,destinations) 

Dvars_name = []
Dvars_value = []
if model.status == GRB.OPTIMAL:   
    for var in model.getVars():
        Dvars_name.append(var.varName)
        Dvars_value.append(var.x)
else:
    print("No optimal solution found.")







