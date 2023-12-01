"""
Created on Wed Oct 11 11:51:41 2023

@author: levi1
"""
# Import packages
import osmnx as ox
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
[G_taxi, gates, list_runway_nodes, node_mapping ]= LoadOSMdata(airport_name, operational_runways, operational_gates)


destination_gate = gates['ref']
dest_list = ox.distance.nearest_nodes(G_taxi, gates.geometry.x , gates.geometry.y , return_dist=False)
destinations = dest_list
orig = list(G_taxi)[300]
origins = []
for i in range(len(destinations)):
    origins.append(orig)

from Functions import appeartimes
appear_times = appeartimes(appear_times_T, date_of_interest)

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







