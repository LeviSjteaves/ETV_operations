# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:00:27 2024

@author: levi1
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Folder containing the saved files
folder_name = "2024-02-23_13-47-55"

# List all files in the folder
files = os.listdir(folder_name)

#close all open plots
plt.close('all')
# Initialize an empty list to store the loaded dictionaries
iterations = []

# Load each file from the folder
for file in files:
    if file.endswith('.npy'):  # Check if the file is a numpy file
        loaded_dict = np.load(os.path.join(folder_name, file), allow_pickle=True).item()  # Load the dictionary
        iterations.append(loaded_dict)  # Append the loaded dictionary to the list



Runtime = [iteration['Runtime'] for iteration in iterations] 

taxi_delay = [iteration['Taxi_delay'] for iteration in iterations]
Kg_kerosene = [iteration['Kg_kerosene'] for iteration in iterations]  
Costs_etvs = [iteration['Costs_etvs'] for iteration in iterations]  
ETVs_cat1 = [iteration['p']['N_etvs_cat1'] for iteration in iterations]
ETVs_cat2 = [iteration['p']['N_etvs_cat2'] for iteration in iterations]
allowed_delay = [iteration['p']['start_delay'] for iteration in iterations]
N_aircraft = [iteration['p']['N_aircraft'] for iteration in iterations]


fig, ax = plt.subplots()	    
for i in range(len(iterations)):         
    plt.plot(taxi_delay[i], Kg_kerosene[i], 'o')    
    ax.text(taxi_delay[i], Kg_kerosene[i], f'{ETVs_cat1[i] }ETVs cat.1 \n {ETVs_cat2[i] }ETVs cat.2', color='black',
        fontweight='normal', fontsize= 10)
    ax.set_xlabel('Average Taxi delay [min]')
    ax.set_ylabel('Fuel consumption [kg kerosene]')
    ax.grid(True)
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_pie_marker(sizes, colors, x, y, start_angle=0, width_radius=10):
    total = sum(sizes)
    patchespie = []
    start_angle = start_angle
    for size, color in zip(sizes, colors):
        angle = 360 * size / total
        patchespie.append(patches.Wedge((x, y), width_radius, start_angle, start_angle + angle, fc=color))
        start_angle += angle
    return patchespie

pie_marker_list = []
# Plot a point at (0, 0) and add custom pie marker
for i in range(len(iterations)):  
    # Define your data 
    total_etvs = sum([ETVs_cat1[i], ETVs_cat2[i]])
    if total_etvs==0:
        no_etv=1
    else:
        no_etv=0
        
    x = taxi_delay[i]
    y = Kg_kerosene[i]
    # Define colors for the pie chart
    colors = ['skyblue', 'lightgreen', 'grey']
    # Create custom pie marker
    pie_marker = create_pie_marker([ETVs_cat1[i], ETVs_cat2[i], no_etv], colors, x, y) 
    pie_marker_list.append(pie_marker)
    # Create a plot

fig, ax = plt.subplots()
for i in range(len(iterations)):  
    plt.plot(taxi_delay[i], Kg_kerosene[i])    
    for patch in pie_marker_list[i]:
        ax.add_patch(patch)  # manually add the wedge patches to the plot

# Show the plot
ax.grid(True)
plt.show()

    
    
fig, ax = plt.subplots()
plt.plot(Costs_etvs, Runtime, 'o')    
ax.set_xlabel('Costs ETVs [Euro]')
ax.set_ylabel('Runtime [s]')
ax.grid(True)
plt.show()   
