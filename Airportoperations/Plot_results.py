# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:00:27 2024

@author: levi1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from importlib import import_module

module = import_module('AirportOp_Func_1')
Plotting = module.Plotting

# Folder containing the saved files
folder_name = "2024-03-02_14-08-41"

# plot the schedules for each iteration
plot_schedule = False

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
total_delay = [iteration['Total_delay'] for iteration in iterations]
Kg_kerosene = [iteration['Kg_kerosene'] for iteration in iterations]  
Costs_etvs = [iteration['Costs_etvs'] for iteration in iterations]  
ETVs_cat1 = [iteration['p']['N_etvs_cat1'] for iteration in iterations]
ETVs_cat2 = [iteration['p']['N_etvs_cat2'] for iteration in iterations]
allowed_delay = [iteration['p']['start_delay'] for iteration in iterations]
N_aircraft = [iteration['p']['N_aircraft'] for iteration in iterations]
obj = [iteration['obj_values'] for iteration in iterations]
gap = [iteration['gap_values'] for iteration in iterations]
bounds = [iteration['bound_values'] for iteration in iterations]
times = [iteration['times'] for iteration in iterations]
gap_per = []


# Define the maximum number of plots per window
max_plots_per_window = 4

# Calculate the total number of windows needed
num_windows = (len(iterations) + max_plots_per_window - 1) // max_plots_per_window

# Create and iterate over windows
for window_index in range(num_windows):
    # Calculate the start and end index for the plots in this window
    start_index = window_index * max_plots_per_window
    end_index = min((window_index + 1) * max_plots_per_window, len(iterations))

    # Create a single window and subplots
    fig, axes = plt.subplots(end_index - start_index, 1, figsize=(8, 6 * (end_index - start_index)))

    # Iterate over plots in this window
    for i, ax in enumerate(axes):
        index = start_index + i
        ax2 = ax.twinx()
        gap_per = [abs(bounds[index][a] - obj[index][a]) / abs(obj[index][a]) * 100 for a in range(len(obj[index]))]

        ax.plot(times[index], obj[index], color='tab:blue', label='Objective')
        ax.plot(times[index], bounds[index], color='k', label='Bound')

        ax2.plot(times[index], gap_per, color='tab:red', label='Gap %')

        ax.fill_between(times[index], obj[index], bounds[index], color='lightgray', alpha=0.7)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Objective Value', color='tab:blue')
        ax2.set_ylabel('Gap %', color='tab:red')
        ax.grid(True)

        ax.set_title(f'Iteration {index} with {ETVs_cat1[index]} ETVs cat 1 and {ETVs_cat2[index]} ETVs cat 2')

        # Add legend
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def create_pie_marker(sizes, x, y, x_max, y_max, markersize, cat1max, cat2max):
    patchespie = []
    x_nor = markersize/y_max*1.5
    y_nor = markersize/x_max
    colorsb = plt.cm.Blues(np.linspace(0.25, 0.75, cat1max))
    colorsg = plt.cm.Greens(np.linspace(0.25, 0.75, cat2max))
    patchespie.append(patches.Rectangle((x, y-y_nor), -x_nor, 2*y_nor, color=colorsb[sizes[0]]))
    patchespie.append(patches.Rectangle((x, y-y_nor), x_nor, 2*y_nor, color=colorsg[sizes[1]])) 
    return patchespie
'''     
    if sizes[0] == 0 and sizes[1] == 0:
        patchespie.append(patches.Rectangle((x, y-y_nor), x_nor, 2*y_nor, color='grey'))
        patchespie.append(patches.Rectangle((x, y-y_nor), -x_nor, 2*y_nor, color='grey'))
    elif sizes[0] == 0:
        colorsg = plt.cm.Greens(np.linspace(0, 1, cat2max))
        patchespie.append(patches.Rectangle((x, y-y_nor), x_nor, 2*y_nor, color=colorsg[sizes[1]+1]))
        patchespie.append(patches.Rectangle((x, y-y_nor), -x_nor, 2*y_nor, color='grey')) 
    elif sizes[1] == 0:
        colorsb = plt.cm.Blues(np.linspace(0, 1, cat1max))
        patchespie.append(patches.Rectangle((x, y-y_nor), -x_nor, 2*y_nor, color=colorsb[sizes[0]+1]))
        patchespie.append(patches.Rectangle((x, y-y_nor), x_nor, 2*y_nor, color='grey'))  
    else:
'''
    
pie_marker_list = []
fig, ax = plt.subplots()
cat1max =max(ETVs_cat1)+1
cat2max =max(ETVs_cat2)+1
# Plot a point at (0, 0) and add custom pie marker
for i in range(len(iterations)):  
    # Define your data  
    x_set = taxi_delay
    y_set = Kg_kerosene       
    x = x_set[i]
    y = y_set[i]
    x_max = max(x_set)
    y_max = max(y_set)
    x_min = min(x_set)
    y_min = min(y_set)
    markersize = 150
    # Create custom pie marker
    pie_marker = create_pie_marker([ETVs_cat1[i], ETVs_cat2[i]], x, y, x_max ,y_max, markersize, cat1max, cat2max) 
    pie_marker_list.append(pie_marker)
    plt.plot(taxi_delay[i], Kg_kerosene[i], 'o', markersize=3) 

    positions = [[Kg_kerosene[i]] * len(total_delay[i]) for i in range(len(total_delay))]

    # Plot violin/boxplot
    #plt.boxplot(total_delay[i],positions=range(int(Kg_kerosene[i]),int(Kg_kerosene[i])+1),widths= markersize/x_max,vert=False, whis=100)
    #plt.violinplot(total_delay[i],positions=range(int(Kg_kerosene[i]),int(Kg_kerosene[i])+1),vert=False,widths=200)
    for patch in pie_marker_list[i]:
        ax.add_patch(patch)
        ax.text(taxi_delay[i], Kg_kerosene[i], f'{ETVs_cat1[i]}  {ETVs_cat2[i]}', ha='center', va='center', fontsize=10)

ax.set_xlabel('Average Taxi delay [min]')
ax.set_ylabel('Fuel consumption [kg kerosene]')
legend_elements = [
    patches.Patch(facecolor=plt.cm.Blues(np.linspace(0.25, 0.75, cat1max))[int(cat1max/2)], label='ETVs cat. 1'),
    patches.Patch(facecolor=plt.cm.Greens(np.linspace(0.25, 0.75, cat1max))[int(cat1max/2)], label='ETVs cat. 2')
]
ax.legend(handles=legend_elements)
plt.xlim(0, None)
plt.ylim(0, None)
ax.grid(True)

# Show plot Runtime
fig, ax = plt.subplots()
plt.plot(Costs_etvs, Runtime, 'o')  
ax.set_xlabel('Costs ETVs [Euro]')
ax.set_ylabel('Runtime [s]')
ax.grid(True)

# Show plot allocation
if plot_schedule == True:
    for i in range(len(iterations)):  
        I_up = iterations[i]['I_up'] 
        I_do = iterations[i]['I_do']
        t_min = iterations[i]['t_min']
        appear_times = iterations[i]['appear_times'] 
        G_a = iterations[i]['G_a']
        cat = iterations[i]['cat']
        dep = iterations[i]['dep'] 
        E_a_dis =iterations[i]['E_a_dis']
        E_e_return = iterations[i]['E_e_return']
        variable_values = iterations[i]['variable_values']
        P = iterations[i]['P']
        p = iterations[i]['p']
        Plotting(variable_values, P, I_up, p, appear_times, G_a, cat, dep, E_a_dis, E_e_return, t_min)  
 
    
