# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:00:27 2024

@author: levi1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
from importlib import import_module
import networkx as nx
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

global iterations,Runtime,taxi_delay,total_delay,Kg_kerosene,Costs_etvs,ETVs_cat1,ETVs_cat2,allowed_delay,N_aircraft,obj,gap,bounds,times,gap_per,Kg_frac,gap_percentage

# Define global variables
iterations = []
Runtime = []
taxi_delay = []
total_delay = []
Kg_kerosene = []
Costs_etvs = []
ETVs_cat1 = []
ETVs_cat2 = []
allowed_delay = []
N_aircraft = []
obj = []
gap = []
bounds = []
times = []
gap_per = []
Kg_frac = []
gap_percentage = []

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

module = import_module('AirportOp_Func_1')
Plotting = module.Plotting
plot_scheme = True
comp_plots = False

#close all open plots
plt.close('all')

def Load_new_file(folder_name):
    global iterations,Runtime,taxi_delay,total_delay,Kg_kerosene,Costs_etvs,ETVs_cat1,ETVs_cat2,allowed_delay,N_aircraft,obj,gap,bounds,times,gap_per,Kg_frac,gap_percentage
    # Initialize an empty list to store the loaded dictionaries
    iterations.clear()
    Runtime.clear()
    taxi_delay.clear()
    total_delay.clear()
    Kg_kerosene.clear()
    Costs_etvs.clear()
    ETVs_cat1.clear()
    ETVs_cat2.clear()
    allowed_delay.clear()
    N_aircraft.clear()
    obj.clear()
    gap.clear()
    bounds.clear()
    times.clear()
    gap_per.clear()
    Kg_frac.clear()
    gap_percentage.clear()
    
    files = os.listdir(folder_name)
    # Load each file from the folder
    for file in files:
        if file.endswith('.npy'):  # Check if the file is a numpy file
            loaded_dict = np.load(os.path.join(folder_name, file), allow_pickle=True).item()  # Load the dictionary
            iterations.append(loaded_dict)  # Append the loaded dictionary to the list
    # Define global variables
    # Clear previous data

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
        
    gap_percentage = []
    for i in range(len(iterations)):
        gap_per = [abs(bounds[i][-1] - obj[i][-1]) / abs(obj[i][-1])*100]
        gap_percentage.append(gap_per)


#################################################################
#COMPUTATION PLOTS
if comp_plots == True:
    # Define the maximum number of plots per window
    max_plots_per_window =5
    
    # Calculate the total number of windows needed
    num_windows = (len(iterations) + max_plots_per_window - 1) // max_plots_per_window
    
    for i in range(len(iterations)):
        first_normal = next((i for i, value in enumerate(obj[i]) if value < np.exp(100)), None)
        obj[i] = [value if value < np.exp(100) else obj[i][first_normal] for value in obj[i]]
    
    # Create and iterate over windows
    for window_index in range(num_windows):
        # Calculate the start and end index for the plots in this window
        start_index = window_index * max_plots_per_window
        end_index = min((window_index + 1) * max_plots_per_window, len(iterations))
    
        # Create figure and axes
        fig, axes = plt.subplots(end_index - start_index, 1, figsize=(8, 6 * (end_index - start_index)))
    
        if end_index - start_index == 1:  # If only one iteration, adjust axes to be iterable
            axes = [axes]
    
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
    
            ax.set_title(f'Iteration {index} with {ETVs_cat1[index]} normal ETVs and {ETVs_cat2[index]} heavy ETVs, {N_aircraft[index]}')
    
            # Add legend
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
        # Adjust layout
        plt.tight_layout()
    
        # Show plot
        plt.show()


#################################################################
#PARETO PLOT
# Folder containing the saved files
folder_name = "2024-03-22_10-35-15"
Load_new_file(folder_name) 

Kg_frac_save = []
Kg_frac = []
        
for k in range(len(iterations)):
    Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == 60), None)]
    Kg_frac.append((1-Kg_kerosene[k]/Kg_kerosene_max)*100) 

fig, ax = plt.subplots(figsize=(9/2.54, 8/2.54))

blue_cmap_tot = plt.get_cmap('Blues')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.2, 1, 256)))
color_dict = {value: blue_cmap(i / len(allowed_delay)) for i, value in enumerate(allowed_delay) if ETVs_cat2[i] == 2 }
handles = []
labels = []
#plt.plot(taxi_delay, Kg_kerosene , color='C0')
for i in range(len(allowed_delay)):
    if ETVs_cat2[i] == 2:
        handle = plt.scatter(taxi_delay[i], Kg_frac[i],s=25, color=color_dict[allowed_delay[i]])
        plt.arrow(taxi_delay[i]+0.1, Kg_frac[i], 0.4, 0, width = 0.0005, head_width = 0.2, head_length = 0.1, color='k')
        if i == 1:
            ax.text(taxi_delay[i]+1, Kg_frac[i], str(f'{round(max(total_delay[i]),2)}'), color='k',
                ha='center', va='bottom', fontweight='bold', fontsize= 8)
        else:   
            ax.text(taxi_delay[i]+1, Kg_frac[i], str(f'{round(max(total_delay[i]),2)}'), color='k',
                ha='center', va='top', fontweight='bold', fontsize= 8)
        handles.append(handle)
        labels.append(r'$t_\textrm{w}^a=$'f'{allowed_delay[i]}')

# Create legend using handles for unique N_aircraft values
ax.legend(handles=handles, labels=labels, fontsize=8)
ax.set_ylabel(r'Fuel savings [\%]',fontsize=10)
ax.set_xlabel(r'Avg. Taxi delay [min]',fontsize=10)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
ax.grid(True)
plt.tight_layout()  
plt.xlim(0, 6)

##################################################################
#ETV COMPOSITION PLOT fuel consumption
# Folder containing the saved files
folder_name = "2024-03-08_22-30-15_2"
Load_new_file(folder_name)

Kg_frac_save = []
Kg_frac = []

for k in range(len(iterations)):
    Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == 200), None)]
    Kg_frac.append((1-Kg_kerosene[k]/Kg_kerosene_max)*100) 
Kg_frac_dual = []
Kg_kerosene_dual = []

for k in range(len(iterations)):
    variable_values = iterations[k]['variable_values']
    P = iterations[k]['P']
    t_min = iterations[k]['t_min']
    Kg_kerosene_dual.append(((bounds[k][-1]-0.01*sum((variable_values['t'][a][len(P[a])-1]-(t_min[a][len(P[a])-1])) for a in range(N_aircraft[k])))))

for k in range(len(iterations)):    
    Kg_kerosene_max_dual = Kg_kerosene_dual[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == N_aircraft[k]), None)]
    Kg_frac_dual.append((1-Kg_kerosene_dual[k]/Kg_kerosene_max_dual)*100) 
    
fig, ax = plt.subplots(figsize=(9/2.54, 5.3/2.54))

#Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == N_aircraft[k]), None)]
cat = iterations[0]['cat']
dep = iterations[0]['dep']
for i in range(len(iterations)):
    if Costs_etvs[i] == 0:
        emissions = [iterations[i]['variable_values']['Emission'][k] for k in range(N_aircraft[i])]
Max_cons = sum(emissions) 

Max_reduction_1 = (1-(Max_cons-sum(emission for emission, cat_val in zip(emissions, cat) if cat_val < 5))/Kg_kerosene_max)
Max_reduction_2 = (1-(Max_cons-sum(emission for emission, cat_val in zip(emissions, cat) if cat_val < 8))/Kg_kerosene_max)

for i in range(len(Kg_frac)):
    if ETVs_cat1[i] > 0:
        Kg_frac_save.append(Kg_frac[i]/ETVs_cat1[i])

# Assign color
blue_cmap_tot = plt.get_cmap('Blues_r')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.0, 0.8, 256)))

# Assign color
O_cmap_tot = plt.get_cmap('Oranges_r')
O_cmap = ListedColormap(O_cmap_tot(np.linspace(0.0, 0.8, 256)))

pie_marker_list = []
cat1max =max(ETVs_cat1)+1
cat2max =max(ETVs_cat2)+1
# Plot a point at (0, 0) and add custom pie marker

# Define your data  
x_set = [ETVs_cat1[i]+ETVs_cat2[i] for i in range(len(iterations))]
y_set = [Kg_frac[k] for k in range(len(Kg_frac))]
   
handles = []        
handle1 = plt.scatter([x_set[i]for i in range(len(iterations)) if x_set[i] != 0], [y_set[i]for i in range(len(iterations)) if x_set[i] != 0], c=[gap_percentage[i]for i in range(len(iterations)) if x_set[i] != 0], cmap=blue_cmap,s=20)
 
handle2 = plt.scatter([ETVs_cat2[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], [y_set[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], c=[gap_percentage[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], cmap=O_cmap,s=20)

plt.scatter([ETVs_cat1[i]for i in range(len(iterations)) if ETVs_cat1[i] != 0], [Kg_frac_dual[i]for i in range(len(iterations)) if ETVs_cat1[i] != 0], color = 'C0',marker=MarkerStyle(1),s=5)
plt.scatter([ETVs_cat2[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], [Kg_frac_dual[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], color = 'peru',marker=MarkerStyle(1),s=5)
   
plt.axhline(y=Max_reduction_1, color='C0', linestyle='--', label='Horizontal Line', linewidth=0.5)
plt.axhline(y=Max_reduction_2, color='peru', linestyle='--', label='Horizontal Line', linewidth=0.5)
   
ax.set_xlabel(r'ETV-fleet size $|\mathcal{I}_e|$',fontsize=10)
ax.set_ylabel(r'Fuel savings [\%]',fontsize=10)
#ax.set_ylabel('$\sum_{a \in \mathcal{I}_a}(t^a_{\mathrm{end}}-t_a^{\mathrm{min}})$',fontsize=10)
legend_elements = [
    patches.Patch(facecolor=plt.cm.Blues(np.linspace(0.25, 0.75, cat1max))[int(cat1max/2)], label='ETVs normal class'),
    patches.Patch(facecolor=plt.cm.Oranges(np.linspace(0.25, 0.75, cat1max))[int(cat1max/2)], label='ETVs heavy class'),
]
ax.legend(handles=legend_elements,fontsize=8)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
plt.xlim(0, None)
plt.ylim(0, None)
ax.grid(True)
plt.tight_layout()

cbar = plt.colorbar(handle1)
cbar.set_label(r'Optimality gap \%')
#yticks = [25,50,75]
#cbar.set_ticks(yticks)
cbar = plt.colorbar(handle2)
#yticks = [25,50,75]
#cbar.set_ticks(yticks)
plt.show()

##################################################################
#ETV COMPOSITION PLOT taxi delay
# Folder containing the saved files
folder_name = "2024-03-08_22-30-15_2"
Load_new_file(folder_name)

fig, ax = plt.subplots(figsize=(9/2.54, 5.3/2.54))

# Assign color
blue_cmap_tot = plt.get_cmap('Blues_r')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.0, 0.8, 256)))

# Assign color
O_cmap_tot = plt.get_cmap('Oranges_r')
O_cmap = ListedColormap(O_cmap_tot(np.linspace(0.0, 0.8, 256)))

pie_marker_list = []
cat1max =max(ETVs_cat1)+1
cat2max =max(ETVs_cat2)+1
 
# Define your data  
x_set = [ETVs_cat1[i]+ETVs_cat2[i] for i in range(len(iterations))]
y_set = [taxi_delay[k] for k in range(len(taxi_delay))]

handles = []        
handle1 = plt.scatter([x_set[i]for i in range(len(iterations)) if x_set[i] != 0], [y_set[i]for i in range(len(iterations)) if x_set[i] != 0], c=[gap_percentage[i]for i in range(len(iterations)) if x_set[i] != 0], cmap=blue_cmap,s=20)
 
handle2 = plt.scatter([ETVs_cat2[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], [y_set[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], c=[gap_percentage[i]for i in range(len(iterations)) if ETVs_cat2[i] != 0], cmap=O_cmap,s=20)
   
cbar = plt.colorbar(handle1)
cbar.set_label(r'Optimality gap \%')
cbar = plt.colorbar(handle2)
   
ax.set_xlabel(r'ETV-fleet size $|\mathcal{I}_e|$',fontsize=10)
ax.set_ylabel(r'Avg. Taxi delay [min]',fontsize=10)
legend_elements = [
    patches.Patch(facecolor=plt.cm.Blues(np.linspace(0.25, 0.75, cat1max))[int(cat1max/2)], label='ETVs normal class'),
    patches.Patch(facecolor=plt.cm.Oranges(np.linspace(0.25, 0.75, cat1max))[int(cat1max/2)], label='ETVs heavy class'),
]
ax.legend(handles=legend_elements,fontsize=8)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
plt.xlim(0, None)
plt.ylim(0, None)
ax.grid(True)
plt.tight_layout()   

##################################################################
#ETV COMPOSITION PLOT fuel consumption DIFFERENT APLPHA
# Folder containing the saved files
folder_name = "2024-03-08_22-30-15_4"
Load_new_file(folder_name)

fig, ax = plt.subplots(figsize=(9/2.54, 4.5/2.54))


blue_cmap_tot = plt.get_cmap('Blues_r')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.0, 0.8, 256)))
alpha = []
for i in range(len(iterations)):
    alpha.append(iterations[i]['p']['F_delay'])

import matplotlib.colors as mcolors    
# Plot the points
handles = []
handle = plt.scatter([ETVs_cat1[i]+ETVs_cat2[i] for i in range(len(iterations))],  taxi_delay, c=alpha, cmap=blue_cmap,s=10,norm=mcolors.LogNorm())


#cbar = plt.colorbar(handle)
cbar = plt.colorbar(handle, norm=mcolors.LogNorm())
cbar.set_label(r'alpha')
plt.show()
 
ax.set_xlabel(r'ETV-fleet size $|\mathcal{I}_e|$',fontsize=10)
ax.set_ylabel(r'$m_f$',fontsize=10)
#ax.set_ylabel('$\sum_{a \in \mathcal{I}_a}(t^a_{\mathrm{end}}-t_a^{\mathrm{min}})$',fontsize=10)

ax.legend(handles=legend_elements,fontsize=8)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
plt.xlim(0, None)
plt.ylim(0, None)
ax.grid(True)
plt.tight_layout()

##################################################################
#ETV COMPOSITION PLOT (Multiple with patches)
folder_name = "2024-03-22_15-53-18"
Load_new_file(folder_name)

Kg_frac_save = []
Kg_frac = []
        
for k in range(len(iterations)):
    Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == 200), None)]
    Kg_frac.append((1-Kg_kerosene[k]/Kg_kerosene_max)*100) 


fig, ax = plt.subplots(figsize=(9/2.54, 8/2.54))
pie_marker_list = []
cat1max =max(ETVs_cat1)+1
cat2max =max(ETVs_cat2)+1
colorsb = plt.cm.Blues(np.linspace(0.1, 0.8, cat1max))
colorsg = plt.cm.Oranges(np.linspace(0.1, 0.8, cat2max))
# Plot a point at (0, 0) and add custom pie marker
for i in range(len(iterations)):  
    # Define your data  
    x_set = [ETVs_cat1[i]+ETVs_cat2[i] for i in range(len(iterations))]
    y_set = [Kg_frac[i] for k in range(len(taxi_delay))]
    x = x_set[i]
    y = y_set[i]
    x_max = max(x_set)
    y_max = 0.6
    x_min = min(x_set)
    y_min = min(y_set)
    plt.scatter(x_set[i], y_set[i], marker=MarkerStyle('o', fillstyle='left'), s=250, color=colorsb[ETVs_cat1[i]]) 
    plt.scatter(x_set[i], y_set[i], marker=MarkerStyle('o', fillstyle='right'), s=250, color=colorsg[ETVs_cat2[i]])
    plt.scatter(x_set[i], y_set[i], s=2.5,  color='k')
    #positions = [[Kg_kerosene[i]] * len(total_delay[i]) for i in range(len(total_delay))]
    ax.text(x_set[i], y_set[i], f'{ETVs_cat1[i]}  {ETVs_cat2[i]}', ha='center', va='center', fontsize=8)

for i in range(len(iterations)):  
    # Define your data  
    x_set = [ETVs_cat1[i]+ETVs_cat2[i] for i in range(len(iterations))]
    y_set = [Kg_frac[i] for k in range(len(taxi_delay))]
    x = x_set[i]
    y = y_set[i]
    plt.scatter(x_set[i], y_set[i], s=2.5,  color='k')
    
ax.set_xlabel(r'ETV-fleet size $|\mathcal{I}_e|$',fontsize=10)
ax.set_ylabel(r'Fuel savings $[\%]$',fontsize=10)
legend_elements = [
    patches.Patch(facecolor='C0', label='ETVs normal class'),
    patches.Patch(facecolor='orange', label='ETVs heavy class')
]
ax.legend(handles=legend_elements,fontsize=8)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
#plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
#plt.minorticks_on()
plt.xlim(0, None)
plt.ylim(0, None)
ax.grid(True)
plt.tight_layout() 

##################################################################  
# DISTRIBUTION PLOT VIOLIN
folder_name = "2024-03-08_22-30-15_3"
Load_new_file(folder_name)

fig, ax = plt.subplots(figsize=(9/2.54, 9/2.54))
# Plot a point at (0, 0) and add custom pie marker
for i in range(len(iterations)):  
    # Define your data  
    x_set = taxi_delay
    y_set = ETVs_cat2
    x = x_set[i]
    y = y_set[i]
    x_max = max(x_set)
    y_max = max(y_set)
    x_min = min(x_set)
    y_min = min(y_set)

    # Plot violin/boxplot
    #plt.boxplot(total_delay[i],positions=range(int(y_set[i]),int(y_set[i])+1),widths= 0.5,vert=False, whis=10)
    #vp = plt.violinplot(total_delay[i],positions=range(int(y_set[i]),int(y_set[i])+1),vert=False,widths=1,)
    vp = ax.violinplot(total_delay[i],positions=range(int(y_set[i]),int(y_set[i])+1),vert=False,widths=1.5,showmedians=False,showmeans=True)
    for pc in vp['bodies']:
        pc.set_color('C0')
        pc.set_facecolor('C0')
        pc.set_alpha(0.80)
    for partname in ('cbars', 'cmins', 'cmaxes','cmeans'):
        vp[partname].set_edgecolor('black')
        vp[partname].set_linewidth(1.5)  # Optional: Set the linewidth of the edges


ax.set_xlabel('$t$ [min]',fontsize=10)
ax.set_ylabel( r'Heavy ETV-fleet size $|\mathcal{I}_e|$',fontsize=10)
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
#ax.legend(handles=legend_elements)
plt.xlim(0, None)
plt.ylim(0, None)
plt.tight_layout()
ax.grid(True)


################################################################
#AMOUNT OF AIRCRAFT: SINGLE ETV
folder_name = "2024-03-11_14-18-49_3"
Load_new_file(folder_name)

N_aircraft_c = []
charges = []


for i in range(len(iterations)):
    if iterations[i]['p']['N_etvs'] > 0  :
        charges.append(sum([iterations[i]['variable_values']['C'][0][a] for a in range(N_aircraft[i])]))
        N_aircraft_c.append(N_aircraft[i])
        
for k in range(len(iterations)):
    Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == N_aircraft[k]), None)]
    Kg_frac.append((1-Kg_kerosene[k]/Kg_kerosene_max)*100) 

Kg_frac_dual = []
Kg_kerosene_dual = []

for k in range(len(iterations)):
    variable_values = iterations[k]['variable_values']
    P = iterations[k]['P']
    t_min = iterations[k]['t_min']
    Kg_kerosene_dual.append(((bounds[k][-1]-0.01*sum((variable_values['t'][a][len(P[a])-1]-(t_min[a][len(P[a])-1])) for a in range(N_aircraft[k])))))

for k in range(len(iterations)):    
    Kg_kerosene_max_dual = Kg_kerosene_dual[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == N_aircraft[k]), None)]
    Kg_frac_dual.append((1-Kg_kerosene_dual[k]/Kg_kerosene_max_dual)*100) 

fig, ax = plt.subplots(figsize=(9/2.54, 4.5/2.54))

# Assign color
blue_cmap_tot = plt.get_cmap('Blues_r')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.0, 0.8, 256)))

# Plot the points
handles = []
handle = plt.scatter([N_aircraft],  [Kg_frac], c=gap_percentage, cmap=blue_cmap,s=10)
plt.scatter([N_aircraft],  [Kg_frac_dual], color = 'k',marker=MarkerStyle(1),s=5)
#ax2 = ax.twinx()  # Create a twin axes sharing the same x-axis
#ax2.plot(N_aircraft_c, charges, color='grey')  # Plot second scatter plot on the twin axes


ax.set_xlabel( r'$|\mathcal{I}_a|$',fontsize=10)
ax.set_ylabel(r'Fuel savings [\%]',fontsize=10)

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10)) 
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
plt.tight_layout()

# Plot the points
cbar = plt.colorbar(handle)
cbar.set_label(r'Gap \%')
yticks = [25,50,72]
cbar.set_ticks(yticks)
plt.show()

################################################################
#AMOUNT OF AIRCRAFT: MULTIPLE ETV
folder_name = "2024-03-12_09-37-00"
Load_new_file(folder_name)
Kg_frac_b =[]
for k in range(len(iterations)):
    Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == N_aircraft[k]), None)]
    Kg_frac.append((1-Kg_kerosene[k]/Kg_kerosene_max)) 
    Kg_frac_b.append((1-bounds[k][-1]/Kg_kerosene_max)*100) 

fig, ax = plt.subplots(figsize=(9/2.54, 8/2.54))

unique_values_cat1 = sorted(list(set(ETVs_cat1)))
unique_values_cat2 = sorted(list(set(ETVs_cat2)))
#unique_values_combined = [(cat1, cat2) for cat1 in unique_values_cat1 for cat2 in unique_values_cat2]
unique_values_combined = [(1,0),(2,0),(0,1),(1,1),(0,2),(2,1),(1,2),(2,2)]

# Assign colors to each unique value combination
blue_cmap_tot = plt.get_cmap('Blues')
blue_cmap = ListedColormap(blue_cmap_tot(np.linspace(0.2, 1, 256)))
color_dict = {value: blue_cmap(i / len(unique_values_combined)) for i, value in enumerate(unique_values_combined)}

# Plot the points
handles = []
labels = []
for value in unique_values_combined:
    mask_cat1 = [ETVs_cat1[i] == value[0] for i in range(len(iterations))]
    mask_cat2 = [ETVs_cat2[i] == value[1] for i in range(len(iterations))]
    mask_combined = [mask_cat1[i] and mask_cat2[i] for i in range(len(iterations))]
    handle = plt.scatter([N_aircraft[i] for i, match in enumerate(mask_combined) if match], 
                         [Kg_frac[i] for i, match in enumerate(mask_combined) if match], 
                         color=color_dict[value],
                         s=10,
                         label=r'$|\mathcal{I}_e|=$'f'{value}')
    #new = plt.scatter([N_aircraft[i] for i, match in enumerate(mask_combined) if match], 
    #                     [Kg_frac_b[i] for i, match in enumerate(mask_combined) if match], 
    #                     color=color_dict[value],
    #                     s=3,
    #                     label=r'$|\mathcal{I}_e|=$'f'{value}')
    handles.append(handle)
    labels.append(r'$|\mathcal{I}_e|=$'f'{value}')
'''
    X = np.array(np.sort([N_aircraft[i] for i, match in enumerate(mask_combined) if match])).reshape(-1, 1)
    y = np.array(np.sort([Kg_frac[i] for i, match in enumerate(mask_combined) if match])[::-1])
    np.delete(X, 0, axis=0)
    np.delete(y, 0, axis=0)
    poly = PolynomialFeatures(degree=2)  # You can change the degree of the polynomial
    X_poly = poly.fit_transform(X)    
    model = Ridge(alpha=2000)  # Ridge regression with regularization
    #model = LinearRegression()
    model.fit(X_poly, y)
    x_values = np.linspace(min(N_aircraft), max(N_aircraft), 100)
    x_values_poly = poly.fit_transform(x_values.reshape(-1, 1))
    y_values = model.predict(x_values_poly)
    plt.plot(x_values, y_values, color=color_dict[value], linestyle='-', linewidth=1) 
'''

ax.set_xlabel( r'$|\mathcal{I}_a|$',fontsize=10)
ax.set_ylabel(r'Fuel savings [\%]',fontsize=10)

# Create legend using handles for unique N_aircraft values
ax.legend(handles=handles, labels=labels,fontsize=8)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10)) 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.3)
plt.minorticks_on()
plt.tight_layout()
plt.grid(True)
plt.show()


###################################################################
# OCCURANCES AIRCRAFT
folder_name = "2024-03-04_09-41-05"
Load_new_file(folder_name)

fig, ax = plt.subplots(figsize=(3.504, 2))
categories = ['1','2','3','4','5','6','7','8','9','10']
cat = iterations[0]['cat']
dep = iterations[0]['dep']
if type(cat) == np.ndarray:
    cat = cat.tolist()

arr_cat = [cat[i] for i in range(len(cat)) if dep[i] != 1]
dep_cat = [cat[i] for i in range(len(cat)) if dep[i] != 0]

emissions = []
for i in range(len(iterations)):
    if Costs_etvs[i] == 0:
        emissions = [iterations[i]['variable_values']['Emission'][k] for k in range(N_aircraft[i])]
    
arr_values = [arr_cat.count(0),arr_cat.count(1),arr_cat.count(2),arr_cat.count(3),arr_cat.count(4),arr_cat.count(5),arr_cat.count(6),arr_cat.count(7),arr_cat.count(8),arr_cat.count(9)]
dep_values = [dep_cat.count(0),dep_cat.count(1),dep_cat.count(2),dep_cat.count(3),dep_cat.count(4),dep_cat.count(5),dep_cat.count(6),dep_cat.count(7),dep_cat.count(8),dep_cat.count(9)]

plt.bar(categories, arr_values,  label='Arrivals')
plt.bar(categories, dep_values, color='peru', label='Departures', bottom=arr_values)
ax.legend(loc='upper right',fontsize=10)
ax.set_xlabel('Aircraft category',fontsize=10)
ax.set_ylabel( r'$|\mathcal{I}_a|$',fontsize=10)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.tight_layout()
plt.show()

# Plot traffic per hour
appear_times = iterations[0]['appear_times']
appearance_hours = [time / 60 for time in appear_times]

arr_appearance_hours = [appearance_hours[i] for i in range(len(appearance_hours)) if dep[i] != 1]
dep_appearance_hours = [appearance_hours[i] for i in range(len(appearance_hours)) if dep[i] != 0]

# Count occurrences per hour
hourly_counts_arr = [0] * 24  # Initialize a list to store counts for each hour
for hour in arr_appearance_hours:
    if hour < 24:
        hourly_counts_arr[int(hour)] += 1
        
hourly_counts_dep = [0] * 24  # Initialize a list to store counts for each hour
for hour in dep_appearance_hours:
    if hour < 24:
        hourly_counts_dep[int(hour)] += 1
    
fig, ax = plt.subplots(figsize=(3.504, 3))
# Plotting
hours = range(24)  # Hours of the day
plt.bar(hours, hourly_counts_arr,  label='Arr.')
plt.bar(hours, hourly_counts_dep, color='peru', label='Dep.', bottom=hourly_counts_arr)
ax.legend(loc='upper left',fontsize=10)
plt.xlabel('Hour of the day',fontsize=10)
plt.ylabel(r'$|\mathcal{I}_a|$',fontsize=10)
plt.yticks(fontsize=10)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10)) 
plt.xticks(hours,rotation=45,fontsize=7)  # Set x-axis ticks to display all hours
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.tight_layout()
plt.show()

###################################################################
# Show plot allocation
folder_name = "2024-04-02_11-58-09"
Load_new_file(folder_name)

#for k in range(len(iterations)):
#    Kg_kerosene_max = Kg_kerosene[next((i for i, (c1, c2, N_a) in enumerate(zip(ETVs_cat1, ETVs_cat2, N_aircraft)) if c1 == 0 and c2 == 0 and N_a == 200), None)]
#    Kg_frac.append((1-Kg_kerosene[k]/Kg_kerosene_max)) 

if plot_scheme == True:
    #for i in [15,16]:  
        i=0  
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
 

#################################################################
#Info
#folder_name = "2024-03-23_00-09-53"
#Load_new_file(folder_name)

towed=[]   
for a in range(N_aircraft[0]):
    towed.append(sum(variable_values['X'][a][i] + sum(variable_values['O'][a][I_up[a][b]][i] for b in range(len(I_up[a]))) for i in range(ETVs_cat1[0]+ETVs_cat2[0])))

cat_polder =[]
cat_polder_tot=[]
cat_list =  []
cat_tow_list =[]
for i in range(N_aircraft[0]):
    cat_list.append(cat[i]+1)
    if towed[i] >0:
        cat_tow_list.append(cat[i]+1)
        if P[i][0] == 3:
            cat_polder.append(cat[i]+1)
    if P[i][0] == 3:
        cat_polder_tot.append(cat[i]+1)
 
cat_dict = {}

cat_dict['total'] = {i: f'{cat_list.count(i)}' for i in [3, 4, 6, 7, 8, 9]}
cat_dict['towed'] = {i: f'{cat_tow_list.count(i)}' for i in [3, 4, 6, 7, 8, 9]}
cat_dict['per'] = {i: f'{cat_tow_list.count(i) / cat_list.count(i)*100}' for i in [3, 4, 6, 7, 8, 9]}

cat_dict_polder = {}

cat_dict_polder['total'] = {i: f'{cat_polder_tot.count(i)}' for i in [3, 4, 6, 7, 8, 9]}
cat_dict_polder['towed'] = {i: f'{cat_polder.count(i)}' for i in [3, 4, 6, 7, 8, 9]}
cat_dict_polder['per'] = {i: f'{cat_polder.count(i) / cat_polder_tot.count(i)*100}' for i in [3, 4, 7, 8, 9]}


            
            