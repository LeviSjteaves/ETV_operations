# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:17:48 2023

@author: levi1
"""

import gurobipy as gp
from gurobipy import GRB
import networkx as nx


def create_model(N_aircraft, P, N_etvs, max_speed_a, min_speed_a):
    model = gp.Model("Aircraft_Taxiing")

    # Definitions
    I_col = [] #set of aircraft with possible collisions
    I_col_nodes = []
    for a in range(N_aircraft):
        for b in range(N_aircraft):
            for n in range(1, len(P[a])):
                for m in range(1, len(P[b])):
                    e_col = []
                    #possibility to be at the same node at the same time
                    if a != b and P[a][n] == P[b][m] and (t_min[a][n] <= t_max[b][m] and t_min[b][m] <= t_max[a][n]):
                        e_col = [a,b]
                        I_col_nodes.append(P[a][n])
                        I_col.append(e_col)
    idx_col = []
    for a in range(len(I_col_nodes)):
        for b in range(len(I_col_nodes)):
            if I_col[a][0] == I_col[b][1] and I_col[a][1] == I_col[b][0] and I_col_nodes[a] == I_col_nodes[b] and a<b:
                idx_col.append([a,b])
                
    # Decision variables
    t = {}  # Arrival times nodes
    X = {}  # Towing tasks
    Z = {}  # Order of solving collisions
    
    for a in range(N_aircraft):
        for n in range(len(P[a])):
            t[a, n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{a}_{n}")

    for a in range(N_aircraft):
        for i in range(N_etvs):
            X[a, i] = model.addVar(vtype=GRB.BINARY, name=f"X_{a}_{i}")
            
    for a in range(len(idx_col)):
            Z[a] = model.addVar(vtype=GRB.BINARY, name=f"Z_{a}")
            
    # Objective function: minimize emissions (only rolling resistance) + total taxitime
    model.setObjective(gp.quicksum(mu*m_a*g*d_a[a]*eta*(1-sum(X[a,i] for i in range(N_etvs))) for a in range(N_aircraft))+gp.quicksum(t[a, n] for a in range(N_aircraft) for n in range(1, len(P[a])))+100000000*(1-gp.quicksum(Z[a] for a in range(len(idx_col)))), sense=GRB.MINIMIZE)
    
                         
    # Constraints
    for a in range(N_aircraft):
        #Min start time
        model.addConstr(t[a, 0] >= tO_a[a], f"appear_constraint_{a}")
        # max one etv per aircraft
        model.addConstr(gp.quicksum(X[a, i] for i in range(N_etvs)) <= 1, f"one_towing_task_{a}")
        for n in range(1, len(P[a])):
            #Min/max time to arrive at each node (from min/max speed)
            model.addConstr(t[a, n] >= t[a, n-1] + short_path_dist(G_a, P[a][n-1], P[a][n]) / max_speed_a, f"arrival_constraint_min_{a}_{n}")
            model.addConstr(t[a, n] <= t[a, n-1] + short_path_dist(G_a, P[a][n-1], P[a][n]) / min_speed_a, f"arrival_constraint_max_{a}_{n}")
    
    # Collision
    I_col_filter = []
    I_col_nodes_filter = []
    idx_choice=[0,1]    
    idx_ind = [0]*len(idx_col)
    for a in range(len(idx_col)):
        model.addConstr(idx_ind[a] == Z[a] * idx_choice[1] + (1 - Z[a]) * idx_choice[0], f"order_constraint_{a}")
        I_col_filter.append(I_col[idx_col[a][idx_ind[a]]])
        I_col_nodes_filter.append(I_col_nodes[idx_col[a][idx_ind[a]]])
    
    for n in range(len(I_col_nodes_filter)):
            model.addConstr(t[I_col_filter[n][1],P[I_col_filter[n][1]].index(I_col_nodes_filter[n])] >= t[I_col_filter[n][0],P[I_col_filter[n][0]].index(I_col_nodes_filter[n])]+1, f"collision_constraint_node_{a}_{n}")
    
    # ETV constraints availability
    for k in range(len(I_col_filter)):  
        a =I_col_filter[k][0]
        b =I_col_filter[k][1]
        for i in range(N_etvs):
            if tO_a[b] <= tO_a[a] +short_path_dist(G_a, O_a[a], D_a[a]) / max_speed_a + short_path_dist(G_e, D_a[a], O_a[b]) / speed_e: #assume fastest path instead of actual path time
                    model.addConstr(X[a, i]+ X[b, i] <= 1)
                    #e=1

    model.update()
    return model, I_col, I_col_nodes, idx_col, idx_ind, I_col_nodes_filter

def short_path_dist(G, n1, n2):
    dist = nx.shortest_path_length(G, source=n1, target=n2, weight='weight')
    return dist


if __name__ == "__main__":
    
    g=9.81

    # Airport info
    G_a = nx.Graph()
    G_a.add_edges_from([(0, 2,{'weight': 10}), (1, 2,{'weight': 10}), (2, 3,{'weight': 10}), (3, 4,{'weight': 10}), (3, 5,{'weight': 10})])
    G_e = nx.Graph()
    G_e.add_edges_from([(0, 1,{'weight': 10}), (1, 5,{'weight': 10}), (5, 4,{'weight': 10}), (4, 0,{'weight': 10})])


    # Aircraft info
    N_aircraft = 3
    max_speed_a = 1
    min_speed_a = 0.01
    O_a = [0, 1, 4]   # Origins
    D_a = [4, 5, 0]   # Destinations
    tO_a = [50, 0, 0]  # Appearing times
    mu = 0.02      # Rolling resistance
    m_a = 40000    # Airplane mass
    eta = 0.3      # Turbine efficiency

    # ETV info
    N_etvs = 1
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
    model, I_col, I_col_nodes, idx_col, idx_ind, I_col_nodes_filter = create_model(N_aircraft, P, N_etvs, max_speed_a, min_speed_a)

    # Optimize the model
    model.optimize()

    # Display results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for var in model.getVars():
            print(f"{var.varName}: {var.x}")

        print("Optimal objective value:", model.objVal)
    else:
        print("No optimal solution found.")