# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:31:14 2023

@author: levi1
"""
import networkx as nx
import plotly.graph_objects as go

# Create a sample graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (5, 6)])

# Get node positions
pos = nx.spring_layout(G)

# Create a plotly figure
fig = go.Figure()

# Add nodes to the figure
for node, (x, y) in pos.items():
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', marker=dict(size=10), text=[str(node)]))

# Add edges to the figure
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines'))

# Define a callback for node clicks
def remove_node(trace, points, selector):
    if points.point_inds:
        node_to_remove = G.nodes[points.point_inds[0]]
        G.remove_node(node_to_remove)
        update_figure()

# Set up the callback
fig.data[0].on_click(remove_node)

# Define a function to update the figure after removing nodes
def update_figure():
    fig.data = []

    # Add nodes to the figure
    for node, (x, y) in pos.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', marker=dict(size=10), text=[str(node)]))

    # Add edges to the figure
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines'))

    # Update layout
    fig.update_layout(title_text='Click on a node to remove it', showlegend=False)

# Show the figure
fig.show()