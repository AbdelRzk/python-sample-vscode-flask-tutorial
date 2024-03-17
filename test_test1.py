#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
get_ipython().run_line_magic('matplotlib', 'notebook')



n = int(input("Enter the number of individuals for this experiment: "))
Matrix_score = [[0 for j in range(n)] for i in range(n)]
PHQ_9 = [0 for k in range(n)]

for i in range(n):
    for j in range(n):
        if i ==j:
            Matrix_score[i][j] = 0
        Matrix_score[i][j] = float(input(f"For individual {i+1} in the experiment, enter the score for dimension {j+1}: "))

for k in range(n):
    PHQ_9[k] = float(input(f"For individual {k+1} in the experiment, enter the PHQ-9 score: "))


# Creating a Directed Network Graph
G = nx.DiGraph()

# Assuming interaction_scores_df's columns and rows both start from 0
for i in range(n):  # Rows
    G.add_node(i+1)  # Nodes are labeled starting from 1
    for j in range(n):  # Columns
        if i != j:  # Avoid self-loops, adjusting indices since nodes start from 1
            weight = Matrix_score[i][j]  # Accessing DataFrame with zero-based indices
            if not np.isnan(weight):  # Check if weight exists
                G.add_edge(i+1, j+1, weight=weight)  # Adjust indices to match node labels
                
                
# Setup the figure for 3D plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Generate positions in a circular layout
pos = nx.circular_layout(G, dim=2, scale=1)
pos_3d = {node: (pos[node][0], pos[node][1], PHQ_9[node-1]) for node in G.nodes()}  # Extend to 3D

# Normalize edge weights for color mapping
weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
norm = Normalize(vmin=weights.min(), vmax=weights.max())
cmap = plt.get_cmap('coolwarm')
sm = ScalarMappable(norm=norm, cmap=cmap)

# Draw nodes with scatter in 3D
for node, (x, y, z) in pos_3d.items():
    ax.scatter(x, y, z, color='skyblue', s=100, edgecolors='black', depthshade=False)
    ax.text(x, y, z, f'{node}', color='red')

# Draw edges with color gradient
for (u, v, d) in G.edges(data=True):
    x_vals = [pos_3d[u][0], pos_3d[v][0]]
    y_vals = [pos_3d[u][1], pos_3d[v][1]]
    z_vals = [pos_3d[u][2], pos_3d[v][2]]
    ax.plot(x_vals, y_vals, z_vals, color=sm.to_rgba(d['weight']), lw=2)

# Add color bar
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
cbar.set_label('Connection Strength')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PHQ-9 Score')
ax.set_title('3D Network Graph with PHQ-9 Score')

plt.show()


# def test_mock():
#   assert True
