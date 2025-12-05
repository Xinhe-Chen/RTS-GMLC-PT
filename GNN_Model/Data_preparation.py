import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


# define the root path of all the csv files 
data_path = os.path.join(os.getcwd(), '..', 'Data')

# read bus and branch data
bus = pd.read_csv(os.path.join(data_path, 'bus.csv'))
bus_ids = bus['Bus ID'].to_list()
bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}

num_nodes = len(bus_ids)

# read edges from the branch data
branch = pd.read_csv(os.path.join(data_path, 'branch.csv'))
edge_u = []
edge_v = []
edge_attr_list = []

for _, row in branch.iterrows():
    from_bus = row["From Bus"]
    to_bus   = row["To Bus"]
    u = bus_id_to_idx[from_bus]
    v = bus_id_to_idx[to_bus]
    edge_u += [u, v]  # undirected: add both directions
    edge_v += [v, u]

    # optional edge features (same in both directions)
    edge_attr_list += [[row["X"], row["STE Rating"]]] * 2  # example: reactance, limit

edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long)
edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)

# load by bus
df_bus_load = pd.read_csv(os.path.join(data_path, 'bus_load.csv'))

# renewable generation by bus (solar/wind/hydro)
df_bus_renewable = pd.read_csv(os.path.join(data_path, 'bus_renewable.csv'))

# read static features
df_static_features = pd.read_csv(os.path.join(data_path, 'bus_static.csv'))
X_static_np = df_static_features.to_numpy().T  # shape: (N, F_static)
X_static = torch.tensor(X_static_np, dtype=torch.float)  # shape: (N, F_static)

# build dynamic feature
ts_features_list = [
    df_bus_load.to_numpy(),  # shape: (T, N)
    df_bus_renewable.to_numpy(),  # shape: (T, N)
]
X_dyn_np = np.stack(ts_features_list, axis=-1)  # shape: (T, N, F)
X_dyn = torch.tensor(X_dyn_np, dtype=torch.float)   # shape: (T, N, F_dyn)

# Stack into [T, N, F]
T, N, F_dyn = X_dyn.shape
N2, F_static = X_static.shape
assert N == N2, "Mismatch in number of nodes between dynamic and static features"

# [N, F_static] → [1, N, F_static] → [T, N, F_static]
X_static_expanded = X_static.unsqueeze(0).expand(T, -1, -1)

# Concatenate
X_all = torch.cat([X_dyn, X_static_expanded], dim=-1)  # [T, N, F_dyn + F_static]

# print(X_all.shape)  # shape: (T, N, F_dyn + F_static)

# standardize features
# do not do train/test split here for debugging convenience
mu = X_all.mean(axis=(0, 1), keepdims=True)
sigma = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
X_norm = (X_all - mu) / sigma

# Extract LMP labels
y_np = pd.read_csv(os.path.join(data_path, 'lmp_by_bus.csv')).to_numpy()  # shape: (T, N)

# Build PyTorch Geometric Data objects
T, N, F = X_norm.shape

graphs = []
for t in range(T):
    x_t = X_norm[t]               # [N, F]
    y_t = torch.tensor(y_np[t], dtype=torch.float).unsqueeze(-1)  # [N, 1]
    data = Data(
        x=x_t,
        edge_index=edge_index,
        edge_attr=edge_attr,      # optional
        y=y_t,                    # node-level regression target
    )
    graphs.append(data)

print(graphs[0])  # print the first graph for verification