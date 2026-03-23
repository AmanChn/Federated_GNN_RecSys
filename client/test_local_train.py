import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import torch

from client_simulator import *
from client_train import *
from utils.graph_builder import *
from models.lightgcn import LightGCN


print("Loading dataset...")

with open("data/processed_data.pkl", "rb") as f:
    train_df, test_df, n_users, n_items, train_dict = pickle.load(f)


print("Creating clients...")
clients = create_clients(train_df, num_clients=5)
client_dicts = build_client_dicts(clients)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Building graph...")
adj = build_adj_matrix(train_df, n_users, n_items)
norm_adj = normalize_adj_matrix(adj)
norm_adj = convert_to_torch_sparse(norm_adj).to(device)


print("Initializing global model...")
global_model = LightGCN(n_users, n_items, 128, norm_adj).to(device)


print("Training one client...")

client_id = 0
client_dict = client_dicts[client_id]

updated_weights = train_client(
    global_model,
    client_dict,
    n_users,
    n_items,
    norm_adj,
    device
)

print("Client training complete!")