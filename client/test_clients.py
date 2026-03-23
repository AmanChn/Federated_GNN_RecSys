import pickle
from client_simulator import *

print("Loading dataset...")

with open("data/processed_data.pkl", "rb") as f:
    train_df, test_df, n_users, n_items, train_dict = pickle.load(f)

clients = create_clients(train_df, num_clients=20)

client_dicts = build_client_dicts(clients)

print("Number of clients:", len(clients))

for cid in clients:
    print(f"Client {cid} users:", len(client_dicts[cid]))