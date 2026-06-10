import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import torch
from tqdm import tqdm

from client.client_simulator import *
from client.client_train import *
from utils.graph_builder import *
from utils.data_loader import build_interaction_dict
from models.lightgcn import LightGCN
from server.fedavg import federated_avg


# Load Dataset
print("Loading dataset...")

with open("data/processed_data.pkl", "rb") as f:
    (
        train_df,
        test_df,
        n_users,
        n_items,
        train_dict,
        user_map,
        item_map
    ) = pickle.load(f)

test_dict = build_interaction_dict(test_df)


# Create Clients
print("Creating clients...")

clients = create_clients(train_df, num_clients=20)
client_dicts = build_client_dicts(clients)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Build Graph
print("Building graph...")

adj = build_adj_matrix(train_df, n_users, n_items)
norm_adj = normalize_adj_matrix(adj)
norm_adj = convert_to_torch_sparse(norm_adj).to(device)


# Initialize Global Model
embedding_dim = 128
global_model = LightGCN(
    n_users,
    n_items,
    embedding_dim,
    norm_adj
).to(device)


# Federated Training
rounds = 10
federated_losses = []

for r in range(rounds):

    print(f"\n--- Federated Round {r+1} ---")

    client_weights = []
    client_sizes = []
    round_loss = 0

    for cid in tqdm(client_dicts):

        client_dict = client_dicts[cid]

        # train_client now returns:
        # weights + avg_local_loss
        weights, local_loss = train_client(
            global_model,
            client_dict,
            n_users,
            n_items,
            norm_adj,
            device
        )

        client_weights.append(weights)

        # Weighted FedAvg using number of interactions
        client_sizes.append(
            sum(len(v) for v in client_dict.values())
        )

        round_loss += local_loss

    avg_round_loss = round_loss / len(client_dicts)
    federated_losses.append(avg_round_loss)

    print(f"Round {r+1} Loss: {avg_round_loss:.4f}")

    # Weighted Federated Averaging
    new_global_weights = federated_avg(
        client_weights,
        client_sizes
    )

    global_model.load_state_dict(new_global_weights)


# Save Federated Loss Values
with open("federated_losses.txt", "w") as f:
    for i, loss in enumerate(federated_losses):
        f.write(f"Round {i+1}: {loss:.4f}\n")

print("Loss values saved to federated_losses.txt")


# Evaluation
print("\nEvaluating Federated Model...")

global_model.eval()

with torch.no_grad():
    user_emb, item_emb = global_model.propagate()


K = 10
precisions = []
recalls = []
ndcgs = []

for user in tqdm(test_dict.keys()):

    if user not in train_dict:
        continue

    ground_truth = test_dict[user]

    scores = torch.matmul(
        user_emb[user],
        item_emb.T
    )

    # Remove already seen items
    seen_items = train_dict[user]
    scores[seen_items] = -1e9

    _, recommended = torch.topk(scores, K)
    recommended = recommended.cpu().numpy()

    precision = len(
        set(recommended) & set(ground_truth)
    ) / K

    recall = len(
        set(recommended) & set(ground_truth)
    ) / len(ground_truth)

    # =====================
    # NDCG Calculation
    # =====================
    dcg = 0
    for i, item in enumerate(recommended):
        if item in ground_truth:
            dcg += 1 / torch.log2(
                torch.tensor(i + 2.0)
            )

    idcg = sum([
        1 / torch.log2(
            torch.tensor(i + 2.0)
        )
        for i in range(
            min(len(ground_truth), K)
        )
    ])

    ndcg = dcg / idcg if idcg > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    ndcgs.append(ndcg.item())


# =====================
# Final Results
# =====================
print("\nFederated Results:")

print(
    f"Precision@10: "
    f"{sum(precisions)/len(precisions):.4f}"
)

print(
    f"Recall@10: "
    f"{sum(recalls)/len(recalls):.4f}"
)

print(
    f"NDCG@10: "
    f"{sum(ndcgs)/len(ndcgs):.4f}"
)