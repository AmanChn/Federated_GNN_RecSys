import torch
import numpy as np
import pickle
from tqdm import tqdm

from utils.data_loader import *
from utils.graph_builder import *
from models.lightgcn import LightGCN


# =====================
# Metrics
# =====================
def precision_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]) & set(ground_truth)) / k


def recall_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]) & set(ground_truth)) / len(ground_truth)


def ndcg_at_k(recommended, ground_truth, k):
    dcg = 0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)

    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])

    return dcg / idcg if idcg > 0 else 0


# =====================
# Load Processed Data
# =====================

print("Loading processed dataset...")

with open("data/processed_data.pkl", "rb") as f:
    train_df, test_df, n_users, n_items, train_dict = pickle.load(f)

test_dict = build_interaction_dict(test_df)


# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Build Graph
# =====================
adj = build_adj_matrix(train_df, n_users, n_items)
norm_adj = normalize_adj_matrix(adj)
norm_adj = convert_to_torch_sparse(norm_adj).to(device)


# =====================
# Load Model
# =====================
embedding_dim = 128
model = LightGCN(n_users, n_items, embedding_dim, norm_adj).to(device)
model.load_state_dict(torch.load("lightgcn.pth"))
model.eval()


# =====================
# Get embeddings
# =====================
with torch.no_grad():
    user_emb, item_emb = model.propagate()


# =====================
# Evaluation
# =====================
K = 10

precisions = []
recalls = []
ndcgs = []

for user in tqdm(test_dict.keys()):

    if user not in train_dict:
        continue

    ground_truth = test_dict[user]

    scores = torch.matmul(user_emb[user], item_emb.T)

    # remove seen items
    seen_items = train_dict[user]
    scores[seen_items] = -1e9

    _, recommended = torch.topk(scores, K)
    recommended = recommended.cpu().numpy()

    precisions.append(precision_at_k(recommended, ground_truth, K))
    recalls.append(recall_at_k(recommended, ground_truth, K))
    ndcgs.append(ndcg_at_k(recommended, ground_truth, K))


print("\nEvaluation Results:")
print(f"Precision@{K}: {np.mean(precisions):.4f}")
print(f"Recall@{K}: {np.mean(recalls):.4f}")
print(f"NDCG@{K}: {np.mean(ndcgs):.4f}")