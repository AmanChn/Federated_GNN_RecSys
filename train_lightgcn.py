import torch
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm

from utils.data_loader import *
from utils.graph_builder import *
from models.lightgcn import LightGCN
from utils.loss import bpr_loss


# =====================
# Load Data
# =====================
print("Loading processed dataset...")

with open("data/processed_data.pkl", "rb") as f:
    train_df, test_df, n_users, n_items, train_dict = pickle.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# Build Graph
# =====================
print("Building graph...")

adj = build_adj_matrix(train_df, n_users, n_items)
norm_adj = normalize_adj_matrix(adj)
# norm_adj = convert_to_torch_sparse(norm_adj)
norm_adj = convert_to_torch_sparse(norm_adj).to(device)


# =====================
# Model
# =====================
embedding_dim = 128
model = LightGCN(n_users, n_items, embedding_dim, norm_adj).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)

# =====================
# Training
# =====================
def sample_batch(train_dict, n_items, batch_size=1024):

    users = np.random.choice(list(train_dict.keys()), batch_size)

    pos_items = []
    neg_items = []

    for u in users:

        pos = np.random.choice(train_dict[u])

        neg = np.random.randint(0, n_items)

        tries = 0
        while neg in train_dict[u]:
            neg = np.random.randint(0, n_items)
            tries += 1
            if tries > 5:   # IMPORTANT: reduce hardness
                break

        pos_items.append(pos)
        neg_items.append(neg)

    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.LongTensor(neg_items)
    )


print("Training started...")

epochs = 15
# epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for _ in tqdm(range(200)):  # batches per epoch
    # for _ in tqdm(range(200)):  # batches per epoch

        users, pos_items, neg_items = sample_batch(train_dict, n_items)

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        pos_scores, neg_scores = model(users, pos_items, neg_items)

        loss = bpr_loss(pos_scores, neg_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


print("Training complete!")

torch.save(model.state_dict(), "lightgcn.pth")