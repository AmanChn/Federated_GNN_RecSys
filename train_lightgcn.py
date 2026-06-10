import torch
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm

from utils.data_loader import *
from utils.graph_builder import *
from models.lightgcn import LightGCN
from utils.loss import bpr_loss



# Load Data
print("Loading processed dataset...")

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Build Graph
print("Building graph...")

adj = build_adj_matrix(train_df, n_users, n_items)
norm_adj = normalize_adj_matrix(adj)
norm_adj = convert_to_torch_sparse(norm_adj).to(device)


# Model
embedding_dim = 128
model = LightGCN(n_users, n_items, embedding_dim, norm_adj).to(device)

# Slightly stable LR for better convergence
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Training Batch Sampling
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
            if tries > 5:
                break

        pos_items.append(pos)
        neg_items.append(neg)

    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.LongTensor(neg_items)
    )


# Training
print("Training started...")

epochs = 15
batches_per_epoch = 200

epoch_losses = []

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for _ in tqdm(range(batches_per_epoch)):

        users, pos_items, neg_items = sample_batch(
            train_dict,
            n_items,
            batch_size=1024
        )

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        pos_scores, neg_scores = model(users, pos_items, neg_items)

    
        # BPR Loss
        ranking_loss = bpr_loss(pos_scores, neg_scores)


        # L2 Regularization Loss
        reg_loss = (
            model.user_embedding.weight.norm(2).pow(2) +
            model.item_embedding.weight.norm(2).pow(2)
        ) / 2

        # Final Loss
        loss = ranking_loss + 1e-4 * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / batches_per_epoch
    epoch_losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# Save Model
torch.save(model.state_dict(), "lightgcn.pth")

print("Training complete!")


# Save Loss Values
with open("centralized_losses.txt", "w") as f:
    for i, loss in enumerate(epoch_losses):
        f.write(f"Epoch {i+1}: {loss:.4f}\n")


print("Loss values saved to centralized_losses.txt")