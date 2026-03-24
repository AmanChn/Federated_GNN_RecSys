import torch
import torch.optim as optim
import numpy as np

from models.lightgcn import LightGCN
from utils.loss import bpr_loss


def sample_batch(train_dict, n_items, batch_size=512):
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


# def train_client(global_model, client_dict, n_users, n_items, norm_adj, device,
                #  local_epochs=1, batch_size=512):
def train_client(global_model, client_dict, n_users, n_items, norm_adj, device,
                 local_epochs=3, batch_size=512):

    # Create local model
    local_model = LightGCN(n_users, n_items, 128, norm_adj).to(device)

    # Load global weights
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(local_model.parameters(), lr=0.0005)

    local_model.train()

    for epoch in range(local_epochs):

        total_loss = 0

        for _ in range(50):  # small local updates

            users, pos_items, neg_items = sample_batch(client_dict, n_items, batch_size)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_scores, neg_scores = local_model(users, pos_items, neg_items)

            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return local_model.state_dict()