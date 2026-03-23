import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, norm_adj):
        super(LightGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.norm_adj = norm_adj

        # embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self):
        """
        LightGCN propagation
        """

        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])

        embeddings_list = [all_embeddings]

        for _ in range(3):  # number of layers
        # for _ in range(2):  # number of layers
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)

        # average embeddings
        final_embedding = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)

        users, items = torch.split(
            final_embedding,
            [self.n_users, self.n_items]
        )

        return users, items

    def forward(self, users, pos_items, neg_items):
        """
        Compute BPR loss inputs
        """

        user_emb, item_emb = self.propagate()

        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        return pos_scores, neg_scores