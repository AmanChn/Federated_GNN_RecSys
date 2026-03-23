from data_loader import *
from graph_builder import *

print("Loading dataset...")

ratings = load_movielens32m()

ratings = convert_to_implicit(ratings)

ratings = filter_users_items(ratings)

ratings, n_users, n_items = encode_ids(ratings)

train_df, test_df = split_train_test(ratings)

print("Building adjacency matrix...")

adj = build_adj_matrix(train_df, n_users, n_items)

print("Adj shape:", adj.shape)

print("Normalizing adjacency matrix...")

norm_adj = normalize_adj_matrix(adj)

print("Converting to torch sparse tensor...")

norm_adj = convert_to_torch_sparse(norm_adj)

print("Graph construction complete!")