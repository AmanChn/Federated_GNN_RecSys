import pickle
from data_loader import *

print("Loading dataset...")

ratings = load_movielens32m()

print("Original size:", len(ratings))

ratings = convert_to_implicit(ratings)

print("After implicit conversion:", len(ratings))

ratings = filter_users_items(ratings)

print("After filtering:", len(ratings))

ratings, n_users, n_items = encode_ids(ratings)

print("Users:", n_users)
print("Items:", n_items)

train_df, test_df = split_train_test(ratings)

print("Train interactions:", len(train_df))
print("Test interactions:", len(test_df))

train_dict = build_interaction_dict(train_df)

print("Dataset preparation complete.")


with open("data/processed_data.pkl", "wb") as f:
    pickle.dump((train_df, test_df, n_users, n_items, train_dict), f)

print("Saved processed dataset!")          