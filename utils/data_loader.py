import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_movielens32m(path="data/ml-32m/ratings.csv"):
    """
    Load MovieLens 32M dataset
    """

    ratings = pd.read_csv(path)

    ratings = ratings[['userId', 'movieId', 'rating']]

    return ratings


def convert_to_implicit(ratings, threshold=4):
    """
    Convert ratings to implicit interactions
    """

    ratings = ratings[ratings["rating"] >= threshold]

    ratings = ratings[['userId', 'movieId']]

    return ratings


def filter_users_items(df, min_user_interactions=20, min_item_interactions=20, max_users=50000):
    """
    Filter sparse users/items and limit dataset size
    """

    # remove sparse users/items
    user_counts = df['userId'].value_counts()
    item_counts = df['movieId'].value_counts()

    df = df[df['userId'].isin(user_counts[user_counts >= min_user_interactions].index)]
    df = df[df['movieId'].isin(item_counts[item_counts >= min_item_interactions].index)]

    # limit number of users (VERY IMPORTANT)
    unique_users = df['userId'].unique()

    if len(unique_users) > max_users:
        selected_users = np.random.choice(unique_users, max_users, replace=False)
        df = df[df['userId'].isin(selected_users)]

    return df


def encode_ids(df):
    """
    Map user and item IDs to continuous indices
    """

    user_map = {u: i for i, u in enumerate(df['userId'].unique())}
    item_map = {i: j for j, i in enumerate(df['movieId'].unique())}

    df['user'] = df['userId'].map(user_map)
    df['item'] = df['movieId'].map(item_map)

    n_users = len(user_map)
    n_items = len(item_map)

    return df[['user','item']], n_users, n_items


def split_train_test(df, test_ratio=0.2):

    df = df.sample(frac=1, random_state=42)

    test_size = int(len(df) * test_ratio)

    test_df = df.iloc[:test_size]
    train_df = df.iloc[test_size:]

    return train_df, test_df


def build_interaction_dict(df):

    interaction_dict = {}

    for row in df.itertuples():

        user = row.user
        item = row.item

        if user not in interaction_dict:
            interaction_dict[user] = []

        interaction_dict[user].append(item)

    return interaction_dict