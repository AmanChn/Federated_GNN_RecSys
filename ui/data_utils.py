import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.data_utils import *

import pandas as pd
import pickle


def load_data():

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

    return train_df, test_df, n_users, n_items, train_dict


def load_movies():

    movies = pd.read_csv("data/ml-32m/movies.csv")

    return movies


def get_dataset_stats():

    train_df, test_df, n_users, n_items, train_dict = load_data()

    stats = {
        "Users": n_users,
        "Items": n_items,
        "Train Interactions": len(train_df),
        "Test Interactions": len(test_df)
    }

    return stats