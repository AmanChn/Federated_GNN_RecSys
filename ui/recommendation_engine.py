import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
import torch

from models.lightgcn import LightGCN
from utils.graph_builder import *


class RecommendationEngine:

    def __init__(self):

        print("Loading recommendation engine...")

        # =====================
        # Load processed data
        # =====================
        with open("data/processed_data.pkl", "rb") as f:
            (
                self.train_df,
                self.test_df,
                self.n_users,
                self.n_items,
                self.train_dict,
                self.user_map,
                self.item_map
            ) = pickle.load(f)

        # =====================
        # Load movies
        # =====================
        self.movies = pd.read_csv("data/ml-32m/movies.csv")

        # =====================
        # Rebuild mappings
        # =====================
        self.item_to_movieid = {
            encoded_id: original_movie_id
            for original_movie_id, encoded_id in self.item_map.items()
        }

        # =====================
        # Device
        # =====================
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # =====================
        # Graph
        # =====================
        adj = build_adj_matrix(
            self.train_df,
            self.n_users,
            self.n_items
        )

        norm_adj = normalize_adj_matrix(adj)

        self.norm_adj = convert_to_torch_sparse(
            norm_adj
        ).to(self.device)

        # =====================
        # Load Model
        # =====================
        self.model = LightGCN(
            self.n_users,
            self.n_items,
            128,
            self.norm_adj
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                "lightgcn.pth",
                map_location=self.device
            )
        )

        self.model.eval()

        with torch.no_grad():
            self.user_emb, self.item_emb = self.model.propagate()

    # =====================================
    # Watched Movies
    # =====================================
    def get_watched_movies(self, user_id):

        if user_id not in self.train_dict:
            return []

        watched_items = self.train_dict[user_id]

        watched_movies = []

        for item in watched_items[:10]:

            if item not in self.item_to_movieid:
                continue

            movie_id = self.item_to_movieid[item]

            row = self.movies[
                self.movies["movieId"] == movie_id
            ]

            if len(row) > 0:
                watched_movies.append(
                    row.iloc[0]["title"]
                )

        return watched_movies

    # =====================================
    # Recommendations
    # =====================================
    def recommend(self, user_id, top_k=10):

        if user_id not in self.train_dict:
            return []

        scores = torch.matmul(
            self.user_emb[user_id],
            self.item_emb.T
        )

        seen_items = self.train_dict[user_id]

        scores[seen_items] = -1e9

        _, recommended = torch.topk(
            scores,
            top_k
        )

        recommended = recommended.cpu().numpy()

        recommendations = []

        for item in recommended:

            if item not in self.item_to_movieid:
                continue

            movie_id = self.item_to_movieid[item]

            row = self.movies[
                self.movies["movieId"] == movie_id
            ]

            if len(row) > 0:
                recommendations.append(
                    row.iloc[0]["title"]
                )

        return recommendations