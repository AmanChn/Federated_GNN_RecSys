import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from ui.data_utils import *
from ui.recommendation_engine import RecommendationEngine


# =====================
# Page Config
# =====================

st.set_page_config(
    page_title="Federated GNN Recommender",
    layout="wide"
)

st.title("🎬 Federated LightGCN Recommendation System")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Overview",
        "Recommendations",
        "Training Analytics"
    ]
)

# =====================
# Dataset Statistics
# =====================
if page == "Overview":

    st.info(
    """
    Federated LightGCN Recommendation System

    • Dataset: MovieLens 32M (filtered)

    • Users: 50,000

    • Items: 20,033

    • Federated Clients: 20

    • Precision@10: 0.1411

    • Recall@10: 0.1009

    • NDCG@10: 0.1686
    """
    )

    comparison_df = pd.DataFrame({
        "Model": [
            "Collaborative Filtering",
            "Centralized LightGCN",
            "Federated LightGCN"
        ],
        "Precision@10": [
            0.1243,
            0.1654,
            0.1411
        ],
        "Recall@10": [
            0.0865,
            0.1200,
            0.1009
        ],
        "NDCG@10": [
            0.1501,
            0.1999,
            0.1686
        ]
    })

    st.subheader("Model Comparison")

    st.table(comparison_df)

    st.success(
        "Centralized LightGCN achieved the highest recommendation accuracy "
        "(P@10 = 0.1654, NDCG@10 = 0.1999), while Federated LightGCN "
        "maintained privacy-preserving training with competitive performance "
        "(P@10 = 0.1411, NDCG@10 = 0.1686)."
    )

    st.header("📊 Dataset Statistics")

    stats = get_dataset_stats()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Users", f"{stats['Users']:,}")
    col2.metric("Items", f"{stats['Items']:,}")
    col3.metric("Train Interactions", f"{stats['Train Interactions']:,}")
    col4.metric("Test Interactions", f"{stats['Test Interactions']:,}")


    # =====================
    # Model Performance
    # =====================
    st.header("📈 Model Performance")

    metrics_df = pd.DataFrame({
        "Metric": ["Precision@10", "Recall@10", "NDCG@10"],
        "Score": [0.1411, 0.1009, 0.1686]
    })

    st.subheader("Federated Model Metrics")

    st.table(metrics_df)


# =====================
# Recommendation Engine
# =====================

elif page == "Recommendations":

    st.header("👤 Recommendation Demo")

    @st.cache_resource
    def load_engine():
        return RecommendationEngine()

    engine = load_engine()

    available_users = sorted(engine.train_dict.keys())

    if "selected_user" not in st.session_state:
        st.session_state.selected_user = available_users[100]

    col1, col2 = st.columns([4,1])

    with col1:
        user_id = st.selectbox(
            "Select User",
            available_users,
            index=available_users.index(st.session_state.selected_user)
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("🎲 Random"):
            st.session_state.selected_user = random.choice(available_users)
            st.rerun()

    st.session_state.selected_user = user_id

    if st.button("Generate Recommendations"):

        st.subheader("Previously Watched Movies")

        watched = engine.get_watched_movies(user_id)

        if len(watched) == 0:
            st.warning("No interactions found for this user.")
        else:
            for movie in watched:
                st.write("•", movie)

        st.subheader("Top-10 Recommendations")

        recommendations = engine.recommend(user_id)

        for idx, movie in enumerate(recommendations, start=1):
            st.write(f"{idx}. {movie}")


# =====================
# Loss Curves
# =====================
elif page == "Training Analytics":

    st.header("📉 Training Analytics")

    col1, col2 = st.columns(2)


    def read_losses(filename):

        if not os.path.exists(filename):
            return []

        losses = []

        with open(filename, "r") as f:

            for line in f:

                value = float(
                    line.strip().split(":")[1]
                )

                losses.append(value)

        return losses


    centralized_losses = read_losses(
        "centralized_losses.txt"
    )

    federated_losses = read_losses(
        "federated_losses.txt"
    )


    with col1:

        st.subheader("Centralized Training Loss")

        if len(centralized_losses) > 0:

            fig, ax = plt.subplots()

            ax.plot(
                range(1, len(centralized_losses)+1),
                centralized_losses
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            st.pyplot(fig)

        else:
            st.info("centralized_losses.txt not found")


    with col2:

        st.subheader("Federated Training Loss")

        if len(federated_losses) > 0:

            fig, ax = plt.subplots()

            ax.plot(
                range(1, len(federated_losses)+1),
                federated_losses
            )

            ax.set_xlabel("Round")
            ax.set_ylabel("Loss")

            st.pyplot(fig)

        else:
            st.info("federated_losses.txt not found")