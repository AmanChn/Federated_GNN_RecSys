Federated Temporal-Aware Graph Neural Network with Weighted Aggregation for Explainable Recommendation Systems
FTA-WAGNN: Federated Temporal-Aware Weighted GNN Recommender

Example Recommendations with Explanation:

User 13069 recommended items: [ 92 184  91  96 117]
Item 92 → score 16.6711 (high similarity to user embedding)
Item 184 → score 15.3659 (high similarity to user embedding)
Item 91 → score 15.1741 (high similarity to user embedding)
Item 96 → score 15.1603 (high similarity to user embedding)
Item 117 → score 14.3910 (high similarity to user embedding)

Federated Results:
Precision@10: 0.1411
Recall@10: 0.1009
NDCG@10: 0.1686


Important Commands:
python utils/prepare_dataset.py
python train_lightgcn.py
python evaluate.py
python client/test_clients.py
python client/test_local_train.py
python server/test_fedavg.py
python server/federated_main.py


1. Why Changes Are Needed

Your current system is built specifically for:

MovieLens → userId, movieId, rating

But datasets like:

Ciao
Epinions
Douban

have:

Different formats
Different column names
Extra info (social links, trust, reviews)
📊 2. What Will Change (IMPORTANT)
🔴 1. Data Loader (MOST IMPORTANT)
File:
utils/data_loader.py
Current (MovieLens):
df = pd.read_csv("ratings.csv")
For new datasets:

You must:

✔ Read correct file format
✔ Rename columns → user, item
✔ Handle missing values
Example: Epinions

Dataset might look like:

user_id, item_id, rating, timestamp

👉 Modify:

df.rename(columns={
    "user_id": "user",
    "item_id": "item"
}, inplace=True)
🔴 2. Implicit Conversion

Your code:

convert_to_implicit()

👉 Works for most datasets, but you may adjust threshold:

rating >= 3 → positive
🔴 3. Filtering Function
filter_users_items()

👉 May need tuning:

min interactions = 10–20 depending on dataset
🔴 4. ID Encoding
encode_ids()

👉 This part works unchanged

🔴 5. Graph Builder
utils/graph_builder.py

👉 Works unchanged as long as:

user, item columns exist
🔴 6. Model (LightGCN)
models/lightgcn.py

👉 ✅ No change needed

🔴 7. Federated Code
client/
server/

👉 ✅ No change needed

🔥 3. Special Case: Social Datasets (IMPORTANT)

Datasets like:

Ciao
Epinions

also have:

User-user trust graph
🧠 You Have Two Options
✅ Option A (Simple — Recommended)

Ignore social part:

Use only user-item interactions

👉 Your current code works with small changes

🚀 Option B (Advanced — Research Boost)

Use social graph:

User-user edges + user-item graph

👉 Requires:

modifying graph_builder
multi-relational GNN
📊 4. Summary of Changes
Component	Change Needed
data_loader.py	✅ YES (major)
implicit conversion	⚠️ maybe
filtering	⚠️ tune
graph_builder.py	❌ no
LightGCN model	❌ no
training code	❌ no
federated code	❌ no
🧠 5. Best Strategy for YOU
Step 1 (Recommended)

👉 Try:

Epinions OR Douban

Using:

ONLY user-item data
Step 2 (Advanced — if time permits)

👉 Extend to:

Social-aware recommendation
🎯 6. Thesis-Level Idea (VERY STRONG)

You can extend your work as:

Federated Social-Aware GNN Recommendation System

👉 This is publishable-level upgrade

🚀 7. Example Workflow for New Dataset

1️⃣ Replace dataset file
2️⃣ Modify data_loader.py
3️⃣ Run:

python utils/prepare_dataset.py
python train_lightgcn.py
python evaluate.py
python server/federated_main.py
🏆 Final Answer

👉 Your system is:

✔ Reusable
✔ Modular
✔ Easily adaptable

👉 Only data loading + preprocessing changes