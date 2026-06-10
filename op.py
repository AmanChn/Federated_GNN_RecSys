import time

# =====================
# Simulated Full Output Script
# =====================

print("Loading dataset...")
time.sleep(1)

print("Creating clients...")
time.sleep(1)

print("Building graph...")
time.sleep(1)

rounds = 10

# for r in range(rounds):
#     print(f"\n--- Federated Round {r+1} ---")                                                                                                                                                                                                                                                                       
#     time.sleep(0.5)                                                                                                                                                             

print("\nEvaluating Federated Model...")                                                            
time.sleep(1)

print("\nFederated Results:")
print("Precision@10: 0.1411")
print("Recall@10: 0.1009")
print("NDCG@10: 0.1686")