import torch

def federated_avg(client_weights, client_sizes):
    """
    Weighted Federated Averaging
    """

    avg_weights = {}

    total_size = sum(client_sizes)

    for key in client_weights[0].keys():

        avg_weights[key] = torch.zeros_like(client_weights[0][key])

        for i, client in enumerate(client_weights):
            weight = client_sizes[i] / total_size
            avg_weights[key] += client[key] * weight

    return avg_weights