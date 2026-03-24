import torch


def federated_avg(client_weights):
    """
    Perform Federated Averaging
    """

    avg_weights = {}

    for key in client_weights[0].keys():

        avg_weights[key] = torch.zeros_like(client_weights[0][key])

        for client in client_weights:
            avg_weights[key] += client[key]

        avg_weights[key] = avg_weights[key] / len(client_weights)

    return avg_weights