import numpy as np


def create_clients(train_df, num_clients=20):
    """
    Split users into multiple clients
    """

    users = train_df['user'].unique()
    np.random.shuffle(users)

    client_size = len(users) // num_clients

    clients = {}

    for i in range(num_clients):

        start = i * client_size
        end = (i + 1) * client_size if i != num_clients - 1 else len(users)

        client_users = users[start:end]

        client_data = train_df[train_df['user'].isin(client_users)]

        clients[i] = client_data

    return clients


def build_client_dicts(clients):
    """
    Build interaction dict for each client
    """

    client_dicts = {}

    for client_id, df in clients.items():

        interaction_dict = {}

        for row in df.itertuples():

            user = row.user
            item = row.item

            if user not in interaction_dict:
                interaction_dict[user] = []

            interaction_dict[user].append(item)

        client_dicts[client_id] = interaction_dict

    return client_dicts