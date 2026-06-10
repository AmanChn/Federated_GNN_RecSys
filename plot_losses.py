import matplotlib.pyplot as plt


def read_losses(filename):
    losses = []

    with open(filename, "r") as f:
        for line in f:
            value = float(line.strip().split(":")[1])
            losses.append(value)

    return losses


centralized_losses = read_losses("centralized_losses.txt")
federated_losses = read_losses("federated_losses.txt")


# Plot Centralized Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(centralized_losses) + 1), centralized_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Centralized LightGCN Training Loss")
plt.grid(True)
plt.savefig("centralized_loss_plot.png")
plt.show()


# Plot Federated Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(federated_losses) + 1), federated_losses)

plt.xlabel("Federated Round")
plt.ylabel("Loss")
plt.title("Federated LightGCN Training Loss")
plt.grid(True)
plt.savefig("federated_loss_plot.png")
plt.show()