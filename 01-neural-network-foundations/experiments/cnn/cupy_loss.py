# coding: utf-8
"""
Train MyConvNet with CuPy and plot accuracy after every epoch.

Place this file in the same directory as:
    cupy_myconvnet.py

Outputs:
    epoch_accuracy.png
    epoch_accuracy.csv
"""
import time
import csv
import sys
import os

import cupy as cp
import matplotlib.pyplot as plt

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.mnist import load_mnist
from cupy_myconvnet import MyConvNet, Adam


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnn" / "cupy_epoch_accuracy"
EPOCHS = 20
BATCH_SIZE = 100
LEARNING_RATE = 0.001


def evaluate_accuracy(network, x, t, batch_size=100):
    """Evaluate accuracy with dropout disabled."""
    return network.accuracy(x, t, batch_size=batch_size)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("CuPy version:", cp.__version__)

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print("GPU:", gpu_name)

    # Load MNIST on CPU
    (x_train_np, t_train_np), (x_test_np, t_test_np) = load_mnist(
        normalize=True,
        flatten=False,
        one_hot_label=True,
    )

    # Copy dataset once to GPU
    x_train = cp.asarray(x_train_np, dtype=cp.float32)
    t_train = cp.asarray(t_train_np, dtype=cp.float32)
    x_test = cp.asarray(x_test_np, dtype=cp.float32)
    t_test = cp.asarray(t_test_np, dtype=cp.float32)

    print("x_train:", x_train.shape)
    print("x_test :", x_test.shape)

    network = MyConvNet()
    optimizer = Adam(lr=LEARNING_RATE)

    train_size = x_train.shape[0]
    iterations_per_epoch = train_size // BATCH_SIZE

    train_accuracy_history = []
    test_accuracy_history = []
    loss_history = []

    cp.cuda.Stream.null.synchronize()
    total_start = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0

        for _ in range(iterations_per_epoch):
            batch_mask = cp.random.choice(train_size, BATCH_SIZE)

            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            # Keep this extra forward pass so we can record an epoch loss curve.
            loss = network.loss(x_batch, t_batch)
            epoch_loss += float(loss.item())

        average_loss = epoch_loss / iterations_per_epoch

        # Accuracy measured at the END of this epoch
        train_accuracy = evaluate_accuracy(
            network, x_train, t_train, batch_size=BATCH_SIZE
        )
        test_accuracy = evaluate_accuracy(
            network, x_test, t_test, batch_size=BATCH_SIZE
        )

        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)
        loss_history.append(average_loss)

        cp.cuda.Stream.null.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f} | "
            f"Time: {epoch_time:.2f} s"
        )

    cp.cuda.Stream.null.synchronize()
    total_time = time.perf_counter() - total_start

    print()
    print(f"Total time: {total_time:.2f} s")
    print(f"Total time: {total_time / 60:.2f} min")

    # Save numerical results
    with open(OUTPUT_DIR / "epoch_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_accuracy", "test_accuracy"])

        for epoch in range(EPOCHS):
            writer.writerow(
                [
                    epoch + 1,
                    loss_history[epoch],
                    train_accuracy_history[epoch],
                    test_accuracy_history[epoch],
                ]
            )

    # Plot accuracy curve
    epochs = list(range(1, EPOCHS + 1))

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        train_accuracy_history,
        marker="o",
        label="Train Accuracy",
    )
    plt.plot(
        epochs,
        test_accuracy_history,
        marker="o",
        label="Test Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MyConvNet Accuracy vs Epoch")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "epoch_accuracy.png", dpi=200)
    plt.show()

    print("Saved: epoch_accuracy.png")
    print("Saved: epoch_accuracy.csv")


if __name__ == "__main__":
    main()
