# coding: utf-8
import os
import sys

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


# ============================================================
# 1. Load MNIST
# ============================================================

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce training data intentionally to reproduce overfitting
x_train = x_train[:300]
t_train = t_train[:300]


# ============================================================
# 2. Experiment settings
# ============================================================

# NEW VARIABLE:
# Different weight decay strengths to compare
weight_decay_lambda_list = [
    0.0,
    0.001,
    0.01,
    0.05,
    0.1
]

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

SEED = 1234

iter_per_epoch = max(train_size // batch_size, 1)


# NEW VARIABLE:
# Stores results from every weight_decay_lambda
results = {}


# ============================================================
# 3. Run experiments
# ============================================================

for weight_decay_lambda in weight_decay_lambda_list:

    print("\n")
    print("====================================================")
    print("weight_decay_lambda =", weight_decay_lambda)
    print("====================================================")

    # Reset random seed so each experiment starts
    # from the same initialization and batch sequence
    np.random.seed(SEED)

    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100, 100],
        output_size=10,
        weight_decay_lambda=weight_decay_lambda
    )

    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    test_acc_list = []

    epoch_cnt = 0
    iteration = 0

    while epoch_cnt < max_epochs:

        # --------------------------------------------
        # Mini-batch sampling
        # --------------------------------------------

        batch_mask = np.random.choice(
            train_size,
            batch_size
        )

        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]


        # --------------------------------------------
        # Gradient + SGD update
        # --------------------------------------------

        grads = network.gradient(
            x_batch,
            t_batch
        )

        optimizer.update(
            network.params,
            grads
        )


        # --------------------------------------------
        # Evaluate once per epoch
        # --------------------------------------------

        if iteration % iter_per_epoch == 0:

            train_acc = network.accuracy(
                x_train,
                t_train
            )

            test_acc = network.accuracy(
                x_test,
                t_test
            )

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print(
                "epoch:",
                epoch_cnt,
                "train acc:",
                round(train_acc, 4),
                "test acc:",
                round(test_acc, 4)
            )

            epoch_cnt += 1

        iteration += 1


    # --------------------------------------------
    # Save results
    # --------------------------------------------

    results[weight_decay_lambda] = {
        "train_acc_list": train_acc_list,
        "test_acc_list": test_acc_list
    }


# ============================================================
# 4. Print summary
# ============================================================

print("\n")
print("====================================================")
print("SUMMARY")
print("====================================================")

for weight_decay_lambda in weight_decay_lambda_list:

    train_acc_list = results[weight_decay_lambda]["train_acc_list"]
    test_acc_list = results[weight_decay_lambda]["test_acc_list"]

    final_train_acc = train_acc_list[-1]
    final_test_acc = test_acc_list[-1]

    best_test_acc = max(test_acc_list)
    best_test_epoch = np.argmax(test_acc_list)

    print(
        f"lambda = {weight_decay_lambda:<6}"
        f" | final train = {final_train_acc:.4f}"
        f" | final test = {final_test_acc:.4f}"
        f" | best test = {best_test_acc:.4f}"
        f" | best epoch = {best_test_epoch}"
    )


# ============================================================
# 5. Plot TEST accuracy
# ============================================================

plt.figure(figsize=(10, 6))

for weight_decay_lambda in weight_decay_lambda_list:

    test_acc_list = results[weight_decay_lambda]["test_acc_list"]

    x = np.arange(len(test_acc_list))

    plt.plot(
        x,
        test_acc_list,
        label=f"lambda = {weight_decay_lambda}"
    )

plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Weight Decay - Test Accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.grid()

plt.show()


# ============================================================
# 6. Plot TRAIN accuracy
# ============================================================

plt.figure(figsize=(10, 6))

for weight_decay_lambda in weight_decay_lambda_list:

    train_acc_list = results[weight_decay_lambda]["train_acc_list"]

    x = np.arange(len(train_acc_list))

    plt.plot(
        x,
        train_acc_list,
        label=f"lambda = {weight_decay_lambda}"
    )

plt.xlabel("Epochs")
plt.ylabel("Train Accuracy")
plt.title("Weight Decay - Train Accuracy")
plt.ylim(0, 1.0)
plt.legend()
plt.grid()

plt.show()