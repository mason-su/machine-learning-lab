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
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer


# ============================================================
# 1. Load data
# ============================================================

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce training data intentionally to reproduce overfitting
x_train = x_train[:300]
t_train = t_train[:300]


# ============================================================
# 2. Experiment settings
# ============================================================

dropout_ratio = 0.2

epochs = 1001
mini_batch_size = 100
learning_rate = 0.01

SEED = 1234


# ============================================================
# 3. Training function
# ============================================================

def run_experiment(use_dropout):

    # Use the same random seed so the two networks start
    # from the same initialization as much as possible
    np.random.seed(SEED)

    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100, 100],
        output_size=10,
        use_dropout=use_dropout,
        dropout_ration=dropout_ratio
    )

    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        optimizer='sgd',
        optimizer_param={'lr': learning_rate},
        verbose=True
    )

    trainer.train()

    return trainer.train_acc_list, trainer.test_acc_list


# ============================================================
# 4. Without Dropout
# ============================================================

print("\n========================================")
print("Training WITHOUT Dropout")
print("========================================")

train_acc_without_dropout, test_acc_without_dropout = run_experiment(
    use_dropout=False
)


# ============================================================
# 5. With Dropout
# ============================================================

print("\n========================================")
print("Training WITH Dropout")
print("========================================")

train_acc_with_dropout, test_acc_with_dropout = run_experiment(
    use_dropout=True
)


# ============================================================
# 6. Print final results
# ============================================================

print("\n========================================")
print("Final Results")
print("========================================")

print(
    f"Without Dropout:"
    f" Train = {train_acc_without_dropout[-1]:.4f},"
    f" Test = {test_acc_without_dropout[-1]:.4f}"
)

print(
    f"With Dropout:"
    f" Train = {train_acc_with_dropout[-1]:.4f},"
    f" Test = {test_acc_with_dropout[-1]:.4f}"
)


# ============================================================
# 7. Plot comparison
# ============================================================

x_without_dropout = np.arange(len(train_acc_without_dropout))
x_with_dropout = np.arange(len(train_acc_with_dropout))

plt.figure(figsize=(10, 6))

plt.plot(
    x_without_dropout,
    train_acc_without_dropout,
    label='Train - Without Dropout'
)

plt.plot(
    x_without_dropout,
    test_acc_without_dropout,
    label='Test - Without Dropout'
)

plt.plot(
    x_with_dropout,
    train_acc_with_dropout,
    linestyle='--',
    label='Train - With Dropout'
)

plt.plot(
    x_with_dropout,
    test_acc_with_dropout,
    linestyle='--',
    label='Test - With Dropout'
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.title("Dropout vs No Dropout")
plt.legend()
plt.grid()

plt.show()