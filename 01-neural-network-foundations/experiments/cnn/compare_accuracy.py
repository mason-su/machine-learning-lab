# coding: utf-8
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from awesome_net import MyConvNet


# Load MNIST test data
(_, _), (x_test, t_test) = load_mnist(flatten=False)


# =========================
# Original DeepConvNet
# =========================
deep_network = DeepConvNet()
deep_network.load_params(str(PROJECT_ROOT / "results" / "checkpoints" / "deep_convnet_params.pkl"))

print("Calculating DeepConvNet accuracy...")
deep_accuracy = deep_network.accuracy(
    x_test,
    t_test,
    batch_size=100
)


# =========================
# MyConvNet
# =========================
my_network = MyConvNet()
my_network.load_params(str(PROJECT_ROOT / "results" / "checkpoints" / "my_convnet_params.pkl"))

print("Calculating MyConvNet accuracy...")
my_accuracy = my_network.accuracy(
    x_test,
    t_test,
    batch_size=100
)


# =========================
# Compare
# =========================
print()
print("========== Accuracy Comparison ==========")

print(
    "DeepConvNet accuracy:",
    deep_accuracy,
    "(" + str(deep_accuracy * 100) + "%)"
)

print(
    "MyConvNet accuracy:",
    my_accuracy,
    "(" + str(my_accuracy * 100) + "%)"
)

print()

if my_accuracy > deep_accuracy:
    print("MyConvNet has higher accuracy.")
    print(
        "Difference:",
        (my_accuracy - deep_accuracy) * 100,
        "percentage points"
    )

elif deep_accuracy > my_accuracy:
    print("DeepConvNet has higher accuracy.")
    print(
        "Difference:",
        (deep_accuracy - my_accuracy) * 100,
        "percentage points"
    )

else:
    print("Both networks have the same accuracy.")