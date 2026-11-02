# coding: utf-8

import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import time
import csv

from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet


# ============================================================
# 1. 读取 MNIST
# ============================================================

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,
    one_hot_label=True
)

train_size = x_train.shape[0]


# ============================================================
# 2. 实验参数
# ============================================================

# ------------------------------------------------------------
# 新变量：
# hidden_layer_count = 隐藏层数量
# ------------------------------------------------------------

hidden_layer_counts = [
    1,
    2,
    5,
    10
]


# ------------------------------------------------------------
# hidden_size 的含义保持不变：
# 每一个隐藏层中的神经元数量
# ------------------------------------------------------------

hidden_sizes = [
    1,
    5,
    20,
    40,
    50,
    100
]


# ------------------------------------------------------------
# Learning Rate 减少
# ------------------------------------------------------------

learning_rates = [
    0.05,
    0.1
]


# ------------------------------------------------------------
# Batch Size 减少
# ------------------------------------------------------------

batch_sizes = [
    100,
    500
]


epochs = 16

SEED = 1234


# ============================================================
# 3. 保存实验结果
# ============================================================

# 每一组超参数最终结果
results = []

# 每一个 epoch 的详细结果
epoch_results = []


# ============================================================
# 4. 四个超参数一起扫描
# ============================================================

for hidden_layer_count in hidden_layer_counts:

    for hidden_size in hidden_sizes:

        for learning_rate in learning_rates:

            for batch_size in batch_sizes:


                # =================================================
                # 自动生成 hidden_sizes
                #
                # 例如：
                #
                # hidden_layer_count = 1
                # hidden_size = 20
                #
                # network_hidden_sizes = [20]
                #
                #
                # hidden_layer_count = 5
                # hidden_size = 20
                #
                # network_hidden_sizes =
                # [20, 20, 20, 20, 20]
                # =================================================

                network_hidden_sizes = (
                    [hidden_size]
                    * hidden_layer_count
                )


                print("\n" + "=" * 90)

                print(
                    f"Layers={hidden_layer_count}, "
                    f"Hidden={hidden_size}, "
                    f"LR={learning_rate}, "
                    f"Batch={batch_size}"
                )

                print(
                    "Structure: "
                    f"784 -> "
                    + " -> ".join(
                        str(size)
                        for size in network_hidden_sizes
                    )
                    + " -> 10"
                )

                print("=" * 90)


                # =================================================
                # 每组实验使用相同的初始随机种子
                # =================================================

                np.random.seed(SEED)


                # =================================================
                # 创建网络
                # =================================================

                network = MultiLayerNet(

                    input_size=784,

                    hidden_sizes=
                        network_hidden_sizes,

                    output_size=10
                )


                # =================================================
                # 参数数量
                # =================================================

                num_params = sum(

                    param.size

                    for param
                    in network.params.values()
                )


                # =================================================
                # 初始状态
                # =================================================

                initial_train_acc = network.accuracy(
                    x_train,
                    t_train
                )

                initial_test_acc = network.accuracy(
                    x_test,
                    t_test
                )

                initial_train_loss = network.loss(
                    x_train,
                    t_train
                )

                initial_test_loss = network.loss(
                    x_test,
                    t_test
                )


                print(
                    f"Initial: "
                    f"train_acc={initial_train_acc:.4f}, "
                    f"test_acc={initial_test_acc:.4f}, "
                    f"train_loss={initial_train_loss:.4f}, "
                    f"test_loss={initial_test_loss:.4f}"
                )


                # =================================================
                # Epoch 0
                # =================================================

                epoch_results.append({

                    "hidden_layer_count":
                        hidden_layer_count,

                    "hidden_size":
                        hidden_size,

                    "learning_rate":
                        learning_rate,

                    "batch_size":
                        batch_size,

                    "epoch":
                        0,

                    "train_acc":
                        initial_train_acc,

                    "test_acc":
                        initial_test_acc,

                    "train_loss":
                        initial_train_loss,

                    "test_loss":
                        initial_test_loss
                })


                best_test_acc = initial_test_acc
                best_epoch = 0

                final_train_acc = initial_train_acc
                final_test_acc = initial_test_acc

                final_train_loss = initial_train_loss
                final_test_loss = initial_test_loss


                start_time = time.time()


                # =================================================
                # 5. 正式训练
                # =================================================

                for epoch in range(
                    1,
                    epochs + 1
                ):


                    # ---------------------------------------------
                    # 每一个 epoch 重新 shuffle
                    # ---------------------------------------------

                    epoch_rng = np.random.RandomState(
                        SEED + epoch
                    )

                    indices = epoch_rng.permutation(
                        train_size
                    )


                    # ---------------------------------------------
                    # 按 batch_size 切分训练集
                    # ---------------------------------------------

                    for start in range(
                        0,
                        train_size,
                        batch_size
                    ):

                        end = min(
                            start + batch_size,
                            train_size
                        )

                        batch_mask = indices[
                            start:end
                        ]

                        x_batch = x_train[
                            batch_mask
                        ]

                        t_batch = t_train[
                            batch_mask
                        ]


                        # =========================================
                        # Backpropagation
                        # =========================================

                        grad = network.gradient(
                            x_batch,
                            t_batch
                        )


                        # =========================================
                        # SGD
                        #
                        # 这里不能再写死：
                        #
                        # W1 b1 W2 b2
                        #
                        # 因为现在层数会变化。
                        # =========================================

                        for key in network.params.keys():

                            network.params[key] -= (
                                learning_rate
                                * grad[key]
                            )


                    # =================================================
                    # 一个完整 Epoch 结束
                    # =================================================

                    train_acc = network.accuracy(
                        x_train,
                        t_train
                    )

                    test_acc = network.accuracy(
                        x_test,
                        t_test
                    )

                    train_loss = network.loss(
                        x_train,
                        t_train
                    )

                    test_loss = network.loss(
                        x_test,
                        t_test
                    )


                    final_train_acc = train_acc
                    final_test_acc = test_acc

                    final_train_loss = train_loss
                    final_test_loss = test_loss


                    # =================================================
                    # 最佳 Test Accuracy
                    # =================================================

                    if test_acc > best_test_acc:

                        best_test_acc = test_acc
                        best_epoch = epoch


                    # =================================================
                    # 保存当前 Epoch
                    # =================================================

                    epoch_results.append({

                        "hidden_layer_count":
                            hidden_layer_count,

                        "hidden_size":
                            hidden_size,

                        "learning_rate":
                            learning_rate,

                        "batch_size":
                            batch_size,

                        "epoch":
                            epoch,

                        "train_acc":
                            train_acc,

                        "test_acc":
                            test_acc,

                        "train_loss":
                            train_loss,

                        "test_loss":
                            test_loss
                    })


                    print(

                        f"Epoch {epoch:2d}: "

                        f"train_acc="
                        f"{train_acc:.4f}, "

                        f"test_acc="
                        f"{test_acc:.4f}, "

                        f"train_loss="
                        f"{train_loss:.4f}, "

                        f"test_loss="
                        f"{test_loss:.4f}"
                    )


                # =================================================
                # 6. 当前实验结束
                # =================================================

                elapsed_time = (
                    time.time()
                    - start_time
                )


                results.append({

                    "hidden_layer_count":
                        hidden_layer_count,

                    "hidden_size":
                        hidden_size,

                    "learning_rate":
                        learning_rate,

                    "batch_size":
                        batch_size,

                    "epochs":
                        epochs,

                    "parameters":
                        num_params,

                    "initial_train_acc":
                        initial_train_acc,

                    "initial_test_acc":
                        initial_test_acc,

                    "initial_train_loss":
                        initial_train_loss,

                    "initial_test_loss":
                        initial_test_loss,

                    "final_train_acc":
                        final_train_acc,

                    "final_test_acc":
                        final_test_acc,

                    "final_train_loss":
                        final_train_loss,

                    "final_test_loss":
                        final_test_loss,

                    "best_test_acc":
                        best_test_acc,

                    "best_epoch":
                        best_epoch,

                    "time_seconds":
                        elapsed_time
                })


# ============================================================
# 7. Shell 输出最终总表
# ============================================================

print("\n\n")

print("=" * 140)
print("FINAL RESULTS")
print("=" * 140)


print(

    f"{'Layers':>8} "

    f"{'Hidden':>8} "

    f"{'LR':>8} "

    f"{'Batch':>8} "

    f"{'Params':>10} "

    f"{'Final Train':>12} "

    f"{'Final Test':>12} "

    f"{'Train Loss':>12} "

    f"{'Test Loss':>12} "

    f"{'Best Test':>12} "

    f"{'Best Epoch':>10} "

    f"{'Time(s)':>10}"
)


print("-" * 140)


for r in results:

    print(

        f"{r['hidden_layer_count']:>8} "

        f"{r['hidden_size']:>8} "

        f"{r['learning_rate']:>8.3f} "

        f"{r['batch_size']:>8} "

        f"{r['parameters']:>10} "

        f"{r['final_train_acc']:>12.4f} "

        f"{r['final_test_acc']:>12.4f} "

        f"{r['final_train_loss']:>12.4f} "

        f"{r['final_test_loss']:>12.4f} "

        f"{r['best_test_acc']:>12.4f} "

        f"{r['best_epoch']:>10} "

        f"{r['time_seconds']:>10.1f}"
    )


# ============================================================
# 8. 保存最终结果 CSV
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hyperparameters" / "depth_width"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_filename = str(OUTPUT_DIR / "depth_width_experiment.csv")


fieldnames = [

    "hidden_layer_count",

    "hidden_size",

    "learning_rate",

    "batch_size",

    "epochs",

    "parameters",

    "initial_train_acc",

    "initial_test_acc",

    "initial_train_loss",

    "initial_test_loss",

    "final_train_acc",

    "final_test_acc",

    "final_train_loss",

    "final_test_loss",

    "best_test_acc",

    "best_epoch",

    "time_seconds"
]


with open(

    csv_filename,

    "w",

    newline="",

    encoding="utf-8"

) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 9. 保存每一个 Epoch 的详细数据
# ============================================================

epoch_csv_filename = str(OUTPUT_DIR / "depth_width_epoch_history.csv")


epoch_fieldnames = [

    "hidden_layer_count",

    "hidden_size",

    "learning_rate",

    "batch_size",

    "epoch",

    "train_acc",

    "test_acc",

    "train_loss",

    "test_loss"
]


with open(

    epoch_csv_filename,

    "w",

    newline="",

    encoding="utf-8"

) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=epoch_fieldnames
    )

    writer.writeheader()

    writer.writerows(
        epoch_results
    )


# ============================================================
# 10. 显示保存位置
# ============================================================

print(
    "\nFinal results CSV saved to:"
)

print(
    os.path.abspath(
        csv_filename
    )
)


print(
    "\nEpoch history CSV saved to:"
)

print(
    os.path.abspath(
        epoch_csv_filename
    )
)