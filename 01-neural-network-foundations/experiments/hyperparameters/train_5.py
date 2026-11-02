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

# 隐藏层数量
hidden_layer_counts = [
    1,
    2,
    5,
    10
]


# hidden_size 的含义保持不变：
# 它是构造网络宽度时使用的“基础隐藏层宽度”
hidden_sizes = [
    1,
    5,
    20,
    40,
    50,
    100
]


# 新增维度：
# constant   = 每层神经元数量相同
# decreasing = 从输入端到输出端逐渐变窄
# increasing = 从输入端到输出端逐渐变宽
hidden_size_patterns = [
    "constant",
    "decreasing",
    "increasing"
]


# 为了把重点放在 Depth × Width × Width Pattern，
# 这一轮先固定较少的 learning rate / batch size。
# 如果之后想研究交互作用，可以再加回更多值。
learning_rates = [
    0.1
]

batch_sizes = [
    100
]


epochs = 16

SEED = 1234


# ============================================================
# 3. 根据层数、基础宽度和结构模式生成每层神经元数量
# ============================================================

def build_network_hidden_sizes(
    hidden_layer_count,
    hidden_size,
    hidden_size_pattern
):

    # 只有一个隐藏层时，不存在“越来越宽/越来越窄”
    if hidden_layer_count == 1:
        return [hidden_size]


    # --------------------------------------------------------
    # constant
    #
    # hidden_layer_count = 5
    # hidden_size = 100
    #
    # [100, 100, 100, 100, 100]
    # --------------------------------------------------------

    if hidden_size_pattern == "constant":

        return (
            [hidden_size]
            * hidden_layer_count
        )


    # --------------------------------------------------------
    # 为 increasing / decreasing 定义另一端宽度
    #
    # 这里取 hidden_size 的 1/4，
    # 最少保持 1 个神经元。
    #
    # hidden_size = 100
    # narrow_hidden_size = 25
    #
    # hidden_size = 20
    # narrow_hidden_size = 5
    # --------------------------------------------------------

    narrow_hidden_size = max(
        1,
        int(round(hidden_size / 4))
    )


    # --------------------------------------------------------
    # decreasing
    #
    # hidden_layer_count = 5
    # hidden_size = 100
    #
    # 大约：
    # [100, 81, 62, 44, 25]
    # --------------------------------------------------------

    if hidden_size_pattern == "decreasing":

        network_hidden_sizes = np.linspace(
            hidden_size,
            narrow_hidden_size,
            hidden_layer_count
        )

        network_hidden_sizes = np.rint(
            network_hidden_sizes
        ).astype(int).tolist()

        return network_hidden_sizes


    # --------------------------------------------------------
    # increasing
    #
    # hidden_layer_count = 5
    # hidden_size = 100
    #
    # 大约：
    # [25, 44, 62, 81, 100]
    # --------------------------------------------------------

    if hidden_size_pattern == "increasing":

        network_hidden_sizes = np.linspace(
            narrow_hidden_size,
            hidden_size,
            hidden_layer_count
        )

        network_hidden_sizes = np.rint(
            network_hidden_sizes
        ).astype(int).tolist()

        return network_hidden_sizes


    raise ValueError(
        "Unknown hidden_size_pattern: "
        + str(hidden_size_pattern)
    )


# ============================================================
# 4. 保存实验结果
# ============================================================

results = []

epoch_results = []


# ============================================================
# 5. 五个实验维度一起扫描
#
# hidden_layer_count
# hidden_size
# hidden_size_pattern
# learning_rate
# batch_size
# ============================================================

for hidden_layer_count in hidden_layer_counts:

    for hidden_size in hidden_sizes:

        # 用于避免重复结构
        #
        # 例如 hidden_size=1 时：
        #
        # constant
        # decreasing
        # increasing
        #
        # 实际都会生成 [1, 1, 1, ...]
        #
        # 这种情况只跑一次。
        used_network_hidden_sizes = set()

        for hidden_size_pattern in hidden_size_patterns:


            network_hidden_sizes = (
                build_network_hidden_sizes(
                    hidden_layer_count,
                    hidden_size,
                    hidden_size_pattern
                )
            )


            network_hidden_sizes_tuple = tuple(
                network_hidden_sizes
            )


            if (
                network_hidden_sizes_tuple
                in used_network_hidden_sizes
            ):

                continue


            used_network_hidden_sizes.add(
                network_hidden_sizes_tuple
            )


            # CSV 里保存实际网络结构
            network_hidden_sizes_text = (
                "-".join(
                    str(size)
                    for size in network_hidden_sizes
                )
            )


            for learning_rate in learning_rates:

                for batch_size in batch_sizes:


                    print("\n" + "=" * 100)

                    print(
                        f"Layers={hidden_layer_count}, "
                        f"Hidden={hidden_size}, "
                        f"Pattern={hidden_size_pattern}, "
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

                    print("=" * 100)


                    # =================================================
                    # 每组实验使用相同的初始随机种子
                    # =================================================

                    np.random.seed(SEED)


                    # =================================================
                    # 创建网络
                    # =================================================

                    network = MultiLayerNet(
                        input_size=784,
                        hidden_sizes=network_hidden_sizes,
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

                        "hidden_size_pattern":
                            hidden_size_pattern,

                        "network_hidden_sizes":
                            network_hidden_sizes_text,

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
                    # 6. 正式训练
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

                            "hidden_size_pattern":
                                hidden_size_pattern,

                            "network_hidden_sizes":
                                network_hidden_sizes_text,

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
                    # 7. 当前实验结束
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

                        "hidden_size_pattern":
                            hidden_size_pattern,

                        "network_hidden_sizes":
                            network_hidden_sizes_text,

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
# 8. Shell 输出最终总表
# ============================================================

print("\n\n")

print("=" * 175)
print("FINAL RESULTS")
print("=" * 175)


print(

    f"{'Layers':>7} "

    f"{'Hidden':>7} "

    f"{'Pattern':>11} "

    f"{'Structure':>30} "

    f"{'LR':>7} "

    f"{'Batch':>7} "

    f"{'Params':>10} "

    f"{'Final Train':>12} "

    f"{'Final Test':>12} "

    f"{'Train Loss':>12} "

    f"{'Test Loss':>12} "

    f"{'Best Test':>12} "

    f"{'Best Epoch':>10} "

    f"{'Time(s)':>10}"
)


print("-" * 175)


for r in results:

    print(

        f"{r['hidden_layer_count']:>7} "

        f"{r['hidden_size']:>7} "

        f"{r['hidden_size_pattern']:>11} "

        f"{r['network_hidden_sizes']:>30} "

        f"{r['learning_rate']:>7.3f} "

        f"{r['batch_size']:>7} "

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
# 9. 保存最终结果 CSV
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hyperparameters" / "depth_width_pattern"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_filename = str(OUTPUT_DIR / "depth_width_pattern_experiment.csv")


fieldnames = [

    "hidden_layer_count",

    "hidden_size",

    "hidden_size_pattern",

    "network_hidden_sizes",

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
# 10. 保存每一个 Epoch 的详细数据
# ============================================================

epoch_csv_filename = str(OUTPUT_DIR / "depth_width_pattern_epoch_history.csv")


epoch_fieldnames = [

    "hidden_layer_count",

    "hidden_size",

    "hidden_size_pattern",

    "network_hidden_sizes",

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
# 11. 显示保存位置
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