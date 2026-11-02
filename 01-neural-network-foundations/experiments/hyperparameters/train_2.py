# coding: utf-8
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import time

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


# ============================================================
# 1. 读取 MNIST
# ============================================================

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,
    one_hot_label=True
)


# ============================================================
# 2. 实验参数
# ============================================================

hidden_sizes = [1, 2, 5, 10, 20, 40, 50, 100]

iters_num = 10000
batch_size = 100
learning_rate = 0.05

train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size


# ============================================================
# 3. 所有网络使用相同的 mini-batch
# ============================================================

batch_rng = np.random.RandomState(1234)

batch_masks = batch_rng.randint(
    0,
    train_size,
    size=(iters_num, batch_size)
)


# ============================================================
# 4. 保存最终结果
# ============================================================

results = []


# ============================================================
# 5. 测试不同 hidden size
# ============================================================

for hidden_size in hidden_sizes:

    print("\n" + "=" * 60)
    print("Training hidden_size =", hidden_size)
    print("=" * 60)

    np.random.seed(1234)

    network = TwoLayerNet(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10
    )

    # 参数数量
    num_params = (
        network.params['W1'].size +
        network.params['b1'].size +
        network.params['W2'].size +
        network.params['b2'].size
    )

    # 初始准确率
    initial_train_acc = network.accuracy(x_train, t_train)
    initial_test_acc = network.accuracy(x_test, t_test)

    print(
        f"Initial accuracy: "
        f"train={initial_train_acc:.4f}, "
        f"test={initial_test_acc:.4f}"
    )

    best_test_acc = initial_test_acc
    best_epoch = 0

    final_train_acc = initial_train_acc
    final_test_acc = initial_test_acc

    start_time = time.time()

    # ========================================================
    # 训练
    # ========================================================

    for i in range(iters_num):

        batch_mask = batch_masks[i]

        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 反向传播
        grad = network.gradient(x_batch, t_batch)

        # SGD 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 每个 epoch 检查一次
        if (i + 1) % iter_per_epoch == 0:

            epoch = (i + 1) // iter_per_epoch

            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)

            final_train_acc = train_acc
            final_test_acc = test_acc

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch

            print(
                f"Epoch {epoch:2d}: "
                f"train={train_acc:.4f}, "
                f"test={test_acc:.4f}"
            )

    elapsed_time = time.time() - start_time

    results.append({
        "hidden_size": hidden_size,
        "parameters": num_params,
        "initial_train_acc": initial_train_acc,
        "initial_test_acc": initial_test_acc,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "time_seconds": elapsed_time
    })


# ============================================================
# 6. 最终汇总
# ============================================================

print("\n\n")
print("=" * 115)
print("FINAL RESULTS")
print("=" * 115)

print(
    f"{'Hidden':>8} "
    f"{'Params':>10} "
    f"{'Init Train':>12} "
    f"{'Init Test':>12} "
    f"{'Final Train':>12} "
    f"{'Final Test':>12} "
    f"{'Best Test':>12} "
    f"{'Best Epoch':>10} "
    f"{'Time(s)':>10}"
)

print("-" * 115)

for r in results:
    print(
        f"{r['hidden_size']:>8} "
        f"{r['parameters']:>10} "
        f"{r['initial_train_acc']:>12.4f} "
        f"{r['initial_test_acc']:>12.4f} "
        f"{r['final_train_acc']:>12.4f} "
        f"{r['final_test_acc']:>12.4f} "
        f"{r['best_test_acc']:>12.4f} "
        f"{r['best_epoch']:>10} "
        f"{r['time_seconds']:>10.1f}"
    )