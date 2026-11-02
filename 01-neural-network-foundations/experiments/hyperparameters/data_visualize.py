# coding: utf-8
"""
MNIST 超参数实验可视化脚本
================================

输入:
    results/hyperparameters/hyperparameter_plots/00_summary/all_results_ranked.csv

输出:
    outputs/hyperparameter_plots_regenerated/
        00_summary/
        01_marginal_effects/
        02_final_test_vs_batch/
        03_final_test_vs_learning_rate/
        04_final_test_vs_hidden_size/
        05_final_test_heatmaps/
        06_best_test_heatmaps/
        07_tradeoffs/

说明:
- 每张图单独保存，不使用 subplot。
- Batch Size 和 Learning Rate 跨度很大，所以很多图使用对数横轴。
- Accuracy 图统一用百分比显示。
- "Marginal mean" 表示对其他超参数组合取平均，用于看整体趋势；
  详细的交互作用请看后面的分组曲线和 heatmap。
"""

import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 基本设置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = PROJECT_ROOT / "results" / "hyperparameters" / "hyperparameter_plots" / "00_summary" / "all_results_ranked.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hyperparameter_plots_regenerated"

# MNIST train set size
TRAIN_SIZE = 60000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_SUMMARY = OUTPUT_DIR / "00_summary"
DIR_MARGINAL = OUTPUT_DIR / "01_marginal_effects"
DIR_BATCH = OUTPUT_DIR / "02_final_test_vs_batch"
DIR_LR = OUTPUT_DIR / "03_final_test_vs_learning_rate"
DIR_HIDDEN = OUTPUT_DIR / "04_final_test_vs_hidden_size"
DIR_HEAT_FINAL = OUTPUT_DIR / "05_final_test_heatmaps"
DIR_HEAT_BEST = OUTPUT_DIR / "06_best_test_heatmaps"
DIR_TRADEOFF = OUTPUT_DIR / "07_tradeoffs"

for folder in [
    DIR_SUMMARY, DIR_MARGINAL, DIR_BATCH, DIR_LR, DIR_HIDDEN,
    DIR_HEAT_FINAL, DIR_HEAT_BEST, DIR_TRADEOFF
]:
    folder.mkdir(exist_ok=True)


# ============================================================
# 1. 读取数据
# ============================================================

df = pd.read_csv(CSV_FILE)

required_columns = [
    "batch_size",
    "hidden_size",
    "learning_rate",
    "epochs",
    "parameters",
    "initial_train_acc",
    "initial_test_acc",
    "final_train_acc",
    "final_test_acc",
    "best_test_acc",
    "best_epoch",
    "time_seconds"
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"CSV 缺少这些列: {missing}")


# ============================================================
# 2. 添加派生指标
# ============================================================

df["generalization_gap"] = (
    df["final_train_acc"] - df["final_test_acc"]
)

df["test_improvement"] = (
    df["final_test_acc"] - df["initial_test_acc"]
)

df["best_to_final_drop"] = (
    df["best_test_acc"] - df["final_test_acc"]
)

df["updates_per_epoch"] = np.ceil(
    TRAIN_SIZE / df["batch_size"]
).astype(int)

df["total_updates"] = (
    df["updates_per_epoch"] * df["epochs"]
)

df["improvement_per_second"] = (
    df["test_improvement"] / df["time_seconds"]
)


batch_sizes = sorted(df["batch_size"].unique())
hidden_sizes = sorted(df["hidden_size"].unique())
learning_rates = sorted(df["learning_rate"].unique())


# ============================================================
# 3. 工具函数
# ============================================================

def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def percent_axis():
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        lambda value, pos: f"{value * 100:.0f}%"
    )


def plot_single_line(
    x,
    y,
    xlabel,
    ylabel,
    title,
    path,
    xscale=None,
    y_percent=False,
    marker="o"
):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)

    if xscale:
        plt.xscale(xscale)

    if y_percent:
        percent_axis()

    save_current_figure(path)


def plot_group_lines(
    data,
    x_col,
    y_col,
    group_col,
    xlabel,
    ylabel,
    title,
    path,
    xscale=None,
    y_percent=False
):
    plt.figure(figsize=(9, 5.5))

    for group_value, group in data.groupby(group_col):
        group = group.sort_values(x_col)
        plt.plot(
            group[x_col],
            group[y_col],
            marker="o",
            label=f"{group_col}={group_value}"
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)

    if xscale:
        plt.xscale(xscale)

    if y_percent:
        percent_axis()

    save_current_figure(path)


def save_heatmap(
    matrix,
    x_labels,
    y_labels,
    title,
    xlabel,
    ylabel,
    path,
    as_percent=True
):
    plt.figure(figsize=(8, 6))

    image = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(image)

    plt.xticks(
        np.arange(len(x_labels)),
        [str(x) for x in x_labels]
    )

    plt.yticks(
        np.arange(len(y_labels)),
        [str(y) for y in y_labels]
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # 在格子内写数字
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]

            if np.isnan(value):
                text = "-"
            elif as_percent:
                text = f"{value * 100:.1f}%"
            else:
                text = f"{value:.3f}"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8
            )

    save_current_figure(path)


# ============================================================
# 4. 汇总表
# ============================================================

# 按 best test accuracy 排名
ranked = df.sort_values(
    ["best_test_acc", "final_test_acc"],
    ascending=False
).reset_index(drop=True)

ranked.to_csv(
    DIR_SUMMARY / "all_results_ranked.csv",
    index=False,
    encoding="utf-8-sig"
)

ranked.head(20).to_csv(
    DIR_SUMMARY / "top20_configurations.csv",
    index=False,
    encoding="utf-8-sig"
)

# 各超参数值能达到的最好结果
best_by_batch = (
    df.loc[df.groupby("batch_size")["best_test_acc"].idxmax()]
    .sort_values("batch_size")
)
best_by_batch.to_csv(
    DIR_SUMMARY / "best_configuration_by_batch_size.csv",
    index=False,
    encoding="utf-8-sig"
)

best_by_lr = (
    df.loc[df.groupby("learning_rate")["best_test_acc"].idxmax()]
    .sort_values("learning_rate")
)
best_by_lr.to_csv(
    DIR_SUMMARY / "best_configuration_by_learning_rate.csv",
    index=False,
    encoding="utf-8-sig"
)

best_by_hidden = (
    df.loc[df.groupby("hidden_size")["best_test_acc"].idxmax()]
    .sort_values("hidden_size")
)
best_by_hidden.to_csv(
    DIR_SUMMARY / "best_configuration_by_hidden_size.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. Marginal effect：对其他参数取平均
# ============================================================

# ---------- Batch Size ----------

batch_mean = (
    df.groupby("batch_size", as_index=False)
    .mean(numeric_only=True)
    .sort_values("batch_size")
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["final_test_acc"],
    "Batch size",
    "Mean final test accuracy",
    "Mean Final Test Accuracy vs Batch Size",
    DIR_MARGINAL / "01_mean_final_test_accuracy_vs_batch_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["best_test_acc"],
    "Batch size",
    "Mean best test accuracy",
    "Mean Best Test Accuracy vs Batch Size",
    DIR_MARGINAL / "02_mean_best_test_accuracy_vs_batch_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["final_train_acc"],
    "Batch size",
    "Mean final train accuracy",
    "Mean Final Train Accuracy vs Batch Size",
    DIR_MARGINAL / "03_mean_final_train_accuracy_vs_batch_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["generalization_gap"],
    "Batch size",
    "Mean train-test gap",
    "Mean Generalization Gap vs Batch Size",
    DIR_MARGINAL / "04_mean_generalization_gap_vs_batch_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["best_to_final_drop"],
    "Batch size",
    "Mean best-to-final drop",
    "Mean Best-to-Final Accuracy Drop vs Batch Size",
    DIR_MARGINAL / "05_mean_best_to_final_drop_vs_batch_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["time_seconds"],
    "Batch size",
    "Mean training time (s)",
    "Mean Training Time vs Batch Size",
    DIR_MARGINAL / "06_mean_training_time_vs_batch_size.png",
    xscale="log"
)

plot_single_line(
    batch_mean["batch_size"],
    batch_mean["best_epoch"],
    "Batch size",
    "Mean best epoch",
    "Mean Best Epoch vs Batch Size",
    DIR_MARGINAL / "07_mean_best_epoch_vs_batch_size.png",
    xscale="log"
)

# 理论上的 updates per epoch
updates = (
    df[["batch_size", "updates_per_epoch"]]
    .drop_duplicates()
    .sort_values("batch_size")
)

plot_single_line(
    updates["batch_size"],
    updates["updates_per_epoch"],
    "Batch size",
    "Updates per epoch",
    "Parameter Updates per Epoch vs Batch Size",
    DIR_MARGINAL / "08_updates_per_epoch_vs_batch_size.png",
    xscale="log"
)

plt.figure(figsize=(8, 5))
plt.plot(
    updates["batch_size"],
    updates["updates_per_epoch"],
    marker="o"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Batch size")
plt.ylabel("Updates per epoch")
plt.title("Parameter Updates per Epoch vs Batch Size (Log-Log)")
plt.grid(True, alpha=0.25)
save_current_figure(
    DIR_MARGINAL / "09_updates_per_epoch_vs_batch_size_loglog.png"
)


# ---------- Learning Rate ----------

lr_mean = (
    df.groupby("learning_rate", as_index=False)
    .mean(numeric_only=True)
    .sort_values("learning_rate")
)

plot_single_line(
    lr_mean["learning_rate"],
    lr_mean["final_test_acc"],
    "Learning rate",
    "Mean final test accuracy",
    "Mean Final Test Accuracy vs Learning Rate",
    DIR_MARGINAL / "10_mean_final_test_accuracy_vs_learning_rate.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    lr_mean["learning_rate"],
    lr_mean["best_test_acc"],
    "Learning rate",
    "Mean best test accuracy",
    "Mean Best Test Accuracy vs Learning Rate",
    DIR_MARGINAL / "11_mean_best_test_accuracy_vs_learning_rate.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    lr_mean["learning_rate"],
    lr_mean["generalization_gap"],
    "Learning rate",
    "Mean train-test gap",
    "Mean Generalization Gap vs Learning Rate",
    DIR_MARGINAL / "12_mean_generalization_gap_vs_learning_rate.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    lr_mean["learning_rate"],
    lr_mean["best_to_final_drop"],
    "Learning rate",
    "Mean best-to-final drop",
    "Mean Best-to-Final Accuracy Drop vs Learning Rate",
    DIR_MARGINAL / "13_mean_best_to_final_drop_vs_learning_rate.png",
    xscale="log",
    y_percent=True
)


# ---------- Hidden Size ----------

hidden_mean = (
    df.groupby("hidden_size", as_index=False)
    .mean(numeric_only=True)
    .sort_values("hidden_size")
)

plot_single_line(
    hidden_mean["hidden_size"],
    hidden_mean["final_test_acc"],
    "Hidden size",
    "Mean final test accuracy",
    "Mean Final Test Accuracy vs Hidden Size",
    DIR_MARGINAL / "14_mean_final_test_accuracy_vs_hidden_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    hidden_mean["hidden_size"],
    hidden_mean["best_test_acc"],
    "Hidden size",
    "Mean best test accuracy",
    "Mean Best Test Accuracy vs Hidden Size",
    DIR_MARGINAL / "15_mean_best_test_accuracy_vs_hidden_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    hidden_mean["hidden_size"],
    hidden_mean["generalization_gap"],
    "Hidden size",
    "Mean train-test gap",
    "Mean Generalization Gap vs Hidden Size",
    DIR_MARGINAL / "16_mean_generalization_gap_vs_hidden_size.png",
    xscale="log",
    y_percent=True
)

plot_single_line(
    hidden_mean["hidden_size"],
    hidden_mean["time_seconds"],
    "Hidden size",
    "Mean training time (s)",
    "Mean Training Time vs Hidden Size",
    DIR_MARGINAL / "17_mean_training_time_vs_hidden_size.png",
    xscale="log"
)

# 参数数量 vs hidden size
param_data = (
    df[["hidden_size", "parameters"]]
    .drop_duplicates()
    .sort_values("hidden_size")
)

plot_single_line(
    param_data["hidden_size"],
    param_data["parameters"],
    "Hidden size",
    "Number of parameters",
    "Model Parameter Count vs Hidden Size",
    DIR_MARGINAL / "18_parameter_count_vs_hidden_size.png",
    xscale="log"
)


# ============================================================
# 6. 每个 Hidden Size:
#    Final Test Accuracy vs Batch Size，线条代表 LR
# ============================================================

for hidden in hidden_sizes:
    subset = df[df["hidden_size"] == hidden]

    plot_group_lines(
        subset,
        x_col="batch_size",
        y_col="final_test_acc",
        group_col="learning_rate",
        xlabel="Batch size",
        ylabel="Final test accuracy",
        title=f"Final Test Accuracy vs Batch Size | Hidden={hidden}",
        path=DIR_BATCH / f"hidden_{hidden}_final_test_vs_batch.png",
        xscale="log",
        y_percent=True
    )


# ============================================================
# 7. 每个 Hidden Size:
#    Final Test Accuracy vs Learning Rate，线条代表 Batch
# ============================================================

for hidden in hidden_sizes:
    subset = df[df["hidden_size"] == hidden]

    plot_group_lines(
        subset,
        x_col="learning_rate",
        y_col="final_test_acc",
        group_col="batch_size",
        xlabel="Learning rate",
        ylabel="Final test accuracy",
        title=f"Final Test Accuracy vs Learning Rate | Hidden={hidden}",
        path=DIR_LR / f"hidden_{hidden}_final_test_vs_learning_rate.png",
        xscale="log",
        y_percent=True
    )


# ============================================================
# 8. 每个 Batch Size:
#    Final Test Accuracy vs Hidden Size，线条代表 LR
# ============================================================

for batch in batch_sizes:
    subset = df[df["batch_size"] == batch]

    plot_group_lines(
        subset,
        x_col="hidden_size",
        y_col="final_test_acc",
        group_col="learning_rate",
        xlabel="Hidden size",
        ylabel="Final test accuracy",
        title=f"Final Test Accuracy vs Hidden Size | Batch={batch}",
        path=DIR_HIDDEN / f"batch_{batch}_final_test_vs_hidden_size.png",
        xscale="log",
        y_percent=True
    )


# ============================================================
# 9. Final Test Accuracy Heatmap
#    每个 hidden size 单独一张
# ============================================================

for hidden in hidden_sizes:
    subset = df[df["hidden_size"] == hidden]

    pivot = subset.pivot(
        index="batch_size",
        columns="learning_rate",
        values="final_test_acc"
    )

    pivot = pivot.reindex(
        index=batch_sizes,
        columns=learning_rates
    )

    save_heatmap(
        pivot.values,
        x_labels=learning_rates,
        y_labels=batch_sizes,
        title=f"Final Test Accuracy Heatmap | Hidden={hidden}",
        xlabel="Learning rate",
        ylabel="Batch size",
        path=DIR_HEAT_FINAL / f"hidden_{hidden}_final_test_heatmap.png",
        as_percent=True
    )


# ============================================================
# 10. Best Test Accuracy Heatmap
#     每个 hidden size 单独一张
# ============================================================

for hidden in hidden_sizes:
    subset = df[df["hidden_size"] == hidden]

    pivot = subset.pivot(
        index="batch_size",
        columns="learning_rate",
        values="best_test_acc"
    )

    pivot = pivot.reindex(
        index=batch_sizes,
        columns=learning_rates
    )

    save_heatmap(
        pivot.values,
        x_labels=learning_rates,
        y_labels=batch_sizes,
        title=f"Best Test Accuracy Heatmap | Hidden={hidden}",
        xlabel="Learning rate",
        ylabel="Batch size",
        path=DIR_HEAT_BEST / f"hidden_{hidden}_best_test_heatmap.png",
        as_percent=True
    )


# ============================================================
# 11. Trade-off 图
# ============================================================

# Accuracy vs training time
plt.figure(figsize=(8, 5.5))
plt.scatter(
    df["time_seconds"],
    df["best_test_acc"]
)
plt.xlabel("Training time (s)")
plt.ylabel("Best test accuracy")
plt.title("Best Test Accuracy vs Training Time")
plt.grid(True, alpha=0.25)
percent_axis()
save_current_figure(
    DIR_TRADEOFF / "01_best_test_accuracy_vs_training_time.png"
)

# Final vs Best: 看训练是否在后期退化
plt.figure(figsize=(7, 6))
plt.scatter(
    df["final_test_acc"],
    df["best_test_acc"]
)

lower = min(
    df["final_test_acc"].min(),
    df["best_test_acc"].min()
)

upper = max(
    df["final_test_acc"].max(),
    df["best_test_acc"].max()
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--"
)

plt.xlabel("Final test accuracy")
plt.ylabel("Best test accuracy")
plt.title("Best vs Final Test Accuracy")
plt.grid(True, alpha=0.25)

ax = plt.gca()
ax.xaxis.set_major_formatter(
    lambda value, pos: f"{value * 100:.0f}%"
)
ax.yaxis.set_major_formatter(
    lambda value, pos: f"{value * 100:.0f}%"
)

save_current_figure(
    DIR_TRADEOFF / "02_best_vs_final_test_accuracy.png"
)

# Best-to-final drop vs LR
drop_lr = (
    df.groupby("learning_rate", as_index=False)
    ["best_to_final_drop"]
    .mean()
    .sort_values("learning_rate")
)

plot_single_line(
    drop_lr["learning_rate"],
    drop_lr["best_to_final_drop"],
    "Learning rate",
    "Mean best-to-final drop",
    "Training Instability vs Learning Rate",
    DIR_TRADEOFF / "03_training_instability_vs_learning_rate.png",
    xscale="log",
    y_percent=True
)

# Accuracy gain per second vs batch
eff_batch = (
    df.groupby("batch_size", as_index=False)
    ["improvement_per_second"]
    .mean()
    .sort_values("batch_size")
)

plot_single_line(
    eff_batch["batch_size"],
    eff_batch["improvement_per_second"],
    "Batch size",
    "Mean accuracy improvement per second",
    "Training Efficiency vs Batch Size",
    DIR_TRADEOFF / "04_training_efficiency_vs_batch_size.png",
    xscale="log"
)

# Best achievable test accuracy per batch
batch_envelope = (
    df.groupby("batch_size", as_index=False)
    ["best_test_acc"]
    .max()
    .sort_values("batch_size")
)

plot_single_line(
    batch_envelope["batch_size"],
    batch_envelope["best_test_acc"],
    "Batch size",
    "Best achievable test accuracy",
    "Best Achievable Test Accuracy vs Batch Size",
    DIR_TRADEOFF / "05_best_achievable_accuracy_vs_batch_size.png",
    xscale="log",
    y_percent=True
)

# Best achievable test accuracy per LR
lr_envelope = (
    df.groupby("learning_rate", as_index=False)
    ["best_test_acc"]
    .max()
    .sort_values("learning_rate")
)

plot_single_line(
    lr_envelope["learning_rate"],
    lr_envelope["best_test_acc"],
    "Learning rate",
    "Best achievable test accuracy",
    "Best Achievable Test Accuracy vs Learning Rate",
    DIR_TRADEOFF / "06_best_achievable_accuracy_vs_learning_rate.png",
    xscale="log",
    y_percent=True
)

# Best achievable test accuracy per hidden
hidden_envelope = (
    df.groupby("hidden_size", as_index=False)
    ["best_test_acc"]
    .max()
    .sort_values("hidden_size")
)

plot_single_line(
    hidden_envelope["hidden_size"],
    hidden_envelope["best_test_acc"],
    "Hidden size",
    "Best achievable test accuracy",
    "Best Achievable Test Accuracy vs Hidden Size",
    DIR_TRADEOFF / "07_best_achievable_accuracy_vs_hidden_size.png",
    xscale="log",
    y_percent=True
)


# ============================================================
# 12. 文本总结
# ============================================================

best_row = ranked.iloc[0]

summary_text = f"""
MNIST Hyperparameter Experiment Summary
=======================================

Rows: {len(df)}

Batch sizes:
{batch_sizes}

Hidden sizes:
{hidden_sizes}

Learning rates:
{learning_rates}

Epochs:
{sorted(df['epochs'].unique())}

Best configuration by BEST TEST ACCURACY
----------------------------------------
Batch size     : {best_row['batch_size']}
Hidden size    : {best_row['hidden_size']}
Learning rate  : {best_row['learning_rate']}
Best test acc  : {best_row['best_test_acc']:.4f}
Final test acc : {best_row['final_test_acc']:.4f}
Final train acc: {best_row['final_train_acc']:.4f}
Best epoch     : {best_row['best_epoch']}
Time (s)       : {best_row['time_seconds']:.2f}

Derived metrics
---------------
Generalization gap = final_train_acc - final_test_acc
Best-to-final drop = best_test_acc - final_test_acc
Updates per epoch  = ceil(60000 / batch_size)
"""

with open(
    DIR_SUMMARY / "summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(summary_text)


print("\n完成。")
print(f"读取文件: {CSV_FILE}")
print(f"共分析 {len(df)} 组实验")
print(f"输出目录: {OUTPUT_DIR.resolve()}")

print("\n最佳组合:")
print(
    f"Batch={best_row['batch_size']}, "
    f"Hidden={best_row['hidden_size']}, "
    f"LR={best_row['learning_rate']}, "
    f"Best Test={best_row['best_test_acc']:.4f}"
)

print("\n生成内容包括:")
print("- 各超参数的整体边际趋势")
print("- Final Test Accuracy vs Batch Size")
print("- Final Test Accuracy vs Learning Rate")
print("- Final Test Accuracy vs Hidden Size")
print("- 每个 Hidden Size 的 Batch × LR heatmap")
print("- Best Test Accuracy heatmap")
print("- Train-Test gap")
print("- Best-to-Final drop")
print("- Training time")
print("- Updates per epoch")
print("- Accuracy / training-time trade-off")
print("- Top 20 参数组合和各超参数最优配置 CSV")