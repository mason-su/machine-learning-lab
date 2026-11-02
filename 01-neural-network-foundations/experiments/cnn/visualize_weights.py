# coding: utf-8

import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================
# Configuration
# =========================================

param_file_name = str(PROJECT_ROOT / "results" / "checkpoints" / "my_convnet_params.pkl")
# 你也可以改成:
# "my_params.pkl"
# "deep_convnet_params.pkl"
# "my_convnet_params.pkl"

show_histogram = True
show_conv_filters = True


# =========================================
# Helper functions
# =========================================

def load_params(param_file_name):
    with open(param_file_name, "rb") as f:
        params = pickle.load(f)
    return params


def print_param_shapes(params):
    print("===== Parameter Shapes =====")
    for key in sorted(params.keys()):
        print(f"{key}: {params[key].shape}")
    print()


def plot_weight_histograms(params):
    weight_keys = [key for key in sorted(params.keys()) if key.startswith("W")]

    number_of_layers = len(weight_keys)
    number_of_columns = 2
    number_of_rows = math.ceil(number_of_layers / number_of_columns)

    plt.figure(figsize=(12, 4 * number_of_rows))

    for index, key in enumerate(weight_keys, start=1):
        weight = params[key]
        weight_flat = weight.flatten()

        plt.subplot(number_of_rows, number_of_columns, index)
        plt.hist(weight_flat, bins=50)
        plt.title(f"{key} histogram")
        plt.xlabel("weight value")
        plt.ylabel("count")

        print(
            f"{key}: "
            f"min={weight_flat.min():.6f}, "
            f"max={weight_flat.max():.6f}, "
            f"mean={weight_flat.mean():.6f}, "
            f"std={weight_flat.std():.6f}"
        )

    plt.tight_layout()
    plt.show()


def plot_single_channel_filters(weight, layer_name):
    # weight shape: (filter_num, in_channel, filter_h, filter_w)
    filter_num, in_channel, filter_h, filter_w = weight.shape

    number_of_columns = min(10, filter_num)
    number_of_rows = math.ceil(filter_num / number_of_columns)

    plt.figure(figsize=(1.5 * number_of_columns, 1.5 * number_of_rows))

    weight_min = weight.min()
    weight_max = weight.max()

    print(f"\n===== {layer_name} filter visualization =====")
    print(f"shape: {weight.shape}")
    print(f"weight range: {weight_min:.6f} ~ {weight_max:.6f}")

    for filter_index in range(filter_num):
        plt.subplot(number_of_rows, number_of_columns, filter_index + 1)

        if in_channel == 1:
            filter_image = weight[filter_index, 0]
        else:
            # For multi-channel filters, average over input channels
            filter_image = np.mean(weight[filter_index], axis=0)

        plt.imshow(
            filter_image,
            cmap="gray",
            interpolation="nearest"
        )
        plt.title(str(filter_index + 1), fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(
        f"{layer_name} filters\nweight range: {weight_min:.6f} ~ {weight_max:.6f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()


def plot_all_conv_filters(params):
    conv_weight_keys = []

    for key in sorted(params.keys()):
        if key.startswith("W") and params[key].ndim == 4:
            conv_weight_keys.append(key)

    for key in conv_weight_keys:
        plot_single_channel_filters(params[key], key)


def plot_fc_heatmap(weight, layer_name, max_rows=60, max_cols=60):
    # weight shape: (input_dim, output_dim)
    clipped_weight = weight[:max_rows, :max_cols]

    plt.figure(figsize=(8, 6))
    plt.imshow(clipped_weight, cmap="gray", aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(
        f"{layer_name} heatmap (top-left {clipped_weight.shape[0]}x{clipped_weight.shape[1]})"
    )
    plt.xlabel("output neuron index")
    plt.ylabel("input neuron index")
    plt.tight_layout()
    plt.show()


def plot_all_fc_heatmaps(params):
    fc_weight_keys = []

    for key in sorted(params.keys()):
        if key.startswith("W") and params[key].ndim == 2:
            fc_weight_keys.append(key)

    for key in fc_weight_keys:
        weight = params[key]
        print(f"\n===== {key} fully-connected weight matrix =====")
        print(f"shape: {weight.shape}")
        print(
            f"min={weight.min():.6f}, "
            f"max={weight.max():.6f}, "
            f"mean={weight.mean():.6f}, "
            f"std={weight.std():.6f}"
        )
        plot_fc_heatmap(weight, key)


# =========================================
# Main
# =========================================

def main():
    params = load_params(param_file_name)

    print_param_shapes(params)

    if show_histogram:
        print("===== Weight Statistics =====")
        plot_weight_histograms(params)

    if show_conv_filters:
        plot_all_conv_filters(params)

    plot_all_fc_heatmaps(params)


if __name__ == "__main__":
    main()