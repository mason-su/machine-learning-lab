# Third-party and authorship notice

This module combines textbook-derived material, adapted learning exercises, and personal machine-learning experiments. A file classified as a personal experiment may still contain or build upon textbook code.

## Textbook source and license

**Koki Saitoh, *Deep Learning from Scratch* (O'Reilly Japan)**

- Official repository: https://github.com/oreilly-japan/deep-learning-from-scratch
- License: **MIT License**; Copyright (c) 2016 Koki Saitoh
- Complete original notice: [`third_party/BOOK_LICENSE.md`](third_party/BOOK_LICENSE.md)

The original copyright and MIT permission notice are retained. Redistribution and adaptation of the textbook-derived software are governed by that license.

## Textbook-derived components

The following are textbook source files, textbook baselines, or adaptations of them:

- `common/`: textbook shared layers, activation functions, gradients, optimisers, training utilities, and convolution helpers.
- `dataset/mnist.py`: textbook MNIST dataset loader.
- `experiments/hyperparameters/two_layer_net.py`: textbook two-layer network baseline.
- `experiments/cnn/deep_convnet.py`: textbook DeepConvNet baseline, with packaging changes.
- `experiments/learning_exercises/ch04/gradient_2d.py`: textbook gradient visualisation baseline.
- `experiments/learning_exercises/ch03/sample_weight.pkl`: **official textbook sample weights**, byte-for-byte identical to the corresponding Git blob in the official repository.
- Several `experiments/learning_exercises/` scripts and `experiments/regularization/` scripts: modified or closely followed textbook examples.
- Some training and CNN experiment scripts also adapt or use textbook examples and helper code. A modified script is not necessarily wholly original.

See [`FILE_SELECTION.md`](FILE_SELECTION.md) and [`FILE_MANIFEST.csv`](FILE_MANIFEST.csv) for the original file paths, selected files, and archive provenance.

## Owner's experiment work

The hyperparameter sweeps, architecture-width/depth studies, CNN experiments, NumPy/CuPy/PyTorch comparisons, experiment analyses, and archived results come from the repository owner's project archive. These are described as **personal experiments**, not as proof that every underlying implementation or line of code was independently authored.

## Checkpoint provenance

- `results/checkpoints/deep_convnet_params.pkl`: the repository owner confirms these weights were **trained by the owner** using a textbook-style DeepConvNet. They are **not** the official textbook author's pretrained weights. The architecture and reused textbook code remain attributed to Koki Saitoh.
- `results/checkpoints/my_convnet_params.pkl`: the repository owner confirms these weights were **trained by the owner** in their NumPy CNN experiments.
- `results/checkpoints/my_params_cupy.pkl`: the repository owner confirms these weights were **trained by the owner** in their CuPy CNN experiments.

These three training-provenance statements are based on the owner's confirmation, not independent reproduction of the historical training runs.
- `experiments/learning_exercises/ch03/sample_weight.pkl`: the official textbook sample checkpoint, retained under the textbook's MIT License.

## Data and licensing scope

MNIST raw data are not committed to this repository; the loader downloads them separately. The repository owner does not claim authorship or ownership of MNIST.

**No blanket MIT license or other open-source license has been granted for the owner's original contributions.** The original textbook MIT license continues to apply to its own code and sample assets, including portions in adapted scripts. See the [repository-wide copyright notice](../NOTICE.md).
