# Copyright, licensing, and third-party attribution

This is a **mixed-origin educational portfolio**, not a wholly independently authored codebase and not a mirror of the textbook source repository. The presence of a file in this repository does not imply that the repository owner wrote every line of it or owns all rights to it.

## Textbook-derived code and sample weights (MIT License)

Some code, examples, and a sample checkpoint were copied or adapted from:

- Koki Saitoh, *Deep Learning from Scratch* (O'Reilly Japan)
- Official source repository: https://github.com/oreilly-japan/deep-learning-from-scratch
- Original license: **MIT License**; Copyright (c) 2016 Koki Saitoh
- Full original copyright and permission notice: [01-neural-network-foundations/third_party/BOOK_LICENSE.md](01-neural-network-foundations/third_party/BOOK_LICENSE.md)

The original MIT license and copyright notice are retained. The original MIT-licensed material remains subject to that license, including portions incorporated into adapted files. Some files under `01-neural-network-foundations/common/`, `dataset/`, and `experiments/` reproduce or modify textbook source. The official textbook sample checkpoint is `01-neural-network-foundations/experiments/learning_exercises/ch03/sample_weight.pkl`.

For a more detailed inventory, see [the module's third-party notice](01-neural-network-foundations/THIRD_PARTY_NOTICE.md), [file selection notes](01-neural-network-foundations/FILE_SELECTION.md), and [source manifest](01-neural-network-foundations/FILE_MANIFEST.csv).

## Repository owner's contributions

The repository also contains the owner's educational exercises, model training, experiment scripts, analyses, charts, and other project material. Some of these contributions build on or adapt MIT-licensed textbook code; they must not be described as wholly independently implemented where that is not the case.

**No repository-wide open-source license has been granted for the owner's original contributions.** Unless separately stated, rights in those original contributions are reserved by their respective copyright holder(s). This does not limit the rights already granted under the textbook's MIT License to the original MIT-licensed portions.

Making the repository publicly visible is not an assertion that all its contents are the owner's original work, nor a blanket permission to reuse all contents under MIT.

## Model checkpoints

- `01-neural-network-foundations/experiments/learning_exercises/ch03/sample_weight.pkl`: copied from the official MIT-licensed textbook repository; its Git blob matches the official sample checkpoint.
- `01-neural-network-foundations/results/checkpoints/deep_convnet_params.pkl`: the repository owner confirms these weights were **trained by the owner**, using a textbook-style DeepConvNet. They are not the textbook author's pretrained weights. The underlying textbook-derived architecture and code remain attributed to Koki Saitoh.
- `01-neural-network-foundations/results/checkpoints/my_convnet_params.pkl`: the repository owner confirms these weights were **trained by the owner** in their NumPy CNN experiments.
- `01-neural-network-foundations/results/checkpoints/my_params_cupy.pkl`: the repository owner confirms these weights were **trained by the owner** in their CuPy CNN experiments.

The provenance of these three owner-trained checkpoints is based on the owner's confirmation; the historical training runs were not independently reproduced during this attribution review.

Authorship of trained weights does not imply ownership of any third-party training dataset.

## Datasets, figures, and dependencies

MNIST is a third-party dataset used for training and evaluation. Raw MNIST dataset files are **not distributed** in this repository; scripts download the dataset separately when needed. The dataset is not claimed as an original contribution.

The retained PNG figures and CSV files are archived experiment outputs from the owner's project archive, not reproduced textbook pages or figures. External Python dependencies (including NumPy, PyTorch, and CuPy) are installed separately and remain subject to their own licenses.

For any future module, check the origin and license of code, images, data, pretrained models, and other assets before publishing them.
