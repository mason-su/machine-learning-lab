# File selection decisions

The uploaded archive mixed textbook source, personal experiments, outputs, data and caches. Modified timestamps and code content help identify edited files but do **not** prove authorship. This selection preserves the work most useful for a personal GitHub portfolio.

## A — Main personal experiments (keep)

- `ch05/train_2.py` → hidden-size study (8 configurations).
- `ch05/train_3.py` → 3D hyperparameter sweep (150 configurations in current script).
- `ch05/train_4.py` → depth/width/learning-rate/batch-size sweep (96 configurations).
- `ch05/train_5.py` → increasing/decreasing/constant-width architecture study (54 unique configurations).
- `ch05/multi_layer_net.py` → custom multi-hidden-layer network used by train_4/5.
- `ch05/data_visualize.py` → multi-plot hyperparameter analysis.
- `ch06/overfit_dropout.py`, `overfit_weight_decay.py`, `weight_init_compare.py`, `weight_init_activation_histogram.py` → regularisation and initialisation investigations.
- `ch08/awesome_net.py`, `train_deepnet.py`, `my_convnet_with_pytorch.py`, `cupy_myconvnet.py`, `cupy_loss.py` → custom CNN training and acceleration.
- `ch08/nueralnet_weight.py` (renamed `visualize_weights.py`), `misclassified_mnist.py`, `compare_accuracy.py` → weight/error/baseline analysis.

## B — Early edited exercises (keep but de-emphasise)

- `ch01/try.py` → numeric differentiation experiment.
- `ch03/mnist_show.py`, `neuralnet_mnist_batch.py` → MNIST exploration and inference with textbook sample weights.
- `ch04/gradient_method.py`, `gradient_simplenet.py`, `train_neuralnet.py`, `two_layer_net.py` → numerical gradient, training and early MLP work.
- `ch05/train_neuralnet.py` → minor textbook training-script adaptation.
- `common/my_layers.py` → hand-written Affine, ReLU and SoftmaxWithLoss exercise (not used by main experiments; batch-scaling typo corrected in curated copy).

## C — Dependencies / textbook baseline (keep with attribution)

- `common/` (excluding the personal exercise `my_layers.py`), `dataset/mnist.py`, `dataset/__init__.py`.
- `ch05/two_layer_net.py`, `ch04/gradient_2d.py`, `ch03/sample_weight.pkl`, `ch08/deep_convnet.py`.
- `LICENSE.md` retained in `third_party/BOOK_LICENSE.md`.

## D — Original results (keep unmodified)

- All seven CSVs in `ch05/` and **all** files from `ch05/hyperparameter_plots/` (57 PNGs plus five summary CSVs and a TXT).
- `ch08/epoch_accuracy.csv`, `epoch_accuracy.png`, `cupy_training_timing.txt`.
- Three checkpoints: `my_convnet_params.pkl`, `my_params_cupy.pkl`, `deep_convnet_params.pkl`. The owner confirms that all three checkpoints were trained by the owner; `deep_convnet_params.pkl` was not copied from the textbook pretrained checkpoint.

## E — Excluded from curated project

- Original `.git/` metadata and GitHub funding configuration.
- `__pycache__/`, `.pyc`, local downloaded MNIST `.gz` and `dataset/mnist.pkl` (~55 MB), duplicate `ch08/data/MNIST/raw/` files.
- Original book notebooks `notebooks/` and unchanged textbook chapter demonstrations (most `ch01`–`ch07` files), including `ch07/simple_convnet.py`, `ch07/train_convnet.py`, `ch06/hyperparameter_optimization.py`, `ch08/half_float_network.py`.
- Book example checkpoint `ch07/params.pkl`, unused `dataset/lena.png` and `dataset/lena_gray.png`.

See `FILE_MANIFEST.csv` for the full exact selected-file mapping and original SHA-256 values.
