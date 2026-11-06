# Deep Learning from Scratch — Personal Experiments (Pre-NLP)

A curated collection of my **pre-NLP deep-learning experiments** based on *Deep Learning from Scratch* (Koki Saitoh): manually implemented neural networks, MNIST hyperparameter studies, regularisation experiments, and NumPy / PyTorch / CuPy CNN implementations.

This is **not** a mirror of the textbook repository. Personal experiments are separated from retained textbook-derived dependencies, and saved experimental results are kept unchanged. See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) and [FILE_SELECTION.md](FILE_SELECTION.md).

## Project map

| Location | Contents |
| --- | --- |
| `experiments/learning_exercises/` | Early numerical differentiation, gradient descent, basic MNIST and hand-written layers; some textbook adaptations |
| `experiments/hyperparameters/` | Four experiment scripts, a custom `MultiLayerNet`, and 240-grid visualisation |
| `experiments/regularization/` | Dropout, weight decay, and weight-initialisation experiments |
| `experiments/cnn/` | Custom NumPy CNN, PyTorch CNN, CuPy CNN, training, accuracy and weight visualisation |
| `common/`, `dataset/` | **Book-derived** shared modules and MNIST loader needed by the experiments |
| `results/hyperparameters/` | Original CSVs and 57 original hyperparameter charts |
| `results/cnn/` | Original epoch accuracy CSV/PNG and CuPy timing log |
| `results/checkpoints/` | Archived model checkpoints, including an owner-trained textbook-style DeepConvNet baseline |
| `outputs/` | **Generated locally**, ignored by Git, never overwrites archived results |


## New: MNIST Core grid study (6,300 configurations)

The expanded PyTorch MLP study investigates the full factorial grid of **depth × width × batch size × learning rate**, holding network shape constant. It documents **numerical divergence, effective learning, validation performance, overfitting-like loss trajectories, train–validation generalization gaps, late training collapse and accuracy–compute trade-offs**.

**[Read the complete Core study with 15 figures →](experiments/hyperparameters/core_study/README.md)**

Results: **6,300 configurations; 5,169 completed; 1,131 non-finite-loss failures**. Each configuration was run once for up to 20 epochs with a fixed seed. **These are validation results, not independent test-set results.** The original large CSV outputs and local training runner are not yet part of the published repository; the report contains aggregate figure data, not a claim of full end-to-end reproduction.

## Main experiments

1. **Hidden layer size:** `experiments/hyperparameters/train_2.py` → `results/hyperparameters/hidden_size_experiment.csv` (8 configurations). The archived CSV exists, but the current script prints rather than exports results.
2. **Batch size × hidden size × learning rate:** `train_3.py` → 150-config CSV + 2,550 epoch records. A **240-config full-grid summary** and 57 original plots are in `results/hyperparameters/hyperparameter_plots/`.
3. **Depth × width × learning rate × batch size:** `train_4.py` → 96-config CSV + 1,632 epoch records.
4. **Depth × width × width pattern:** `train_5.py` → 54-config CSV + 918 epoch records (one-hidden-layer duplicate patterns are skipped).
5. **Regularisation:** comparison scripts for dropout and weight decay, plus weight-initialisation demonstrations; no saved numeric CSVs were found for these scripts.
6. **CNN:** `awesome_net.py` (NumPy), `my_convnet_with_pytorch.py` (PyTorch), and `cupy_myconvnet.py` (CuPy). `cupy_loss.py` records accuracy/loss by epoch.

### Important: the 150 vs 240 hyperparameter results

The 150 configurations in `hyperparameter_experiment.csv` are an **exact configuration subset** of the 240 configurations in `hyperparameter_plots/00_summary/all_results_ranked.csv`. For all 150 overlapping configurations, initial/final accuracy, best test accuracy and best epoch match exactly. **The recorded `time_seconds` values differ**, so both original files are preserved without merging or rewriting their timing fields. The 150-config CSV also has loss columns and the associated per-epoch history; the 240-config summary contains the full grid and historical plots. The present `train_3.py` reflects the **150-config** grid, not the entire 240-config sweep. The exact historical version of the script that produced all 240 entries was not present in the archive.

Run `python verify_results.py` to confirm this relationship.

## Recorded CNN results

- `results/cnn/epoch_accuracy.csv`: 20 epochs, best recorded test accuracy **99.45%** (epoch 15); epoch 20 **99.38%**.
- `results/cnn/cupy_training_timing.txt`: **389.40 seconds / 6.49 minutes** for the recorded CuPy training run, final test accuracy **99.42%**; entire program **6.56 minutes**.
- Performance numbers from other runs (e.g. approximate NumPy CPU / PyTorch CPU / PyTorch GPU comparisons) are not present as raw timing logs in this archive, so they are not presented here as reproducible benchmark measurements.

The recorded runs do not establish a controlled benchmark: implementations, timing boundaries and evaluation work can differ.

## Setup and execution

Python 3.10+ recommended. In the **repository root**:

```powershell
python -m pip install -r requirements.txt
python verify_results.py
python smoke_test.py
```

The first call to `dataset.mnist.load_mnist` downloads MNIST and creates a local cache in `dataset/`; downloaded dataset files are **not** included in this repository.

**Inspect existing results without training:**

```powershell
python experiments/hyperparameters/data_visualize.py
python experiments/cnn/visualize_weights.py
python experiments/cnn/compare_accuracy.py
python experiments/cnn/misclassified_mnist.py
```

The plotting script regenerates the 240-grid figures under `outputs/hyperparameter_plots_regenerated/` and does **not** overwrite the 57 archived figures. `visualize_weights.py` and the CNN evaluation scripts read archived checkpoints. Only load pickle checkpoints from sources you trust.

**Run CPU experiments (may take a long time):**

```powershell
python experiments/hyperparameters/train_2.py
python experiments/hyperparameters/train_3.py
python experiments/hyperparameters/train_4.py
python experiments/hyperparameters/train_5.py
python experiments/regularization/overfit_dropout.py
python experiments/regularization/overfit_weight_decay.py
python experiments/cnn/train_deepnet.py
```

New CSVs and checkpoints are written to `outputs/`, where applicable. Some early learning scripts display plots interactively.

**PyTorch:**

```powershell
python -m pip install -r requirements-pytorch.txt
python experiments/cnn/my_convnet_with_pytorch.py
```

The PyTorch script automatically selects CUDA when available. For NVIDIA GPU use, install the correct PyTorch CUDA build for your machine.

**CuPy (CUDA 12.x example):**

```powershell
python -m pip install -r requirements-cupy.txt
python experiments/cnn/cupy_myconvnet.py
python experiments/cnn/cupy_loss.py
```

Use a CuPy build matching your CUDA environment; GPU execution was **not** re-tested during this packaging pass.

## Architecture

The custom `MyConvNet` is: Conv(1→32, 3×3) → ReLU → Conv(32→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU → MaxPool → Dense(3136→50) → ReLU → Dropout(0.5) → Dense(50→10). The NumPy and CuPy implementations implement forward/backward operations manually, using book-inspired helper layers and utilities.

## Scope and attribution

Only **pre-NLP** work is included. No CBOW, word embeddings, RNN, Transformer or other second-book NLP work is included.

`common/`, `dataset/mnist.py`, `experiments/hyperparameters/two_layer_net.py`, `experiments/cnn/deep_convnet.py`, `experiments/learning_exercises/ch04/gradient_2d.py`, and the small pretrained example checkpoint are textbook-derived or baseline materials. Other scripts may also be adapted from textbook exercises. See the original book license in `third_party/BOOK_LICENSE.md`. A license for the owner's original contributions has **not** been selected. The three checkpoints under `results/checkpoints/` were trained by the owner, while textbook-style architectures and reused code remain attributed to Koki Saitoh. See the [repository-wide copyright notice](../NOTICE.md).
