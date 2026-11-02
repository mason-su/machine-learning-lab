# Third-party and authorship notice

This curated repository contains material from Koki Saitoh's *Deep Learning from Scratch* source repository, originally published under the MIT License. The original copyright and permission notice are reproduced verbatim in [`third_party/BOOK_LICENSE.md`](third_party/BOOK_LICENSE.md).

**Textbook-derived components**: the shared `common/` package, `dataset/mnist.py`, the baseline `experiments/hyperparameters/two_layer_net.py`, the baseline `experiments/cnn/deep_convnet.py`, `experiments/learning_exercises/ch04/gradient_2d.py`, and `experiments/learning_exercises/ch03/sample_weight.pkl`. Some early learning exercises and training scripts are also adapted from the book, even if edited.

**User experiment code**: the major hyperparameter sweeps, architecture-width/depth experiments, regularisation comparisons, custom CNN, CuPy port, PyTorch implementation, and analysis scripts were identified from this user's archive. "User experiment code" is a classification of work in the archive, not a claim that every line was independently written without using book examples or earlier assistant assistance.

**Checkpoints**: `results/checkpoints/deep_convnet_params.pkl` is used as a book-style DeepConvNet baseline; the exact provenance of this checkpoint cannot be established from the archive alone. The other two are archived NumPy/CuPy MyConvNet weights.

The user has not selected a license for their original contributions. **Do not interpret the retained textbook MIT license as an explicit license for all original material in this curated repository.**
