# Third-party and authorship notice

This curated repository contains material from Koki Saitoh's *Deep Learning from Scratch* source repository, originally published under the MIT License. The original copyright and permission notice are reproduced verbatim in [`third_party/BOOK_LICENSE.md`](third_party/BOOK_LICENSE.md).

**Textbook-derived components**: the shared `common/` package, `dataset/mnist.py`, the baseline `experiments/hyperparameters/two_layer_net.py`, the baseline `experiments/cnn/deep_convnet.py`, `experiments/learning_exercises/ch04/gradient_2d.py`, and `experiments/learning_exercises/ch03/sample_weight.pkl`. Some early learning exercises and training scripts are also adapted from the book, even if edited.

**User experiment code**: the major hyperparameter sweeps, architecture-width/depth experiments, regularisation comparisons, custom CNN, CuPy port, PyTorch implementation, and analysis scripts were identified from this user's archive. "User experiment code" is a classification of work in the archive, not a claim that every line was independently written without using book examples or earlier assistant assistance.

**Checkpoints**: The repository owner confirms that `results/checkpoints/deep_convnet_params.pkl` contains weights produced by their own training of a textbook-style DeepConvNet; it is not claimed to be the book's pretrained checkpoint. The underlying architecture and any reused textbook code remain attributed to Koki Saitoh. The other two files are archived NumPy/CuPy MyConvNet weights. This provenance statement is based on the owner's confirmation, not independent reproduction.

The user has not selected a license for their original contributions. **Do not interpret the retained textbook MIT license as an explicit license for all original material in this curated repository.**
