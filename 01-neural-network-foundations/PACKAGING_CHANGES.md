# Packaging changes (code only)

All CSVs, original figures, checkpoints and the book license were copied without alteration. Source code was adjusted to make relocated scripts usable and to avoid overwriting the original results.

- `experiments/hyperparameters/train_2.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/hyperparameters/train_3.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/hyperparameters/train_4.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/hyperparameters/train_5.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/hyperparameters/multi_layer_net.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/regularization/overfit_dropout.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/regularization/overfit_weight_decay.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/regularization/weight_init_compare.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/awesome_net.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/train_deepnet.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/cupy_myconvnet.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/cupy_loss.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/misclassified_mnist.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/compare_accuracy.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/cnn/deep_convnet.py` — Use script-relative project-root import instead of working-directory-dependent sys.path
- `experiments/hyperparameters/train_5.py` — Fix missing comment marker in source encoding declaration
- `experiments/learning_exercises/ch03/neuralnet_mnist_batch.py` — Fix project root for relocated early learning exercise
- `experiments/learning_exercises/ch03/mnist_show.py` — Fix project root for relocated early learning exercise
- `experiments/learning_exercises/ch04/gradient_simplenet.py` — Fix project root for relocated early learning exercise
- `experiments/learning_exercises/ch04/two_layer_net.py` — Fix project root for relocated early learning exercise
- `experiments/learning_exercises/ch04/train_neuralnet.py` — Fix project root for relocated early learning exercise
- `experiments/learning_exercises/ch05/train_neuralnet.py` — Use shared two-layer baseline and reliable project root
- `experiments/learning_exercises/ch05/my_layers.py` — Correct one-hot SoftmaxWithLoss backward scaling and imports; source had /dout instead of /batch_size
- `experiments/hyperparameters/train_3.py` — Write new experiment CSVs under outputs/hyperparameters instead of alongside archived originals
- `experiments/hyperparameters/train_4.py` — Write new experiment CSVs under outputs/hyperparameters instead of alongside archived originals
- `experiments/hyperparameters/train_5.py` — Write new experiment CSVs under outputs/hyperparameters instead of alongside archived originals
- `experiments/hyperparameters/data_visualize.py` — Read full historical 240-grid dataset and write regenerated plots to separate outputs directory
- `experiments/cnn/train_deepnet.py` — Write trained NumPy checkpoint to outputs, leaving archived checkpoint unchanged
- `experiments/cnn/cupy_myconvnet.py` — Write CuPy weights and timings to outputs/cnn/cupy
- `experiments/cnn/cupy_loss.py` — Write new epoch accuracy plots and CSV to outputs instead of overwriting original results
- `experiments/cnn/compare_accuracy.py` — Use archived NumPy and baseline checkpoints via stable absolute paths
- `experiments/cnn/misclassified_mnist.py` — Use archived NumPy checkpoint via stable path; remove unused baseline import
- `experiments/cnn/visualize_weights.py` — Default to archived NumPy checkpoint using stable path; rename typo in filename
- `experiments/cnn/my_convnet_with_pytorch.py` — Store downloaded torchvision MNIST under ignored outputs/mnist_torch
