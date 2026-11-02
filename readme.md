# Machine Learning Lab

A collection of my machine learning implementations, experiments, and personal projects.

This repository documents my learning and practical exploration of machine learning, from neural network fundamentals to advanced AI topics.

## Projects

### 01 — Neural Network Foundations
Neural network implementations, hyperparameter experiments, CNNs, and CPU/GPU performance comparisons using NumPy, PyTorch, and CuPy.

*More projects and experiments will be added as my learning progresses.*

## Featured experiment results

The [neural network foundations project](01-neural-network-foundations/README.md) includes a unified runner for four historical hyperparameter studies and [24 new charts with a findings report](01-neural-network-foundations/results/hyperparameters/analysis/findings.md), based on archived MNIST experiments.

- **240 hyperparameter combinations:** best recorded final test accuracy **98.03%**.
- **96 depth/width configurations:** under the tested initialization and training setup, all 5- and 10-hidden-layer models ended at **11.35%** test accuracy, highlighting a training limitation worth investigating.
- **54 width-pattern configurations:** compare constant, increasing and decreasing hidden-layer widths.

These are historical exploratory results, not controlled benchmarks; the test set was also used for hyperparameter monitoring.

## Copyright and attribution

This repository includes textbook-derived MIT-licensed code and sample weights alongside my own experiments. See [NOTICE.md](NOTICE.md) for copyright ownership, third-party attribution, and license scope. My original contributions are not currently offered under a repository-wide open-source license.
