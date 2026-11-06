# Machine Learning Lab

A personal machine-learning and deep-learning experiment repository, with an emphasis on **implementing models, running controlled experiments, and analyzing training behavior**.

## Projects

### [01 — Neural Network Foundations](01-neural-network-foundations/README.md)

Hands-on neural-network implementations, MNIST experiments, regularization and convolutional-network work. See the project README for implementation details, dependencies, historical results and attribution.

**Featured study: [6,300-run MNIST MLP Core hyperparameter grid](01-neural-network-foundations/experiments/hyperparameters/core_study/README.md)**

A systematic four-factor study of network depth, hidden-layer width, mini-batch size and learning rate, including:

- Numerical stability and parameter interactions
- Effective learning and convergence behavior
- Final validation accuracy and learning-rate/batch-size trade-offs
- Train–validation generalization gaps and overfitting-like behavior
- Late training collapse and accuracy–training-time trade-offs

The study includes **15 linked figures**, methodological caveats and aggregated figure data. **All reported accuracies in the Core report are validation accuracies, not independent test accuracies.**

## Reproducibility and attribution

Refer to each project's README for its setup, experiment instructions, source-code provenance and third-party notices. Archived experiments and the new Core report are distinct; the Core report does not imply that its local training runner or full raw outputs are committed.

See [NOTICE.md](NOTICE.md) for repository-wide attribution information.
