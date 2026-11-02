# Machine Learning Lab

A personal project exploring machine learning through implementation, experimentation, and analysis.

## About This Project

I started this project while studying the *Deep Learning from Scratch* series by Koki Saitoh.

As I worked through the books and implemented different neural networks, I became curious about many of the design choices involved in machine learning. How much does the learning rate affect training? Why do some network architectures perform better than others? How do different optimisation algorithms influence convergence? And how much difference does the underlying implementation make to training speed?

Rather than simply following textbook examples, I wanted to investigate these questions through my own implementations and experiments.

This repository documents that process, from exploring neural network fundamentals and systematically comparing training behaviour to eventually developing a small machine learning framework of my own.

The project is organised into three main parts.

## Project Structure

### 01 — Neural Network Foundations

**Current focus: MLPs, CNNs, hyperparameter experiments, and computational performance.**

The first part of this project focuses on understanding how neural networks work, how they learn, and how different design choices affect their performance.

#### Multilayer Perceptrons (MLPs)

Using MNIST as a common benchmark, I implement and train fully connected neural networks while investigating the effects of:

- Learning rate and batch size
- Number of hidden layers and neurons per layer
- Network architecture and layer-width patterns
- Optimisation algorithms, including SGD, Momentum, and Adam
- Weight initialisation and activation functions
- Regularisation techniques, including dropout and weight decay
- Training convergence, accuracy, computational cost, and generalisation

The aim is not simply to identify the best-performing configuration, but to understand why different choices lead to different training behaviours.

The project includes exploratory hyperparameter studies, tools for visualising results, and ongoing work towards more systematic and reproducible experiments.

#### Convolutional Neural Networks (CNNs)

I also designed and implemented my own convolutional neural network for MNIST classification.

Beyond building and training the model, I am interested in how the same neural network can be implemented using different computational approaches:

- **NumPy:** implementing neural network operations and forward/backward propagation manually
- **PyTorch:** using automatic differentiation and GPU acceleration
- **CuPy:** exploring GPU-accelerated array operations and their computational performance

Comparing these approaches helps me understand not only neural network algorithms, but also the relationship between software implementation, computational efficiency, and hardware acceleration.

#### Experiments and Analysis

Alongside the implementations, this part of the repository includes experimental results and visualisations examining the relationships between model configuration, training behaviour, and performance.

Earlier exploratory experiments provide a starting point for further investigation. I am extending this work with more comprehensive experiments and clearer evaluation procedures.

The detailed implementations, experiment descriptions, and available results can be found here:

**[Explore Neural Network Foundations](01-neural-network-foundations/README.md)**

### 02 — Natural Language Processing (NLP)

*To be developed.*

### 03 — Machine Learning Framework

*To be developed.*

## Project Philosophy

My goal is to move beyond knowing how to use machine learning models and towards understanding how they work internally.

I approach this through three connected activities:

- **Implementation:** understanding the mathematics and algorithms by building them myself
- **Experimentation:** investigating how design choices influence training and performance
- **Analysis:** interpreting results and connecting theoretical concepts with practical observations

I am particularly interested in understanding not only *what* works, but also *why* it works, when it fails, and what trade-offs are involved.

This repository is an ongoing learning and experimentation project rather than a collection of finished models. Implementations, experiments, and documentation will continue to evolve as I explore new topics.

## References and Attribution

This project was initially inspired by the *Deep Learning from Scratch* series by Koki Saitoh.

Some foundational implementations and utilities are adapted from the textbook's MIT-licensed source code. Other implementations, experiments, analyses, and documentation reflect my own learning and development.

Third-party code and materials retain their respective copyright and licensing terms. My original contributions are not currently offered under a repository-wide open-source license.

See [NOTICE.md](NOTICE.md) and the relevant project-level attribution files for further copyright and licensing information.