# Copyright, licensing, and third-party attribution

This repository is a **mixed-origin educational portfolio**, not a wholly original codebase or a mirror of the textbook source. Publishing it does not mean that every file was independently authored by the repository owner.

## 1. Textbook source (MIT License)

Some files reproduce or adapt code and example assets from:

- **Koki Saitoh**, *Deep Learning from Scratch* (O'Reilly Japan).
- Official source: https://github.com/oreilly-japan/deep-learning-from-scratch
- Original license: **MIT License**, Copyright (c) 2016 Koki Saitoh.
- **Full original copyright and permission notice:** [01-neural-network-foundations/third_party/BOOK_LICENSE.md](01-neural-network-foundations/third_party/BOOK_LICENSE.md).

The original MIT license permits redistribution and modification subject to retaining its copyright and permission notice. The MIT license continues to apply to the original textbook-derived material, including portions incorporated into adapted scripts.

See the module's [detailed third-party notice](01-neural-network-foundations/THIRD_PARTY_NOTICE.md) and [file selection record](01-neural-network-foundations/FILE_SELECTION.md) for examples and provenance notes. Some experiments build on textbook examples; descriptions such as “personal experiment” do **not** mean that every line is independently authored.

## 2. Other contributions

The repository also includes experimental code, analysis, figures, and checkpoints from the repository owner's project archive, sometimes built upon the textbook material.

**No separate open-source license has yet been granted for the repository owner's original contributions.** Unless otherwise stated, rights to such contributions are reserved by their respective copyright holders. This does **not** restrict rights already granted by the textbook's MIT license for its own code and assets.

Do not assume that the textbook MIT license applies to every file in this repository, or that the repository owner claims ownership of third-party material.

## 3. Data and dependencies

MNIST training/test data are downloaded separately when scripts are run; the dataset files are **not redistributed** in this repository. Third-party Python libraries are installed separately and retain their own licenses.

## 4. Model checkpoint provenance

The repository owner confirms that `01-neural-network-foundations/results/checkpoints/deep_convnet_params.pkl` contains **model weights produced by their own training**, using a textbook-style DeepConvNet implementation. It is not presented as the textbook author's pretrained checkpoint. The architecture and any reused or adapted source code remain attributed to the textbook under its MIT License. Ownership of the trained weights does **not** imply ownership of any underlying third-party training dataset.

This provenance statement is based on the owner's confirmation; the training run itself has not been independently reproduced as part of this licensing review.

For future modules, review all third-party code, datasets, images, and model weights before publishing and record applicable licenses and credits.
