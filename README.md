Domain-Aware and Diffusion-Augmented FixMatch for Semi-Supervised Hematology Image Classification

📌 Project Overview

This repository hosts a semi-supervised learning (SSL) framework designed for
peripheral blood cell classification on the BloodMNIST dataset under extreme
label scarcity. Recognizing that manual annotation of microscopic blood smears
is a slow, costly, and structurally bottlenecked clinical task, we evaluate deep
neural networks under a highly challenging 1% labeled split—representing only
about 15 labeled images per class.

Built on top of the robust FixMatch baseline, the proposed methodology
introduces two key innovations tailored specifically to the biological,
physical, and optical properties of peripheral blood smears:

1.  Hema-Aug (Domain-Aware Strong Augmentation): A custom three-layer
    augmentation pipeline designed to preserve clinical color semantics (Giemsa
    staining) and cellular morphology while introducing biologically plausible
    spatial, morphological, and occlusion variations.
2.  Stochastic Diffusion-Based Augmentation: A regularization policy that
    replaces hand-crafted coordinate distortions with a mathematically grounded
    closed-form Denoising Diffusion Probabilistic Model (DDPM) forward noise
    process inside the active consistency loop.

🔬 Key Scientific Discoveries

  - The Labeled Memorization Trap: Under extreme data scarcity (1% labels),
    brute-force oversampling via a class-weighted sampler degrades long-term SSL
    performance. Because the model only has 15 unique minority cells,
    oversampling forces the model to memorize those exact 15 images, destroying
    its test-set generalization. Removing the sampler forces the model to rely
    on the large unlabeled pool, acting as a massive regularizer.
  - The DA / Sampler Mathematical Conflict: While standard FixMatch benefits
    from Distribution Alignment (DA), the integration of class-weighted sampling
    creates a uniform batch update that directly conflicts with the imbalanced
    prior in the DA equation, causing prediction distortion and over-correction.
  - The Pure Noise Failure Mode: Standard isotropic diffusion noise collapses
    the unsupervised consistency loss toward zero, indicating trivial
    consistency satisfaction. Hybridizing diffusion noise with Hema-Aug's
    geometric and morphological operations resolves this failure by forcing
    spatial invariance on top of pixel noise.

## Performance Leaderboard (1% Labels, 300 Epochs)

The table below compiles our final, 300-epoch, pretrained ResNet-18 test-set
results, showing the progression of our ablation study:

| Method / Configuration                        | Test Accuracy      | Macro Precision  | Macro Recall     | Macro F1-Score    | Monocyte F1-Score |
| :-------------------------------------------- | :----------------: | :--------------: | :--------------: | :---------------: | :---------------: |
| **Supervised Baseline (1%)**                  | $86.73\%$          | $0.850$          | $0.850$          | $0.8460$          | $0.630$           |
| **Standard FixMatch (1%)**                    | $93.07\%$          | $0.934$          | $0.923$          | $0.9322$          | $0.800$           |
| **FixMatch + Hema-Aug Only**                  | $\mathbf{95.21\%}$ | $\mathbf{0.950}$ | $\mathbf{0.950}$ | $\mathbf{0.9475}$ | $\mathbf{0.890}$  |
| **FixMatch + Diffusion ($t=300$)**            | $92.28\%$          | $0.912$          | $0.912$          | $0.9115$          | $0.720$           |
| **FixMatch + Hema-Aug + Diffusion ($t=300$)** | **$94.45\%$**      | **$0.939$**      | **$0.939$**      | **$0.9390$**      | **$0.860$**       |
| *Supervised Upper Bound (10% Labeled)*        | *$97.46\%$*        | *$0.980$*        | *$0.970$*        | *$0.9748$*        | *$0.940$*         |

## Core Implementations

1. Hema-Aug: Domain-Aware Augmentation Policy

Hema-Aug is implemented using a deterministic sequential pipeline and wrapped in
a custom PyTorch adapter class to handle real-time PIL-to-Tensor formatting.

The pipeline contains specific clinical configurations:

  - Rotation with Reflection: Uses a background-reflection border mode to mirror
    background pink/white smear colors, avoiding introducing black border
    artifacts.
  - Strict Hue Boundary: Restricts color jitter hue shifts strictly to a low
    threshold (\pm 0.1 radians) to preserve the distinct staining color profiles
    of eosinophilic and basophilic granules.
  - Morphological Deformation: Uses subtle elastic transformations to simulate
    natural membrane flattening and stretching during smear preparation.

2. Diffusion-Based Augmentation: Mathematical Noise Schedule

This method replaces hand-crafted strong augmentations with the closed-form
forward diffusion process derived from the linear noise schedule in Denoising
Diffusion Probabilistic Models (DDPM). This closed-form calculation requires no
model training or reverse-denoising overhead during active training loops,
keeping training fast.

3. Preventing Cloud Instance Memory Leaks & Process Crashes

Due to container boundaries, multi-process data loading easily triggers shared
memory and RAM Out-of-Memory restarts. We engineered two safeguards:

  - Memory-Safe DataLoader: We set the dataloader worker processes to zero. This
    completely bypasses the shared partition and avoids the Linux Copy-on-Write
    (CoW) page-duplication memory leak in PyTorch multiprocess forks.
  - Non-Caching Cycle Generator: Standard cyclic iterators cache every yielded
    tensor in memory, causing RAM usage to expand over 300 epochs. We replace
    them with a lightweight, non-caching custom generator.

### How to Reproduce

1. Environment Setup

Install the necessary dependencies in your local Python environment: medmnist,
albumentations, timm, diffusers, accelerate, scikit-learn, pandas, numpy,
matplotlib, and seaborn.

2. Prepare the Data

Ensure BloodMNIST (MedMNIST v2) is loaded and preprocessed at 128 \times 128
resolution. The dataset will automatically download to your cache directory upon
the first dataloader instantiation.

3. Run Training & Evaluation

Open and execute your target notebooks sequentially. Start with your unweighted
pretrained baseline runs, progress to your class-weighted adaptive runs, and
complete your diffusion-augmented runs. Each of these evaluation steps will
dynamically write your training history, final leaderboards, and raw test
predictions to local CSV files.

Finally, load these CSVs into a centralized plotting notebook to render
high-resolution validation curves, per-class F1 comparisons, and delta confusion
matrices.

## References

[1] Sohn, K., et al. "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." Advances in Neural Information Processing Systems (NeurIPS), 2020. Link
[2] MedMNIST v2 Benchmark. Nature Scientific Data, 2023. Link
[3] Wang, Y., et al. "FreeMatch: Self-Adaptive Thresholding for Semi-Supervised Learning." International Conference on Learning Representations (ICLR), 2023. Link
[4] Ho, J., et al. "Denoising Diffusion Probabilistic Models." Advances in Neural Information Processing Systems (NeurIPS), 2020. Link
[5] Albumentations. Fast and Flexible Image Augmentations, 2020. Link
[6] timm (PyTorch Image Models). Hugging Face, 2019. Link