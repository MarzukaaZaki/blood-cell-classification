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

📊 Performance Leaderboard (1% Labels, 300 Epochs)

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

⚙️ Core Implementations

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

🚀 How to Reproduce

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

📃 References

  - [1] K. Sohn, D. Berthelot, C.-L. Li, Z. Zhang, N. Carlini, E. D. Cubuk, A.
    Kurakin, H. Zhang, and C. Raffel, "FixMatch: Simplifying Semi-Supervised
    Learning with Consistency and Confidence," Advances in Neural Information
    Processing Systems (NeurIPS), vol. 33, pp. 596–608, 2020.
  - [2] D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and C.
    Raffel, "MixMatch: A Holistic Approach to Semi-Supervised Learning,"
    Advances in Neural Information Processing Systems (NeurIPS), vol. 32,
    pp. 5049–5059, 2019.
  - [3] Q. Xie, Z. Dai, E. Hovy, T.-M. Luong, and Q. V. Le, "Unsupervised Data
    Augmentation for Consistency Training," Advances in Neural Information
    Processing Systems (NeurIPS), vol. 33, pp. 6277–6288, 2020.
  - [4] B. Zhang, Y. Wang, W. Hou, H. Wu, J. Wang, M. Okumura, and T. Shinozaki,
    "FlexMatch: Boosting Semi-Supervised Learning with Curriculum
    Pseudo-Labeling," Advances in Neural Information Processing Systems
    (NeurIPS), vol. 34, pp. 18408–18419, 2021.
  - [5] Y. Wang, H. Chen, Q. Heng, W. Hou, Y. Fan, Z. Wu, J. Wang, M. Savvides,
    T. Shinozaki, B. Raj, B. Schiele, and X. Xie, "FreeMatch: Self-adaptive
    Thresholding for Semi-Supervised Learning," in International Conference on
    Learning Representations (ICLR), 2023.
  - [6] D. Berthelot, N. Carlini, E. D. Cubuk, A. Kurakin, K. Sohn, H. Zhang,
    and C. Raffel, "ReMixMatch: Semi-Supervised Learning with Distribution
    Matching and Augmentation Anchoring," in International Conference on
    Learning Representations (ICLR), 2020.
  - [7] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, "RandAugment: Practical
    Automated Data Augmentation with a Reduced Search Space," in Proceedings of
    the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
    (CVPRW), pp. 702–703, 2020.
  - [8] D. Tellez, G. Litjens, P. Bándi, W. Bulten, J.-M. Bokhorst, F. Ciompi,
    and J. van der Laak, "Quantifying the Effects of Data Augmentation and Stain
    Color Normalization in Convolutional Neural Networks for Computational
    Pathology," Medical Image Analysis, vol. 58, p. 101544, 2019.
  - [9] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic
    Models," Advances in Neural Information Processing Systems (NeurIPS),
    vol. 33, pp. 6840–6851, 2020.
  - [10] A. Kazerouni, E. K. Aghdam, M. Heidari, R. Azad, M. Fayyaz, I.
    Hacihaliloglu, and D. Merhof, "Diffusion Models in Medical Imaging: A
    Comprehensive Survey," Medical Image Analysis, vol. 88, p. 102846, 2023.
  - [11] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image
    Recognition," in Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), pp. 770–778, 2016.
  - [12] R. Wightman, "PyTorch Image Models," GitHub, 2019. [Online]. Available:
    https://github.com/huggingface/pytorch-image-models
  - [13] J.-C. Su, Z. Cheng, and S. Maji, "A Realistic Evaluation of
    Semi-Supervised Learning for Fine-Grained Classification," in Proceedings of
    the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    pp. 12966–12976, 2021.
  - [14] S. C. Huang, A. Pareek, R. Zamanian, I. Banerjee, and M. P. Lungren,
    "ReVisiting Semi-supervised Learning for Medical Image Classification," in
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition Workshops (CVPRW), 2023.
