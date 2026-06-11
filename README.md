 Linear Mode Connectivity and Model Merging

Permutation Alignment, REPAIR, and Multi-Model Fusion across Architectures.

Course project for **Deep Learning and Applied AI** (M.Sc. in Computer Science, Sapienza University of Rome).

Authors: Alberto Rivas Casal and Carlos Fernandez Fernandez (Erasmus exchange from the University of Oviedo).

Code: https://github.com/carlosgrl/DeepLearning-AppliedAI

## Overview

Independently trained networks that reach the same low loss are usually separated by a high loss barrier along their linear interpolation, even though they live in a single connected basin once the permutation symmetry of hidden units is taken into account. This project is a unified, reproducible study of linear mode connectivity (LMC) and weight-space model merging across three architecture/dataset pairs, each chosen to stress a different part of the theory:

- MLP-3 on KMNIST: no normalisation, pure permutation symmetry.
- SimpleConvBN on SVHN: BatchNorm handling and activation variance collapse.
- MobileNetV3-small on EuroSAT: a realistic backbone for task arithmetic.

The pipeline (i) reduces the interpolation barrier by neuron alignment (weight matching, activation matching, Procrustes), (ii) explains and corrects the variance collapse of interpolated activations with REPAIR, (iii) visualises the 2D loss landscape before and after alignment, (iv) scales merging to n in {3, 5, 8} models with anchor and iterative alignment, and (v) studies task arithmetic and TIES-merging.
