# Default model architecture

`GalaxyClassifier_ResNet` is a wide pre-activation ResNet with bottleneck residual units,
adapted for 128×128 galaxy images.

### Image


### Key parameters (default)

| Parameter | Value |
|---|---|
| Input size | 128 × 128 × 3 |
| Width multiplier `k` | 2 |
| Units per group `N` | 2 |
| Output classes | 5 |
| FC layer size | 4096 (2048 × k) |

### Layer-by-layer summary

| Stage | Module | Output shape | Notes |
|---|---|---|---|
| **Stem** | Conv2d 3→64, 6×6 s=2 p=2 | 64 × 64 × 64 | Large kernel for galaxy texture |
| | MaxPool2d 2×2 s=2 | 64 × 32 × 32 | Early aggressive downsampling |
| **Group 2** | ConvGroup 64 → 512k | 512 × 16 × 16 | Mid: 128k; last unit stride=2 |
| **Group 3** | ConvGroup 512k → 1024k | 1024 × 8 × 8 | Mid: 256k; last unit stride=2 |
| **Group 4** | ConvGroup 1024k → 2048k | 2048 × 4 × 4 | Mid: 512k; last unit stride=2 |
| **Group 5** | ConvGroup 2048k → 4096k | 4096 × 4 × 4 | Mid: 1024k; no downsampling |
| **Head** | AdaptiveAvgPool2d(1,1) | 4096 × 1 × 1 | Variable input size safe |
| | Dropout p=0.2 | 4096 | After flatten |
| | Linear 4096 → `num_classes` | 5 | Raw logits; softmax at inference |

### Design decisions

| Choice | Detail |
|---|---|
| Pre-activation order | BN → ReLU → Conv (vs. standard post-activation) |
| Bottleneck block | 1×1 expand → 3×3 spatial → 1×1 project; dropout after the 3×3 |
| Downsampling | Stride=2 on `conv2` of last unit in each group; shortcut via 1×1 conv when dims change |
| Initialisation | Kaiming normal for conv layers; constant 1/0 for BN weight/bias |
| Domain augmentation | `RandomRotation(180)` — galaxies have no canonical orientation |
| Normalisation stats | GZ2-specific mean/std, not ImageNet |

# Hubble variant
Just training the head of the default model on Hubble images (with the same 5 classes) yields a model that performs poorly on Galaxy Zoo images. The `hubble` variant is a modified architecture that is more suitable for Hubble images, with a smaller width multiplier and fewer units per group.

# Dirlichlet variant
The dirichlet variant uses multi head output that ouputs a vector with dirichlet alphas. For training this model we have to use a dirichlet multinominal negative log likelyhood loss function. Since pythorch doesn't have this I had to customly code this function. 