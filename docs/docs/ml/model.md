# Models

## WRN-26-2 ([Zhu et al., 2019](https://arxiv.org/abs/1807.10406))


### Model Overview

This architecture is a custom deep residual convolutional neural network designed for 5-class galaxy image classification.  
The network follows a bottleneck-style ResNet structure with residual skip connections, progressive spatial downsampling, batch normalization, dropout regularization, and global average pooling.

---

### Architecture Summary

| Property | Value |
|---|---|
| Input Resolution | 64 × 64 |
| Architecture Type | Residual CNN (Bottleneck ResNet) |
| Output Classes | 5 |
| Activation | ReLU |
| Normalization | Batch Normalization |
| Regularization | Dropout |
| Pooling | Max Pool + Global Average Pool |
| Final Classifier | Fully Connected + Softmax |

---

## Detailed Layer Configuration

### 1. Stem Block

| Layer | Operation | Kernel Size | Stride | Padding | Output Shape | Description |
|---|---|---|---|---|---|---|
| Conv1 | Conv2D | 6×6 | 1 | same | 64×64×64 | Initial feature extraction |
| MaxPool | Max Pooling | 2×2 | 2 | 0 | 32×32×64 | Spatial downsampling |

---

### 2. Residual Block Group — Conv2

#### Residual Unit 2.1

| Layer | Operation | Kernel | Stride | Channels | Output Shape | Notes |
|---|---|---|---|---|---|---|
| Conv2.1a | 1×1 Conv | 1×1 | 1 | 128 | 32×32×128 | Bottleneck reduction |
| Conv2.1b | 3×3 Conv | 3×3 | 1 | 128 | 32×32×128 | Spatial feature learning |
| Conv2.1c | 1×1 Conv | 1×1 | 1 | 512 | 32×32×512 | Channel expansion |
| Skip2.1 | Projection Shortcut | 1×1 | 1 | 512 | 32×32×512 | Dimension matching |

### Notes
- Batch Normalization and ReLU applied before convolutions
- Dropout applied after middle 3×3 convolution

---

#### Residual Unit 2.2 (Downsampling)

| Layer | Operation | Kernel | Stride | Channels | Output Shape | Notes |
|---|---|---|---|---|---|---|
| Conv2.2a | 1×1 Conv | 1×1 | 1 | 128 | 32×32×128 | Bottleneck reduction |
| Conv2.2b | 3×3 Conv | 3×3 | 1 | 128 | 32×32×128 | Feature extraction |
| Conv2.2c | 1×1 Conv | 1×1 | 2 | 512 | 16×16×512 | Spatial downsampling |
| Skip2.2 | Projection Shortcut | 1×1 | 2 | 512 | 16×16×512 | Residual alignment |

---

### 3. Residual Block Group — Conv3

#### Residual Unit 3.1

| Layer | Operation | Kernel | Stride | Channels | Output Shape |
|---|---|---|---|---|---|
| Conv3.1a | 1×1 Conv | 1×1 | 1 | 256 | 16×16×256 |
| Conv3.1b | 3×3 Conv | 3×3 | 1 | 256 | 16×16×256 |
| Conv3.1c | 1×1 Conv | 1×1 | 1 | 1024 | 16×16×1024 |
| Skip3.1 | Projection Shortcut | 1×1 | 1 | 1024 | 16×16×1024 |

---

#### Residual Unit 3.2 (Downsampling)

| Layer | Operation | Kernel | Stride | Channels | Output Shape |
|---|---|---|---|---|---|
| Conv3.2a | 1×1 Conv | 1×1 | 1 | 256 | 16×16×256 |
| Conv3.2b | 3×3 Conv | 3×3 | 1 | 256 | 16×16×256 |
| Conv3.2c | 1×1 Conv | 1×1 | 2 | 1024 | 8×8×1024 |
| Skip3.2 | Projection Shortcut | 1×1 | 2 | 1024 | 8×8×1024 |

---

### 4. Residual Block Group — Conv4

#### Residual Unit 4.1

| Layer | Operation | Kernel | Stride | Channels | Output Shape |
|---|---|---|---|---|---|
| Conv4.1a | 1×1 Conv | 1×1 | 1 | 512 | 8×8×512 |
| Conv4.1b | 3×3 Conv | 3×3 | 1 | 512 | 8×8×512 |
| Conv4.1c | 1×1 Conv | 1×1 | 1 | 2048 | 8×8×2048 |
| Skip4.1 | Projection Shortcut | 1×1 | 1 | 2048 | 8×8×2048 |

---

#### Residual Unit 4.2 (Downsampling)

| Layer | Operation | Kernel | Stride | Channels | Output Shape |
|---|---|---|---|---|---|
| Conv4.2a | 1×1 Conv | 1×1 | 1 | 512 | 8×8×512 |
| Conv4.2b | 3×3 Conv | 3×3 | 1 | 512 | 8×8×512 |
| Conv4.2c | 1×1 Conv | 1×1 | 2 | 2048 | 4×4×2048 |
| Skip4.2 | Projection Shortcut | 1×1 | 2 | 2048 | 4×4×2048 |

---

### 5. Residual Block Group — Conv5

#### Residual Unit 5.1

| Layer | Operation | Kernel | Stride | Channels | Output Shape |
|---|---|---|---|---|---|
| Conv5.1a | 1×1 Conv | 1×1 | 1 | 1024 | 4×4×1024 |
| Conv5.1b | 3×3 Conv | 3×3 | 1 | 1024 | 4×4×1024 |
| Conv5.1c | 1×1 Conv | 1×1 | 1 | 4096 | 4×4×4096 |
| Skip5.1 | Projection Shortcut | 1×1 | 1 | 4096 | 4×4×4096 |

---

#### Residual Unit 5.2

| Layer | Operation | Kernel | Stride | Channels | Output Shape |
|---|---|---|---|---|---|
| Conv5.2a | 1×1 Conv | 1×1 | 1 | 1024 | 4×4×1024 |
| Conv5.2b | 3×3 Conv | 3×3 | 1 | 1024 | 4×4×1024 |
| Conv5.2c | 1×1 Conv | 1×1 | 1 | 4096 | 4×4×4096 |
| Skip5.2 | Identity Shortcut | — | — | 4096 | 4×4×4096 |

---

### 6. Classification Head

| Layer | Operation | Parameters | Output Shape | Description |
|---|---|---|---|---|
| AvgPool | Global Average Pooling | 4×4 | 1×1×4096 | Spatial aggregation |
| Dropout | Dropout | Keep Prob = 0.8 | 1×1×4096 | Regularization |
| FC | Fully Connected | 5 outputs | 5 | Final classifier |
| Softmax | Softmax Activation | — | 5 | Probability distribution |

---

# Architectural Characteristics

## Residual Learning
The network employs residual shortcut connections to:
- Improve gradient propagation
- Reduce vanishing gradient problems
- Enable deeper architectures

## Bottleneck Design
Each residual unit follows:
1. 1×1 reduction convolution
2. 3×3 feature extraction convolution
3. 1×1 expansion convolution

This reduces computational complexity while preserving representational power.

## Downsampling Strategy
Spatial resolution reduction occurs progressively:
- 64×64 → 32×32 → 16×16 → 8×8 → 4×4

## Regularization
Dropout layers are inserted within bottleneck units and before the classifier head to reduce overfitting.

---

# Final Output

The model produces:
- A 5-dimensional probability vector
- Corresponding galaxy class prediction via softmax activation

## WRN-26-2 with Dirichlet Output Head 
