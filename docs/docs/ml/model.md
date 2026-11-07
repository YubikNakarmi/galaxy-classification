## Default model architecture

`GalaxyClassifier_ResNet` is a wide pre-activation ResNet with bottleneck residual units, adapted for 128×128 galaxy images. This model is largly based off the ResNet architecture[@Zhu_2019,], with the following modifications:

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

## Hubble variant
Just training the head of the default model on Hubble images (with the same 5 classes) yields a model that performs poorly on Galaxy Zoo images. The `hubble` variant is a modified architecture that is more suitable for Hubble images, with a smaller width multiplier and fewer units per group.

## Dirlichlet variant

### Dirichlet Loss

The **Dirichlet loss** can be derived from the **Dirichlet-multinomial distribution**, which arises by integrating a multinomial likelihood over a Dirichlet prior. See the Dirichlet-multinomial likelihood formulation[@yu2014efficient] for further background.

#### 1. Observed Vote Counts

For an input sample, let the observed vote counts across \(K\) classes be

$$
\mathbf{k} = (k_1, k_2, \ldots, k_K),
\qquad
n = \sum_{i=1}^{K} k_i,
$$

where \(k_i\) is the number of votes assigned to class \(i\), and \(n\) is the total number of votes.

We model the underlying class probabilities using a Dirichlet distribution with parameters

$$
\mathbf{\alpha}
=
(\alpha_1, \alpha_2, \ldots, \alpha_K),
\qquad
S = \sum_{i=1}^{K} \alpha_i.
$$

Here, \(\alpha_i > 0\) are the concentration parameters of the Dirichlet distribution.

---

#### 2. Multinomial Likelihood

Given a vector of class probabilities

$$
\mathbf{p} = (p_1, p_2, \ldots, p_K),
\qquad
\sum_{i=1}^{K}p_i = 1,
$$

the probability of observing the vote counts \(\mathbf{k}\) is given by the multinomial distribution:

$$
P(\mathbf{k}\mid\mathbf{p})
=
\frac{\Gamma(n+1)}
{\prod_{i=1}^{K}\Gamma(k_i+1)}
\prod_{i=1}^{K}p_i^{k_i}.
$$

For integer-valued counts, this is equivalent to the familiar form

$$
P(\mathbf{k}\mid\mathbf{p})
=
\frac{n!}{\prod_{i=1}^{K}k_i!}
\prod_{i=1}^{K}p_i^{k_i}.
$$

---

#### 3. Dirichlet Distribution

Instead of treating \(\mathbf{p}\) as fixed, we place a Dirichlet prior over the class probabilities:

$$
P(\mathbf{p}\mid\mathbf{\alpha})
=
\frac{\Gamma(S)}
{\prod_{i=1}^{K}\Gamma(\alpha_i)}
\prod_{i=1}^{K}p_i^{\alpha_i-1}.
$$

The Dirichlet distribution is defined over the probability simplex, where

$$
p_i \geq 0,
\qquad
\sum_{i=1}^{K}p_i = 1.
$$

---

#### 4. Dirichlet-Multinomial Distribution

The Dirichlet-multinomial distribution is obtained by marginalizing the class probabilities \(\mathbf{p}\):

$$
P(\mathbf{k}\mid\mathbf{\alpha})
=
\int
P(\mathbf{k}\mid\mathbf{p})
P(\mathbf{p}\mid\mathbf{\alpha})
\,d\mathbf{p}.
$$

Substituting the multinomial likelihood and Dirichlet distribution gives

$$
P(\mathbf{k}\mid\mathbf{\alpha})
=
\frac{\Gamma(n+1)}
{\prod_i\Gamma(k_i+1)}
\frac{\Gamma(S)}
{\Gamma(n+S)}
\prod_i
\frac{\Gamma(k_i+\alpha_i)}
{\Gamma(\alpha_i)}.
$$

For integer vote counts, this can also be written as

$$
P(\mathbf{k}\mid\mathbf{\alpha})
=
\frac{n!}{\prod_i k_i!}
\frac{\Gamma(S)}
{\Gamma(n+S)}
\prod_i
\frac{\Gamma(k_i+\alpha_i)}
{\Gamma(\alpha_i)}.
$$

This distribution gives the probability of observing the complete vote-count vector \(\mathbf{k}\) given the Dirichlet parameters \(\mathbf{\alpha}\).

---

#### 5. Negative Log-Likelihood

During training, we minimize the negative log-likelihood:

$$
\mathcal{L}(\mathbf{\alpha})
=
-\log P(\mathbf{k}\mid\mathbf{\alpha}).
$$

Taking the logarithm of the Dirichlet-multinomial likelihood gives

$$
\begin{aligned}
\mathcal{L}(\mathbf{\alpha})
={}&
-\log\Gamma(n+1)
+
\sum_i\log\Gamma(k_i+1)
\\
&-\log\Gamma(S)
+
\log\Gamma(n+S)
\\
&-
\sum_i
\left[
\log\Gamma(k_i+\alpha_i)
-
\log\Gamma(\alpha_i)
\right].
\end{aligned}
$$

The first two terms,

$$
-\log\Gamma(n+1)
+
\sum_i\log\Gamma(k_i+1),
$$

depend only on the observed vote counts \(\mathbf{k}\) and not on the model parameters \(\mathbf{\alpha}\). Therefore, they are constant with respect to the model and can be omitted during optimization.

The resulting loss is

$$
\boxed{
\mathcal{L}(\mathbf{\alpha})
=
\log\Gamma(n+S)
-
\log\Gamma(S)
-
\sum_{i=1}^{K}
\left[
\log\Gamma(k_i+\alpha_i)
-
\log\Gamma(\alpha_i)
\right]
}
$$

up to an additive constant independent of \(\mathbf{\alpha}\).

#### 6. Why This Works for Vote-Based Labels

Unlike standard classification, where the target is typically represented as a single class label, vote-based datasets provide a **distribution of votes** across classes.

For example, with five classes, an observation might have

$$
\mathbf{k} = (2, 5, 1, 0, 2),
\qquad n=10.
$$

The Dirichlet-multinomial likelihood models these counts directly. The neural network therefore learns Dirichlet parameters \(\mathbf{\alpha}\) that describe the underlying uncertainty over the class probabilities.

The corresponding expected class probabilities are

$$
\mathbb{E}[p_i]
=
\frac{\alpha_i}{S}.
$$

Thus, the model provides both:

* **Class probabilities:** determined by the normalized parameters
  \(\alpha_i/S\).
* **Uncertainty:** influenced by the total concentration
  \(S=\sum_i\alpha_i\).

A larger \(S\) corresponds to a more concentrated Dirichlet distribution, while a smaller \(S\) corresponds to a more diffuse distribution.



