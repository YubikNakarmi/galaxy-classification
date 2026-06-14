# training factors
| Component | Value | Notes |
|---|---|---|
| Loss | `CrossEntropyLoss` | `label_smoothing=0.1` |
| Optimizer | `AdamW` | `lr=1e-3`, `weight_decay=1e-4`, `betas=(0.9, 0.999)` |
| LR Warmup | Linear, 5–10 epochs | LR rises from ~0 to `1e-3` |
| LR Scheduler | Cosine Annealing | Decays smoothly to ~0 after warmup |
| Epochs | 50–100 | No pretrained weights,|
| Batch Size | 32–64 | Balance gradient quality vs memory |
| Grad Clipping | `clip_grad_norm_` | `max_norm=1.0`, before `optimizer.step()` |



## checkpoints
- eval checkpoints
- training epoch based checkpoints

## visualizations
- Mlflow and tensorboard, and reports functions

## Training options

### Default
    Default model variant with standard training settings. With 5 classes and using default model

### Hubble
    Model variant optimized for Hubble Space Telescope images., with 10 hubble classes and defautl model.

### Dirilichet
    Model variant tailored for Dirilichet telescope images. With 10 dirilichet classes and dirilcihlet model.

## Default training model

### Experiments
Each training run should log its own experiment name with ```{time}_{model_variant}```
