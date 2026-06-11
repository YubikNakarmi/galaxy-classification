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
-tensorboard with mlflow
