

# Machine learning

This section documents the complete machine-learning workflow, from preparing
labels and images through training, evaluation, and explainability.

- [Preprocessing](preprocessing.md): transform labels and prepare dataset
  splits.
- [Model](model.md): classifier architectures and the Dirichlet formulation.
- [Training loop](training.md): run training, resume checkpoints, and evaluate
  MLflow runs.
- [Evaluation and metrics](evaluation.md): compare predictions and report
  performance.
- [Grad-CAM](gradcam.md): inspect image regions that influence predictions.
- [Runs](runs.md): review experiment outputs and run artifacts.

The active profile is [`configs/ml.yaml`](../../../configs/ml.yaml). Use
`galax pipeline train --status` to inspect the resolved settings before a run.
