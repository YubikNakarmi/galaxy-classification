# MLOps

The MLOps layer manages data versions, experiments, model promotion, serving,
and monitoring across the model lifecycle.

- [DVC data versioning](dvc.md): track datasets and remote Azure Blob storage.
- [MLflow experiment tracking](mlflow.md): record parameters, metrics, and
  artifacts.
- [Model registry](registry.md): register and promote evaluated models.
- [BentoML inference service](bentoml.md): package and serve trained models.
- [Drift monitoring](drift.md): monitor prediction and data behavior over time.

Training configuration and checkpoint behavior are documented in
[ML training](../ml/training.md).