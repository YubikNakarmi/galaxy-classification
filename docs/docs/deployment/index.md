# Deployment

This section covers deploying the classification services and their supporting
infrastructure.

- [Azure VM](azure.md): host training and MLflow-related workloads on Azure.
- [Environment variables](env-vars.md): configure storage, databases, MLflow,
  and service URLs.

For local development, start the supporting services with Docker Compose. For
production deployments, review the container and Kubernetes guidance under
[Infrastructure](../infra/index.md).