# Infrastructure

The infrastructure documentation describes how the project runs locally and in
deployed environments.

- [Containers](containers.md): service images and required environment values.
- [Docker Compose](docker-compose.md): local service orchestration.
- [Kubernetes](k8.md): cluster manifests and ingress setup.
- [Training infrastructure](train.md): the MLflow and training VM workflow.
- [Observability](observability.md): metrics, dashboards, and log collection.
- [CI/CD](cicd.md): automated testing and documentation workflows.

Choose Docker Compose for a reproducible local stack and Kubernetes when
independent service scaling and cluster scheduling are required.