# Prerequisites

Before running the project locally, make sure your machine has the tools and runtime versions required by the project configuration.

## Required software

The project currently targets Python 3.12.x and declares support for versions from 3.12 up to, but not including, 3.14 in [pyproject.toml](../../pyproject.toml).

- Python 3.12.x
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Git
- Docker and Docker Compose for local infrastructure services
- Azure CLI for DVC/Azure integration
- Optional: [Azurite](https://github.com/Azure/Azurite) for local blob emulation

## Install Python

If Python 3.12 is not already available, install it with your preferred method. For example, using `pyenv`:

```bash
pyenv install 3.12.11
pyenv local 3.12.11
```

## Create a virtual environment

This project uses modern Python packaging and optional dependency groups. The recommended workflow is with `uv`.

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create and activate the environment
uv venv
source .venv/bin/activate
```

## Install project dependencies

Install the project and the dependency groups you need. The project defines extras such as `api`, `ml`, `bentoml`, `dev`, and `docs`.

### Install everything

```bash
uv sync --all-extras --frozen
```

### Install only the API stack

```bash
uv sync --extra api
```

### Install training and ML tooling

```bash
uv sync --extra ml
```

### Install docs dependencies

```bash
uv sync --extra docs
```

## Alternative: pip-based setup

If you prefer pip instead of uv, create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[api,ml,bentoml,dev,docs]"
```

You can also install the base requirements file directly:

```bash
pip install -r requirements.txt
```

## Optional local services

Some workflows rely on supporting services that are not required for every task:

- DVC for dataset/model versioning
- Azure Blob Storage or Azurite for remote storage integration
- MLflow for experiment tracking
- Docker Compose for the app and observability stack

## Recommended next step

After installing dependencies, continue with the local setup docs and then run the project services or CLI according to your use case.
