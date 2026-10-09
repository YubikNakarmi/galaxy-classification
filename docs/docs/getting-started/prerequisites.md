# Prerequisites

Before running the project locally, make sure your machine has the tools and runtime versions required by the project configuration.

## Required software

The project currently targets Python 3.12.x and declares support for versions from 3.12 up to, but not including, 3.14 in `pyproject.toml`.

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

## Database and storage setup

### MySQL database

The API and Alembic use the `DATABASE_URL` environment variable. Create a MySQL database and user, then add the connection URL to the project root `.env` file:

```dotenv
DATABASE_URL=mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
```

For a local MySQL server, the URL commonly looks like this:

```dotenv
DATABASE_URL=mysql+pymysql://galaxy_user:your_password@localhost:3306/galaxy_classification
```

The `pymysql` driver is included in the project requirements. From the repository root, apply the database migrations:

```bash
alembic upgrade head
```

You can verify the migration state with:

```bash
alembic current
```

Keep the database credentials in `.env` and do not commit that file.

### Azure Blob Storage

For a real Azure Storage account, set its connection string in `.env` and choose the container used for uploaded images:

```dotenv
AZURE_STORAGE_CONNECTION_STRING="<azure-storage-connection-string>"
```

The backend defaults to the `image-uploads` container. Create that container in the storage account before starting the API. If you use DVC or MLflow, configure their containers and environment variables as described in the [configuration guide](configuration.md).

### Azurite for local development

Azurite is included in `docker-compose.yaml` and exposes Blob Storage on port `10000`. Start it with:

```bash
docker compose up -d azureite
```

For applications running on the host machine, use this connection string in `.env`:

```dotenv
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
```

When the API runs inside Docker Compose, use `azureite` instead of `127.0.0.1` in the Blob endpoint:

```dotenv
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azureite:10000/devstoreaccount1;"
```

The Compose service does not currently mount a volume, so Azurite data is temporary and will be lost when its container is removed.


## Recommended next step

After installing dependencies, continue with the local setup docs and then run the project services or CLI according to your use case.
