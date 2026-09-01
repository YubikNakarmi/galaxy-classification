## Environment Variables

```dotenv title=".env.template"
AZURE_STORAGE_CONNECTION_STRING="" # In quotes because Linux environment variables may contain special characters
DVC_AZURE_CONTAINER_NAME=galaxy-classification-data

MYSQLURL_REMOTE=<local/cloud MySQL URL for MLflow backend store>

MLFLOW_BACKEND_STORE_URI=mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
MLFLOW_ARTIFACT_ROOT=wasbs://<container_name>@<storage_acc>.blob.core.windows.net/
MLFLOW_TRACKING_URI=http://mlflow:5000

LOG_JSON=True/False

DATABASE_URL=mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
```

### Azure Connections

```dotenv
AZURE_STORAGE_CONNECTION_STRING=""
DVC_AZURE_CONTAINER_NAME=galaxy-classification-data

...
```

These two environment variables are used to connect to the **Azure Blob Storage** account.

* `AZURE_STORAGE_CONNECTION_STRING` contains the connection string for the Azure Storage account.
* `DVC_AZURE_CONTAINER_NAME` specifies the name of the container where DVC stores the project data.

!!! note

```
These can be configured locally using [Azurite](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite) or Azure Storage Explorer.
```

### MLflow Connections

```dotenv
MLFLOW_BACKEND_STORE_URI=<db>+<driver>://<username>:<password>@<host>:<port>/<database_name>

MLFLOW_ARTIFACT_ROOT=wasbs://<container_name>@<storage_account>.blob.core.windows.net/
```

MLflow requires a **backend store** and an **artifact store** when hosting a tracking server.

The backend store is used to store MLflow metadata such as experiments, runs, parameters, and metrics. It can be a locally hosted or remote database supported by MLflow.

The artifact store is used to store files produced by MLflow runs, such as model artifacts and other outputs. It can use a cloud object store or local storage.

!!! note

    
    The artifact store can be configured locally using [Azurite](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite) or Azure Storage Explorer.

    Similarly, the backend store can be configured locally using any [database supported by MLflow](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/).
    

!!! warning

    Database connection URLs must use the following format:

    ```text
    <database>+<driver>://<username>:<password>@<host>:<port>/<database_name>
    ```

    For example, when using MySQL with the PyMySQL driver:

    ```text
    mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
    ```

    Here, `mysql` specifies the database type and `pymysql` specifies the Python database driver.
