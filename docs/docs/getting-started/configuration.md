
## Environment Variables


``` title = ".env.template"
AZURE_STORAGE_CONNECTION_STRING="" #in string because of linux env vars for special chars
DVC_AZURE_CONTAINER_NAME=galaxy-classification-data


MYSQLURL_REMOTE= <local/cloud mysql url for mlflow backend store>
MLFLOW_BACKEND_STORE_URI=mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
MLFLOW_ARTIFACT_ROOT=wasbs://<container_name>@<storage_acc>.blob.core.windows.net/
MLFLOW_TRACKING_URI=http://mlflow:5000
LOG_JSON=True/ False

DATABASE_URL = mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>

```

### Azure connections

``` 
AZURE_STORAGE_CONNECTION_STRING="" #in string because of linux env vars for special chars
DVC_AZURE_CONTAINER_NAME=galaxy-classification-data
...
```
These two environment variables are used to connect to the Azure Blob Storage account. The `AZURE_STORAGE_CONNECTION_STRING` variable contains the connection string for the storage account, while the `DVC_AZURE_CONTAINER_NAME` variable specifies the name of the container where the data is stored for dvc containers. 

!!! note
    These can be set locally using [Azurite](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite) or using Azure Storage Explorer.

### Mlflow connections

```
....
MLFLOW_BACKEND_STORE_URI=<db>+<driver>://<username>:<password>@<host>:<port>/<database_name>
MLFLOW_ARTIFACT_ROOT=wasbs://<container_name>@<storage_acc>.blob
....
```
Mlflow requires backend store and artifact root to host a server. Backend store can be set locally or any database, similarly artifact root must be a object store in cloud or locally

!!! note
    Artifact root can be set locally using [Azurite](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite) or using Azure Storage Explorer.

    Similarly, backend store can be set locally using mlflow supported databases.

!!! warning
    Python drivers must be uset in the format of ```<db>+<driver>:// ...```. I used pymysql for mysql database, so the backend store uri is set as ```mysql+pymysql:// ...```

###

# File
