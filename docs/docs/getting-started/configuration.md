
## Environment Variables


``` title = ".env.template"
AZURE_STORAGE_CONNECTION_STRING="" #in string because of linux env vars for special chars
DVC_AZURE_CONTAINER_NAME=galaxy-classification-data

ROOT_DIR= D:\galaxy classification

MYSQLURL_REMOTE= <local/cloud mysql url>
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
These two environment variables are used to connect to the Azure Blob Storage account. The `AZURE_STORAGE_CONNECTION_STRING` variable contains the connection string for the storage account, while the `DVC_AZURE_CONTAINER_NAME` variable specifies the name of the container where the data is stored.


# File
