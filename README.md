# DVC setup
```sh
dvc init
```

```sh
az storage blob upload --account-name <storage_account_name> --container-name <container_name> --name <blob_name> --file <file_path>
```


```sh
dvc add <file_path>
```
```sh
dvc remote add -d azure_remote azure://<container_name>
```
```sh
dvc remote modify azure_remote account_name <storage_account_name>
```
```sh
dvc remote modify myremote --local connection_string "DefaultEndpointsProtocol=https;AccountName=<storage_account_name>;AccountKey=<storage_account_key>;EndpointSuffix=core.windows.net"
```

```sh
dvc remote modify azure_remote sas_token <generated_sas_token>
```

# File setup
 make dirs `/logs`,`/models`,`/data` in the root of the project and add .gitkeep files to them.

# Azure blob setup
1. Create a storage account in Azure. or emulate using Azurite
2. Create a container in the storage account.
3. Generate a SAS token or Connection string for authentication.

## Required containers
 - `mlflow` (for mlflow tracking backend)
 - `dvc-storage` (for dvc remote storage)
 - 