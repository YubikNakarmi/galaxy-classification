### DVC setup
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