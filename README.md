### DVC setup
```dvc init```
``` az storage blob upload --account-name <storage_account_name> --container-name <container_name> --name <blob_name> --file <file_path>```
```dvc add <file_path>```
```dvc remote add -d azure_remote azure://<container_name>```
``` dvc remote modify azure_remote account_name <storage_account_name>```
``` ddvc remote modify myremote --local connection_string "DefaultEndpointsProtocol=https;AccountName=<storage_account_name>;AccountKey=<storage_account_key>;EndpointSuffix=core.windows.net"```
 
```dvc remote modify azure_remote sas_token <generated_sas_token>```