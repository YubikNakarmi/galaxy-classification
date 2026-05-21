### DVC setup
```dvc init```
``` az storage blob upload --account-name <storage_account_name> --container-name <container_name> --name <blob_name> --file <file_path>```
```dvc add <file_path>```
```dvc remote add -d azure_remote azure://<storage_account_name>.blob.core.windows.net/<container_name>```
```dvc push```

```az storage container generate-sas \
  --account-name mlopsstorage2026 \
  --name dvc-storage \
  --permissions acdlrw \
  --expiry 2026-12-31 \
  --auth-mode login \
  -o tsv``` 
```dvc remote modify azure_remote sas_token <generated_sas_token>```