

# need to add stack 
- hydra or omegaconf for config management.


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



## containers 
- dvc-storage
- mlflow
- galaxy-classification-experiments
- galaxy-classification-logs




# improvement
- cascading module loading problem while running scropts. Since the cutom moduels also realy on other modules the runtime is slow especially during importing of modules. This is not really a problem for ml (as they are mostly training eval, etc which take considerable time anyway).part of the projoect. But thsi can be improved. 


# vm trianin

do nvdia-smi 

```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

ssh -i .\test-vm_key.pem -L 5000:localhost:5000 azureuser@20.204.16.165
```

