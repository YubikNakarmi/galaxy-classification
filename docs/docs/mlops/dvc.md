
## Configurations
For DVC I used `Azule Blob Storage` as the remote storage for data and model versioning. The configuration is stored in the `.dvc/config` file. The following is an example of the configuration:

```ini 
['remote "azure_remote"']
    url = azure://<container_name>
    account_name = <storage_account_name>
    connection_string = DefaultEndpointsProtocol=https;AccountName=<storage_account_name>;AccountKey=<storage_account_key>;EndpointSuffix=core.windows.net
```

Check connection
```sh
dvc pull
```

### Alternate Configurations
If you wannt ot use locally you can use emulated azure blob storage using [Azurite](https://github.com/Azure/Azurite). 