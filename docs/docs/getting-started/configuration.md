
# Dvc Config
 For setting up DVC with Azure Blob Storage, you need to configure your DVC remote storage. This involves specifying the connection string and container name in your `.env` file, and then modifying your DVC remote configuration to use these values.

 A second option is hosting azure blob storage emulator like Azurite locally, which can be useful for development and testing without incurring Azure costs.

 ```sh
  dvc init
    dvc remote add -d azure_remote azure://$AZURE_CONTAINER_NAME
    dvc remote modify azure_remote account_name $AZURE_STORAGE_ACCOUNT_NAME   

    export AZURE_STORAGE_CONNECTION_STRING=$(echo $AZURE_STORAGE_CONNECTION_STRING) #or
    #dvc remote modify azure_remote --local connection_string $AZURE_STORAGE_CONNECTION_STRING 
    
    # --local is used to avoid commiting the secret
    dvc pull
    echo "DVC setup completed with Azure remote storage."
```

# File Dirs
