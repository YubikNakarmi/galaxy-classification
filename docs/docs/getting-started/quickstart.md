# Quickstart Guide
### Clone repo

``` bash
git clone https://github.com/galaxy-classification/galaxy-classification.git
```

### Environment variables

`.env` contains all the environment varibles the format can be found in `.env.example`. You can create your own `.env` file or copy the example file.

=== "uv"

    ```bash title="Install dependencies"
    uv sync --frozen
    ```

=== "pip"

    ```bash title="Install dependencies"
    python3 -m venv .venv
    pip install -r requirements.txt
    ```


### Setup file structure

``` bash title="powershell"
bash -c ". ./setup.sh; dir_setup"
```

``` bash title="bash"
chmod +x ./setup.sh
(source ./setup.sh && dir_setup)
```

##

### Setup DVC remote storage


For setting up DVC with Azure Blob Storage, you need to configure your DVC remote storage. This involves specifying the connection string and container name in your `.env` file, and then modifying your DVC remote configuration to use these values.

A second option is hosting azure blob storage emulator like Azurite locally, which can be useful for development and testing without incurring Azure costs.

``` sh title="DVC Azure Remote Setup Commands" hl_lines="5"
dvc init
dvc remote add -d azure_remote azure://$AZURE_CONTAINER_NAME
dvc remote modify azure_remote account_name $AZURE_STORAGE_ACCOUNT_NAME   

export AZURE_STORAGE_CONNECTION_STRING=$(echo $AZURE_STORAGE_CONNECTION_STRING) #or
#dvc remote modify azure_remote --local connection_string $AZURE_STORAGE_CONNECTION_STRING 

# --local is used to avoid commiting the secret
dvc pull
echo "DVC setup completed with Azure remote storage."
```


If you want to init dvc for your project and setup remote storage, you can run the following commands. Make sure to replace the placeholders with your actual values.

``` bash title="powershell"
bash -c ". ./setup.sh; dvc_setup"
```

``` bash title="bash
(source ./setup.sh && dvc_setup)
```
