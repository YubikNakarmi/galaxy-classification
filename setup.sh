'#!/bin/bash'

# directory setup
source .env

dir_setup()
{

     echo "Creating project directories..."
    mkdir -p 
    
             logs/{reports,system,tensorboard,training} \
             models/{checkpoints,temp} \
             data/{raw,preprocessed,split} \
             .venv
}

}

#DVC setup (Assumes DVC is already installed and Storage account and container are aready setup)

dvc_setup() 
{
    dvc init
    dvc remote add -d azure_remote azure://$AZURE_CONTAINER_NAME
    dvc remote modify azure_remote account_name $AZURE_STORAGE_ACCOUNT_NAME   

    export AZURE_STORAGE_CONNECTION_STRING=$(echo $AZURE_STORAGE_CONNECTION_STRING) #or
    #dvc remote modify azure_remote --local connection_string $AZURE_STORAGE_CONNECTION_STRING 
    
    # --local is used to avoid commiting the secret
    dvc pull
    echo "DVC setup completed with Azure remote storage for root"

    cd data

    dvc pull
    echo "Data pulled for /data"
        
}

dir_setup
