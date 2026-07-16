#!/bin/bash

# directory setup

if [ -f .env ]; then
    source .env
fi

dir_setup()
{

     echo "Creating project directories..."
    mkdir -p 
    
             logs/{reports,system,tensorboard,training} \
             models/{checkpoints,temp} \
             data/{raw,preprocessed,split} \
             logs/{reports,system,tensorboard}





#DVC setup (Assumes DVC is already installed and Storage account and container are aready setup)

dvc_setup() 


{


    dvc init
    dvc remote add -d azure_remote azure://$AZURE_CONTAINER_NAME
    dvc remote modify azure_remote account_name $AZURE_STORAGE_ACCOUNT_NAME   

    #export AZURE_STORAGE_CONNECTION_STRING=$(echo $AZURE_STORAGE_CONNECTION_STRING) #or
    dvc remote modify azure_remote --local connection_string $AZURE_STORAGE_CONNECTION_STRING 
    
    # --local is used to avoid commiting the secret
    
        
}

bentoml_build()
{
    try:
    echo "Building BentoML container..."
    bentoml build --version 0.6.0 --containerize || bentoml update --version 0.6.0 --containerize

}

