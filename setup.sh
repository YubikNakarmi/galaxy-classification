'#!/bin/bash'

# directory setup
source .env

dir_setup()
{

    mkdir logs models data


}


dvc_setup()
{
    dvc init
    dvc remote add -d azure_remote azure://$AZURE_CONTAINER_NAME
    echo "DVC setup completed with Azure remote storage."

    dvc remote 
}
# DVC setup (Assumes DVC is already installed and Storage account and container are aready setup)
