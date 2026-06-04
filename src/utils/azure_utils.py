import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
from pathlib import Path
from kombu.log import setup_logging
from loguru import logger
from yaml_utils import load_yaml
from logging_config import setup_app_logging


root = Path(__file__).resolve().parent.parent.parent
load_dotenv(root / '.env') 
log_config = config = load_yaml(root / 'configs' / 'core.yaml')

setup_app_logging(log_config)



AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_ACCOUNT_URL = os.getenv('AZURE_STORAGE_ACCOUNT_URL') #default env var entries``

class AzureBlobStorage:

    ''' Azure Blob helper module to write,create container, list blobs and delete blobs from Azure Blob Storage.'''

    def __init__(self , container_name=None,account_url = AZURE_STORAGE_ACCOUNT_URL,
                 storage_connection_string = AZURE_STORAGE_CONNECTION_STRING):
        

        if not storage_connection_string:
            raise ValueError("Azure Storage connection string is not set in environment variables.")
        
        
        self.storage_connection_string = storage_connection_string
        self.account_url = account_url
        self.container_name = container_name

        #connect
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str = self.storage_connection_string)

        self.container_client = None #None client

        if container_name:# Not None create client
            self.container_client = self.blob_service_client.get_container_client(
                container_name
            )


    def create_container(self,name=None):
        '''
        Creates a container in Azure Blob Storage if it does not already exist.
        Parameters:
        name (str): The name of the container to be created. If None, uses the container name provided during initialization. Default is None.
        '''

        if name:
            self.container_name = name
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        try:
            self.container_client.create_container()
            logger.info(f"Container '{self.container_name}' created successfully.")
        except Exception as e:
            logger.exception(f"Error creating container: {e}")

    def upload_blob(self, blob_name, data,Overwrite=False):

        ''' Uploads a blob to the specified container.
        
        Parameters:
        blob_name (str): The name of the blob to be created or overwritten.
        data (bytes): The data to be uploaded.
        Overwrite (bool): If True, overwrites the blob if it already exists. Default is False.
        
        '''
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=Overwrite)
            logger.info(f"Blob '{blob_name}' uploaded successfully.")
        except Exception as e:
            logger.exception(f"Error uploading blob: {e}")

   

    def list_blobs(self):
        try:
            blobs = self.container_client.list_blobs()
            blob_names = [blob.name for blob in blobs]
            return blob_names
        except Exception as e:
            logger.exception(f"Error listing blobs: {e}")
            return []
    
    def list_containers(self):
        try:
            containers = self.blob_service_client.list_containers()
            container_names = [container.name for container in containers]
            return container_names
        except Exception as e:
            logger.exception(f"Error listing containers: {e}")
            return []

    def print_info(self):
        logger.info(f"Container Name: {self.container_name}")
        logger.info(f"storage_name: {self.blob_service_client.account_name}")

        for blob in self.list_blobs():
            logger.info(f"Blob Name: {blob}")
        
        for container in self.list_containers():
            logger.info(f"Container Name: {container}")

    


