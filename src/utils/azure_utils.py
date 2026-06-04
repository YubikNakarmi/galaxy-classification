import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
from pathlib import Path


root = Path(__file__).parent.parent
load_dotenv(root / '.env') 

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_ACCOUNT_URL = os.getenv('AZURE_STORAGE_ACCOUNT_URL') #default env var entries

class AzureBlobStorage:

    ''' Azure Blob helper module to write,create container, list blobs and delete blobs from Azure Blob Storage.'''

    def __init__(self , container_name,account_url = AZURE_STORAGE_ACCOUNT_URL,
                 storage_connection_string = AZURE_STORAGE_CONNECTION_STRING):
        

        if not storage_connection_string:
            raise ValueError("Azure Storage connection string is not set in environment variables.")
        
        
        self.storage_connection_string = storage_connection_string
        self.account_url = account_url
        self.container_name = container_name


        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str = self.storage_connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
    
    def create_container(self):
        try:
            self.container_client.create_container()
            print(f"Container '{self.container_name}' created successfully.")
        except Exception as e:
            print(f"Error creating container: {e}")