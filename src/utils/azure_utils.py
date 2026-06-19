"""
Azure Blob Storage helper.

Provides a wrapper around the Azure Blob Storage SDK for creating
containers, uploading files, listing blobs, and deleting blobs.

Example:
    >>> storage = AzureBlobStorage("images")
    >>> storage.upload_blob("cat.jpg", image_bytes)
"""


import os, uuid
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_app_logging


root = Path(__file__).resolve().parent.parent.parent
load_dotenv(root / '.env') 

log_config = load_yaml(root / 'configs' / 'core.yaml')

setup_app_logging(log_config)



AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_ACCOUNT_URL = os.getenv('AZURE_STORAGE_ACCOUNT_URL') #default env var entries``

class AzureBlobStorage:

    """Client for interacting with Azure Blob Storage.

    This class provides a high-level wrapper around the Azure Storage Blob SDK
    for managing containers and blobs. It simplifies common operations such as
    creating containers, uploading blobs, listing resources, and deleting blobs.

    Args:
        container_name: Name of the container to connect to. If omitted, no
            container is selected until one is specified.
        account_url: Azure Storage account URL.
        storage_connection_string: Azure Storage connection string used for
            authentication.

    Raises:
        ValueError: If the storage connection string is not provided.

    Example:
        >>> storage = AzureBlobStorage("images")
        >>> storage.list_blobs()
    """

    def __init__(self , container_name=None,account_url = AZURE_STORAGE_ACCOUNT_URL,
                 storage_connection_string = AZURE_STORAGE_CONNECTION_STRING):
        """
        Initialize an Azure Blob Storage client.

        Parameters
        ----------
        container_name : str, optional
            Container to connect to immediately.
        account_url : str, optional
            Storage account URL.
        storage_connection_string : str
            Azure Storage connection string.

        Raises
        ------
        ValueError
            If the connection string is missing.
        """

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


    def check_container_exists(self, name=None):
            
        """Check whether a container exists.

        Args:
            name: Container name to check. If omitted, the currently configured
                container is used.

        Returns:
            True if the container exists; otherwise False.
        """

        container_name = name or self.container_name
        client = self.blob_service_client.get_container_client(container_name)
        try:
            return client.exists()
        except Exception as e:
            logger.exception(f"Error checking container existence: {e}")
            return False
     
    def create_container(self,name=None,unique_suffix=False):
        """Create a new container if it does not already exist.

        Args:
            name: Name of the container to create. If omitted, uses the current
                container.
            unique_suffix: Whether to append a UUID suffix to the container name.

        Notes:
            If the container already exists, no action is taken.

        Example:
            >>> storage.create_container("datasets")
        """

        if self.check_container_exists(name):
            logger.info(f"Container '{name}' already exists.")
            return
        
        if name:
            self.container_name = name
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        if unique_suffix:
            self.container_name += f"-{uuid.uuid4()}"
        try:
            self.container_client.create_container()
            logger.info(f"Container '{self.container_name}' created successfully.")
        except Exception as e:
            logger.exception(f"Error creating container: {e}")


    def upload_blob(self, blob_name, data,Overwrite=False)-> str:

        """Upload a blob to the current container.

        Args:
            blob_name: Name of the destination blob.
            data: Bytes or file-like object to upload.
            overwrite: Whether to overwrite an existing blob.

        Returns:
            URL of the uploaded blob if the upload succeeds; otherwise ``None``.

        Example:
            >>> with open("image.jpg", "rb") as f:
            ...     storage.upload_blob("image.jpg", f)
        """

        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=Overwrite)
            logger.info(f"Blob '{blob_name}' uploaded successfully.")

            return str(blob_client.url)
        
        except Exception as e:
            logger.exception(f"Error uploading blob: {e}")
            return None
        
    def load_blob(self,blob_name):
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            

    def list_blobs(self):

        """List all blobs in the current container.

        Returns:
            A list containing the names of all blobs. Returns an empty list if
            the operation fails.
        """

        try:
            blobs = self.container_client.list_blobs()
            blob_names = [blob.name for blob in blobs]
            return blob_names
        except Exception as e:
            logger.exception(f"Error listing blobs: {e}")
            return []
    
    def list_containers(self):

        """List all blobs in the current container.

        Returns:
            A list containing the names of all blobs. Returns an empty list if
            the operation fails.

        """
        try:
            containers = self.blob_service_client.list_containers()
            container_names = [container.name for container in containers]
            return container_names
        except Exception as e:
            logger.exception(f"Error listing containers: {e}")
            return []

    def delete_blob(self, blob_name):

        """Delete a blob from the current container.

        Args:
            blob_name: Name of the blob to delete.
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            logger.info(f"Blob '{blob_name}' deleted successfully.")
        except Exception as e:
            logger.exception(f"Error deleting blob: {e}")

    def print_info(self):

        """Log information about the current storage account.

        Logs the active container, storage account name, available blobs, and
        available containers.
        """

        logger.info(f"Container Name: {self.container_name}")
        logger.info(f"storage_name: {self.blob_service_client.account_name}")

        for blob in self.list_blobs():
            logger.info(f"Blob Name: {blob}")
        
        for container in self.list_containers():
            logger.info(f"Container Name: {container}")

    
