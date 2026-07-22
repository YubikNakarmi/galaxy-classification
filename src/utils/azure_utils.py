"""
Azure Blob Storage helper.

Provides a wrapper around the Azure Blob Storage SDK for creating
containers, uploading files, listing blobs, and deleting blobs.

Example:
    ``` python
    with AzuleBlobStorage("images") as storage:
        storage.create_container("images")
    ```
    
"""

import os, uuid
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
from typing import BinaryIO,Optional
from typing import Optional


root = Path(__file__).resolve().parent.parent.parent
load_dotenv(root / '.env') 


_AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING') #default env var entries``

class AzureBlobStorage:

    """ Client for interacting with Azure Blob Storage.

    This class provides a high-level wrapper around the Azure Storage Blob SDK
    for managing containers and blobs. It simplifies common operations such as
    creating containers, uploading blobs, listing resources, and deleting blobs.


    Example:
        ``` python
        with AzureBlobStorage("images") as storage:
            storage.create_container("images") 
            storage.upload_blob("image.jpg", b"image data")
            storage.print_info()
        ```
    """

    def __init__(self , container_name:Optional[str] = None, 
                 storage_connection_string:Optional[str] = _AZURE_STORAGE_CONNECTION_STRING):
        """
        Initialize an Azure Blob Storage client.

        Args:
            container_name: Name of the container to connect to. If omitted, no container is selected until one is specified.
            account_url: Azure Storage account URL.
            storage_connection_string: Azure Storage connection string used for authentication.

        Raises:
            ValueError: If the storage connection string is not provided.
            RuntimeError: If the container name is not provided when required
            ------
       
        """

        if not storage_connection_string:
            raise ValueError("Azure Storage connection string is not set in environment variables.")
       
        self.__storage_connection_string = storage_connection_string
        self.container_name = container_name

        #connect
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_str = self.__storage_connection_string)

        self.container_client = None #None client

        if container_name:# Not None create client
            self.container_client = self.blob_service_client.get_container_client(
                container_name
                
            )
    #context managers
    def __enter__(self):
        logger.info("AzureBlobStorage client initialized...")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Closing AzureBlobStorage client...")
        self.blob_service_client.close()
        
    #Main methods
    def check_container_exists(self, name:str)-> bool:
            
        """Check whether a container exists.

        Args:
            name: Container name to check. If omitted, the currently configured container is used.

        Returns:
            True if the container exists; otherwise False.
        """

        container_name = name or self.container_name
        if container_name is None:
            raise RuntimeError(
                "Container name must be specified either in the constructor or as an argument."
            )
        client = self.blob_service_client.get_container_client(container_name)
        try:
            return client.exists()
        except Exception as e:
            logger.exception(f"Error checking container existence: {e}")
            return False
     
    def create_container(self,name:Optional[str] = None,unique_suffix:bool=False):
        """Create a new container if it does not already exist.

        Args:
            name: Name of the container to create. If omitted, uses the current
                container.
            unique_suffix: Whether to append a UUID suffix to the container name.

        Notes:
            If the container already exists, no action is taken.

        Example:
            ``` python
                storage.create_container("datasets")
            ```
        """
        container_name = name or self.container_name
        if container_name is None:
            logger.exception("Container name must be specified either in the constructor or as an argument.")
            return

        if unique_suffix:
            container_name = container_name + f"-{uuid.uuid4()}"

        if self.check_container_exists(container_name):
            self.container_name = container_name
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            logger.info(f"Container '{container_name}' already exists.")
            return

        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        try:
            self.container_client.create_container()
            logger.info(f"Container '{self.container_name}' created successfully.")
        except Exception as e:
            logger.exception(f"Error creating container: {e}")


    def upload_blob(self, blob_name:str, 
                    data:bytes | BinaryIO | str | Path,
                    Overwrite:bool=True)-> Optional[str]:

        """Upload a blob to the current container.

        Args:
            blob_name: Name of the destination blob.
            data: Bytes, a file-like object, or a local file path.
            overwrite: Whether to overwrite an existing blob.

        Returns:
            URL of the uploaded blob if the upload succeeds; otherwise ``None``.

        Example:
            ```python
            with open("image.jpg", "rb") as f:
                 storage.upload_blob("image.jpg", f)
            ```

        """

        try:
            container_client = self._ensure_container_client()
            blob_client = container_client.get_blob_client(blob_name)


            if isinstance(data, (str, Path)):
                with open(data, "rb") as f:
                    blob_client.upload_blob(f, overwrite=Overwrite)

            else: #option to upload blob directly
                blob_client.upload_blob(data, overwrite=Overwrite)

            logger.info(f"Blob '{blob_name}' uploaded successfully.")

            return str(blob_client.url) #returns uri of blob
        
        except Exception as e:
            logger.exception(f"Error uploading blob: {e}")
            return None
        
    def load_blob(self, blob_name:str, 
                  local_path:Optional[str | Path] = None) -> Optional[str|bytes|BinaryIO]:
        '''Download a blob from the current container to a local file.

        Args:
            blob_name: Name of the blob to download.
            local_path: Local file path to save the downloaded blob.

        Returns:
            The URL of the downloaded blob if successful; otherwise None.

        Example:
            ```python
             storage.load_blob("image.jpg", "local_image.jpg")
            ```
        '''

        try:
           
            container_client = self._ensure_container_client()
            blob_client = container_client.get_blob_client(blob_name)

            if local_path is None:
                logger.warning("No local path provided. just returning blob")
                return blob_client.download_blob().readall()
            
            else:
                with open(local_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                logger.info(f"Blob '{blob_name}' downloaded to {local_path}")

                return str(blob_client.url) 
         
             # Return the blob URL for reference
        except Exception as e:
            logger.exception(f"Error downloading blob: {e}")
            return None

    def list_blobs(self):

        """List all blobs in the current container.

        Returns:
            A list containing the names of all blobs. Returns an empty list if
            the operation fails.
        """

        try:
            container_client = self._ensure_container_client()
            blobs = container_client.list_blobs()
            blob_names = [blob.name for blob in blobs]
            return blob_names
        except Exception as e:
            logger.exception(f"Error listing blobs: {e}")
            return []
    
    def list_containers(self)->Optional[list[str]]:

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
            return None

    def delete_blob(self, blob_name:str):

        """Delete a blob from the current container.

        Args:
            blob_name: Name of the blob to delete.
        """
        try:
            container_client = self._ensure_container_client()
            blob_client = container_client.get_blob_client(blob_name)
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
        
        container_names = self.list_containers()
        if container_names is not None:
            for container in container_names:
                logger.info(f"Container Name: {container}")
                
    #helpers
    def _ensure_container_client(self) -> ContainerClient:
        if self.container_client is None:
            if not self.container_name:
                raise RuntimeError(
                    "AzureBlobStorage has no container configured. Pass container_name to the constructor or call create_container(name=...)."
                )
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        return self.container_client
