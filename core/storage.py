from src.utils.azure_utils import AzureBlobStorage
from core.settings import settings


_blob_connection: AzureBlobStorage | None = None

def get_storage() -> AzureBlobStorage:
    ''' For api level persistnace of service client connection to azure blob storage. 
    This is a  pattern to avoid creating multiple connections to the storage service. '''

    global _blob_connection

    if _blob_connection is None:
        _blob_connection = AzureBlobStorage(
            storage_connection_string=settings.azure_storage.connection_string,
            container_name=settings.azure_storage.container_name
        )

        _blob_connection.create_container()  # Ensure the container exists

    return _blob_connection

def close_storage_connection() -> None:
    ''' Close the connection to Azure Blob Storage. '''
    global _blob_connection
    if _blob_connection is not None:
        _blob_connection.blob_service_client.close()
        _blob_connection = None