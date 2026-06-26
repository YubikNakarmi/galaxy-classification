from src.utils.azure_utils import AzureBlobStorage
from pathlib import Path


root = Path(__file__).parent.parent
r = r"checkpoints/acfa1451290b4a8d880870fe4c21557a/best.pt"


helper = AzureBlobStorage(container_name="galaxy-experiments")
helper.load_blob(r, root / "models/temp/best.pt")