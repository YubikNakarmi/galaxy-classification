from mlflow import MlflowClient
import mlflow
from dotenv import load_dotenv
from pathlib import Path
import os
root = Path(__file__).parent.parent.parent
load_dotenv(root / ".env")  # Load environment variables from .env file


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

client.set_terminated(
    run_id="9aec5b204b7840dab69ec9c8250148f6",
    status="KILLED"
)