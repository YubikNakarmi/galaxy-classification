import mlflow
from dotenv import load_dotenv
import os

load_dotenv("../env")
uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(uri)
mlflow

