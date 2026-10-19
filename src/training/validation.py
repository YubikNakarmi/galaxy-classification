from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
import os,base64
import mlflow
from mlflow.exceptions import MlflowException
from loguru import logger
from dotenv import load_dotenv

root = Path(__file__).parent.parent.parent
try:
    load_dotenv(root / ".env")
    
except Exception as e:
    logger.error(f"Error loading .env file: {e}")




def validate_mlflow(mlflow_tracking_uri: str) -> bool:
    stage = "setup"
    try:
        user = (os.environ.get("MLFLOW_TRACKING_USERNAME") or "").strip()
        pwd = (os.environ.get("MLFLOW_TRACKING_PASSWORD") or "").strip()
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
        
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

        logger.info(f"URI={mlflow_tracking_uri!r} user_set={bool(user)} pwd_set={bool(pwd)}")

        headers = {"User-Agent": "galaxy-classification-validator"}
        if user and pwd:
            cred = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            headers["Authorization"] = f"Basic {cred}"

        stage = "health check"
        request = Request(f"{mlflow_tracking_uri.rstrip('/')}/health",
                          headers=headers, method="GET")
        with urlopen(request, timeout=10):
            pass

        stage = "set_experiment"
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Connection Test")

        stage = "start_run/log_param"
        with mlflow.start_run(run_name="setup_validation"):
            mlflow.log_param("validation", "success")

        logger.info(f"MLflow connection OK ({mlflow_tracking_uri})")
        return True

    except Exception as e:
        logger.error(f"MLflow failed at [{stage}]: {type(e).__name__}: {e}")
        return False


    
    
def validate_setup(
    required_paths: list[Path],
    mlflow_tracking_uri: str,
    create_dirs: list[Path]| None =None,
) -> bool:

    """
    Validate training environment before any training code runs.

    Checks:
    - Required input paths exist (data files, log dir)
    - Output directories are created if missing
    - MLflow tracking server is reachable

    Args:
        required_paths: paths that must already exist (train csv, val csv, image dir, log dir)
        create_dirs:    directories to create if missing (checkpoints, reports, tensorboard)
        mlflow_tracking_uri: MLflow server URI to test connection against

    Returns:
        True if all checks pass, False otherwise.
    """
    ok = True


    # paths that must already exist
    for p in required_paths:
        p = Path(p)
        if not p.exists():
            logger.error(f"Required path missing: {p}")
            ok = False

    if not ok:
        return False

    #create dirs if missing
    if create_dirs:
        for d in create_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directories ready ({len(create_dirs)} dirs)")
        
    if not validate_mlflow(mlflow_tracking_uri=mlflow_tracking_uri):
        return False

    logger.info("All setup checks passed.")
    return True





# validate_mlflow(mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))