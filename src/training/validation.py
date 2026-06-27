from __future__ import annotations

from pathlib import Path

import mlflow
from loguru import logger


def validate_setup(
    required_paths: list[Path],
    create_dirs: list[Path],
    mlflow_tracking_uri: str,
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


    #paths
    for p in required_paths:
        if not Path(p).exists():
            logger.error(f"Required path missing: {p}")
            ok = False

    if not ok:
        return False

    #create dirs if missing
    for d in create_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready ({len(create_dirs)} dirs)")

    # check mlflow
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Connection Test")
        with mlflow.start_run(run_name="setup_validation"):
            mlflow.log_param("validation", "success")
        logger.info(f"MLflow connection OK ({mlflow_tracking_uri})")
    except Exception as e:
        logger.error(f"MLflow not reachable: {e}")
        return False

    logger.info("All setup checks passed.")
    return True