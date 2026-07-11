from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import mlflow
from loguru import logger


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

    # check mlflow
    try:
        request = Request(
            f"{mlflow_tracking_uri.rstrip('/')}/health",
            headers={"User-Agent": "galaxy-classification-validator"},
            method="GET",
        )
        with urlopen(request, timeout=2) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Connection Test")
        with mlflow.start_run(run_name="setup_validation"):
            mlflow.log_param("validation", "success")
        logger.info(f"MLflow connection OK ({mlflow_tracking_uri})")
    except (URLError, HTTPError, TimeoutError, RuntimeError, OSError) as e:
        logger.error(f"MLflow not reachable: {e}")
        return False

    logger.info("All setup checks passed.")
    return True