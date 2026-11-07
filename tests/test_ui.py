import logging

from src.cli.style import UI
from src.utils.logging_config import suppress_mlflow_console_logs


def test_status_runs_body_when_level_is_hidden():
    ui = UI(verbose=0, log=False)

    with ui.status("hidden", level=1) as status:
        status.update("still hidden")


def test_suppress_mlflow_console_logs_keeps_warnings_visible():
    mlflow_logger = logging.getLogger("mlflow")
    previous_level = mlflow_logger.level
    try:
        suppress_mlflow_console_logs()
        assert mlflow_logger.level == logging.WARNING
    finally:
        mlflow_logger.setLevel(previous_level)
