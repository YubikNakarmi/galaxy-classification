''' Parser and overwritting default logging with file stream configuration

- setup_app_logging: for application logs
- setup_ml_logging: for machine learning logs

Example:

```python
    from src.utils.logging_config import setup_app_logging, setup_ml_logging
    setup_app_logging(config)
````
'''

from loguru import logger
from pathlib import Path
import sys
import logging


root = Path(__file__).resolve().parent.parent.parent


_FMT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
    "{extra[service]} | {name}:{function}:{line} - {message}"
)
_FMT_CONSOLE = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[service]}</cyan> | {message}"
)


def _configure(cfg: dict, service: str, json_logs: bool):
    logger.remove()
    logger.configure(extra={"service": service})

    if json_logs:# in docker container, JSON format
        logger.add(
            sys.stdout,
            level=cfg["level"],
            serialize=True,
            colorize=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=cfg["level"],
            format=_FMT_CONSOLE,
            colorize=True,
        )

    if cfg.get("file"):# for human readable in log file
        logger.add(
            root / cfg["file"],
            level=cfg["level"],
            rotation=cfg["rotation"],
            retention=cfg["retention"],
            compression=cfg["compression"],
            format=_FMT_FILE,
            enqueue=True,
        )

    logger.info(f"logging configured (service={service}, json={json_logs})")


def setup_app_logging(config: dict, service: str = "app", json_logs: bool = None):
    cfg = config["logging"]["app"]
    json_logs = json_logs if json_logs is not None else cfg.get("json", False)
    _configure(cfg, service, json_logs)


def setup_ml_logging(config: dict, service: str = "ml", json_logs: bool = None):
    cfg = config["logging"]["ml"]
    json_logs = json_logs if json_logs is not None else cfg.get("json", False)
    _configure(cfg, service, json_logs)


class InterceptHandler(logging.Handler):
    '''Redirects stdlib logging (uvicorn, sqlalchemy, celery) into loguru.'''
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_back and "logging" in frame.f_code.co_filename:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def intercept_stdlib_logging(names: list[str] = None):
    names = names or ["uvicorn", "uvicorn.access", "uvicorn.error", "sqlalchemy", "celery"]
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in names:
        lg = logging.getLogger(name)
        lg.handlers = [InterceptHandler()]
        lg.propagate = False